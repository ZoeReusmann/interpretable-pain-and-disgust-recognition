from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
"""
Ziel: analyse der manuellen AU-Timeline-Annotationen des PFR-n40-Datensatzes.

Eingabe:
    pfr_n40_pressure_pain_vs_disgust_frontal.csv
    im Ordner data_analysis->outputs

Ausgabe (standardmäßig unter data_analysis/outputs/annotation_analysis/):
    class_video_counts.csv
    au_statistics_by_class.csv
    au_intensity_distribution_by_class.csv
    au_cooccurrence_by_class.csv
    au_duration_by_class.csv
    au_combinations_by_class.csv
    per_video_au_statistics.csv
    resolved_annotation_files.csv
    missing_or_ambiguous_annotation_files.csv
    summary.csv

Aktiv, wenn: Intensität > 0
"""

TARGET_AUS = ("AU04", "AU06", "AU09", "AU17", "AU25", "AU26")
DEFAULT_FPS = 25.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = SCRIPT_DIR / "pfr_n40_pressure_pain_vs_disgust_frontal.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "annotation_analysis"

# Suchverzeichnisse relativ zum Ordner von analyze_annotations.py.
DEFAULT_ANNOTATION_DIR_CANDIDATES = (
    SCRIPT_DIR / "au_annotations_n40",
    SCRIPT_DIR.parent / "au_annotations_n40",
    SCRIPT_DIR.parent / "PFR_dataset" / "au_annotations_n40",
    Path("/path/to/PFR_dataset/au_annotations_n40"),
)
SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysiert manuelle AU-Annotationen für Pressure Pain vs. Disgust."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Metadaten-CSV (Standard: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Wurzelordner der Timeline-Dateien. Falls nicht gesetzt: automatische Suche.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ausgabeordner (Standard: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Bildrate für Dauer in Sekunden (Standard: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Bei der ersten fehlenden/uneindeutigen/ungültigen Datei abbrechen.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\.[a-z0-9]{1,5}$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_au(value: object) -> str | None:
    text = str(value).upper().strip()
    match = re.search(r"\bAU[\s_\-]*0*(4|6|9|17|25|26)\b", text)
    if not match:
        match = re.fullmatch(r"0*(4|6|9|17|25|26)", re.sub(r"\D", "", text))
    if not match:
        return None
    return f"AU{int(match.group(1)):02d}"


def find_column(
    columns: Iterable[str],
    exact_candidates: Iterable[str],
    contains_candidates: Iterable[str] = (),
) -> str | None:
    normalized = {col: normalize_text(col) for col in columns}

    exact_set = {normalize_text(x) for x in exact_candidates}
    for col, norm in normalized.items():
        if norm in exact_set:
            return col

    contains_set = tuple(normalize_text(x) for x in contains_candidates)
    for col, norm in normalized.items():
        if any(token and token in norm for token in contains_set):
            return col

    return None


def detect_metadata_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)

    class_col = find_column(
        columns,
        exact_candidates=(
            "class", "label", "condition", "category", "target",
            "emotion", "stimulus_type", "video_class",
        ),
        contains_candidates=("class", "label", "condition", "emotion"),
    )

    video_col = find_column(
        columns,
        exact_candidates=(
            "video", "video_id", "video_name", "filename", "file_name",
            "video_filename", "video_path", "path",
        ),
        contains_candidates=("video", "filename", "file_name"),
    )

    subject_col = find_column(
        columns,
        exact_candidates=("subject", "subject_id", "vp_code", "participant", "participant_id"),
        contains_candidates=("subject", "vp_code", "participant"),
    )

    annotation_col = find_column(
        columns,
        exact_candidates=(
            "annotation_file", "annotation_path", "timeline_file",
            "timeline_path", "au_file", "au_path",
        ),
        contains_candidates=("annotation", "timeline"),
    )

    if class_col is None:
        raise ValueError(
            "Keine Klassenspalte gefunden. Erwartet z. B. 'class', 'label' oder 'condition'. "
            f"Vorhandene Spalten: {columns}"
        )
    if video_col is None:
        raise ValueError(
            "Keine Videospalte gefunden. Erwartet z. B. 'video', 'video_id', "
            f"'filename' oder 'video_path'. Vorhandene Spalten: {columns}"
        )

    return {
        "class": class_col,
        "video": video_col,
        "subject": subject_col,
        "annotation": annotation_col,
    }


def normalize_class_label(value: object) -> str:
    text = normalize_text(value)
    if any(token in text for token in ("pressure_pain", "druck_pain", "pressure", "druck", "pain")):
        return "pressure_pain"
    if any(token in text for token in ("disgust", "ekel", "bild_ekel")):
        return "disgust"
    return text or "unknown"


def discover_annotation_roots(explicit_root: Path | None) -> list[Path]:
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Annotationsordner existiert nicht: {root}")
        return [root]

    roots = []
    for candidate in DEFAULT_ANNOTATION_DIR_CANDIDATES:
        expanded = candidate.expanduser()
        if expanded.exists():
            roots.append(expanded.resolve())

    if not roots:
        searched = "\n  - ".join(str(p) for p in DEFAULT_ANNOTATION_DIR_CANDIDATES)
        raise FileNotFoundError(
            "Kein Ordner mit AU-Annotationen gefunden. Geprüft wurden:\n"
            f"  - {searched}\n"
            "Alternativ --annotation-dir /pfad/zu/au_annotations_n40 angeben."
        )

    return roots


def build_file_index(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved = path.resolve()
                if resolved not in seen:
                    files.append(resolved)
                    seen.add(resolved)
    return sorted(files)


def candidate_keys(row: pd.Series, columns: dict[str, str | None]) -> list[str]:
    keys: list[str] = []

    video_value = row[columns["video"]]
    video_path = Path(str(video_value))
    video_name = video_path.name
    video_stem = video_path.stem

    for value in (video_name, video_stem, video_value):
        norm = normalize_text(value)
        if norm and norm not in keys:
            keys.append(norm)

    subject_col = columns.get("subject")
    if subject_col:
        subject = normalize_text(row[subject_col])
        if subject:
            combined = normalize_text(f"{subject}_{video_stem}")
            if combined and combined not in keys:
                keys.append(combined)

    return keys


def resolve_annotation_file(
    row: pd.Series,
    columns: dict[str, str | None],
    all_files: list[Path],
    roots: list[Path],
) -> tuple[Path | None, str, list[str]]:
    annotation_col = columns.get("annotation")
    if annotation_col and pd.notna(row[annotation_col]):
        raw = Path(str(row[annotation_col])).expanduser()
        direct_candidates = [raw]
        if not raw.is_absolute():
            direct_candidates.extend(root / raw for root in roots)
            direct_candidates.append(SCRIPT_DIR / raw)

        existing = sorted({p.resolve() for p in direct_candidates if p.exists() and p.is_file()})
        if len(existing) == 1:
            return existing[0], "explicit_metadata_path", [str(existing[0])]
        if len(existing) > 1:
            return None, "ambiguous_explicit_path", [str(p) for p in existing]

    keys = candidate_keys(row, columns)
    if not keys:
        return None, "no_video_key", []

    scored: list[tuple[int, Path]] = []
    for path in all_files:
        stem = normalize_text(path.stem)
        full = normalize_text(path.as_posix())

        best_score = 0
        for key in keys:
            if stem == key:
                best_score = max(best_score, 100)
            elif stem.startswith(key + "_") or stem.endswith("_" + key):
                best_score = max(best_score, 90)
            elif key in stem:
                best_score = max(best_score, 80)
            elif key in full:
                best_score = max(best_score, 60)

        if best_score:
            scored.append((best_score, path))

    if not scored:
        return None, "missing", []

    max_score = max(score for score, _ in scored)
    best = sorted({path for score, path in scored if score == max_score})

    if len(best) == 1:
        return best[0], f"auto_match_score_{max_score}", [str(best[0])]

    return None, f"ambiguous_auto_match_score_{max_score}", [str(p) for p in best]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": "\t", "encoding": "utf-8-sig"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
    ]

    errors = []
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) > 1 or not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    raise ValueError(f"Datei konnte nicht gelesen werden: {path}. Fehler: {errors[-2:]}")


def numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        )
        values = pd.to_numeric(cleaned, errors="coerce")

    # Intensitäten außerhalb des plausiblen Bereichs werden als fehlend behandelt.
    return values.where(values.between(0, 5))


def detect_wide_au_columns(df: pd.DataFrame) -> dict[str, str]:
    candidates: dict[str, list[tuple[int, str]]] = {au: [] for au in TARGET_AUS}

    for col in df.columns:
        au = canonical_au(col)
        if au not in candidates:
            continue

        norm = normalize_text(col)
        score = 0
        if norm in {normalize_text(au), normalize_text(au.replace("0", "", 1))}:
            score = 100
        elif "intens" in norm:
            score = 90
        elif re.search(r"(^|_)au_?0?\d+($|_)", norm):
            score = 80
        else:
            score = 60

        candidates[au].append((score, col))

    result: dict[str, str] = {}
    for au, cols in candidates.items():
        if cols:
            cols.sort(key=lambda item: (-item[0], str(item[1])))
            result[au] = cols[0][1]

    return result


def timeline_to_wide(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"Leere Annotationstabelle: {path}")

    wide_columns = detect_wide_au_columns(df)
    if wide_columns:
        out = pd.DataFrame(index=np.arange(len(df)))
        for au in TARGET_AUS:
            if au in wide_columns:
                out[au] = numeric_series(df[wide_columns[au]])
            else:
                out[au] = 0.0
        return out.fillna(0.0)

    au_col = find_column(
        df.columns,
        exact_candidates=("au", "action_unit", "actionunit", "facs", "label"),
        contains_candidates=("action_unit", "actionunit"),
    )
    intensity_col = find_column(
        df.columns,
        exact_candidates=("intensity", "intensity_value", "value", "score", "level"),
        contains_candidates=("intens", "score"),
    )
    frame_col = find_column(
        df.columns,
        exact_candidates=("frame", "frame_id", "frame_number", "frame_no", "index"),
        contains_candidates=("frame",),
    )
    time_col = find_column(
        df.columns,
        exact_candidates=("time", "time_s", "timestamp", "seconds", "sec"),
        contains_candidates=("timestamp", "time"),
    )

    if au_col is None or intensity_col is None:
        raise ValueError(
            f"AU-Spalten konnten in {path} nicht erkannt werden. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    long_df = df.copy()
    long_df["_au"] = long_df[au_col].map(canonical_au)
    long_df = long_df[long_df["_au"].isin(TARGET_AUS)].copy()
    long_df["_intensity"] = numeric_series(long_df[intensity_col]).fillna(0.0)

    if long_df.empty:
        # Datei ist formal lesbar, enthält aber keine Ziel-AUs.
        return pd.DataFrame({au: np.zeros(len(df), dtype=float) for au in TARGET_AUS})

    if frame_col:
        long_df["_frame_key"] = long_df[frame_col]
    elif time_col:
        long_df["_frame_key"] = long_df[time_col]
    else:
        # Falls jede Zeile eine AU-Beobachtung ist, aber keine Frame-Spalte existiert, wird die ursprüngliche Zeilennummer verwendet.
        long_df["_frame_key"] = long_df.index

    pivot = long_df.pivot_table(
        index="_frame_key",
        columns="_au",
        values="_intensity",
        aggfunc="max",
        fill_value=0.0,
    ).sort_index()

    for au in TARGET_AUS:
        if au not in pivot.columns:
            pivot[au] = 0.0

    return pivot.loc[:, TARGET_AUS].reset_index(drop=True)


def active_runs(active: np.ndarray) -> list[int]:
    """Längen zusammenhängender True-Sequenzen."""
    if active.size == 0:
        return []

    padded = np.concatenate(([False], active.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return (ends - starts).astype(int).tolist()

def analyse_video(
    wide: pd.DataFrame,
    video_id: str,
    class_label: str,
    annotation_path: Path,
    fps: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    Counter[tuple[str, ...]],
    dict[tuple[str, str], dict[str, int]],
]:
    n_frames = len(wide)
    per_video_rows: list[dict[str, object]] = []
    duration_rows: list[dict[str, object]] = []

    active_matrix = wide.loc[:, TARGET_AUS].to_numpy(dtype=float) > 0
    intensity_matrix = wide.loc[:, TARGET_AUS].to_numpy(dtype=float)

    for idx, au in enumerate(TARGET_AUS):
        intensities = intensity_matrix[:, idx]
        active = active_matrix[:, idx]
        active_values = intensities[active]
        runs = active_runs(active)

        per_video_rows.append(
            {
                "class": class_label,
                "video_id": video_id,
                "annotation_file": str(annotation_path),
                "au": au,
                "total_frames": n_frames,
                "active_frames": int(active.sum()),
                "active_frame_share": float(active.mean()) if n_frames else np.nan,
                "video_has_au": int(active.any()),
                "mean_active_intensity": float(active_values.mean()) if active_values.size else np.nan,
                "max_intensity": float(intensities.max()) if intensities.size else np.nan,
                "activation_count": len(runs),
                "mean_activation_duration_frames": float(np.mean(runs)) if runs else np.nan,
                "mean_activation_duration_seconds": float(np.mean(runs) / fps) if runs else np.nan,
            }
        )

        for run_index, frames in enumerate(runs, start=1):
            duration_rows.append(
                {
                    "class": class_label,
                    "video_id": video_id,
                    "au": au,
                    "activation_index": run_index,
                    "duration_frames": int(frames),
                    "duration_seconds": float(frames / fps),
                }
            )

    combinations: Counter[tuple[str, ...]] = Counter()
    for frame_active in active_matrix:
        combination = tuple(au for au, is_active in zip(TARGET_AUS, frame_active) if is_active)
        if combination:
            combinations[combination] += 1

    cooccurrence: dict[tuple[str, str], dict[str, int]] = {}
    for i, j in itertools.combinations(range(len(TARGET_AUS)), 2):
        pair = (TARGET_AUS[i], TARGET_AUS[j])
        both = active_matrix[:, i] & active_matrix[:, j]
        cooccurrence[pair] = {
            "coactive_frames": int(both.sum()),
            "video_has_cooccurrence": int(both.any()),
        }

    return per_video_rows, duration_rows, combinations, cooccurrence


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.replace(0, np.nan).astype(float))


def aggregate_statistics(
    per_video: pd.DataFrame,
    timelines: list[tuple[str, str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_video_counts = (
        per_video[["class", "video_id"]]
        .drop_duplicates()
        .groupby("class", as_index=False)
        .agg(number_of_videos=("video_id", "nunique"))
        .sort_values("class")
    )
    rows = []
    for (class_label, au), group in per_video.groupby(["class", "au"], sort=True):
        total_frames = int(group["total_frames"].sum())
        active_frames = int(group["active_frames"].sum())
        active_intensities: list[np.ndarray] = []

        for timeline_class, _, wide in timelines:
            if timeline_class != class_label:
                continue
            values = wide[au].to_numpy(dtype=float)
            active_intensities.append(values[values > 0])

        concatenated = (
            np.concatenate(active_intensities)
            if active_intensities and any(arr.size for arr in active_intensities)
            else np.array([], dtype=float)
        )

        rows.append(
            {
                "class": class_label,
                "au": au,
                "number_of_videos": int(group["video_id"].nunique()),
                "total_frames": total_frames,
                "active_frames": active_frames,
                "active_frame_share": active_frames / total_frames if total_frames else np.nan,
                "videos_with_au": int(group["video_has_au"].sum()),
                "video_share_with_au": (
                    float(group["video_has_au"].mean()) if len(group) else np.nan
                ),
                "mean_active_intensity": (
                    float(concatenated.mean()) if concatenated.size else np.nan
                ),
                "max_intensity": (
                    float(concatenated.max()) if concatenated.size else 0.0
                ),
                "number_of_activations": int(group["activation_count"].sum()),
                "mean_activation_duration_frames": (
                    float(
                        np.average(
                            group.loc[group["activation_count"] > 0, "mean_activation_duration_frames"],
                            weights=group.loc[group["activation_count"] > 0, "activation_count"],
                        )
                    )
                    if group["activation_count"].sum() > 0
                    else np.nan
                ),
                "mean_activation_duration_seconds": (
                    float(
                        np.average(
                            group.loc[group["activation_count"] > 0, "mean_activation_duration_seconds"],
                            weights=group.loc[group["activation_count"] > 0, "activation_count"],
                        )
                    )
                    if group["activation_count"].sum() > 0
                    else np.nan
                ),
            }
        )

    return class_video_counts, pd.DataFrame(rows)


def intensity_distribution(
    timelines: list[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    classes = sorted({class_label for class_label, _, _ in timelines})
    for class_label in classes:
        class_timelines = [wide for cls, _, wide in timelines if cls == class_label]
        for au in TARGET_AUS:
            arrays = []
            for wide in class_timelines:
                values = wide[au].to_numpy(dtype=float)
                arrays.append(values[values > 0])

            values = (
                np.concatenate(arrays)
                if arrays and any(arr.size for arr in arrays)
                else np.array([], dtype=float)
            )

            # Wissen: Manuelle Annotationen sollten diskrete Intensitäten 1–5 besitzen (gehe von PFR aus)
            # Falls es Dezimalwerte gibt: auf nächste ordinale Stufe runden
            rounded = np.rint(values).astype(int) if values.size else np.array([], dtype=int)
            total_active = len(rounded)

            for intensity in range(1, 6):
                count = int((rounded == intensity).sum())
                rows.append(
                    {   "class": class_label,
                        "au": au,
                        "intensity": intensity,
                        "frame_count": count,
                        "share_of_active_frames": count / total_active if total_active else np.nan,
                    }
                )

    return pd.DataFrame(rows)

def aggregate_cooccurrences(
    cooccurrence_rows: list[dict[str, object]],
    class_frame_totals: dict[str, int],
    class_video_totals: dict[str, int],
) -> pd.DataFrame:
    df = pd.DataFrame(cooccurrence_rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "class", "au_1", "au_2", "coactive_frames",
                "share_of_class_frames", "videos_with_cooccurrence",
                "share_of_class_videos",
            ]
        )

    agg = (
        df.groupby(["class", "au_1", "au_2"], as_index=False)
        .agg(
            coactive_frames=("coactive_frames", "sum"),
            videos_with_cooccurrence=("video_has_cooccurrence", "sum"),
        )
    )
    agg["share_of_class_frames"] = agg.apply(
        lambda row: (
            row["coactive_frames"] / class_frame_totals[row["class"]]
            if class_frame_totals.get(row["class"], 0)
            else np.nan
        ),
        axis=1,
    )
    agg["share_of_class_videos"] = agg.apply(
        lambda row: (
            row["videos_with_cooccurrence"] / class_video_totals[row["class"]]
            if class_video_totals.get(row["class"], 0)
            else np.nan
        ),
        axis=1,
    )
    return agg.sort_values(
        ["class", "coactive_frames", "au_1", "au_2"],
        ascending=[True, False, True, True],
    )


def aggregate_combinations(
    combination_rows: list[dict[str, object]],
    class_frame_totals: dict[str, int],
    class_video_totals: dict[str, int],
) -> pd.DataFrame:
    df = pd.DataFrame(combination_rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "class", "combination", "number_of_aus",
                "active_frames", "share_of_class_frames",
                "videos_with_combination", "share_of_class_videos",
                "rank_within_class",
            ]
        )

    agg = (
        df.groupby(["class", "combination", "number_of_aus"], as_index=False)
        .agg(
            active_frames=("active_frames", "sum"),
            videos_with_combination=("video_has_combination", "sum"),
        )
    )
    agg["share_of_class_frames"] = agg.apply(
        lambda row: (
            row["active_frames"] / class_frame_totals[row["class"]]
            if class_frame_totals.get(row["class"], 0)
            else np.nan
        ),
        axis=1,
    )
    agg["share_of_class_videos"] = agg.apply(
        lambda row: (
            row["videos_with_combination"] / class_video_totals[row["class"]]
            if class_video_totals.get(row["class"], 0)
            else np.nan
        ),
        axis=1,
    )
    agg = agg.sort_values(
        ["class", "active_frames", "number_of_aus", "combination"],
        ascending=[True, False, False, True],
    )
    agg["rank_within_class"] = agg.groupby("class").cumcount() + 1
    return agg


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps muss größer als 0 sein.")

    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadaten-Datei nicht gefunden: {metadata_path}")

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    columns = detect_metadata_columns(metadata)

    metadata = metadata.copy()
    metadata["_class"] = metadata[columns["class"]].map(normalize_class_label)
    metadata["_video_id"] = metadata[columns["video"]].map(
        lambda x: normalize_text(Path(str(x)).stem)
    )

    relevant_classes = {"pressure_pain", "disgust"}
    unknown_classes = sorted(set(metadata["_class"]) - relevant_classes)
    if unknown_classes:
        print(
            "Warnung: Folgende Klassen werden ignoriert, da sie nicht als "
            f"Pressure Pain oder Disgust erkannt wurden: {unknown_classes}",
            file=sys.stderr,
        )

    metadata = metadata[metadata["_class"].isin(relevant_classes)].copy()
    if metadata.empty:
        raise ValueError("Keine Pressure-Pain- oder Disgust-Zeilen in den Metadaten gefunden.")

    roots = discover_annotation_roots(args.annotation_dir)
    all_annotation_files = build_file_index(roots)
    if not all_annotation_files:
        raise FileNotFoundError(
            "In den gefundenen Annotationsordnern wurden keine unterstützten "
            "Timeline-Dateien gefunden."
        )

    print(f"Metadaten: {metadata_path}")
    print(f"Zeilen: {len(metadata)}")
    print(f"Annotationsdateien im Index: {len(all_annotation_files)}")
    print(f"Ausgabe: {output_dir}")

    resolution_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    duration_rows: list[dict[str, object]] = []
    cooccurrence_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    timelines: list[tuple[str, str, pd.DataFrame]] = []

    for row_number, (_, row) in enumerate(metadata.iterrows(), start=1):
        video_id = row["_video_id"]
        class_label = row["_class"]

        path, status, candidates = resolve_annotation_file(
            row=row,
            columns=columns,
            all_files=all_annotation_files,
            roots=roots,
        )

        if path is None:
            problem = {
                "metadata_row": row_number,
                "video_id": video_id,
                "class": class_label,
                "status": status,
                "candidate_files": " | ".join(candidates),
            }
            problem_rows.append(problem)
            print(
                f"[{row_number}/{len(metadata)}] Problem: {video_id} -> {status}",
                file=sys.stderr,
            )
            if args.strict:
                raise RuntimeError(str(problem))
            continue

        try:
            raw = read_table(path)
            wide = timeline_to_wide(raw, path)
        except Exception as exc:  # noqa: BLE001
            problem = {
                "metadata_row": row_number,
                "video_id": video_id,
                "class": class_label,
                "status": "read_or_parse_error",
                "candidate_files": str(path),
                "error": str(exc),
            }
            problem_rows.append(problem)
            print(
                f"[{row_number}/{len(metadata)}] Fehler: {video_id}: {exc}",
                file=sys.stderr,
            )
            if args.strict:
                raise
            continue

        resolution_rows.append(
            {
                "metadata_row": row_number,
                "video_id": video_id,
                "class": class_label,
                "annotation_file": str(path),
                "resolution_status": status,
                "timeline_frames": len(wide),
            }
        )
        timelines.append((class_label, video_id, wide))

        video_stats, video_durations, combinations, cooccurrences = analyse_video(
            wide=wide,
            video_id=video_id,
            class_label=class_label,
            annotation_path=path,
            fps=args.fps,
        )
        per_video_rows.extend(video_stats)
        duration_rows.extend(video_durations)

        for (au_1, au_2), values in cooccurrences.items():
            cooccurrence_rows.append(
                {
                    "class": class_label,
                    "video_id": video_id,
                    "au_1": au_1,
                    "au_2": au_2,
                    **values,
                }
            )

        for combination, frame_count in combinations.items():
            combination_rows.append(
                {
                    "class": class_label,
                    "video_id": video_id,
                    "combination": "+".join(combination),
                    "number_of_aus": len(combination),
                    "active_frames": int(frame_count),
                    "video_has_combination": 1,
                }
            )

        print(
            f"[{row_number}/{len(metadata)}] OK: {class_label} | {video_id} | "
            f"{len(wide)} Frames"
        )

    resolved_df = pd.DataFrame(resolution_rows)
    problems_df = pd.DataFrame(problem_rows)
    per_video_df = pd.DataFrame(per_video_rows)
    durations_df = pd.DataFrame(duration_rows)

    write_csv(resolved_df, output_dir / "resolved_annotation_files.csv")
    write_csv(problems_df, output_dir / "missing_or_ambiguous_annotation_files.csv")

    if per_video_df.empty:
        raise RuntimeError(
            "Keine Timeline-Datei konnte erfolgreich analysiert werden. "
            f"Details: {output_dir / 'missing_or_ambiguous_annotation_files.csv'}"
        )

    class_video_counts, au_stats = aggregate_statistics(per_video_df, timelines)
    intensity_df = intensity_distribution(timelines)

    class_frame_totals = {
        class_label: int(sum(len(wide) for cls, _, wide in timelines if cls == class_label))
        for class_label in sorted({cls for cls, _, _ in timelines})
    }
    class_video_totals = (
        resolved_df.groupby("class")["video_id"].nunique().astype(int).to_dict()
    )

    cooccurrence_df = aggregate_cooccurrences(
        cooccurrence_rows,
        class_frame_totals=class_frame_totals,
        class_video_totals=class_video_totals,
    )
    combinations_df = aggregate_combinations(
        combination_rows,
        class_frame_totals=class_frame_totals,
        class_video_totals=class_video_totals,
    )

# Excel Dateien modular :)
    write_csv(class_video_counts, output_dir / "class_video_counts.csv")
    write_csv(au_stats, output_dir / "au_statistics_by_class.csv")
    write_csv(intensity_df, output_dir / "au_intensity_distribution_by_class.csv")
    write_csv(cooccurrence_df, output_dir / "au_cooccurrence_by_class.csv")
    write_csv(durations_df, output_dir / "au_duration_by_class.csv")
    write_csv(combinations_df, output_dir / "au_combinations_by_class.csv")
    write_csv(per_video_df, output_dir / "per_video_au_statistics.csv")
    print("\nFertig.")
    print(f"Erfolgreich analysierte Videos: {resolved_df['video_id'].nunique()}")
    print(f"Problemfälle: {len(problems_df)}")
    print("\nVideos pro Klasse:")
    print(class_video_counts.to_string(index=False))
    print(f"\nErgebnisse: {output_dir}")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAbgebrochen.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        raise SystemExit(1)
