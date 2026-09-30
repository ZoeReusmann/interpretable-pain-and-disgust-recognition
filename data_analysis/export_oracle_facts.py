from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
"""
Ziel: Exportiert manuell annotierte PFR-n40-AUs als Oracle-Fakten für FastLAS/ILP
Selbes Setup wie vorher auch, neuer Versuch mit den Oracle Dateien
Oracle Dateien: manuell annotierte PFR 40N Datensatz Daten

Die erzeugte Syntax entspricht den Fakten und Beispielen aus
task_pain_combined.las - damit keine Fehler auftauchen (wg alter Syntax Fehler)

    video(vp000001_001).
    frame(vp000001_001,0).
    original_frame(vp000001_001,0,38).
    active(vp000001_001,au4,0).

    #pos(ex_vp000001_001, {pain(vp000001_001)}, {}, {}).
    #neg(ex_vp000001_005, {pain(vp000001_005)}, {}, {}).

Für einen fairen Vergleich mit den CNN-Fakten wird standardmäßig nur das
bereits verwendete Zeitfenster der Originalframes 38 bis 212 exportiert
Die exportierten Frame-IDs beginnen pro Video wieder bei 0

Der bestehende Train-/Test-Split wird aus diesen Dateien übernommen:

    ILP/data/pfr_facts_train.pl
    ILP/data/pfr_facts_test.pl

Ausgaben:

    ILP/oracle/data/pfr_oracle_facts_train.pl
    ILP/oracle/data/pfr_oracle_facts_test.pl

    ILP/oracle/examples/examples_pain_oracle_train.las
    ILP/oracle/examples/examples_pain_oracle_test.las
    ILP/oracle/examples/examples_disgust_oracle_train.las
    ILP/oracle/examples/examples_disgust_oracle_test.las

    ILP/oracle/oracle_export_summary.csv
    ILP/oracle/missing_or_ambiguous_oracle_files.csv
"""

# beide Skripte müssen im Ordner data_analysis liegen
from analyze_annotations import (
    TARGET_AUS,
    build_file_index,
    detect_metadata_columns,
    discover_annotation_roots,
    normalize_class_label,
    normalize_text,
    read_table,
    resolve_annotation_file,
    timeline_to_wide,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_METADATA = (
    SCRIPT_DIR / "outputs" / "pfr_n40_pressure_pain_vs_disgust_frontal.csv"
)
DEFAULT_TRAIN_REFERENCE = PROJECT_ROOT / "ILP" / "data" / "pfr_facts_train.pl"
DEFAULT_TEST_REFERENCE = PROJECT_ROOT / "ILP" / "data" / "pfr_facts_test.pl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ILP" / "oracle"
# Entspricht dem bereits für die CNN-Fakten verwendeten Ausschnitt
DEFAULT_FIRST_ORIGINAL_FRAME = 38
DEFAULT_LAST_ORIGINAL_FRAME = 212

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exportiert manuelle PFR-n40-Annotationen als Oracle-Fakten "
            "und FastLAS-Beispiele."
        )
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
        help=(
            "Wurzelordner der Timeline-Dateien. Falls nicht angegeben, "
            "wird derselbe automatische Suchmechanismus wie bei "
            "analyze_annotations.py verwendet."
        ),
    )
    parser.add_argument(
        "--train-reference",
        type=Path,
        default=DEFAULT_TRAIN_REFERENCE,
        help=(
            "Bestehende Trainings-Faktendatei, aus der die Videos des "
            "Train-Splits übernommen werden."
        ),
    )
    parser.add_argument(
        "--test-reference",
        type=Path,
        default=DEFAULT_TEST_REFERENCE,
        help=(
            "Bestehende Test-Faktendatei, aus der die Videos des "
            "Test-Splits übernommen werden."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Ausgabeordner (Standard: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--first-frame",
        type=int,
        default=DEFAULT_FIRST_ORIGINAL_FRAME,
        help=(
            "Erster zu exportierender Originalframe, inklusive "
            f"(Standard: {DEFAULT_FIRST_ORIGINAL_FRAME})."
        ),
    )
    parser.add_argument(
        "--last-frame",
        type=int,
        default=DEFAULT_LAST_ORIGINAL_FRAME,
        help=(
            "Letzter zu exportierender Originalframe, inklusive "
            f"(Standard: {DEFAULT_LAST_ORIGINAL_FRAME})."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Beim ersten Problemfall abbrechen.",
    )
    return parser.parse_args()

def read_video_ids_from_facts(path: Path) -> set[str]:
    """Liest alle video(...).-Fakten aus einer bestehenden Faktendatei."""
    if not path.exists():
        raise FileNotFoundError(f"Split-Referenz nicht gefunden: {path}")

    video_pattern = re.compile(r"^\s*video\(([a-zA-Z0-9_]+)\)\.\s*$")
    videos: set[str] = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = video_pattern.match(line)
            if match:
                videos.add(match.group(1).lower())

    if not videos:
        raise ValueError(f"Keine video(...).-Fakten in {path} gefunden.")

    return videos

def determine_split(
    video_id: str,
    train_videos: set[str],
    test_videos: set[str],
) -> str | None:
    """Ordnet ein Video eindeutig dem bestehenden Train- oder Test-Split zu."""
    in_train = video_id in train_videos
    in_test = video_id in test_videos

    if in_train and not in_test:
        return "train"
    if in_test and not in_train:
        return "test"

    return None

def select_frame_window(
    timeline: pd.DataFrame,
    first_original_frame: int,
    last_original_frame: int,
) -> pd.DataFrame:
    """
    Schneidet das gewünschte Originalframe-Fenster aus
    Die Timeline-Dateien besitzen 351 Zeilen. Dabei wird die erste Zeile als
    Originalframe 0 interpretiert. Für das Fenster 38 bis 212 entstehen somit
    175 exportierte Frames mit neuen Frame-IDs 0 bis 174

    Damit passt es wieder zum alten Aufbau und kann miteinander verglichen werden
    """
    if first_original_frame < 0:
        raise ValueError("--first-frame darf nicht negativ sein.")
    if last_original_frame < first_original_frame:
        raise ValueError("--last-frame muss >= --first-frame sein.")
    if last_original_frame >= len(timeline):
        raise ValueError(
            f"Timeline hat nur {len(timeline)} Frames, angefordert wurde aber "
            f"Originalframe {last_original_frame}."
        )

    selected = timeline.iloc[
        first_original_frame : last_original_frame + 1
    ].copy()
    selected.reset_index(drop=True, inplace=True)
    return selected


def au_atom(au: str) -> str:
    """Wandelt AU04 in das in task_pain_combined.las verwendete au4 um."""
    return f"au{int(au.replace('AU', ''))}"


def create_fact_lines(
    video_id: str,
    timeline: pd.DataFrame,
    first_original_frame: int,
) -> list[str]:
    """
    Erstellt FastLAS-kompatible Fakten

    Eine AU gilt genau dann als aktiv, wenn ihre manuell annotierte
    Intensität größer als 0 ist
    """
    lines = [f"video({video_id})."]

    for frame_id, row in timeline.iterrows():
        original_frame = first_original_frame + int(frame_id)

        lines.append(f"frame({video_id},{frame_id}).")
        lines.append(
            f"original_frame({video_id},{frame_id},{original_frame})."
        )

        for au in TARGET_AUS:
            value = pd.to_numeric(row[au], errors="coerce")
            if pd.notna(value) and float(value) > 0:
                lines.append(
                    f"active({video_id},{au_atom(au)},{frame_id})."
                )

    return lines


def create_example_line(
    video_id: str,
    class_label: str,
    target: str,
) -> str:
    """
    Erstellt Beispiele exakt im Format der bestehenden Combined-Task.

    Für pain sind Pressure-Pain-Videos positiv und Disgust-Videos negativ, oder halt andersherum
    """
    if target not in {"pain", "disgust"}:
        raise ValueError(f"Unbekanntes Target: {target}")

    is_positive = (
        class_label == "pressure_pain"
        if target == "pain"
        else class_label == "disgust"
    )
    example_type = "#pos" if is_positive else "#neg"

    return (
        f"{example_type}(ex_{video_id}, "
        f"{{{target}({video_id})}}, {{}}, {{}})."
    )


def write_lines(path: Path, lines: list[str]) -> None:
    """Schreibt eine Textdatei mit genau einem abschließenden Zeilenumbruch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    train_reference = args.train_reference.expanduser().resolve()
    test_reference = args.test_reference.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadaten-Datei nicht gefunden: {metadata_path}")

    train_videos = read_video_ids_from_facts(train_reference)
    test_videos = read_video_ids_from_facts(test_reference)

    overlap = train_videos & test_videos
    if overlap:
        raise ValueError(
            f"{len(overlap)} Videos kommen sowohl im Train- als auch im "
            "Test-Split vor. Der bestehende Split ist nicht eindeutig."
        )

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    columns = detect_metadata_columns(metadata)

    metadata = metadata.copy()
    metadata["_class"] = metadata[columns["class"]].map(normalize_class_label)
    metadata["_video_id"] = metadata[columns["video"]].map(
        lambda value: normalize_text(Path(str(value)).stem)
    )
    metadata = metadata[
        metadata["_class"].isin({"pressure_pain", "disgust"})
    ].copy()

    if metadata.empty:
        raise ValueError(
            "Keine Pressure-Pain- oder Disgust-Videos in den Metadaten gefunden."
        )

    annotation_roots = discover_annotation_roots(args.annotation_dir)
    annotation_files = build_file_index(annotation_roots)

    if not annotation_files:
        raise FileNotFoundError(
            "Keine unterstützten Timeline-Dateien im Annotationsordner gefunden."
        )

    fact_lines: dict[str, list[str]] = {"train": [], "test": []}
    pain_examples: dict[str, list[str]] = {"train": [], "test": []}
    disgust_examples: dict[str, list[str]] = {"train": [], "test": []}

    summary_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []

    total_rows = len(metadata)

    for number, (_, row) in enumerate(metadata.iterrows(), start=1):
        video_id = row["_video_id"]
        class_label = row["_class"]

        split = determine_split(video_id, train_videos, test_videos)
        if split is None:
            problem = {
                "metadata_row": number,
                "video_id": video_id,
                "class": class_label,
                "status": "video_not_uniquely_assigned_to_existing_split",
                "candidate_files": "",
                "error": "",
            }
            problem_rows.append(problem)
            print(
                f"[{number}/{total_rows}] Problem: {video_id} -> kein "
                "eindeutiger Train/Test-Split",
                file=sys.stderr,
            )
            if args.strict:
                raise RuntimeError(str(problem))
            continue

        annotation_path, resolution_status, candidates = resolve_annotation_file(
            row=row,
            columns=columns,
            all_files=annotation_files,
            roots=annotation_roots,
        )

        if annotation_path is None:
            problem = {
                "metadata_row": number,
                "video_id": video_id,
                "class": class_label,
                "status": resolution_status,
                "candidate_files": " | ".join(candidates),
                "error": "",
            }
            problem_rows.append(problem)
            print(
                f"[{number}/{total_rows}] Problem: {video_id} -> "
                f"{resolution_status}",
                file=sys.stderr,
            )
            if args.strict:
                raise RuntimeError(str(problem))
            continue

        try:
            raw_timeline = read_table(annotation_path)
            full_timeline = timeline_to_wide(raw_timeline, annotation_path)
            selected_timeline = select_frame_window(
                full_timeline,
                first_original_frame=args.first_frame,
                last_original_frame=args.last_frame,
            )
        except Exception as exc:  # noqa: BLE001
            problem = {
                "metadata_row": number,
                "video_id": video_id,
                "class": class_label,
                "status": "read_parse_or_window_error",
                "candidate_files": str(annotation_path),
                "error": str(exc),
            }
            problem_rows.append(problem)
            print(
                f"[{number}/{total_rows}] Fehler: {video_id}: {exc}",
                file=sys.stderr,
            )
            if args.strict:
                raise
            continue

        video_fact_lines = create_fact_lines(
            video_id=video_id,
            timeline=selected_timeline,
            first_original_frame=args.first_frame,
        )
        fact_lines[split].extend(video_fact_lines)
        fact_lines[split].append("")

        pain_examples[split].append(
            create_example_line(video_id, class_label, target="pain")
        )
        disgust_examples[split].append(
            create_example_line(video_id, class_label, target="disgust")
        )

        active_fact_count = sum(
            line.startswith("active(") for line in video_fact_lines
        )

        summary_rows.append(
            {
                "video_id": video_id,
                "class": class_label,
                "split": split,
                "annotation_file": str(annotation_path),
                "full_timeline_frames": len(full_timeline),
                "exported_frames": len(selected_timeline),
                "first_original_frame": args.first_frame,
                "last_original_frame": args.last_frame,
                "active_facts": active_fact_count,
                "resolution_status": resolution_status,
            }
        )

        print(
            f"[{number}/{total_rows}] OK: {split} | {class_label} | "
            f"{video_id} | {len(selected_timeline)} Frames | "
            f"{active_fact_count} active-Fakten"
        )

    # Leere Trennzeile am Ende der Faktendateien entfernen.
    for split in ("train", "test"):
        while fact_lines[split] and fact_lines[split][-1] == "":
            fact_lines[split].pop()

    data_dir = output_root / "data"
    examples_dir = output_root / "examples"

    write_lines(
        data_dir / "pfr_oracle_facts_train.pl",
        fact_lines["train"],
    )
    write_lines(
        data_dir / "pfr_oracle_facts_test.pl",
        fact_lines["test"],
    )

    write_lines(
        examples_dir / "examples_pain_oracle_train.las",
        pain_examples["train"],
    )
    write_lines(
        examples_dir / "examples_pain_oracle_test.las",
        pain_examples["test"],
    )
    write_lines(
        examples_dir / "examples_disgust_oracle_train.las",
        disgust_examples["train"],
    )
    write_lines(
        examples_dir / "examples_disgust_oracle_test.las",
        disgust_examples["test"],
    )

    summary_df = pd.DataFrame(summary_rows)
    problems_df = pd.DataFrame(problem_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        output_root / "oracle_export_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    problems_df.to_csv(
        output_root / "missing_or_ambiguous_oracle_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if summary_df.empty:
        raise RuntimeError(
            "Es konnte kein Oracle-Video exportiert werden. "
            "Siehe missing_or_ambiguous_oracle_files.csv."
        )

    print("\nOracle-Export abgeschlossen.")
    print(f"Exportierte Videos insgesamt: {len(summary_df)}")
    print(f"Problemfälle: {len(problems_df)}")

    print("\nVideos nach Split und Klasse:")
    print(
        summary_df.groupby(["split", "class"])
        .size()
        .rename("videos")
        .reset_index()
        .to_string(index=False)
    )

    print("\nErzeugte Dateien:")
    for path in (
        data_dir / "pfr_oracle_facts_train.pl",
        data_dir / "pfr_oracle_facts_test.pl",
        examples_dir / "examples_pain_oracle_train.las",
        examples_dir / "examples_pain_oracle_test.las",
        examples_dir / "examples_disgust_oracle_train.las",
        examples_dir / "examples_disgust_oracle_test.las",
        output_root / "oracle_export_summary.csv",
        output_root / "missing_or_ambiguous_oracle_files.csv",
    ):
        print(f"  {path}")

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
