from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

"""
Ziel: Zählt pro Action Unit, wie oft in den PainFaceReader-Oracle-Daten ein
kurzer Rückfall auf Intensität 0 zwischen zwei Aktivierungen auftritt (z. B.
2 → 0 → 3 → 1 innerhalb weniger Frames). Das dokumentiert, ob und wie oft
dieses Muster vorkommt - als Hinweis auf Kohärenzverlust durch die
randomisierte Frame-Annotation. Die Rohdaten werden dabei nicht verändert.
Standardmäßig werden nur Pain- (druck_p) und Disgust-Videos (ekel) einbezogen
- welches Video welche Bedingung ist, wird aus den subjective_ratings_all_n168
CSVs abgeleitet (Spalte "art", Reihenfolge = Videonummer 001..084 pro VP).

Ausführung:
    python au_diagnostics.py
    python au_diagnostics.py --input /pfad/zu/au_annotations_n40 --output ./flicker_summary.csv
    python au_diagnostics.py --conditions druck_p ekel heat_p
    python au_diagnostics.py --conditions   # leer = kein Filter, alle Videos

Input: Ordner mit den au_annotations_n40 CSVs (eine Timeline-Datei pro Video,
25 fps, eine Spalte pro AU mit Intensitätswerten). Für den Bedingungsfilter
zusätzlich der Ratings-Ordner (subjective_ratings_all_n168), standardmäßig
relativ zu --input erwartet unter ../Ratings/subjective_ratings_all_n168.

Output: eine CSV (flicker_summary.csv) mit einer Zeile pro AU - Anzahl
gefundener Rückfälle (n_flicker_events), Anzahl betroffener Videos, Anzahl
insgesamt analysierter Aktivierungen (n_bursts) und die daraus resultierende
Rate. Plus eine TOTAL-Zeile über alle AUs zusammen.

WICHTIG: Die Zuordnung Video-Nummer -> Bedingung nimmt an, dass die
Reihenfolge der (Nicht-Trainings-)Trials in der Ratings-CSV pro VP identisch
zur Videonummerierung 001..084 ist. Das Skript meldet am Ende, bei wie vielen
Dateien keine passende Zuordnung gefunden wurde - bei einer auffällig hohen
Zahl sollte diese Annahme an ein paar Videos manuell gegengeprüft werden.
"""

# Konfiguration / Defaults

DEFAULT_INPUT_DIR = (
    "/path/to/PFR_dataset/au_annotations_n40"
)
DEFAULT_OUTPUT_CSV = "./flicker_summary.csv"
DEFAULT_CONDITIONS = ["druck_p", "ekel"]  # Pain (Druck) und Disgust (Ekel)

# AUs, die standardmäßig aus der Analyse ausgeschlossen werden, weil sie aus
# anderen Gründen als Kodier-Rauschen naturgemäß sehr kurze, schnelle An/Aus-
# Muster zeigen (Blinzeln, Blickrichtung) und die Ergebnisse verzerren würden.
# Bei Bedarf anpassen/leeren.
DEFAULT_EXCLUDE_AUS = {"AU.45", "AU.63"}  # AU45 = Blink, AU63 = Blickrichtung

FILENAME_RE = re.compile(r"Block\s+(\d+)\s*-\s*(.+?)_timeline\.csv$", re.IGNORECASE)
RATINGS_FILENAME_RE = re.compile(r"Ekel_([A-Za-z0-9]+)_", re.IGNORECASE)
TRAINING_RE = re.compile(r"train", re.IGNORECASE)


# Kernlogik

def find_au_columns(columns, exclude_aus):
    au_cols = []
    for c in columns:
        name = str(c).strip().strip('"')
        if re.match(r"^AU\.\d+$", name) and name not in exclude_aus:
            au_cols.append(c)
    return au_cols


def parse_filename(path: Path):
    """Gibt recording_id (z.B. 'ANN050628_001') aus dem Dateinamen zurück."""
    m = FILENAME_RE.search(path.name)
    return m.group(2) if m else path.stem


def split_recording_id(recording_id: str):
    """Trennt 'ANN050628_001' in VP-Code und Videonummer (int)."""
    vp_code, _, video_str = recording_id.rpartition("_")
    if vp_code and video_str.isdigit():
        return vp_code, int(video_str)
    return recording_id, None


def load_condition_lookup(ratings_dir: Path):
    """Baut ein Dict {(vp_code, video_num): art} aus den subjective_ratings
    CSVs. Videonummer = Position des Trials in der Datei, nachdem Trainings-
    Trials rausgefiltert wurden (siehe Docstring oben zur Annahme dahinter)."""
    lookup = {}
    files = sorted(ratings_dir.glob("*.csv"))
    for f in files:
        m = RATINGS_FILENAME_RE.search(f.name)
        if not m:
            continue
        vp_code = m.group(1)
        try:
            df = pd.read_csv(f, sep=";", encoding="utf-8-sig", usecols=["art"])
        except Exception:
            continue
        arts = df["art"].dropna().astype(str)
        arts = arts[~arts.str.contains(TRAINING_RE)]
        for video_num, art in enumerate(arts, start=1):
            lookup[(vp_code, video_num)] = art
    return lookup


def find_raw_runs(nonzero_mask: np.ndarray):
    """Gibt Liste von (start_idx, end_idx_inclusive) für zusammenhängende
    True-Läufe in nonzero_mask zurück - die unveränderten Aktivierungs-
    abschnitte (Bursts) einer AU."""
    if nonzero_mask.size == 0:
        return []
    diff = np.diff(nonzero_mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if nonzero_mask[0]:
        starts = [0] + starts
    if nonzero_mask[-1]:
        ends = ends + [len(nonzero_mask) - 1]
    return list(zip(starts, ends))


def count_flicker_events(runs, gap_thresh_frames: int) -> int:
    """Zählt, bei wie vielen Paaren aufeinanderfolgender Bursts derselben AU
    die Nullstrecke dazwischen <= gap_thresh_frames war - also ein kurzer
    Rückfall auf 0 und direkt danach wieder eine Aktivierung."""
    n = 0
    for i in range(len(runs) - 1):
        gap_frames = runs[i + 1][0] - runs[i][1] - 1
        if gap_frames <= gap_thresh_frames:
            n += 1
    return n


def process_file(path: Path, exclude_aus, gap_thresh_s, condition_lookup, allowed_conditions):
    recording_id = parse_filename(path)
    vp_code, video_num = split_recording_id(recording_id)

    if allowed_conditions is not None:
        cond = condition_lookup.get((vp_code, video_num))
        if cond is None:
            return [], "no_lookup"
        if cond not in allowed_conditions:
            return [], "filtered_out"

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    if "time" not in df.columns:
        raise ValueError(f"Keine 'time'-Spalte gefunden in {path.name}")

    au_cols = find_au_columns(df.columns, exclude_aus)
    if not au_cols:
        raise ValueError(f"Keine AU-Spalten gefunden in {path.name}")

    time_vals = df["time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(time_vals))) if len(time_vals) > 1 else 0.04  # Fallback: 25 fps
    gap_thresh_frames = max(0, round(gap_thresh_s / dt))

    rows = []
    for au in au_cols:
        intensity = pd.to_numeric(df[au], errors="coerce").fillna(0).to_numpy(dtype=int)
        runs = find_raw_runs(intensity > 0)
        if not runs:
            continue
        rows.append({
            "au": au,
            "recording_id": recording_id,
            "n_bursts": len(runs),
            "n_flicker_events": count_flicker_events(runs, gap_thresh_frames),
        })
    return rows, "processed"


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Ordner mit den au_annotations_n40 CSVs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Pfad für die Ergebnis-CSV")
    parser.add_argument("--ratings-dir", default=None,
                         help="Ordner mit den subjective_ratings_all_n168 CSVs "
                              "(Default: ../Ratings/subjective_ratings_all_n168 relativ zu --input)")
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS,
                         help="Nur Videos mit dieser 'art' einbeziehen (Default: druck_p ekel). "
                              "Leer lassen (--conditions ohne Werte) fuer keinen Filter.")
    parser.add_argument("--flicker-gap", type=float, default=0.2,
                         help="Max. Nullstrecke in Sekunden, die als kurzer Rückfall zwischen zwei "
                              "Bursts gezählt wird (Default: 0.2s)")
    parser.add_argument("--exclude-aus", nargs="*", default=sorted(DEFAULT_EXCLUDE_AUS),
                         help="AU-Spalten, die ausgeschlossen werden (Default: Blink/Blickrichtung)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Nur die ersten N Dateien verarbeiten (zum Testen)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_aus = set(args.exclude_aus)
    allowed_conditions = set(args.conditions) if args.conditions else None

    condition_lookup = {}
    if allowed_conditions is not None:
        ratings_dir = Path(args.ratings_dir) if args.ratings_dir else input_dir.parent / "Ratings" / "subjective_ratings_all_n168"
        if not ratings_dir.exists():
            print(f"Ratings-Ordner nicht gefunden: {ratings_dir} - bitte --ratings-dir angeben.", file=sys.stderr)
            sys.exit(1)
        print(f"Lade Bedingungs-Zuordnung aus {ratings_dir} ...")
        condition_lookup = load_condition_lookup(ratings_dir)
        print(f"  {len(condition_lookup)} Video->Bedingung Zuordnungen geladen.")
        print(f"  Filter aktiv: nur {sorted(allowed_conditions)}")

    files = sorted(input_dir.glob("*.csv"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"Keine CSV-Dateien gefunden in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Verarbeite {len(files)} Dateien aus {input_dir} ...")
    all_rows = []
    n_errors = 0
    n_filtered_out = 0
    n_no_lookup = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            rows, status = process_file(f, exclude_aus, args.flicker_gap, condition_lookup, allowed_conditions)
            all_rows.extend(rows)
            if status == "filtered_out":
                n_filtered_out += 1
            elif status == "no_lookup":
                n_no_lookup += 1
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  [WARNUNG] Fehler bei {f.name}: {e}", file=sys.stderr)
        if i % 500 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"  {i}/{len(files)} Dateien geprüft ({elapsed:.1f}s, {n_errors} Fehler)")

    print(f"\nGefiltert (nicht in {sorted(allowed_conditions) if allowed_conditions else '-'}): {n_filtered_out}")
    if allowed_conditions is not None:
        print(f"Keine Bedingungs-Zuordnung gefunden: {n_no_lookup} "
              f"(falls hoch: Annahme Videonummer=Trialreihenfolge prüfen)")

    if not all_rows:
        print("Keine Bursts gefunden - bitte Filter/Format prüfen.", file=sys.stderr)
        sys.exit(1)

    raw_df = pd.DataFrame(all_rows)

    summary = raw_df.groupby("au", as_index=False).agg(
        n_bursts=("n_bursts", "sum"),
        n_flicker_events=("n_flicker_events", "sum"),
    )
    videos_with_flicker = (
        raw_df[raw_df["n_flicker_events"] > 0]
        .groupby("au")["recording_id"].nunique()
        .rename("n_videos_with_flicker")
    )
    summary = summary.merge(videos_with_flicker, on="au", how="left")
    summary["n_videos_with_flicker"] = summary["n_videos_with_flicker"].fillna(0).astype(int)
    summary["flicker_rate"] = (summary["n_flicker_events"] / summary["n_bursts"].clip(lower=1)).round(4)
    summary = summary.sort_values("n_flicker_events", ascending=False).reset_index(drop=True)

    total_row = pd.DataFrame([{
        "au": "TOTAL",
        "n_bursts": summary["n_bursts"].sum(),
        "n_flicker_events": summary["n_flicker_events"].sum(),
        "n_videos_with_flicker": raw_df.loc[raw_df["n_flicker_events"] > 0, "recording_id"].nunique(),
        "flicker_rate": round(summary["n_flicker_events"].sum() / max(summary["n_bursts"].sum(), 1), 4),
    }])
    summary = pd.concat([summary, total_row], ignore_index=True)

    summary.to_csv(output_path, index=False)
    print(f"\nGeschrieben: {output_path}")

    total = summary[summary["au"] == "TOTAL"].iloc[0]
    print("\nKurzüberblick:")
    if total["n_flicker_events"] > 0:
        n_aus_affected = (summary[summary["au"] != "TOTAL"]["n_flicker_events"] > 0).sum()
        print(f"Rückfälle kommen vor: JA, insgesamt {int(total['n_flicker_events'])} mal "
              f"({total['flicker_rate'] * 100:.2f}% aller Aktivierungen), "
              f"betrifft {n_aus_affected} verschiedene AUs und "
              f"{int(total['n_videos_with_flicker'])} Videos.")
    else:
        print("Rückfälle kommen nicht vor.")


if __name__ == "__main__":
    main()