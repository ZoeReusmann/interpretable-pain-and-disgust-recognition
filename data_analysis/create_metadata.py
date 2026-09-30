
from pathlib import Path
import re
import pandas as pd

"""
Ziel: Video-Metadaten für l annotierten PFR-n40-Videos, legt fest welche video später relevant sind

Ausgewählt werden ausschließlich:
- frontale Druckschmerz-Videos
- frontale Bildekel-Videos

Die kodierten Video-IDs werden mit den vorhandenen Timeline-CSV-Dateien
im Ordner au_annotations_n40 verknüpft
"""
PFR_ROOT = Path(
    "/path/to/PFR_dataset"
)

VPCODES_FILE = PFR_ROOT / "vpcodes_n40.csv"
KEY_DIR = PFR_ROOT / "videos_all_full-length_n168"
ANNOTATION_DIR = PFR_ROOT / "au_annotations_n40"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data_analysis" / "outputs"
OUTPUT_FILE = (
    OUTPUT_DIR
    / "pfr_n40_pressure_pain_vs_disgust_frontal.csv"
)
MISSING_FILE = OUTPUT_DIR / "missing_or_ambiguous_files.csv"

def normalize_subject(value: str) -> str:
    """Normalisiert einen Probandencode."""
    return str(value).strip().upper()

def find_coding_key(subject_id: str) -> Path | None:
    """
    Findet den Kodierungsschlüssel eines Probanden robust,
    ohne die genaue Unicode-Schreibweise von 'ü' vorauszusetzen.
    """
    matches = sorted(
        path
        for path in KEY_DIR.glob(f"{subject_id}_*.csv")
        if "schl" in path.name.lower()
    )

    if len(matches) == 1:
        return matches[0]

    return None


def extract_video_id(coded_name: str) -> str:
    """
    Wandelt beispielsweise
    AGD080924_005.mp4
    in
    AGD080924_005
    um.
    """
    return Path(str(coded_name).strip()).stem.upper()


def create_timeline_index() -> dict[str, list[Path]]:
    """
    Erstellt einen Index aller Timeline-Dateien anhand der Video-ID.

    Erkennt beispielsweise:
    MSE-Segmente Block 7 - AGD080924_005_timeline.csv
    """
    timeline_index: dict[str, list[Path]] = {}

    pattern = re.compile(
        r"([A-Za-z]{3}\d{6}_\d{3})(?:copy)?_timeline\.csv$",
        flags=re.IGNORECASE,
    )

    for timeline_path in ANNOTATION_DIR.glob("*.csv"):
        match = pattern.search(timeline_path.name)

        if match is None:
            continue

        video_id = match.group(1).upper()

        timeline_index.setdefault(video_id, []).append(
            timeline_path
        )

    return timeline_index


def classify_condition(original_name: str) -> str | None:
    """
    Ordnet ausschließlich die beiden relevanten frontalen Bedingungen zu.
    """
    name = str(original_name).strip().lower()

    if "frontal_druck_pain" in name:
        return "pressure_pain"

    if "frontal_bild_ekel" in name:
        return "disgust"

    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VPCODES_FILE.exists():
        raise FileNotFoundError(
            f"vpcodes_n40.csv nicht gefunden: {VPCODES_FILE}"
        )

    if not KEY_DIR.exists():
        raise FileNotFoundError(
            f"Ordner mit Kodierungsschlüsseln fehlt: {KEY_DIR}"
        )

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(
            f"Annotationsordner fehlt: {ANNOTATION_DIR}"
        )

    subjects_df = pd.read_csv(
        VPCODES_FILE,
        encoding="utf-8-sig",
    )

    subjects_df.columns = [
        str(column).strip().lower()
        for column in subjects_df.columns
    ]

    if "vp_code" not in subjects_df.columns:
        raise ValueError(
            "In vpcodes_n40.csv wurde keine Spalte "
            "'vp_code' gefunden."
        )

    subjects = [
        normalize_subject(value)
        for value in subjects_df["vp_code"].dropna()
    ]

    timeline_index = create_timeline_index()

    metadata_rows = []
    problem_rows = []

    for subject_id in subjects:
        coding_key = find_coding_key(subject_id)

        if coding_key is None:
            problem_rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": "",
                    "problem": (
                        "Kodierungsschlüssel fehlt oder "
                        "ist nicht eindeutig"
                    ),
                    "details": "",
                }
            )
            continue

        try:
            key_df = pd.read_csv(
                coding_key,
                sep=";",
                encoding="utf-8-sig",
            )
        except Exception as error:
            problem_rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": "",
                    "problem": "Kodierungsschlüssel nicht lesbar",
                    "details": str(error),
                }
            )
            continue

        key_df.columns = [
            str(column).strip().lower()
            for column in key_df.columns
        ]

        required_columns = {"unkodiert", "kodiert"}

        if not required_columns.issubset(key_df.columns):
            problem_rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": "",
                    "problem": "Erforderliche Spalten fehlen",
                    "details": (
                        f"Gefundene Spalten: "
                        f"{list(key_df.columns)}"
                    ),
                }
            )
            continue

        selected_count = 0

        for _, row in key_df.iterrows():
            original_name = str(row["unkodiert"]).strip()
            coded_name = str(row["kodiert"]).strip()

            label = classify_condition(original_name)

            if label is None:
                continue

            selected_count += 1
            video_id = extract_video_id(coded_name)

            timeline_matches = timeline_index.get(
                video_id,
                [],
            )

            if len(timeline_matches) == 0:
                problem_rows.append(
                    {
                        "subject_id": subject_id,
                        "video_id": video_id,
                        "problem": "Timeline-Datei fehlt",
                        "details": original_name,
                    }
                )
                continue

            if len(timeline_matches) > 1:
                problem_rows.append(
                    {
                        "subject_id": subject_id,
                        "video_id": video_id,
                        "problem": (
                            "Mehrere Timeline-Dateien gefunden"
                        ),
                        "details": " | ".join(
                            str(path)
                            for path in timeline_matches
                        ),
                    }
                )
                continue

            timeline_path = timeline_matches[0]

            video_number_match = re.search(
                r"_(\d{3})$",
                video_id,
            )

            video_number = (
                video_number_match.group(1)
                if video_number_match
                else ""
            )

            metadata_rows.append(
                {
                    "subject_id": subject_id.lower(),
                    "video_id": video_id.lower(),
                    "video_number": video_number,
                    "label": label,
                    "view": "frontal",
                    "original_name": original_name,
                    "coded_name": coded_name,
                    "coding_key_file": str(coding_key),
                    "timeline_file": str(timeline_path),
                }
            )

        if selected_count == 0:
            problem_rows.append(
                {
                    "subject_id": subject_id,
                    "video_id": "",
                    "problem": (
                        "Keine passenden frontal-Pain-/"
                        "Disgust-Einträge gefunden"
                    ),
                    "details": str(coding_key),
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)

    if not metadata_df.empty:
        metadata_df = metadata_df.sort_values(
            by=["subject_id", "label", "video_number"]
        )

        duplicate_mask = metadata_df.duplicated(
            subset=["video_id"],
            keep=False,
        )

        if duplicate_mask.any():
            for _, row in metadata_df[
                duplicate_mask
            ].iterrows():
                problem_rows.append(
                    {
                        "subject_id": row["subject_id"],
                        "video_id": row["video_id"],
                        "problem": (
                            "Doppelte Video-ID in Metadaten"
                        ),
                        "details": row["timeline_file"],
                    }
                )

        metadata_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    problems_df = pd.DataFrame(problem_rows)

    if not problems_df.empty:
        problems_df.to_csv(
            MISSING_FILE,
            index=False,
        )
    elif MISSING_FILE.exists():
        MISSING_FILE.unlink()

    print()
    print("PFR n40 – Metadata-Erstellung")
    print("=" * 45)
    print(f"Probanden in vpcodes_n40.csv: {len(subjects)}")
    print(
        f"Indexierte Timeline-Video-IDs: "
        f"{len(timeline_index)}"
    )
    print(f"Ausgewählte Videos: {len(metadata_df)}")

    if not metadata_df.empty:
        print()
        print("Videos pro Klasse:")
        print(metadata_df["label"].value_counts().to_string())

        print()
        print("Probanden pro Klasse:")
        print(
            metadata_df.groupby("label")["subject_id"]
            .nunique()
            .to_string()
        )

        videos_per_subject = (
            metadata_df.groupby(
                ["subject_id", "label"]
            )
            .size()
            .unstack(fill_value=0)
        )

        print()
        print("Videos pro Proband – Zusammenfassung:")
        print(videos_per_subject.describe().to_string())

    print()
    print(f"Probleme: {len(problem_rows)}")
    print(f"Metadatei: {OUTPUT_FILE}")

    if problem_rows:
        print(f"Problembericht: {MISSING_FILE}")

    print()

    expected_total = len(subjects) * 12

    print(
        "Orientierungswert bei 6 Pain- und 6 Disgust-Videos "
        f"pro Person: {expected_total}"
    )

    if len(metadata_df) != expected_total:
        print(
            "HINWEIS: Die tatsächliche Anzahl weicht vom "
            "Orientierungswert ab. Prüfe den Problembericht "
            "und die Verteilung pro Proband."
        )
    else:
        print(
            "Die tatsächliche Anzahl entspricht dem "
            "Orientierungswert."
        )


if __name__ == "__main__":
    main()