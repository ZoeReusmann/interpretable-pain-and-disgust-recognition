import argparse
import csv
from pathlib import Path

from pfr_config import (
    PFR_DATASET_ROOT,
    PFR_OUTPUT_CSV
)

PFR_VIDEO_SUFFIX = ".mp4"

PFR_REQUIRE_FRONTAL = True

PFR_TARGET_CONDITIONS = {
    "druck_pain": "pressure_pain",
    "bild_ekel": "disgust",
}

def read_key_file(path: Path):
    try:
        import pandas as pd
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin1", sep=None, engine="python")

    df.columns = [c.strip().lower() for c in df.columns]

    if "unkodiert" not in df.columns or "kodiert" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")

    df = df[["unkodiert", "kodiert"]].copy()
    df["unkodiert"] = df["unkodiert"].str.strip()
    df["kodiert"] = df["kodiert"].str.strip()

    return df

def classify_original_name(original_name: str) -> str | None:
    name = original_name.lower()

    for pattern, label in PFR_TARGET_CONDITIONS.items():
        if pattern in name:
            return label

    return None


def is_valid_video(original_name: str) -> bool:
    name = original_name.lower()

    if not name.endswith(PFR_VIDEO_SUFFIX):
        return False

    if PFR_REQUIRE_FRONTAL and "frontal" not in name:
        return False

    return classify_original_name(original_name) is not None


def find_videos(dataset_root: Path) -> list[dict[str, str]]:
    rows = []

    key_files = sorted(
        p for p in dataset_root.glob("*_Kodierungsschl*.csv")
        if not p.name.startswith("._")
    )

    for key_file in key_files:
        key_df = read_key_file(key_file)

        for _, row in key_df.iterrows():
            original_name = row["unkodiert"]
            coded_name = row["kodiert"]

            if not is_valid_video(original_name):
                continue

            label = classify_original_name(original_name)

            subject_id = key_file.name.split("_Kodierung")[0]
            video_path = dataset_root / subject_id / coded_name

            if not video_path.exists():
                continue

            rows.append(
                {
                    "subject_id": subject_id,
                    "video_path": str(video_path),
                    "original_name": original_name,
                    "coded_name": coded_name,
                    "label": label,
                }
            )

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["subject_id", "video_path", "original_name", "coded_name", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a filtered list of PainFaceReader pressure_pain and disgust videos."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PFR_DATASET_ROOT,
        help="Path to PFR_dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PFR_OUTPUT_CSV,
        help="Output CSV path",
    )

    # parser.add_argument(
    #     "--key-file",
    #     type=Path,
    #     required=True,
    #     help="Path to the key file",
    # )

    args = parser.parse_args()

    rows = find_videos(args.dataset_root)
    write_csv(rows, args.output)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Output CSV: {args.output}")
    print(f"Found videos: {len(rows)}")

    label_counts = {}
    for row in rows:
        label = row["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("Label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()