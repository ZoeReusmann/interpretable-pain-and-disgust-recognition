from pathlib import Path
import pandas as pd

from constants import (
    PFR_CNN_EXPORT,
    OUTPUT_DIR,
)

"""
Ziel: Beispiele für Prolog erstellen mit positiven und negativen beispielen für Pain und disgust
output form sind 2 .las dateien enimal für pain und einmal für disgust
"""

INPUT_CSV = PFR_CNN_EXPORT

PAIN_EXAMPLES = OUTPUT_DIR / "examples_pain.las"
DISGUST_EXAMPLES = OUTPUT_DIR / "examples_disgust.las"


def sanitize_atom(value: str) -> str: # wieder lesbar für Prolog
    value = str(value).lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace(".", "_")
    return value


def main():
    print(f"Loading labels from: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    video_df = df[["video_id", "label"]].drop_duplicates()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # speicherort

    with open(PAIN_EXAMPLES, "w", encoding="utf-8") as pain_file, \
         open(DISGUST_EXAMPLES, "w", encoding="utf-8") as disgust_file:

        for _, row in video_df.iterrows():
            video_id = sanitize_atom(row["video_id"])
            label = row["label"]

            if label == "pressure_pain":
                pain_file.write(
                    f"#pos(ex_{video_id}, {{pain({video_id})}}, {{}}, {{}}).\n"
                )
                disgust_file.write(
                    f"#neg(ex_{video_id}, {{disgust({video_id})}}, {{}}, {{}}).\n"
                )

            elif label == "disgust":
                pain_file.write(
                    f"#neg(ex_{video_id}, {{pain({video_id})}}, {{}}, {{}}).\n"
                )
                disgust_file.write(
                    f"#pos(ex_{video_id}, {{disgust({video_id})}}, {{}}, {{}}).\n"
                )

    print(f"Pain examples: {PAIN_EXAMPLES}")
    print(f"Disgust examples: {DISGUST_EXAMPLES}")
    print(f"Videos: {len(video_df)}")


if __name__ == "__main__":
    main()