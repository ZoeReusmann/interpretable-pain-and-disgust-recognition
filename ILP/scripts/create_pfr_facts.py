from pathlib import Path
import pandas as pd
from constants import (
    PFR_CNN_EXPORT,
    FACTS_FILE,
    TARGET_AUS,
    FPS,
)

"""
wichtig: hier keine label dabei!!!
Ziel: Fakten für Prolog erstellen
"""

INPUT_CSV = PFR_CNN_EXPORT
OUTPUT_FILE = FACTS_FILE

def sanitize_atom(value: str) -> str: # für Prolog nutzbar machen, z.B. die PFR subjects haben bindestriche, die Prolog net akzeptiert (zb AFT080412-001)
    value = str(value).lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace(".", "_")
    return value


def write_fact(file, fact: str): # jeder Prolog Fakt in einer eigenen Zeile, für mehr Übersicht und debugging
    file.write(fact + "\n")


def main():
    print(f"Loading CNN predictions from: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rows: {len(df)}")
    print(f"Videos: {df['video_id'].nunique()}")
    print(df["label"].value_counts())
# pro Video Fakten
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        video_df = df[["video_id", "label"]].drop_duplicates() # hier nur ein Fakt pro Video speichern, nicht pro Frame, sonst wird es zu groß und unübersichtlich

        for _, row in video_df.iterrows():
            video_id = sanitize_atom(row["video_id"])
            label = sanitize_atom(row["label"]) 

            write_fact(f, f"video({video_id}).")
            # write_fact(f, f"label({video_id},{label}).") # diese Fakten braucht es nur einmal pro Video, sonst zu viel doppelt

        f.write("\n")
# pro Frame Fakten
        frame_df = df[["video_id", "frame", "relative_frame", "time_s"]].drop_duplicates() # Frame Fakten

        for _, row in frame_df.iterrows():
            video_id = sanitize_atom(row["video_id"])

            write_fact(f, f"frame({video_id},{int(row['relative_frame'])}).")
            write_fact(
                f,
                f"original_frame({video_id},{int(row['relative_frame'])},{int(row['frame'])})."
            )
            write_fact(
                f,
                f"time_s({video_id},{int(row['relative_frame'])},{round(float(row['time_s']),3)})."
            )

        f.write("\n")

# Action Unit Fakten
        for _, row in df.iterrows():
            video_id = sanitize_atom(row["video_id"])
            relative_frame = int(row["relative_frame"])

            for au in TARGET_AUS:
                if int(row[f"{au}_pred"]) == 1:
                    write_fact(
                        f,
                        f"active({video_id},{au.lower()},{relative_frame})."
                    )

    print(f"Saved facts to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()