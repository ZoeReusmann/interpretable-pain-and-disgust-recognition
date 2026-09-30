import pandas as pd
import os

from config import TARGET_AUS

"""
Ziel: prüfen ob das MOdell lernt, Verteilung der AU Wshk, Problem mit den leeren frames angehen: %  der aktiv vorhergesagten AU'S

next step: das hier als Excel Tabelle ausgeben, damit ich mir die Wrete merken kann anstatt nur im terminal
"""

df = pd.read_csv("outputs/predictions/frame_predictions_best_model_thr0.35_seed11.csv")

print("\nColumns")
print(df.columns)

print("\nFirst rows")
print(df.head())

print("\nRelevant columns")
print(df[["split", "subject", "trial", "frame", "image_path"]].head(20))

print("\nShape")
print(df.shape)

print("\nSplit distribution")
print(df["split"].value_counts())

print("\nSubject distribution")
print(df["subject"].value_counts())

print("\nTrial distribution")
print(df["trial"].value_counts().head(20))

print("\nChronological order check")

# frame "032.jpg" --> 32
df["frame_id"] = (
    df["frame"]
    .astype(str)
    .str.replace(".jpg", "", regex=False)
    .astype(int)
)

# sortieren
df_sorted = df.sort_values(
    by=["split", "subject", "trial", "frame_id"]
).reset_index(drop=True)

print(df_sorted[["subject", "trial", "frame", "frame_id"]].head(20))

# speichern
df_sorted.to_csv(
    "outputs/predictions/frame_predictions_sorted.csv",
    index=False
)

print("\nFrames pro Video")
print(
    df_sorted.groupby(["subject", "trial"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

example = df_sorted[
    (df_sorted["subject"] == "SN004") &
    (df_sorted["trial"] == "A2_AU2_TrailNo_1")
]

print("\nAU4 sequence example")
print(example[["frame_id", "AU4_pred"]].head(30))

print("\nAU4 sequences per video")

for (split, subject, trial), group in df_sorted.groupby(["split", "subject", "trial"]):
    sequence = group["AU4_pred"].tolist()
    frames = group["frame_id"].tolist()

    print("\nVideo:")
    print(split, subject, trial)
    print("Number of frames:", len(sequence))
    print("First 30 frames:", frames[:30])
    print("First 30 AU4 predictions:", sequence[:30])

    # erstmal nur ein Beispiel ausgeben
    break

print("\nWriting temporal facts for all AUs")

output_dir = "outputs/temporal_facts"
os.makedirs(output_dir, exist_ok=True)

for au in TARGET_AUS:
    pred_col = f"{au}_pred"
    prob_col = f"{au}_prob"
    output_file = os.path.join(output_dir, f"{au.lower()}_facts.pl")

    with open(output_file, "w") as f:
        for (split, subject, trial), group in df_sorted.groupby(["split", "subject", "trial"]):
            video_id = f"{split}_{subject}_{trial}".lower().replace("-", "_")

            active_rows = group[group[pred_col] == 1]

            for _, row in active_rows.iterrows():
                frame_id = int(row["frame_id"])
                prob = float(row[prob_col])

                f.write(f"active({video_id}, {au.lower()}, {frame_id}).\n")
                f.write(f"prob({video_id}, {au.lower()}, {frame_id}, {prob:.4f}).\n")

    print(f"{au} facts saved to: {output_file}")

print("\nSorted CSV saved.")