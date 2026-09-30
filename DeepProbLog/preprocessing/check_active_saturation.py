import pandas as pd
import numpy as np

CSV_PATH = "CNN/outputs/predictions/pfr/pfr_cropped_frame_predictions_oversampling_thresholds_seed11.csv"
AUS = ["AU4", "AU6", "AU9", "AU17", "AU25", "AU26"]

df = pd.read_csv(CSV_PATH)

results = []
for video_id, group in df.groupby("video_id"):
    for au in AUS:
        col = f"{au}_prob"
        probs = group[col].to_numpy()
        p_active = 1 - np.prod(1 - probs)  # P(mind. 1 Frame aktiv)
        results.append({"video": video_id, "au": au.lower(), "p_active": p_active})

result_df = pd.DataFrame(results)

print(result_df["p_active"].describe())
print()
for threshold in [0.9, 0.99, 0.999]:
    saturated = (result_df["p_active"] >= threshold).mean()
    print(f"Anteil Video-AU-Paare mit p_active >= {threshold}: {saturated:.1%}")

print()
print("Pro AU aufgeschlüsselt (Median p_active):")
print(result_df.groupby("au")["p_active"].median().sort_values(ascending=False))