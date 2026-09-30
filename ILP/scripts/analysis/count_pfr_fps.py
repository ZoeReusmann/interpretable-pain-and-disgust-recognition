from pathlib import Path
import cv2
import pandas as pd

CSV = Path("outputs/painfacereader_pressure_pain_vs_disgust_frontal.csv")
OUT = Path("outputs/pfr_video_fps_summary.csv")

df = pd.read_csv(CSV)

rows = []

for video_path in df["video_path"]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else None

    rows.append({
        "video_id": video_path.stem,
        "video_path": str(video_path),
        "opened": opened,
        "fps": fps,
        "frames": frames,
        "duration_s": duration,
    })

    cap.release()

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("Videos:", len(out))
print("Opened:", out["opened"].sum())
print("\nFPS values:")
print(out["fps"].value_counts(dropna=False))

print("\nFrames:")
print(out["frames"].describe())

print("\nDuration:")
print(out["duration_s"].describe())

print(f"\nSaved to: {OUT}")