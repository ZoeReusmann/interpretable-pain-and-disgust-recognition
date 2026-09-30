from pathlib import Path
import shutil
import pandas as pd

SOURCE_ROOT = Path("/path/to/data/pfr_cropped_faces_mtcnn")
TARGET_ROOT = Path("/path/to/data/relevant_pfr")
SUMMARY_CSV = Path("outputs/relevant_pfr_summary.csv")

FPS = 25
START_SECOND = 1.5
END_SECOND = 8.5

START_FRAME = int(START_SECOND * FPS)
END_FRAME = int(END_SECOND * FPS)

rows = []

TARGET_ROOT.mkdir(parents=True, exist_ok=True)

for video_dir in sorted(SOURCE_ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    images = sorted(video_dir.glob("*.jpg"))
    target_dir = TARGET_ROOT / video_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for img in images:
        if img.name.startswith("._") or not img.stem.isdigit():
            continue

        frame_id = int(img.stem)

        if START_FRAME <= frame_id < END_FRAME:
            shutil.copy2(img, target_dir / img.name)
            copied += 1

    rows.append({
        "video_id": video_dir.name,
        "original_frames": len(images),
        "copied_frames": copied,
        "start_frame": START_FRAME,
        "end_frame_exclusive": END_FRAME,
        "start_second": START_SECOND,
        "end_second": END_SECOND,
    })

summary = pd.DataFrame(rows)
SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(SUMMARY_CSV, index=False)

print("Source:", SOURCE_ROOT)
print("Target:", TARGET_ROOT)
print("Videos:", len(summary))
print("Total original frames:", summary["original_frames"].sum())
print("Total copied frames:", summary["copied_frames"].sum())
print()
print(summary["copied_frames"].describe())
print()
print(f"Saved summary to: {SUMMARY_CSV}")