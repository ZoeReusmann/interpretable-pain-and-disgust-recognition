from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/data/pfr_cropped_faces_mtcnn")
OUTPUT = Path("outputs/pfr_cropped_faces_counts.csv")

rows = []

for video_dir in sorted(ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    images = sorted(video_dir.glob("*.jpg"))
    rows.append({
        "video_id": video_dir.name,
        "n_images": len(images),
        "first_image": images[0].name if images else None,
        "last_image": images[-1].name if images else None,
    })

df = pd.DataFrame(rows)

print("Videos:", len(df))
print("Total images:", df["n_images"].sum())
print()
print(df["n_images"].describe())
print()
print(df.head())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")