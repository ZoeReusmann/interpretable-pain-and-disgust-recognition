import argparse
import cv2
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


def crop_with_margin(frame, box, margin=0.25):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    x1 = int(max(0, x1 - margin * bw))
    y1 = int(max(0, y1 - margin * bh))
    x2 = int(min(w, x2 + margin * bw))
    y2 = int(min(h, y2 + margin * bh))

    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--every-nth-frame", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        keep_all=True,
        device=device,
        post_process=False,
    )

    df = pd.read_csv(args.csv)

    videos = [Path(p) for p in df["video_path"]]
    print(f"Found videos: {len(videos)}")

    for video_path in tqdm(videos, desc="Cropping PFR videos"):
        video_id = video_path.stem
        out_video_dir = args.output_dir / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        saved_idx = 0
        last_box = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.every_nth_frame != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            boxes, probs = mtcnn.detect(pil_img)

            if boxes is not None and len(boxes) > 0:
                # größtes Gesicht nehmen
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                best_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
                box = boxes[best_idx]
                last_box = box
            elif last_box is not None:
                # falls kurz kein Gesicht erkannt wird: letzte Box verwenden
                box = last_box
            else:
                frame_idx += 1
                continue

            crop = crop_with_margin(rgb, box, margin=args.margin)
            crop = cv2.resize(crop, (224, 224))

            out_path = out_video_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            saved_idx += 1
            frame_idx += 1

        cap.release()


if __name__ == "__main__":
    main()