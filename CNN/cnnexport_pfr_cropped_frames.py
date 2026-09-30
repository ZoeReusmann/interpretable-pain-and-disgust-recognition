import os
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from model import AUModel
from constants import TARGET_AUS
from config import OVERSAMPLING_THRESHOLDS
from transforms import train_transform

"""
Ziel: PFR läuft durch das CNN und exportiert alle Daten (auch Metadaten und zeitliche Informationen) in eine CSV-Datei, die dann in create_pfr_facts.py in Prolog-Fakten umgewandelt wird.

Die CSV-Datei enthält die folgenden Spalten:
- subject_id: ID des Subjekts
- video_id: ID des Videos
- frame: Frame-Nummer im Originalvideo
- relative_frame: Frame-Nummer relativ zum Start des Videos
- time_s: Zeit in Sekunden relativ zum Start des Videos
- image_path: Pfad zum Bild
- original_name: Originalname des Videos
- coded_name: Codierter Name des Videos
- label: Label des Videos (pain oder disgust)
- AU4_prob, AU6_prob, AU9_prob, AU17_prob, AU25_prob, AU26_prob: Wahrscheinlichkeit für jede Action Unit
- AU4_pred, AU6_pred, AU9_pred, AU17_pred, AU25_pred, AU26_pred: Vorhersage für jede Action Unit (1 = aktiv, 0 = inaktiv)

"""


PFR_LABEL_CSV = "../../ILP/outputs/painfacereader_pressure_pain_vs_disgust_frontal.csv"
IMAGE_ROOT = Path("/path/to/data/relevant_pfr")

CHECKPOINT_PATH = "checkpoints/lopo_SN004_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_18-29-32/best_model.pt"

OUTPUT_DIR = "outputs/predictions/pfr"
OUTPUT_FILE = "pfr_cropped_frame_predictions_oversampling_thresholds_seed11.csv"

AU_THRESHOLDS = OVERSAMPLING_THRESHOLDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
FPS = 25.0


def load_model():
    model = AUModel(num_aus=len(TARGET_AUS))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


def predict_batch(model, batch_tensors):
    images = torch.stack(batch_tensors).to(DEVICE)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
        au_probs = probs.max(dim=2).values.cpu()

        thresholds = torch.tensor(
            [AU_THRESHOLDS[au] for au in TARGET_AUS]
        ).view(1, -1)

        au_preds = (au_probs >= thresholds).int()

    return au_probs, au_preds


def find_image_dir(video_id):
    candidates = list(IMAGE_ROOT.rglob(video_id))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def process_video_images(model, row):
    video_path = row["video_path"]
    subject_id = row["subject_id"]
    label = row["label"]
    coded_name = row["coded_name"]
    original_name = row["original_name"]

    video_id = Path(video_path).stem
    image_dir = find_image_dir(video_id)

    if image_dir is None:
        print(f"No image folder found for video_id: {video_id}")
        return []

    image_paths = sorted(image_dir.glob("*.jpg"))

    if len(image_paths) == 0:
        print(f"No images found in: {image_dir}")
        return []

    rows = []
    batch_tensors = []
    batch_meta = []

    for relative_frame, image_path in enumerate(image_paths):
        frame = int(image_path.stem)

        pil_img = Image.open(image_path).convert("RGB")
        tensor = train_transform(pil_img)

        batch_tensors.append(tensor)
        batch_meta.append({
            "subject_id": subject_id,
            "video_id": video_id,
            "frame": frame,
            "relative_frame": relative_frame,
            "time_s": frame / FPS,
            "image_path": str(image_path),
            "original_name": original_name,
            "coded_name": coded_name,
            "label": label,
        })

        if len(batch_tensors) == BATCH_SIZE:
            au_probs, au_preds = predict_batch(model, batch_tensors)

            for i, meta in enumerate(batch_meta):
                out_row = meta.copy()

                for au_idx, au in enumerate(TARGET_AUS):
                    out_row[f"{au}_prob"] = float(au_probs[i, au_idx])
                    out_row[f"{au}_pred"] = int(au_preds[i, au_idx])

                rows.append(out_row)

            batch_tensors = []
            batch_meta = []

    if batch_tensors:
        au_probs, au_preds = predict_batch(model, batch_tensors)

        for i, meta in enumerate(batch_meta):
            out_row = meta.copy()

            for au_idx, au in enumerate(TARGET_AUS):
                out_row[f"{au}_prob"] = float(au_probs[i, au_idx])
                out_row[f"{au}_pred"] = int(au_preds[i, au_idx])

            rows.append(out_row)

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading labels: {PFR_LABEL_CSV}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    labels_df = pd.read_csv(PFR_LABEL_CSV)
    model = load_model()

    all_rows = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing PFR cropped frames"):
        image_rows = process_video_images(model, row)
        all_rows.extend(image_rows)

    out_df = pd.DataFrame(all_rows)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    out_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print(out_df.head())
        print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()