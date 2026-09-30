import os
import torch
import pandas as pd
from tqdm import tqdm

from dataloader import get_dataloaders
from model import AUModel
from constants import TARGET_AUS

from config import (
    IMAGE_DIR,
    LABEL_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    SEED,
    )

"""
Ziel: Daten Transfomrieren! Also: die Vorhersagen des besten CNN Models exportieren, um es für weitere Schritte nutzbar zu machen
Wie? bestes Modell vom Checkpoint laden, Vorhersagen auf dem Validation Set machen, und diese in einem CSV File speichern, mit Spalten für die AU-Wahrscheinlichkeiten und die binären Vorhersagen (basierend auf einem Schwellenwert von 0.35)
Damit sind zeitliche Einordnung der AU's möglich, z.B. Overlap, Duration, Peak etc.
aktuell Problem: es werden noch keine Metadaten gespeichert - als nächstes daran arbeiten!

Auch: Traingin mit SN unterschied anstatt random
"""


CHECKPOINT_PATH = "checkpoints/lopo_SN004_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_18-29-32/best_model.pt"
OUTPUT_DIR = "outputs/predictions"
OUTPUT_FILE = "frame_predictions_best_model_thr0.35_seed11.csv"

PRED_THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def export_predictions(loader, split_name, model):
    rows = []

    with torch.no_grad():
        for images, raw_labels, threshold_targets, metadata in tqdm(loader, desc=f"Exporting {split_name} predictions"):
            images = images.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Modelloutput: [batch, 6 AUs, 5 thresholds]
            # für jeden AU höchsten Threshold-Output als aggregierte AU-Wahrscheinlichkeit nehmen
            au_probs = probs.max(dim=2).values.cpu()

            au_preds = (au_probs >= PRED_THRESHOLD).int()

            for i in range(images.size(0)):
                row = {}

                row["split"] = split_name
                row["subject"] = metadata["subject"][i]
                row["trial"] = metadata["trial"][i]
                row["frame"] = metadata["frame"][i]
                row["image_path"] = metadata["image_path"][i]

                for au_idx, au in enumerate(TARGET_AUS):
                    row[f"{au}_prob"] = float(au_probs[i, au_idx])
                    row[f"{au}_pred"] = int(au_preds[i, au_idx])

                rows.append(row)

    return rows

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
    )

    model = AUModel(num_aus=len(TARGET_AUS))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    all_rows = []

    all_rows.extend(export_predictions(train_loader, "train", model))
    all_rows.extend(export_predictions(val_loader, "val", model))
    all_rows.extend(export_predictions(test_loader, "test", model))

    df = pd.DataFrame(all_rows)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()