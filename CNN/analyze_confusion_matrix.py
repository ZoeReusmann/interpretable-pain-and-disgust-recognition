import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from dataloader import get_dataloaders
from model import AUModel
from config import (
    IMAGE_DIR,
    LABEL_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DECISION_THRESHOLD_MODE,
    OVERSAMPLING_THRESHOLDS,
)
from constants import TARGET_AUS

"""
Ziel: binäre Confusion Matrix pro AU (TP, FP, TN, FN)
Ziel Des Ziels: zu viele FN? FP? welche AU's sind kritisch? werden AU's nie erkannt? AUsreißer! starke Biases?
erstmla nur pro AU und nciht pro Threshold level, weil sonst zu unübersichtlich

problem: 6x5 = zu viele COnfusion Matrices, deswegen [1,1,1,0,0] -> 3
"""

CHECKPOINT_PATH = (
    "checkpoints/facenet_vggface2_au_threshold_partially_unfrozen_bs8_lr0.0001_seed11_2026-05-14_13-31-13/best_model.pt"
)

OUTPUT_DIR = "evaluation_results/confusion_matrix_analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# entweder:
# GLOBAL_THRESHOLD = 0.35

# oder:
AU_THRESHOLDS = {
    "AU4": 0.35,
    "AU6": 0.40,
    "AU9": 0.20,
    "AU17": 0.20,
    "AU25": 0.40,
    "AU26": 0.20,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model():
    model = AUModel()

    print("lade Modell...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    print("Modell geladen")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

def get_predictions(model, loader):

    all_preds = []
    all_targets = []

    with torch.no_grad():

        print("starte Prediction Loop...")

        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

            images = batch[0].to(DEVICE)
            targets = batch[1].float().to(DEVICE) # batch[0] = images, batch[1] = raw_labels, batch[2] = metadata

            logits = model(images)
            probs = torch.sigmoid(logits)

            # global threshold
            if DECISION_THRESHOLD_MODE == "global":

                GLOBAL_THRESHOLD = 0.35

                preds = (
                    probs >= GLOBAL_THRESHOLD
                ).int()

            # AU-specific thresholds
            elif DECISION_THRESHOLD_MODE == "au_specific":

                preds = torch.zeros_like(probs)

                for au_idx, au_name in enumerate(TARGET_AUS):

                    threshold = AU_THRESHOLDS[au_name]

                    preds[:, au_idx, :] = (
                        probs[:, au_idx, :] >= threshold
                    ).int()

            elif DECISION_THRESHOLD_MODE == "oversampling":

                preds = torch.zeros_like(probs)

                for au_idx, au_name in enumerate(TARGET_AUS):

                    threshold = OVERSAMPLING_THRESHOLDS[au_name]

                    preds[:, au_idx, :] = (
                        probs[:, au_idx, :] >= threshold
                    ).int()

            else:
                raise ValueError(
                    f"Unknown DECISION_THRESHOLD_MODE: "
                    f"{DECISION_THRESHOLD_MODE}"
                )

            # Reduce ordinal threshold vectors to binary AU activity
            pred_active = (
                preds.sum(dim=2) > 0
            ).int()

            target_active = targets.int()

            all_preds.append(pred_active.cpu())
            all_targets.append(target_active.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_preds, all_targets

def compute_confusion_matrices(preds, targets, timestamp):
    rows = []
    matrices = {}

    for au_idx, au_name in enumerate(TARGET_AUS):
        y_true = targets[:, au_idx].numpy()
        y_pred = preds[:, au_idx].numpy()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        matrices[au_name] = cm

        rows.append({
            "au": au_name,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "specificity": tn / (tn + fp + 1e-8),
            "f1_score": 2 * tp / (2 * tp + fp + fn + 1e-8),
        })

    df = pd.DataFrame(rows)

    return df, matrices

def save_heatmaps(matrices, timestamp):

    heatmap_dir = os.path.join(
        OUTPUT_DIR,
        "heatmaps",
    )

    os.makedirs(heatmap_dir, exist_ok=True)

    for au_name, cm in matrices.items():

        plt.figure(figsize=(5, 4))

        global_max = max(cm.max() for cm in matrices.values())
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="magma",
            vmin=0,
            vmax=global_max,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )

        plt.title(f"{au_name} Confusion Matrix")

        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")

        plt.tight_layout()

        output_path = os.path.join(
            heatmap_dir,
            f"{au_name}_confusion_matrix_{DECISION_THRESHOLD_MODE}_{timestamp}.png",
        )

        plt.savefig(output_path)

        plt.close()

def save_to_excel(df, timestamp):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_summary_{DECISION_THRESHOLD_MODE}_{timestamp}.xlsx",
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="confusion_matrices",
            index=False,
        )

    print(f"\nExcel gespeichert unter:\n{output_path}")

def main():

    print("analyze_confusion_matrix.py startet")

    set_seed(SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, test_loader = get_dataloaders(
        IMAGE_DIR,
        LABEL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = load_model()

    preds, targets = get_predictions(
        model,
        test_loader,
    )

    df, matrices = compute_confusion_matrices(
        preds,
        targets,
        timestamp=pd.Timestamp.now()
    )
    save_to_excel(df, timestamp)
    save_heatmaps(matrices, timestamp)

    print("\nConfusion Matrix Summary:")
    print(df)


if __name__ == "__main__":
    main()