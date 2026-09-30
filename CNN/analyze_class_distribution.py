import os
import pandas as pd
import torch

from dataloader import get_dataloaders
from config import IMAGE_DIR, LABEL_DIR, BATCH_SIZE, NUM_WORKERS
from constants import TARGET_AUS


OUTPUT_DIR = "evaluation_results"
OUTPUT_FILE = "label_distribution_support_imbalance.xlsx"

"""
Ziel: Analyse der Label-Verteilung pro AU und pro Threshold-Level, um die Imbalance zu verstehen - dann für meine Diskussion oder die Ergebnisse einzuordnen
2 Excel Seiten
1: pro AU: positive labels, total labels, positive rate
2: pro AU und pro Threshold-Level: positive labels, total labels, positive rate
"""

def count_label_distribution(loader, split_name): # count für alles

    rows_per_au = []
    rows_per_au_threshold = []

    n_aus = len(TARGET_AUS)

    positive_per_au = torch.zeros(n_aus)
    total_per_au = torch.zeros(n_aus)

    positive_per_au_threshold = None
    total_per_au_threshold = None

    for batch in loader:
        targets = batch[2] # Ziel: Form [batch_size, n_aus, n_thresholds]
        targets = targets.float()

        if positive_per_au_threshold is None:
            n_thresholds = targets.shape[2]
            positive_per_au_threshold = torch.zeros(n_aus, n_thresholds)
            total_per_au_threshold = torch.zeros(n_aus, n_thresholds)

        au_active = (targets.sum(dim=2) > 0).float() # pro AU: aktiv, wenn mindestens ein Threshold-Level aktiv ist

        positive_per_au += au_active.sum(dim=0)
        total_per_au += torch.ones_like(au_active).sum(dim=0)

        positive_per_au_threshold += targets.sum(dim=0) # check: werden die THresholds mit zunehmenden INtensitätslevel dünner?
        total_per_au_threshold += torch.ones_like(targets).sum(dim=0)

    for au_idx, au_name in enumerate(TARGET_AUS):
        positive = int(positive_per_au[au_idx].item())
        total = int(total_per_au[au_idx].item())
        positive_rate = positive / total if total > 0 else 0

        rows_per_au.append({
            "split": split_name,
            "au": au_name,
            "positive_labels": positive,
            "total_labels": total,
            "positive_rate": positive_rate,
        })

    for au_idx, au_name in enumerate(TARGET_AUS):
        for thr_idx in range(positive_per_au_threshold.shape[1]):
            positive = int(positive_per_au_threshold[au_idx, thr_idx].item())
            total = int(total_per_au_threshold[au_idx, thr_idx].item())
            positive_rate = positive / total if total > 0 else 0

            rows_per_au_threshold.append({
                "split": split_name,
                "au": au_name,
                "threshold_level": thr_idx + 1,
                "positive_labels": positive,
                "total_labels": total,
                "positive_rate": positive_rate,
            })

    return rows_per_au, rows_per_au_threshold


def save_to_excel(per_au_df, per_au_threshold_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        per_au_df.to_excel(writer, sheet_name="per_au_support", index=False)
        per_au_threshold_df.to_excel(
            writer,
            sheet_name="per_au_threshold_support",
            index=False,
        )

def main():
    print("analyze_label_distribution.py startet")

    train_loader, val_loader, test_loader = get_dataloaders(
        IMAGE_DIR,
        LABEL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    all_per_au_rows = []
    all_per_au_threshold_rows = []

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        per_au_rows, per_au_threshold_rows = count_label_distribution(
            loader,
            split_name,
        )

        all_per_au_rows.extend(per_au_rows)
        all_per_au_threshold_rows.extend(per_au_threshold_rows)

    per_au_df = pd.DataFrame(all_per_au_rows)
    per_au_threshold_df = pd.DataFrame(all_per_au_threshold_rows)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    save_to_excel(per_au_df, per_au_threshold_df, output_path)

    print(f"Gespeichert unter: {output_path}")
    print("\nPer-AU Support:")
    print(per_au_df)


if __name__ == "__main__":
    main()