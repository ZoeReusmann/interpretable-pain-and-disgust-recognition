import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import json
"""
Ziel: Confusion Matrix (TP, FP, TN, FN) für eine gelernte FastLAS-Regel
gegen die Testdaten
Visuell: angpasst an das CNN-Analyseskript (analyze_confusion_matrix.py), damit beide Teile der Arbeit konsistent
aussehen (magma-Heatmap, gleiche Label-Konvention, Excel-Export).

Input: das JSON, das evaluate_metrics.py erzeugt hat (enthält tp/fp/tn/fn).
"""

OUTPUT_DIR = "evaluation_results/confusion_matrix_analysis"


def build_matrix(tp, fp, tn, fn):
    """Baut die 2x2-Matrix im gleichen Layout wie sklearn.confusion_matrix
    (Zeilen = tatsaechlich, Spalten = vorhergesagt): [[tn, fp], [fn, tp]]."""
    return np.array([[tn, fp], [fn, tp]])


def compute_row(data):
    tp, fp, tn, fn = data["tp"], data["fp"], data["tn"], data["fn"]
    return {
        "predicate": data.get("predicate", "?"),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": data.get("accuracy"),
        "precision": data.get("precision", tp / (tp + fp + 1e-8)),
        "recall": data.get("recall", tp / (tp + fn + 1e-8)),
        "specificity": tn / (tn + fp + 1e-8),
        "f1_score": data.get("f1", 2 * tp / (2 * tp + fp + fn + 1e-8)),
        "mcc": data.get("mcc"),
    }


def save_heatmap(cm, predicate, timestamp, output_dir):
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        vmin=0,
        vmax=cm.max(),
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )

    plt.title(f"{predicate} Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()

    output_path = os.path.join(
        heatmap_dir,
        f"{predicate}_confusion_matrix_{timestamp}.png",
    )
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap gespeichert unter:\n{output_path}")
    return output_path


def save_to_excel(row, timestamp, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([row])

    output_path = os.path.join(
        output_dir,
        f"confusion_matrix_summary_{row['predicate']}_{timestamp}.xlsx",
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="confusion_matrix", index=False)

    print(f"Excel gespeichert unter:\n{output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="JSON-Datei von evaluate_metrics.py")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    with open(args.metrics) as f:
        data = json.load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cm = build_matrix(data["tp"], data["fp"], data["tn"], data["fn"])
    row = compute_row(data)

    save_heatmap(cm, row["predicate"], timestamp, args.output_dir)
    df = save_to_excel(row, timestamp, args.output_dir)

    print("\nConfusion Matrix Summary:")
    print(df)


if __name__ == "__main__":
    main()
