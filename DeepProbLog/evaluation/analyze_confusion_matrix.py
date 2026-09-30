"""
Ziel: Konfusionsmatrix-Heatmaps für DeepProbLog-Trainingsläufe (Modell A/B/C), im selben visuellen Stil wie analyze_confusion_matrix.py aus dem CNN-Teil
(Colormap magma, gleiche Achsenbeschriftung, gleicher Aufbau) etc

Zwei Modi:
  1. Einzelner Split (aus protocol.py-Output-Datei, Text-Parsing)
  2. LOPO gepoolt (aus der von evaluate_lopo.py erzeugten Excel-Datei,
     Sheet "mean_by_model")

WICHTIG zur Semantik: Anders als die CNN-Konfusionsmatrix (eine Matrix pro AU, Klassen 0/1 = AU inaktiv/aktiv) ist die DeepProbLog-Matrix pro (Modell, Split)
gebaut aus ALLEN Query-Ergebnissen (pain(V) UND disgust(V) zusammen je Video), verglichen mit dem jeweiligen Ziel-Wahrheitswert. "True 1" heißt also "die
Query sollte laut Label wahr sein", nicht "Video ist Klasse X".

Aufruf Modus 1 (einzelner Split):
    python3 DeepProbLog/plot_confusion_matrix.py \
        --output-file DeepProbLog/results/20260727_182437_model_a_run3_relativethresh_output.txt \
        --name model_a_run3

Aufruf Modus 2 (LOPO gepoolt):
    python3 DeepProbLog/plot_confusion_matrix.py \
        --excel-file DeepProbLog/results/lopo_evaluation/lopo_evaluation_2026-07-29_11-05-17.xlsx
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "DeepProbLog/results/confusion_matrix_analysis"

CM_PATTERN = re.compile(
    r"Predicted\s+false\s+(\d+)\s+(\d+)\s*\n\s+true\s+(\d+)\s+(\d+)"
)

SECTIONS = [
    ("train", "--- Accuracy auf TRAIN ---"),
    ("test", "--- Accuracy auf TEST"),
]


def parse_confusion_matrix(text, marker):
    """Liefert eine 2x2-Matrix im sklearn-Format [[TN,FP],[FN,TP]]
    (Zeilen = tatsächlicher Wert, Spalten = Vorhersage), analog zum
    CNN-Skript (cm.ravel() -> tn, fp, fn, tp). Fuer Modus 1 (Text-Output)."""
    idx = text.find(marker)
    if idx == -1:
        return None
    segment = text[idx : idx + 500]
    m = CM_PATTERN.search(segment)
    if not m:
        return None

    pred_false_actual_false, pred_false_actual_true, pred_true_actual_false, pred_true_actual_true = (
        int(x) for x in m.groups()
    )

    cm = np.array(
        [
            [pred_false_actual_false, pred_true_actual_false],
            [pred_false_actual_true, pred_true_actual_true],
        ]
    )
    return cm


def build_matrix_from_counts(tp, fp, fn, tn):
    """Baut dieselbe [[TN,FP],[FN,TP]]-Matrix direkt aus TP/FP/FN/TN-Zahlen,
    fuer Modus 2 (gepoolte LOPO-Ergebnisse aus der Excel-Datei)."""
    return np.array([
        [tn, fp],
        [fn, tp],
    ])


def save_heatmap(cm, name, split, out_dir):
    os.makedirs(out_dir, exist_ok=True)

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

    plt.title(f"{name} ({split}) Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()

    output_path = os.path.join(out_dir, f"{name}_{split}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Gespeichert: {output_path}")
    return output_path


def run_single_split_mode(output_file, name):
    text = Path(output_file).read_text(encoding="utf-8", errors="replace")

    found_any = False
    for split, marker in SECTIONS:
        cm = parse_confusion_matrix(text, marker)
        if cm is None:
            print(f"WARNUNG: Konnte Konfusionsmatrix für '{split}' nicht finden.")
            continue
        save_heatmap(cm, name, split, OUTPUT_DIR)
        found_any = True

    if not found_any:
        print("Keine Konfusionsmatrix gefunden - Datei/Format prüfen.")


def run_lopo_mode(excel_file):
    df = pd.read_excel(excel_file, sheet_name="mean_by_model")

    for _, row in df.iterrows():
        model_name = row["model"]
        cm = build_matrix_from_counts(
            int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])
        )
        save_heatmap(cm, model_name, "lopo_gepoolt", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Konfusionsmatrix-PNGs aus protocol.py-Output ODER LOPO-Excel erzeugen."
    )
    parser.add_argument("--output-file", help="Modus 1: Output-Datei aus protocol.py")
    parser.add_argument("--name", help="Modus 1: Name, z.B. model_a_run3")
    parser.add_argument("--excel-file", help="Modus 2: Excel-Datei aus evaluate_lopo.py")
    args = parser.parse_args()

    if args.excel_file:
        run_lopo_mode(args.excel_file)
    elif args.output_file and args.name:
        run_single_split_mode(args.output_file, args.name)
    else:
        parser.error("Entweder --excel-file (LOPO) ODER --output-file zusammen mit --name (Einzelsplit) angeben.")


if __name__ == "__main__":
    main()