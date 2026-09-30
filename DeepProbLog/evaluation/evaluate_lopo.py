import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_fact_accuracy

from pfr_dataset import PFRDataset
from train import MODEL_PATHS, compute_metrics

"""
Ziel: LOPO Evaluation für Modell A/B/C.

Analog zum CNN-Pendant (evaluate_lopo.py): für jeden Test-Probanden muss
ein eigener Checkpoint existieren (aus run_lopo_training.py). Hier werden
alle Checkpoints geladen, auf dem jeweils zurückgehaltenen Probanden
ausgewertet, und die Ergebnisse zusammengeführt.

Ziel, analog zum CNN-Skript:
- globale LOPO-Metriken (gepoolt über alle Folds/Modelle)
- Metriken pro Modell (Pendant zu "Mittelwerte pro AU"), gepoolt über alle Folds
- alle einzelnen Folds/Probanden separat

Aufruf:
    python3 DeepProbLog/evaluation/evaluate_lopo.py
"""

LABEL_CSV = "data_analysis/outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv"
LABELS_CSV = "DeepProbLog/facts/video_labels.csv"

CHECKPOINT_DIR = "DeepProbLog/lopo_checkpoints"
OUTPUT_DIR = "DeepProbLog/results/lopo_evaluation"


def safe_atom(video_id):
    """Muss identisch zu aggregate_au_probs.py sein, damit Video-Atome übereinstimmen."""
    v = video_id.lower()
    v = "".join(c if (c.isalnum() or c == "_") else "_" for c in v)
    if not v[0].isalpha():
        v = "v_" + v
    return v


def load_subject_video_map():
    df = pd.read_csv(LABEL_CSV)
    mapping = {}
    for _, row in df.iterrows():
        subj = row["subject_id"]
        vid_atom = safe_atom(row["video_id"])
        mapping.setdefault(subj, []).append(vid_atom)
    return mapping


def checkpoint_path(model_name, test_subject):
    return os.path.join(CHECKPOINT_DIR, f"lopo_{model_name}_{test_subject}.dplog")


def load_model_from_checkpoint(model_name, checkpoint_path_):
    program_path = MODEL_PATHS[model_name]
    model = Model(program_path, [])
    model.set_engine(ExactEngine(model))
    model.load_state(checkpoint_path_)
    return model


def metrics_from_counts(tp, fp, fn, tn):
    """Einheitliche Berechnung von Accuracy/Precision/Recall/Specificity/F1
    aus TP/FP/FN/TN - wird sowohl pro Fold als auch für gepoolte Summen
    verwendet, damit die Formel garantiert an nur einer Stelle steht."""
    eps = 1e-8
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / (total + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
    }


def main():
    print("evaluate_lopo.py startet")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    subject_map = load_subject_video_map()
    subjects = sorted(subject_map.keys())

    all_results = []

    for test_subject in subjects:
        print(f"\nLOPO Evaluation für Test-Proband: {test_subject}")

        test_videos = subject_map[test_subject]

        for model_name in MODEL_PATHS:
            ckpt = checkpoint_path(model_name, test_subject)

            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"Checkpoint für {model_name}/{test_subject} nicht gefunden: {ckpt}"
                )

            model = load_model_from_checkpoint(model_name, ckpt)
            test_set = PFRDataset(LABELS_CSV, video_atoms_subset=set(test_videos))

            cm = get_fact_accuracy(model, test_set, verbose=0)
            m = compute_metrics(cm)

            # Fold-Verhältnisse - nur zur Anzeige in "all_folds_per_model",
            # NICHT für die weitere Aggregation verwendet.
            fold_metrics = metrics_from_counts(m["tp"], m["fp"], m["fn"], m["tn"])

            all_results.append({
                "test_subject": test_subject,
                "model": model_name,
                "support_total": m["tp"] + m["fp"] + m["fn"] + m["tn"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "accuracy": fold_metrics["accuracy"],
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
                "specificity": fold_metrics["specificity"],
                "f1": fold_metrics["f1"],
            })

    df_all = pd.DataFrame(all_results)

    pooled_counts = (
        df_all
        .groupby("model", as_index=False)
        .agg({"tp": "sum", "fp": "sum", "fn": "sum", "tn": "sum", "support_total": "sum"})
    )

    mean_by_model_rows = []
    for _, row in pooled_counts.iterrows():
        pooled_metrics = metrics_from_counts(row["tp"], row["fp"], row["fn"], row["tn"])
        mean_by_model_rows.append({
            "model": row["model"],
            "support_total": row["support_total"],
            "tp": row["tp"], "fp": row["fp"], "fn": row["fn"], "tn": row["tn"],
            "accuracy": pooled_metrics["accuracy"],
            "precision": pooled_metrics["precision"],
            "recall": pooled_metrics["recall"],
            "specificity": pooled_metrics["specificity"],
            "f1": pooled_metrics["f1"],
        })
    df_mean_by_model = pd.DataFrame(mean_by_model_rows)

    global_tp = df_all["tp"].sum()
    global_fp = df_all["fp"].sum()
    global_fn = df_all["fn"].sum()
    global_tn = df_all["tn"].sum()
    global_metrics = metrics_from_counts(global_tp, global_fp, global_fn, global_tn)

    df_global = pd.DataFrame([{
        "n_lopo_folds": df_all["test_subject"].nunique(),
        "n_models": df_all["model"].nunique(),
        "tp": global_tp, "fp": global_fp, "fn": global_fn, "tn": global_tn,
        "accuracy": global_metrics["accuracy"],
        "precision": global_metrics["precision"],
        "recall": global_metrics["recall"],
        "specificity": global_metrics["specificity"],
        "f1": global_metrics["f1"],
    }])

    output_path = os.path.join(OUTPUT_DIR, f"lopo_evaluation_{timestamp}.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="global_summary", index=False)
        df_mean_by_model.to_excel(writer, sheet_name="mean_by_model", index=False)
        df_all.to_excel(writer, sheet_name="all_folds_per_model", index=False)

    print("\nLOPO Evaluation gespeichert unter:")
    print(output_path)

    print("\nGlobal Summary (gepoolt über alle Modelle und Folds):")
    print(df_global)

    print("\nMean by Model (gepoolt pro Modell über alle Folds):")
    print(df_mean_by_model)


if __name__ == "__main__":
    main()