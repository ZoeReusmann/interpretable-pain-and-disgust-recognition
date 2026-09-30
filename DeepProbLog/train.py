import argparse
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(__file__))

from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.optimizer import SGD
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model
from deepproblog.evaluate import get_fact_accuracy

from pfr_dataset import PFRDataset
"""
Trainingsskript für Modell A/B/C, subject-wise Train/Test-Split (siehe preprocessing/make_split.py).
Kein neuronales Netz nötig - CNN ist eingefroren, alle AU-Wahrscheinlichkeiten
wurden bereits offline in DeepProbLog/facts/au_facts.pl aggregiert.
Dieses Skript loggt NICHT selbst. das übernimmt protocol.py.
Ausgabe umfasst neben Accuracy auch Precision, Recall, F1 und Specificity
(gleiche Metriken wie im CNN-Teil, analyze_confusion_matrix.py), damit die
Ergebnisse über die gesamte Pipeline hinweg vergleichbar sind.

Direkter Aufruf (zum schnellen Testen/Debuggen, ohne Protokollierung):
    python3 DeepProbLog/train.py --model model_a
Über den Runner (mit Protokollierung):
    python3 DeepProbLog/protocol.py --model model_b --name mein_lauf --dataset "PFR n40"
"""

MODEL_PATHS = {
    "model_a": "DeepProbLog/models/model_a_full.pl",
    "model_b": "DeepProbLog/models/model_b_full.pl",
    "model_c": "DeepProbLog/models/model_c_full.pl",
}

LABELS_CSV = "DeepProbLog/facts/video_labels.csv"
TRAIN_SPLIT = "DeepProbLog/facts/train_videos.txt"
TEST_SPLIT = "DeepProbLog/facts/test_videos.txt"


def load_split(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def compute_metrics(cm):
    """Berechnet Precision/Recall/F1/Specificity aus der DeepProbLog-
    ConfusionMatrix (cm.matrix[predicted_i, actual_i]). Klassen-Reihenfolge
    wird über cm.classes.index(...) ermittelt statt fest anzunehmen, da
    die Einfügereihenfolge je nach Lauf variieren kann."""
    idx_true = cm.classes.index("true")
    idx_false = cm.classes.index("false")

    tp = int(cm.matrix[idx_true, idx_true])
    fp = int(cm.matrix[idx_true, idx_false])
    fn = int(cm.matrix[idx_false, idx_true])
    tn = int(cm.matrix[idx_false, idx_false])

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
    }


def save_raw_probabilities(model, dataset, model_name, split_name, output_dir="DeepProbLog/results/probabilities"):
    """Schreibt pro Query (Video x Prädikat) die rohe DeepProbLog-Wahrscheinlichkeit
    in eine CSV, ohne Thresholding. Ergänzung zu get_fact_accuracy: nutzt dieselbe
    model.solve()-Logik, wirft die Wahrscheinlichkeit aber nicht weg."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    model.eval()
    for query in dataset.to_queries():
        term = query.substitute().query
        video_id = str(term.args[0])
        predicate = term.functor
        answer = model.solve([query])[0]
        predicted_p = float(answer.result[term]) if len(answer.result) > 0 else None
        rows.append({
            "video_id": video_id,
            "predicate": predicate,
            "predicted_probability": predicted_p,
            "actual_label": query.p,
        })

    out_path = os.path.join(output_dir, f"{model_name}_{split_name}_probabilities.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "predicate", "predicted_probability", "actual_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rohwahrscheinlichkeiten gespeichert: {out_path}")


def print_metrics(cm):
    m = compute_metrics(cm)
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall: {m['recall']:.4f}")
    print(f"Specificity: {m['specificity']:.4f}")
    print(f"F1: {m['f1']:.4f}")
    print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def main(model_name, epochs, learning_rate):
    program_path = MODEL_PATHS[model_name]

    model = Model(program_path, [])
    model.set_engine(ExactEngine(model))
    model.optimizer = SGD(model, learning_rate)

    train_videos = load_split(TRAIN_SPLIT)
    test_videos = load_split(TEST_SPLIT)

    train_set = PFRDataset(LABELS_CSV, video_atoms_subset=train_videos)
    test_set = PFRDataset(LABELS_CSV, video_atoms_subset=test_videos)
    print(f"{model_name}: Train {len(train_set)} Queries ({len(train_set)//2} Videos), "
          f"Test {len(test_set)} Queries ({len(test_set)//2} Videos).")

    loader = DataLoader(train_set, 8, False)
    train_model(model, loader, epochs, log_iter=20)

    print("Gelernte Parameter (Reihenfolge wie in der .pl-Datei deklariert):")
    print(model.parameters)

    print("\n--- Accuracy auf TRAIN ---")
    train_cm = get_fact_accuracy(model, train_set, verbose=1)
    print_metrics(train_cm)

    print("\n--- Accuracy auf TEST (ungesehene Probanden) ---")
    test_cm = get_fact_accuracy(model, test_set, verbose=1)
    print_metrics(test_cm)

    save_raw_probabilities(model, train_set, model_name, "train")
    save_raw_probabilities(model, test_set, model_name, "test")

    os.makedirs("DeepProbLog/snapshots", exist_ok=True)
    model.save_state(f"DeepProbLog/snapshots/{model_name}.pth")
    print(f"Modell gespeichert: DeepProbLog/snapshots/{model_name}.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepProbLog Modell A/B/C trainieren.")
    parser.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.epochs, args.learning_rate)