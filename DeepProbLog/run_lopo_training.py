import os
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.optimizer import SGD
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model
from pfr_dataset import PFRDataset
from train import MODEL_PATHS
"""
Ziel: LOPO Trainingslauf für Modell A/B/C.

Analog zum CNN-Pendant (run_lopo_training.sh): für jeden Probanden wird dessen/deren Video(s) als Test-Fold zurückgehalten, alle anderen Probanden
dienen als Training. Anders als beim CNN (ein Subprocess-Aufruf pro Proband über die Shell) läuft hier alles in einem einzigen Python-Prozess,
da kein GPU-Wechsel/CUDA_VISIBLE_DEVICES nötig ist - DeepProbLog trainiert hier ohne neuronales Netz, nur die Regelgewichte.

Pro Fold wird der gelernte Zustand (t(_)-Parameter) als eigene Datei
gespeichert - das Pendant zu einem CNN-Checkpoint pro Test-Proband.

Aufruf:
    python3 DeepProbLog/run_lopo_training.py
"""

LABEL_CSV = "data_analysis/outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv"
LABELS_CSV = "DeepProbLog/facts/video_labels.csv"
EPOCHS = 50
LEARNING_RATE = 1.0
CHECKPOINT_DIR = "DeepProbLog/lopo_checkpoints"

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

def train_fold(model_name, train_videos):
    program_path = MODEL_PATHS[model_name]
    model = Model(program_path, [])
    model.set_engine(ExactEngine(model))
    model.optimizer = SGD(model, LEARNING_RATE)

    train_set = PFRDataset(LABELS_CSV, video_atoms_subset=set(train_videos))
    loader = DataLoader(train_set, 8, False)
    train_model(model, loader, EPOCHS, log_iter=10_000)

    return model

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    subject_map = load_subject_video_map()
    subjects = sorted(subject_map.keys())
    all_videos = [v for vids in subject_map.values() for v in vids]

    total_folds = len(subjects) * len(MODEL_PATHS)
    done = 0
    start_all = time.monotonic()

    for subject in subjects:
        print(f"\nStarte LOPO Training für Test-Proband: {subject}")

        test_videos = subject_map[subject]
        test_lookup = set(test_videos)
        train_videos = [v for v in all_videos if v not in test_lookup]

        for model_name in MODEL_PATHS:
            done += 1
            t0 = time.monotonic()

            model = train_fold(model_name, train_videos)

            out_path = checkpoint_path(model_name, subject)
            model.save_state(out_path)

            dt = time.monotonic() - t0
            elapsed = (time.monotonic() - start_all) / 60
            print(f"  [{done}/{total_folds}] {model_name}: gespeichert nach {out_path} "
                  f"({dt:.1f}s, gesamt {elapsed:.1f} min)")

        print(f"Fertig mit {subject}")

    print(f"\nGesamtlaufzeit: {(time.monotonic() - start_all) / 60:.1f} Minuten")
    print(f"Alle Checkpoints liegen in {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()