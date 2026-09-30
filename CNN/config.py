import os

BASE_DATA_DIR = os.getenv("DATA_DIR", "/path/to/data")

IMAGE_DIR = os.path.join(BASE_DATA_DIR, "disfa_plus", "Images")
LABEL_DIR = os.path.join(BASE_DATA_DIR, "disfa_plus", "Labels")

# Kontext
TARGET_AUS = ["AU4", "AU6", "AU9", "AU17", "AU25", "AU26"]

SPLIT_FILE = "split_indices.pt"

# Trainingszeugs
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_SUBJECTS = ['SN001', 'SN003', 'SN007', 'SN009', 'SN010', 'SN025'] #67 % der Daten
VAL_SUBJECTS   = ['SN013', 'SN027'] # 22 % der Daten
TEST_SUBJECTS  = ['SN004'] # 11 % der Daten
NUM_WORKERS = 0
SEED = 11
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.001
FREEZE_BACKBONE = True

#Scheduler
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MODE = "min"

# Training festhalten
CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs"
SAVE_EVERY = 1

# Evaluation Thresholds
GLOBAL_THRESHOLD = 0.50

# AU_SPECIFIC_THRESHOLDS = { # für freezed backbone
#     "AU4": 0.35,
#     "AU6": 0.40,
#     "AU9": 0.25,
#     "AU17": 0.25,
#     "AU25": 0.40,
#     "AU26": 0.25,
# }

AU_SPECIFIC_THRESHOLDS = { # für unfreezed backbone
    "AU4": 0.25,
    "AU6": 0.25,
    "AU9": 0.35,
    "AU17": 0.30,
    "AU25": 0.35,
    "AU26": 0.25,
}

POS_WEIGHT_THRESHOLDS = {
    "AU4": 0.45,
    "AU6": 0.55,
    "AU9": 0.45,
    "AU17": 0.40,
    "AU25": 0.65,
    "AU26": 0.55,
}

OVERSAMPLING_THRESHOLDS = {
    "AU4": 0.25,
    "AU6": 0.25,
    "AU9": 0.35,
    "AU17": 0.20,
    "AU25": 0.50,
    "AU26": 0.45,
}

# Optionen: "global", "au_specific", "pos_weights", "globalsss", "oversampling"
DECISION_THRESHOLD_MODE = "oversampling"

#evaluation cnn ende
ALL_SUBJECTS = [
    "SN001", "SN003", "SN004", "SN007", "SN009",
    "SN010", "SN013", "SN025", "SN027"
]

LOPO_TEST_SUBJECT = os.getenv("LOPO_TEST_SUBJECT")

if LOPO_TEST_SUBJECT is not None:
    TEST_SUBJECTS = [LOPO_TEST_SUBJECT]

    remaining_subjects = [
        s for s in ALL_SUBJECTS
        if s != LOPO_TEST_SUBJECT
    ]

    VAL_SUBJECTS = [remaining_subjects[-1]]
    TRAIN_SUBJECTS = remaining_subjects[:-1]

# MODEL_NAME = "facenet_vggface2_au_threshold" model name für alles was nicht finale Evaluation ist
MODEL_NAME = f"lopo_{LOPO_TEST_SUBJECT}_facenet_vggface2_au_threshold_oversampling"





