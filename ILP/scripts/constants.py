from pathlib import Path

PFR_CNN_EXPORT = Path(
    "../CNN/outputs/predictions/pfr/pfr_cropped_frame_predictions_oversampling_thresholds_seed11.csv"
)
OUTPUT_DIR = Path("data")
FACTS_FILE = OUTPUT_DIR / "pfr_facts.pl"


TARGET_AUS = ["AU4", "AU6", "AU9","AU17","AU25","AU26"]
TARGET_LABELS = ["pain","disgust"]
FPS = 25

VIDEO_FACT = "video"
LABEL_FACT = "label"
FRAME_FACT = "frame"
ORIGINAL_FRAME_FACT = "original_frame"
TIME_FACT = "time_s"
ACTIVE_FACT = "active"