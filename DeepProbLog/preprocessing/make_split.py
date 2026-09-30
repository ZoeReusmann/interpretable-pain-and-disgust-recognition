"""
Subject-wise Train/Test-Split, analog zum Vorgehen im ILP-Teil
(dort: 196/41 Videos, kein Proband in beiden Splits).

Dies ist KEIN identischer Split zum ILP-Teil (anderer Video-Pool: 479
statt 237), sondern methodisch gleichwertig: subject-wise, kein Proband
in beiden Sets.
"""
import csv
import random

LABEL_CSV = "data_analysis/outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv"
OUT_TRAIN = "DeepProbLog/facts/train_videos.txt"
OUT_TEST = "DeepProbLog/facts/test_videos.txt"

TEST_FRACTION = 0.2
SEED = 25


def safe_atom(video_id):
    v = video_id.lower()
    v = "".join(c if (c.isalnum() or c == "_") else "_" for c in v)
    if not v[0].isalpha():
        v = "v_" + v
    return v


def main():
    rows = []
    with open(LABEL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    subjects = sorted({r["subject_id"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(subjects)

    n_test_subjects = max(1, int(len(subjects) * TEST_FRACTION))
    test_subjects = set(subjects[:n_test_subjects])
    train_subjects = set(subjects[n_test_subjects:])

    train_videos = [safe_atom(r["video_id"]) for r in rows if r["subject_id"] in train_subjects]
    test_videos = [safe_atom(r["video_id"]) for r in rows if r["subject_id"] in test_subjects]

    with open(OUT_TRAIN, "w") as f:
        f.write("\n".join(train_videos) + "\n")
    with open(OUT_TEST, "w") as f:
        f.write("\n".join(test_videos) + "\n")

    def label_counts(video_set, rows):
        counts = {}
        for r in rows:
            if safe_atom(r["video_id"]) in video_set:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
        return counts

    print(f"Subjects: {len(subjects)} total, {len(train_subjects)} train, {len(test_subjects)} test")
    print(f"Videos: {len(train_videos)} train, {len(test_videos)} test")
    print("Label-Verteilung Train:", label_counts(set(train_videos), rows))
    print("Label-Verteilung Test:", label_counts(set(test_videos), rows))
    print(f"Geschrieben: {OUT_TRAIN}, {OUT_TEST}")


if __name__ == "__main__":
    main()
