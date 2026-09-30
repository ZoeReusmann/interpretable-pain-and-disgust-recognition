from pathlib import Path
import random
import re

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
EXAMPLES_DIR = BASE_DIR / "examples"

FACTS_IN = DATA_DIR / "pfr_facts.pl"

PAIN_EXAMPLES_IN = EXAMPLES_DIR / "examples_pain.las"
DISGUST_EXAMPLES_IN = EXAMPLES_DIR / "examples_disgust.las"

FACTS_TRAIN_OUT = DATA_DIR / "pfr_facts_train.pl"
FACTS_TEST_OUT = DATA_DIR / "pfr_facts_test.pl"

PAIN_TRAIN_OUT = EXAMPLES_DIR / "examples_pain_train.las"
PAIN_TEST_OUT = EXAMPLES_DIR / "examples_pain_test.las"

DISGUST_TRAIN_OUT = EXAMPLES_DIR / "examples_disgust_train.las"
DISGUST_TEST_OUT = EXAMPLES_DIR / "examples_disgust_test.las"

TEST_RATIO = 0.2
SEED = 11


def get_video_from_fact(line):
    match = re.search(
        r"^(?:video|frame|original_frame|time_s|active)\(([^,\)]+)",
        line.strip()
    )
    return match.group(1) if match else None


def get_video_from_example(line):
    match = re.search(r"\{(?:pain|disgust)\(([^)]+)\)\}", line)
    return match.group(1) if match else None

def get_subject(video):
    return video.split("_")[0]


def collect_videos_from_facts(path):
    videos = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            video = get_video_from_fact(line)
            if video:
                videos.add(video)

    return videos


def split_subjects(videos):
    subjects = sorted({get_subject(video) for video in videos})

    random.seed(SEED)
    random.shuffle(subjects)

    n_test = max(1, round(len(subjects) * TEST_RATIO))

    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    train_videos = {v for v in videos if get_subject(v) in train_subjects}
    test_videos = {v for v in videos if get_subject(v) in test_subjects}

    return train_subjects, test_subjects, train_videos, test_videos


def write_split_facts(in_path, train_path, test_path, train_videos, test_videos):
    with in_path.open("r", encoding="utf-8") as f, \
         train_path.open("w", encoding="utf-8") as train_file, \
         test_path.open("w", encoding="utf-8") as test_file:

        for line in f:
            video = get_video_from_fact(line)

            if video in train_videos:
                train_file.write(line)
            elif video in test_videos:
                test_file.write(line)


def write_split_examples(in_path, train_path, test_path, train_videos, test_videos):
    with in_path.open("r", encoding="utf-8") as f, \
         train_path.open("w", encoding="utf-8") as train_file, \
         test_path.open("w", encoding="utf-8") as test_file:

        for line in f:
            video = get_video_from_example(line)

            if video in train_videos:
                train_file.write(line)
            elif video in test_videos:
                test_file.write(line)
            elif line.strip() == "":
                train_file.write(line)
                test_file.write(line)
            elif line.strip().startswith("%"):
                train_file.write(line)
                test_file.write(line)


def main():
    videos = collect_videos_from_facts(FACTS_IN)

    train_subjects, test_subjects, train_videos, test_videos = split_subjects(videos)

    write_split_facts(FACTS_IN, FACTS_TRAIN_OUT, FACTS_TEST_OUT, train_videos, test_videos)
    write_split_examples(PAIN_EXAMPLES_IN, PAIN_TRAIN_OUT, PAIN_TEST_OUT, train_videos, test_videos)
    write_split_examples(DISGUST_EXAMPLES_IN, DISGUST_TRAIN_OUT, DISGUST_TEST_OUT, train_videos, test_videos)

    print("Done.")
    print(f"Subjects total: {len(train_subjects) + len(test_subjects)}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")

    print("\nTest subjects:")
    for subject in sorted(test_subjects):
        print(subject)


if __name__ == "__main__":
    main()