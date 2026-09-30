import re
import numpy as np
import pandas as pd

FACTS = "DeepProbLog/facts/au_facts.pl"
LABELS_CSV = "DeepProbLog/facts/video_labels.csv"

labels = pd.read_csv(LABELS_CSV)

fact_pattern = re.compile(r"^([\d.]+)\s*::\s*duration_long\(([^,]+),([^)]+)\)\.")

facts = {}
with open(FACTS) as f:
    for line in f:
        match = fact_pattern.match(line.strip())
        if not match:
            continue
        prob, video, au = match.groups()
        facts[(video.strip(), au.strip())] = float(prob)

print(f"Anzahl geparster Fakten: {len(facts)}")

def rule_prob(video, aus):
    return np.prod([facts.get((video, au), 0.0) for au in aus])

rules = {
    "pain_rule1_au4_6_9": ["au4", "au6", "au9"],
    "pain_rule2_au4_6_9_25": ["au4", "au6", "au9", "au25"],
    "disgust_rule1_au9": ["au9"],
    "disgust_rule2_au9_17": ["au9", "au17"],
    "disgust_rule3_au9_25": ["au9", "au25"],
}

for rule_name, aus in rules.items():
    values_pain, values_disgust = [], []
    for _, row in labels.iterrows():
        video = row["video_atom"]
        p = rule_prob(video, aus)
        if row["label"] == "pressure_pain":
            values_pain.append(p)
        elif row["label"] == "disgust":
            values_disgust.append(p)

    print(f"{rule_name}:")
    print(f"  Pain-Videos    (n={len(values_pain)}): mean={np.mean(values_pain):.4f}, median={np.median(values_pain):.4f}")
    print(f"  Disgust-Videos (n={len(values_disgust)}): mean={np.mean(values_disgust):.4f}, median={np.median(values_disgust):.4f}")
    print()
