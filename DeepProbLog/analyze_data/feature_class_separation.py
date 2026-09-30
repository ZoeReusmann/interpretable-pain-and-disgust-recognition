import re
import os
import pandas as pd
"""
Berechnet pro aggregiertem AU-Merkmal (duration_long, overlap) den
Mittelwertunterschied zwischen den Klassen (disgust vs. pressure_pain)
und schreibt das Ergebnis als CSV weg - für reproduzierbare Diagnose
und direkte Weiterverwendung in der Arbeit (Tabelle/Anhang).
"""
FACTS_FILE = "DeepProbLog/facts/au_facts.pl"
LABELS_FILE = "DeepProbLog/facts/video_labels.csv"
OUT_DIR = "DeepProbLog/analyze_data"
OUT_CSV = os.path.join(OUT_DIR, "feature_class_separation.csv")

FACT_PATTERN = re.compile(
    r"([\d.]+) :: (duration_long|overlap)\(([a-z0-9_]+),([a-z0-9_,]+)\)\."
)

def load_facts(facts_file):
    facts = {}
    with open(facts_file) as f:
        for line in f:
            m = FACT_PATTERN.match(line.strip())
            if not m:
                continue
            p, pred, video, args = m.groups()
            key = f"{pred}({args})"
            facts.setdefault(key, {})[video] = float(p)
    return facts


def compute_separation(facts, labels):
    rows = []
    for key, video_map in facts.items():
        df = pd.DataFrame(list(video_map.items()), columns=["video_atom", "value"])
        merged = df.merge(labels, on="video_atom")
        means = merged.groupby("label")["value"].mean()
        if len(means) != 2:
            continue
        class_names = list(means.index)
        diff = abs(means.iloc[0] - means.iloc[1])
        rows.append(
            {
                "feature": key,
                f"mean_{class_names[0]}": means.iloc[0],
                f"mean_{class_names[1]}": means.iloc[1],
                "abs_diff": diff,
            }
        )
    result = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels = pd.read_csv(LABELS_FILE)
    facts = load_facts(FACTS_FILE)

    result = compute_separation(facts, labels)
    result.to_csv(OUT_CSV, index=False)

    print(result.to_string(index=False))
    print(f"\nGeschrieben nach {OUT_CSV}")


if __name__ == "__main__":
    main()