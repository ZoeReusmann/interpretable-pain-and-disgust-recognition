import argparse
import json
from collections import Counter

from utils import split_rules, split_body_literals, predicate_name, AU_PATTERN, TEMPORAL_PREDICATES

"""
Ziel: strukturelle Interpretierbarkeit der gelernten Regeln analysieren
- Anzahl gelernter Regeln
- Durchschnittliche Regellänge (Anzahl Body-Literale)
- Häufigkeit der verwendeten AUs (au4, au6, ...)
- Häufigkeit der verwendeten Praedikate, insbesondere temporaler
  (overlap, before, duration_long, ...)

Beispiel:
    python3 analyze_rule_structure.py --rule learned_rules/pain_rule.lp --out results/structure_pain.json
"""


def analyze(rule_path):
    with open(rule_path) as f:
        text = f.read()

    rules = split_rules(text)
    rule_lengths = []
    predicate_counter = Counter()
    au_counter = Counter()
    temporal_counter = Counter()

    for rule in rules:
        literals = split_body_literals(rule)
        rule_lengths.append(len(literals))
        for lit in literals:
            pred = predicate_name(lit)
            predicate_counter[pred] += 1
            if pred in TEMPORAL_PREDICATES:
                temporal_counter[pred] += 1
        for au in AU_PATTERN.findall(rule):
            au_counter[au] += 1

    avg_length = sum(rule_lengths) / len(rule_lengths) if rule_lengths else 0.0

    return {
        "rule_file": rule_path,
        "num_rules": len(rules),
        "rule_lengths": rule_lengths,
        "avg_rule_length": avg_length,
        "predicate_frequency": dict(predicate_counter.most_common()),
        "temporal_predicate_frequency": dict(temporal_counter.most_common()),
        "au_frequency": dict(au_counter.most_common()),
        "rules_text": rules,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = analyze(args.rule)

    print(f"Datei: {result['rule_file']}")
    print(f"Anzahl gelernter Regeln: {result['num_rules']}")
    print(f"Durchschnittliche Regellänge (Body-Literale): {result['avg_rule_length']:.2f}")
    print(f"Regellängen einzeln: {result['rule_lengths']}")

    print("\nPrädikat-Häufigkeit im Body:")
    for pred, count in result["predicate_frequency"].items():
        marker = " (temporal)" if pred in TEMPORAL_PREDICATES else ""
        print(f"  {pred}: {count}{marker}")

    print("\nAU- Häufigkeit:")
    for au, count in result["au_frequency"].items():
        print(f"  {au}: {count}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDetails gespeichert in: {args.out}")


if __name__ == "__main__":
    main()