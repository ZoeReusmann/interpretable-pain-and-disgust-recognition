import argparse
import json
import math
import sys
from utils import parse_examples, run_clingo

"""
Ziel: Auswertung des ILP Teils mit Berechnung von 
- Accuracy, 
- Precision, 
- Recall, 
- F1, 
- MCC,
- Confusion-Matrix-Zahlen
(TP/FP/TN/FN) für eine gelernte FastLAS-Regel gegen Testdaten

Speichert die Rohzählungen zusätzlich als JSON, damit plot_confusion_matrix.py
sie separat visualisieren kann (wie gewünscht: getrenntes Skript für die Matrix).

Beispiel auführen 
    python3 evaluate_metrics.py \
        --background background/background_clean.pl \
        --facts data/pfr_facts_test.pl \
        --rule learned_rules/pain_rule.lp \
        --examples examples/examples_pain_test.las \
        --predicate pain \
        --out results/metrics_pain.json
"""

def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# mcc, weil gut für stark ausbalancierte Daten, wie PR_AUC bei dem CNN
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) # nenner der mcc formel
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0

    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "mcc": mcc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--predicate", default="pain")
    parser.add_argument("--out", default=None, help="Pfad fuer JSON-Ausgabe der Rohzaehlungen")
    args = parser.parse_args()

    ground_truth = parse_examples(args.examples, args.predicate)
    if not ground_truth:
        print("WARNUNG: Keine Beispiele geparst - Predicate-Name/Dateiformat pruefen.")
        sys.exit(1)

    derived = run_clingo(args.background, args.facts, args.rule, args.predicate)

    tp = fp = tn = fn = 0
    mismatches = []
    for vid, is_pos in ground_truth.items():
        predicted = vid in derived
        if is_pos and predicted:
            tp += 1
        elif is_pos and not predicted:
            fn += 1
        elif not is_pos and predicted:
            fp += 1
        else:
            tn += 1
        if is_pos != predicted:
            mismatches.append({"video": vid, "actual": is_pos, "predicted": predicted})

    metrics = compute_metrics(tp, fp, tn, fn)
    metrics["mismatches"] = mismatches
    metrics["predicate"] = args.predicate
    metrics["rule_file"] = args.rule

    print(f"Test-Videos gesamt: {metrics['total']}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"MCC:       {metrics['mcc']:.3f}")

    if mismatches:
        print(f"\n{len(mismatches)} Fehlklassifikationen (erste 20):")
        for m in mismatches[:20]:
            print(f"  {m['video']}: wahr={'pos' if m['actual'] else 'neg'}, "
                  f"vorhergesagt={'pos' if m['predicted'] else 'neg'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nRohdaten gespeichert in: {args.out}")


if __name__ == "__main__":
    main()