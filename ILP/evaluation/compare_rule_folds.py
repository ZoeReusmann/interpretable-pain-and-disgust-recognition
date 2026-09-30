import argparse
import itertools
import json
from utils import split_rules, split_body_literals, canonicalize_literal

"""
Ziel: Regeln werden über Lopo Folds hinweg verglichen
Ziel des Ziels: Regelstabilität
Wie? ähnliche Bodyliterale (AU-Kombinationen, Prädikate) prüfen, ob sie in allen Folds vorkommen oder nicht.

Variablen werden kanonisiert (V statt konkretem Variablennamen), Konstanten
(z.B. au4) bleiben erhalten, damit Literale über Folds hinweg vergleichbar sind.

Beispiel ausführen: 
    python3 compare_rule_folds.py \
        --rules learned_rules/pain_fold_1.lp learned_rules/pain_fold_2.lp learned_rules/pain_fold_3.lp \
        --out results/fold_stability.json
"""

def literal_set(rule_path):
    with open(rule_path) as f:
        text = f.read()
    literals = set()
    for rule in split_rules(text):
        for lit in split_body_literals(rule):
            literals.add(canonicalize_literal(lit))
    return literals


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union) if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="+", required=True, help="Ein .lp-File pro Fold")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    fold_literals = {path: literal_set(path) for path in args.rules}

    print("Literale pro Fold:")
    for path, lits in fold_literals.items():
        print(f"  {path}: {sorted(lits)}")

    all_literals = set()
    for lits in fold_literals.values():
        all_literals |= lits

    stable_all = set.intersection(*fold_literals.values()) if fold_literals else set()

    per_literal_count = {
        lit: sum(1 for lits in fold_literals.values() if lit in lits)
        for lit in all_literals
    }
    n_folds = len(fold_literals)

    print(f"\nLiterale in ALLEN {n_folds} Folds (voll stabil): {sorted(stable_all)}")
    print("\nLiteral-Haeufigkeit ueber Folds (Literal: in wie vielen von N Folds):")
    for lit, count in sorted(per_literal_count.items(), key=lambda x: -x[1]):
        print(f"  {lit}: {count}/{n_folds}")

    pairwise = {}
    for a, b in itertools.combinations(fold_literals.keys(), 2):
        pairwise[f"{a} vs {b}"] = jaccard(fold_literals[a], fold_literals[b])

    print("\nPaarweise Jaccard-Aehnlichkeit zwischen Folds:")
    for pair, sim in pairwise.items():
        print(f"  {pair}: {sim:.2f}")

    if args.out:
        result = {
            "n_folds": n_folds,
            "fold_literals": {k: sorted(v) for k, v in fold_literals.items()},
            "stable_in_all_folds": sorted(stable_all),
            "literal_frequency_across_folds": per_literal_count,
            "pairwise_jaccard": pairwise,
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDetails gespeichert in: {args.out}")


if __name__ == "__main__":
    main()