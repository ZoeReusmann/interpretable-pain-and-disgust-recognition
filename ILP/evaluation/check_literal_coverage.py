import argparse
import subprocess
import tempfile
import os
import re
"""
Ziel: Prüft für jede AU, wie viele positive und negative Beispiele ein bestimmtes
Prädikat (z.B. duration_long/2 oder overlap/3) überhaupt erfüllen -
unabhängig von FastLAS. Zeigt VOR dem ILP-Lauf, ob in den Daten selbst
überhaupt ein trennendes Signal steckt, oder ob eine gelernte Regel in
Wahrheit nie oder fast nie feuert.

Beispiel ausführen:
    python3 check_literal_coverage.py \
        --background oracle/background_rules_only.pl \
        --facts oracle/data/pfr_oracle_facts_tiny.pl \
        --examples oracle/examples/examples_pain_oracle_tiny.las \
        --predicate pain \
        --check duration_long/2
"""
def parse_examples(examples_path, predicate):
    labels = {}
    pattern = re.compile(
        r'#(pos|neg)\([^,]+,\s*\{' + re.escape(predicate) + r'\((\w+)\)\}'
    )
    with open(examples_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                kind, vid = m.groups()
                labels[vid] = (kind == 'pos')
    return labels


def run_show(background_path, facts_path, show_predicate, timeout=300):
    arity_pred, arity = show_predicate.split("/")
    combined = tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False)
    try:
        for path in (background_path, facts_path):
            with open(path) as f:
                combined.write(f.read())
                combined.write("\n")
        combined.write(f"#show {arity_pred}/{arity}.\n")
        combined.close()

        result = subprocess.run(
            ["clingo", combined.name, "-n", "1", "--outf=0"],
            capture_output=True, text=True, timeout=timeout,
        )
        atoms = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith(arity_pred + "("):
                atoms.extend(line.split())
        return [a for a in atoms if a.startswith(arity_pred + "(")]
    finally:
        os.unlink(combined.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--predicate", default="pain")
    parser.add_argument("--check", required=True, help="z.B. duration_long/2 oder overlap/3")
    args = parser.parse_args()

    labels = parse_examples(args.examples, args.predicate)
    atoms = run_show(args.background, args.facts, args.check)

    # Für jede AU (letztes Argument des Pädikats) sammeln wir, welche
    # Videos (erstes Argument) das Prädikat erfüllen.
    au_to_videos = {}
    for atom in atoms:
        inner = atom[atom.index("(") + 1:atom.rindex(")")]
        parts = [p.strip() for p in inner.split(",")]
        video = parts[0]
        au = parts[-1]
        au_to_videos.setdefault(au, set()).add(video)

    pos_videos = {v for v, is_pos in labels.items() if is_pos}
    neg_videos = {v for v, is_pos in labels.items() if not is_pos}

    print(f"{args.check} - Abdeckung pro AU (von {len(pos_videos)} pos / {len(neg_videos)} neg Videos):\n")
    print(f"{'AU':<8}{'#pos erfüllt':<16}{'#neg erfüllt':<16}")
    for au in sorted(au_to_videos):
        covered = au_to_videos[au]
        pos_hit = len(covered & pos_videos)
        neg_hit = len(covered & neg_videos)
        print(f"{au:<8}{pos_hit:<16}{neg_hit:<16}")

    print("\nHinweis: AUs, die hier gar nicht auftauchen, werden für KEIN "
          "Video im Datensatz erfüllt - eine Regel darauf feuert nie.")


if __name__ == "__main__":
    main()