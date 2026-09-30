
import re
import subprocess
import tempfile
import os

"""
Ziel: gemeinsame Hilfsfunktionen für die FastLAS-Evaluationsskripte
"""

def parse_examples(examples_path, predicate):
    """
    Liest eine FastLAS .las Beispieldatei und gibt {video_id: True/False} zurück
    True = positives Beispiel (#pos)
    False = negatives Beispiel (#neg)
    """
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


def run_clingo(background_path, facts_path, rule_path, predicate, timeout=600):
    """
    Führt clingo mit Hintergrundwissen + Fakten + gelernter Regel aus und
    gibt die Menge der Video-IDs zurück, für die `predicate` ableitbar ist"""
    combined = tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False)
    try:
        for path in (background_path, facts_path, rule_path):
            with open(path) as f:
                combined.write(f.read())
                combined.write("\n")
        combined.write(f"#show {predicate}/1.\n")
        combined.close()

        result = subprocess.run(
            ["clingo", combined.name, "-n", "1", "--outf=0"],
            capture_output=True, text=True, timeout=timeout
        )
        derived = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith(predicate + "("):
                for atom in line.split():
                    m = re.match(re.escape(predicate) + r'\((\w+)\)', atom)
                    if m:
                        derived.add(m.group(1))
        return derived
    finally:
        os.unlink(combined.name)


def split_rules(rule_text):
    """
    Teilt Code in einzelne Regeln & Fakten auf
    Regelende = '.' gefolgt von Whitespace/Dateiende, aber nicht innerhalb
    von Klammern (damit z.B. 'F1 + 1.0' nicht fälschlich getrennt wird)
    """
    rules = []
    depth = 0
    current = []
    i = 0
    while i < len(rule_text):
        ch = rule_text[i]
        current.append(ch)
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '.' and depth == 0:
            nxt = rule_text[i + 1:i + 2]
            if nxt in ('', '\n', ' ', '\t', '\r'):
                text = ''.join(current).strip()
                if text and not text.startswith('%'):
                    rules.append(text)
                current = []
        i += 1
    remainder = ''.join(current).strip()
    if remainder and not remainder.startswith('%'):
        rules.append(remainder)
    return rules


def split_body_literals(rule_text):
    """
    Trennt den Body einer Regel ('head :- lit1, lit2, ...') in einzelne 
    Literale auf, Klammerbewusst (Kommas in Prädikat-Argumenten trennen nicht)
    """
    if ':-' not in rule_text:
        return []
    body = rule_text.split(':-', 1)[1].rstrip('. \n')
    literals = []
    depth = 0
    current = []
    for ch in body:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            literals.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    last = ''.join(current).strip()
    if last:
        literals.append(last)
    return [l for l in literals if l]


def predicate_name(literal):
    """
    Extrahiert den Praedikatnamen aus einem Literal, z.B.
    'overlap(V,au4,au6)' -> 'overlap'. Entfernt fuehrendes 'not '
    """
    lit = literal.strip()
    if lit.startswith('not '):
        lit = lit[4:].strip()
    m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', lit)
    return m.group(1) if m else lit


def canonicalize_literal(literal):
    """
    Ersetzt Variablen (Grossbuchstaben-Token) durch 'V', behält Konstanten bei, 
    damit Literale über verschiedene Folds vergleichbar sind
    """
    lit = literal.strip()
    if lit.startswith('not '):
        prefix, lit = 'not ', lit[4:].strip()
    else:
        prefix = ''

    def repl(m):
        token = m.group(0)
        return 'V' if token[0].isupper() else token

    canon = re.sub(r'[A-Za-z_][A-Za-z0-9_]*', repl, lit)
    return prefix + canon


AU_PATTERN = re.compile(r'\bau\d+\b')

TEMPORAL_PREDICATES = {
    'overlap', 'coactive', 'au_before', 'au_followed_within_3', 'au_followed_within_5',
    'starts_before', 'duration_long', 'before', 'next', 'active_rank', 'next_active',
    'exists_between', 'first_active', 'earlier_active', 'active_count',
}