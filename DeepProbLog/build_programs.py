"""
Ziel: baut die finalen DeepProbLog-Programme, indem die generierten Fakten (au_facts.pl) mit den Regeldateien (model_a_rules.pl / model_b_rules.pl / model_c_rules.pl)
zusammengefügt werden. Model() akzeptiert nur EINEN Programmpfad, daher werden die Dateien hier konkateniert statt per :- include/consult
eingebunden (das ist ASP/Clingo-Syntax, kein Standard-ProbLog)
"""
import os

FACTS_FILE = "DeepProbLog/facts/au_facts.pl"
RULES_DIR = "DeepProbLog/models"
OUT_DIR = "DeepProbLog/models"

def build(rules_file, out_file):
    with open(FACTS_FILE) as f:
        facts = f.read()
    with open(rules_file) as f:
        rules = f.read()

    combined = facts + "\n\n%Regeln: \n\n" + rules

    with open(out_file, "w") as f:
        f.write(combined)
    print(f"Geschrieben: {out_file} ({len(combined.splitlines())} Zeilen)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build(
        os.path.join(RULES_DIR, "model_a_rules.pl"),
        os.path.join(OUT_DIR, "model_a_full.pl"),
    )
    build(
        os.path.join(RULES_DIR, "model_b_rules.pl"),
        os.path.join(OUT_DIR, "model_b_full.pl"),
    )
    build(
        os.path.join(RULES_DIR, "model_c_rules.pl"),
        os.path.join(OUT_DIR, "model_c_full.pl"),
    )