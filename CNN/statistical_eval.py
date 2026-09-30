import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

MANIFEST = (
    "./"
    "data_analysis/outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv"
)
ORACLE_DIR = (
    "./"
    "ILP/oracle/data"
)
AUS = ["au4", "au6", "au9", "au17", "au25", "au26"]
PAIRS = [("au6", "au9"), ("au6", "au4"), ("au4", "au17")]

FACT_RE = re.compile(
    r"""(?<![\w])active\s*\(\s*"?([\w\-]+)"?\s*,\s*"?(au\d+)"?\s*,\s*(\d+)\s*\)""",
    re.IGNORECASE,
)


def norm(x):
    s = str(x).strip().lower()
    return "disgust" if "disgust" in s else ("pain" if "pain" in s else s)


man = pd.read_csv(MANIFEST)
man["label"] = man["label"].map(norm)
man["key"] = man["video_id"].astype(str).str.strip().str.lower()

recs = []
for name in ["pfr_oracle_facts_train.pl", "pfr_oracle_facts_test.pl"]:
    f = Path(ORACLE_DIR) / name
    found = FACT_RE.findall(f.read_text(errors="ignore"))
    vids = {v.lower() for v, _, _ in found}
    print(f"{name:35s} {len(found):7d} Fakten   {len(vids):4d} Videos")
    recs.extend(found)

facts = pd.DataFrame(recs, columns=["video_id", "au", "frame"])
facts["key"] = facts["video_id"].str.lower()
facts["au"] = facts["au"].str.lower()
facts["frame"] = facts["frame"].astype(int)
facts = facts.drop_duplicates(subset=["key", "au", "frame"])

man["has_facts"] = man["key"].isin(set(facts["key"]))

print("\n" + "=" * 68)
print("FRAGE 1: Sind die fehlenden Videos gleich auf beide Klassen verteilt?")
print("=" * 68)
ct = pd.crosstab(man["label"], man["has_facts"])
ct.columns = ["ohne Fakten", "mit Fakten"]
ct["Anteil mit Fakten"] = (ct["mit Fakten"] / ct.sum(axis=1)).round(3)
print(ct.to_string())

chi2, p_miss, _, _ = chi2_contingency(
    pd.crosstab(man["label"], man["has_facts"]).values
)
print(f"\nChi-Quadrat-Test auf Unabhaengigkeit: chi2 = {chi2:.3f}, p = {p_miss:.4f}")
if p_miss < .05:
    print(">>> ABDECKUNG IST KLASSENABHAENGIG. Die Nullen sind konfundiert.")
    print(">>> Die Ergebnisse aus statistical_eval.py sind so NICHT gueltig.")
else:
    print(">>> Abdeckung ist klassenunabhaengig. Nullauffuellen ist vertretbar,")
    print(">>> muss aber im Text begruendet werden.")

print("\n" + "=" * 68)
print("FRAGE 2: Existieren fuer die fehlenden Videos ueberhaupt Annotationen?")
print("=" * 68)
col = "coding_key_file" if "coding_key_file" in man.columns else None
if col:
    miss = man.loc[~man["has_facts"], col].dropna().astype(str)
    exists = sum(Path(x).exists() for x in miss)
    print(f"{len(miss)} fehlende Videos mit Eintrag in '{col}'")
    print(f"davon Datei vorhanden: {exists}")
    if exists == 0:
        print(">>> Pfade relativ? Dann manuell pruefen:")
        print(f">>> Beispiel: {miss.iloc[0] if len(miss) else '-'}")
    elif exists < len(miss):
        print(">>> Ein Teil der Annotationsdateien fehlt -> fehlende Werte,")
        print(">>> keine strukturellen Nullen. Diese Videos ausschliessen.")
    else:
        print(">>> Alle Annotationsdateien existieren -> die Nullen sind")
        print(">>> vermutlich echt (keine der 6 AUs aktiv).")

print("\n" + "=" * 68)
print("FRAGE 3: Wie sehen die Tests aus, wenn nur die 237 annotierten")
print("         Videos einbezogen werden?")
print("=" * 68)

sub = man[man["has_facts"]].copy()
print(f"Teilstichprobe: {(sub.label=='pain').sum()} pain / "
      f"{(sub.label=='disgust').sum()} disgust")

dur = (facts[facts["au"].isin(AUS)]
       .groupby(["key", "au"]).size().unstack(fill_value=0)
       .reindex(columns=AUS, fill_value=0)
       .reindex(sub["key"].values, fill_value=0)
       .reset_index(names="key").merge(sub[["key", "label"]], on="key"))

ov = pd.DataFrame({"key": sub["key"].values})
for a, b in PAIRS:
    fa = facts[facts["au"] == a][["key", "frame"]]
    fb = facts[facts["au"] == b][["key", "frame"]]
    cnt = fa.merge(fb, on=["key", "frame"]).groupby("key").size().rename(f"{a}-{b}")
    ov = ov.merge(cnt.reset_index(), on="key", how="left")
ov = ov.fillna(0).merge(sub[["key", "label"]], on="key")

rows = []
for feat, frame, col_ in ([(f"duration {a}", dur, a) for a in AUS] +
                          [(f"overlap {a}-{b}", ov, f"{a}-{b}") for a, b in PAIRS]):
    p_ = frame.loc[frame.label == "pain", col_].values
    d_ = frame.loc[frame.label == "disgust", col_].values
    U, pv = mannwhitneyu(p_, d_, alternative="two-sided", method="asymptotic")
    r = 1 - (2 * U) / (len(p_) * len(d_))
    rows.append(dict(feature=feat,
                     pain_md=np.median(p_), dis_md=np.median(d_),
                     pain_mean=round(np.mean(p_), 1), dis_mean=round(np.mean(d_), 1),
                     U=U, p=round(pv, 4), r=round(r, 4),
                     direction="disgust > pain" if np.mean(d_) > np.mean(p_)
                               else "pain > disgust"))
res = pd.DataFrame(rows)
print(res.to_string(index=False))

n_dis = (res["direction"] == "disgust > pain").sum()
print(f"\nRichtungsverteilung: {n_dis} von {len(res)} zugunsten disgust")
if n_dis in (0, len(res)):
    print(">>> immer noch einseitig -- auf systematischen Effekt pruefen")