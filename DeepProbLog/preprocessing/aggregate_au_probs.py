"""
Aggregiert Frame-Ebene AU-Wahrscheinlichkeiten (aus dem eingefrorenen CNN) zu
Video-Ebene probabilistischen Fakten fuer DeepProbLog (duration_long, overlap,
presence) sowie eine Query-Datei mit den Pain/Disgust-Labels.

Nutzt exakt die 479 Videos aus pfr_n40_pressure_pain_vs_disgust_frontal.csv,
gematcht (case-insensitive) gegen die grosse Frame-Prediction-CSV.

duration_long nutzt einen RELATIVEN Schwellenwert: AU muss in >=30% der
Videolaenge aktiv sein. Eine feste Schwelle (>=10 Frames) war bei ~175 Frames
im Schnitt pro Video zu niedrig (~5,7% der Laenge) und fuehrte zu extremer
Saettigung (>=63% aller Werte nahe 1.0).
"""
import os
import numpy as np
import pandas as pd

PRED_CSV = "CNN/outputs/predictions/pfr/pfr_cropped_frame_predictions_oversampling_thresholds_seed11.csv"
LABEL_CSV = "data_analysis/outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv"

AU_LIST = ["au4", "au6", "au9", "au17", "au25", "au26"]
RELATIVE_THRESHOLD = 0.3

OVERLAP_PAIRS = [
    ("au9", "au6"),
    ("au6", "au4"),
    ("au9", "au17"),
]

OUT_DIR = "DeepProbLog/facts"


def prob_at_least_k(frame_probs, k):
    n = len(frame_probs)
    dp = np.zeros(n + 1)
    dp[0] = 1.0
    for p in frame_probs:
        dp_new = np.zeros(n + 1)
        dp_new[1:] += dp[:-1] * p
        dp_new[:] += dp * (1 - p)
        dp = dp_new
    return float(dp[k:].sum())


def prob_overlap(p_au1, p_au2):
    return float(1 - np.prod(1 - p_au1 * p_au2))


def prob_presence(frame_probs):
    return float(1 - np.prod(1 - frame_probs))


def au_col(au):
    return au.upper() + "_prob"


def safe_atom(video_id):
    v = video_id.lower()
    v = "".join(c if (c.isalnum() or c == "_") else "_" for c in v)
    if not v[0].isalpha():
        v = "v_" + v
    return v


def load_labels(label_csv):
    labels = pd.read_csv(label_csv)
    labels["video_id_lower"] = labels["video_id"].str.lower()
    return labels[["video_id_lower", "video_id", "label"]]


def aggregate(pred_csv, video_ids_lower):
    needed_cols = ["video_id", "frame"] + [au_col(au) for au in AU_LIST]
    frames_by_video = {}
    chunksize = 50_000
    for chunk in pd.read_csv(pred_csv, usecols=needed_cols, chunksize=chunksize):
        chunk["video_id_lower"] = chunk["video_id"].str.lower()
        chunk = chunk[chunk["video_id_lower"].isin(video_ids_lower)]
        if chunk.empty:
            continue
        for vid, sub in chunk.groupby("video_id_lower"):
            sub = sub.sort_values("frame")
            if vid not in frames_by_video:
                frames_by_video[vid] = {au: [] for au in AU_LIST}
            for au in AU_LIST:
                frames_by_video[vid][au].append(sub[au_col(au)].to_numpy())
    result = {}
    for vid, au_dict in frames_by_video.items():
        result[vid] = {au: np.concatenate(arrs) for au, arrs in au_dict.items()}
    return result


def compute_facts(au_probs_by_video):
    duration_long = {}
    presence = {}
    overlap = {}
    for vid, au_probs in au_probs_by_video.items():
        n_frames = len(au_probs[AU_LIST[0]])
        k = max(1, int(np.ceil(n_frames * RELATIVE_THRESHOLD)))
        for au in AU_LIST:
            duration_long[(vid, au)] = prob_at_least_k(au_probs[au], k)
            presence[(vid, au)] = prob_presence(au_probs[au])
        for au1, au2 in OVERLAP_PAIRS:
            overlap[(vid, au1, au2)] = prob_overlap(au_probs[au1], au_probs[au2])
    return duration_long, presence, overlap


def write_pl_facts(duration_long, presence, overlap, out_path):
    lines = ["% Automatisch generiert aus " + PRED_CSV, ""]
    lines.append(f"% --- duration_long: AU in >={int(RELATIVE_THRESHOLD*100)}% der Videolaenge aktiv ---")
    for (v, au), p in sorted(duration_long.items()):
        lines.append(f"{p:.6f} :: duration_long({safe_atom(v)},{au}).")
    lines.append("")
    lines.append("% --- presence: AU kommt mindestens einmal vor ---")
    for (v, au), p in sorted(presence.items()):
        lines.append(f"{p:.6f} :: presence({safe_atom(v)},{au}).")
    lines.append("")
    lines.append("% --- overlap: zwei AUs im selben Frame aktiv ---")
    for (v, au1, au2), p in sorted(overlap.items()):
        lines.append(f"{p:.6f} :: overlap({safe_atom(v)},{au1},{au2}).")
    lines.append("")
    lines.append("% --- video/1 Deklaration ---")
    seen_videos = sorted({v for (v, _au) in duration_long.keys()})
    for v in seen_videos:
        lines.append(f"video({safe_atom(v)}).")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} Zeilen geschrieben nach {out_path}")


def write_query_labels(labels_df, out_path):
    labels_df = labels_df.copy()
    labels_df["video_atom"] = labels_df["video_id"].apply(safe_atom)
    labels_df[["video_atom", "label"]].to_csv(out_path, index=False)
    print(f"Labels geschrieben nach {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    labels_df = load_labels(LABEL_CSV)
    video_ids_lower = set(labels_df["video_id_lower"])
    print(f"{len(video_ids_lower)} Videos aus {LABEL_CSV} geladen.")
    au_probs_by_video = aggregate(PRED_CSV, video_ids_lower)
    print(f"{len(au_probs_by_video)} Videos in der Prediction-CSV gefunden und aggregiert.")
    missing = video_ids_lower - set(au_probs_by_video.keys())
    if missing:
        print(f"WARNUNG: {len(missing)} Videos nicht gefunden: {missing}")
    duration_long, presence, overlap = compute_facts(au_probs_by_video)
    write_pl_facts(duration_long, presence, overlap, os.path.join(OUT_DIR, "au_facts.pl"))
    write_query_labels(labels_df, os.path.join(OUT_DIR, "video_labels.csv"))
