import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from dataloader import get_dataloaders
from model import AUModel
from config import (
    IMAGE_DIR,
    LABEL_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    GLOBAL_THRESHOLD,
    AU_SPECIFIC_THRESHOLDS,
    POS_WEIGHT_THRESHOLDS,
    OVERSAMPLING_THRESHOLDS,
    DECISION_THRESHOLD_MODE,
)
from constants import TARGET_AUS


"""
Ziel: das CNN evaluieren pro AU und Threshold-level und herausfinden, ob alle AU'S auch okay erfasst worden sind.

probs: Wahrscheinlichkeiten, dass AU aktiv ist, zwischen 0 und 1
preds: 0 oder 1, je nachdem ob die Wahrscheinlichkeit über oder unter dem Schwellwert liegt
targets: die tatsächlichen Labels, 0 oder 1

Thresholds pro AU
AU4  -> 0.35
AU6  -> 0.40
AU9  -> 0.25
AU17 -> 0.25
AU25 -> 0.40
AU26 -> 0.25

Aufbau der Excel:
sheet run_info: Infos zum Lauf
sheet global_metrics: Gesamtmetrik alle AU's
sheet per_au_metrics: pro AU die Metriken über alle Threshold-levels aggregiert
sheet per_au_per_threshold: pro AU und pro Threshold-level die Metriken
sheet au_subject_table: Zeilen = AUs, Spalten = Subjects, Werte = positive Labels
"""

CHECKPOINT_PATH = "checkpoints/facenet_vggface2_au_threshold_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-18_18-58-25/best_model.pt"
OUTPUT_DIR = "evaluation_results"

def get_threshold_config():
    if DECISION_THRESHOLD_MODE == "global":
        return GLOBAL_THRESHOLD

    if DECISION_THRESHOLD_MODE == "au_specific":
        return AU_SPECIFIC_THRESHOLDS

    if DECISION_THRESHOLD_MODE == "pos_weights":
        return POS_WEIGHT_THRESHOLDS

    if DECISION_THRESHOLD_MODE == "oversampling":
        return OVERSAMPLING_THRESHOLDS

    raise ValueError(
        f"Unknown DECISION_THRESHOLD_MODE: {DECISION_THRESHOLD_MODE}. "
        "Use 'global', 'au_specific', 'oversampling', or 'pos_weights'."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SEED = SEED
EVAL_SPLIT = "val"  # oder "test"

print("evaluate.py startet")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_filename(checkpoint_path: str, seed: int, eval_split: str) -> str:
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if checkpoint_name.startswith("checkpoint_epoch_"):
        filename_core = f"{parent_dir}_{checkpoint_name}"
    else:
        filename_core = checkpoint_name

    filename_core = filename_core.replace(" ", "_")

    return f"evaluation_{filename_core}_{eval_split}_thr{GLOBAL_THRESHOLD}_Unfreezed_seed{seed}_{timestamp}.xlsx"


def load_model(checkpoint_path: str, device: str):
    model = AUModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_epoch = checkpoint.get("epoch", None)
        train_loss = checkpoint.get("train_loss", None)
        val_loss = checkpoint.get("val_loss", None)
    else:
        model.load_state_dict(checkpoint)
        checkpoint_epoch = None
        train_loss = None
        val_loss = None

    model.eval()
    return model, checkpoint_epoch, train_loss, val_loss


def evaluate_model(model, loader, device: str, threshold_override=None):
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                images, _, threshold_targets, _ = batch
                targets = threshold_targets
            elif len(batch) == 3:
                images, _, threshold_targets = batch
                targets = threshold_targets
            elif len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}.")

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            if outputs.shape != targets.shape:
                try:
                    outputs = outputs.view_as(targets)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Could not reshape model outputs from {outputs.shape} "
                        f"to target shape {targets.shape}."
                    ) from e

            probs = torch.sigmoid(outputs)

            # Fall 1: einzelner globaler Threshold wird direkt übergeben
            # wird für globalsss benutzt
            if threshold_override is not None:
                preds = (probs >= threshold_override).int()

            # Fall 2: normaler globaler Threshold aus config.py
            elif DECISION_THRESHOLD_MODE in ["global", "globalsss"]:
                threshold_config = get_threshold_config()
                preds = (probs >= threshold_config).int()

            # Fall 3: AU-spezifische Thresholds aus config.py
            elif DECISION_THRESHOLD_MODE in ["au_specific", "pos_weights", "oversampling"]:
                threshold_config = get_threshold_config()
                preds = torch.zeros_like(probs)

                for au_idx, au_name in enumerate(TARGET_AUS):
                    threshold = threshold_config[au_name]
                    preds[:, au_idx, :] = (
                        probs[:, au_idx, :] >= threshold
                    ).int()

            else:
                raise ValueError(
                    f"Unknown DECISION_THRESHOLD_MODE: {DECISION_THRESHOLD_MODE}"
                )

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_probs, all_preds, all_targets


def compute_auc_scores(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return roc_auc, pr_auc


def compute_metrics_per_au(all_targets, all_preds, all_probs, target_aus):
    rows = []

    for au_idx, au_name in enumerate(target_aus):
        y_true = all_targets[:, au_idx, :].reshape(-1)
        y_pred = all_preds[:, au_idx, :].reshape(-1)
        y_prob = all_probs[:, au_idx, :].reshape(-1)

        roc_auc, pr_auc = compute_auc_scores(y_true, y_prob)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        rows.append({
            "AU": au_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support_positive_labels": int(y_true.sum()),
            "total_labels": int(len(y_true)),
            "gt_positive_rate": float(y_true.mean()),
            "pred_positive_rate": float(y_pred.mean()),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(rows)


def compute_global_metrics(all_targets, all_preds, all_probs):
    y_true = all_targets.reshape(-1)
    y_pred = all_preds.reshape(-1)
    y_prob = all_probs.reshape(-1)

    roc_auc, pr_auc = compute_auc_scores(y_true, y_prob)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    data = [{
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total_labels": int(len(y_true)),
        "positive_labels": int(y_true.sum()),
        "gt_positive_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
    }]

    return pd.DataFrame(data)


def compute_threshold_level_metrics(all_targets, all_preds, all_probs, target_aus):
    rows = []
    n_thresholds = all_targets.shape[2]

    for au_idx, au_name in enumerate(target_aus):
        for thr_idx in range(n_thresholds):
            y_true = all_targets[:, au_idx, thr_idx].reshape(-1)
            y_pred = all_preds[:, au_idx, thr_idx].reshape(-1)
            y_prob = all_probs[:, au_idx, thr_idx].reshape(-1)

            roc_auc, pr_auc = compute_auc_scores(y_true, y_prob)

            rows.append({
                "AU": au_name,
                "threshold_level": thr_idx + 1,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "support_positive_labels": int(y_true.sum()),
                "total_samples": int(len(y_true)),
                "gt_positive_rate": float(y_true.mean()),
                "pred_positive_rate": float(y_pred.mean()),
            })

    return pd.DataFrame(rows)


def compute_au_subject_table(loaders, target_aus):
    """
    Tabelle:
    Zeilen = AUs
    Spalten = alle Subjects
    rechter Rand = SUMME pro AU
    unterer Rand = SUMME pro Subject
    """

    subject_ids = []
    all_active = []

    for loader in loaders:
        for batch in loader:
            if len(batch) != 4:
                raise ValueError(
                    "Für die AU-Subject-Tabelle muss der DataLoader Metadata zurückgeben. "
                    "Erwartet: (images, raw_labels, threshold_targets, metadata)"
                )

            _, _, threshold_targets, metadata = batch

            active = threshold_targets[:, :, 0].cpu().numpy()
            all_active.append(active)

            if isinstance(metadata, dict):
                if "subject" not in metadata:
                    raise KeyError(
                        f"metadata enthält keinen Key 'subject'. Vorhandene Keys: {list(metadata.keys())}"
                    )
                batch_subjects = metadata["subject"]
            else:
                batch_subjects = metadata

            subject_ids.extend(list(batch_subjects))

    active = np.concatenate(all_active, axis=0)
    subject_ids = np.array(subject_ids)

    if len(subject_ids) != active.shape[0]:
        raise ValueError(
            f"Anzahl subject_ids ({len(subject_ids)}) passt nicht zur Anzahl Samples ({active.shape[0]})."
        )

    subjects = [
        "SN001",
        "SN003",
        "SN004",
        "SN007",
        "SN009",
        "SN010",
        "SN013",
        "SN025",
        "SN027"
    ]

    rows = []

    for au_idx, au_name in enumerate(target_aus):
        row = {}

        for subject in subjects:
            subject_mask = subject_ids == subject
            row[subject] = int(active[subject_mask, au_idx].sum())

        row["sum"] = sum(row[subject] for subject in subjects)
        rows.append(row)

    au_subject_df = pd.DataFrame(rows, index=target_aus)
    au_subject_df.index.name = "AU"

    sum_row = au_subject_df.sum(axis=0)
    sum_row.name = "sum"

    au_subject_df = pd.concat([au_subject_df, sum_row.to_frame().T])

    return au_subject_df


def build_run_info_df(
    checkpoint_path,
    output_file,
    device,
    eval_seed,
    checkpoint_epoch,
    train_loss,
    val_loss,
    all_targets,
    eval_split,
):
    n_samples, n_aus, n_thresholds = all_targets.shape

    rows = [
        {"key": "timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"key": "checkpoint_path", "value": checkpoint_path},
        {"key": "output_file", "value": output_file},
        {"key": "device", "value": device},
        {"key": "eval_seed", "value": eval_seed},
        {"key": "decision_threshold_mode", "value": DECISION_THRESHOLD_MODE},
        {"key": "threshold_config", "value": str(get_threshold_config())},
        {"key": "checkpoint_epoch", "value": checkpoint_epoch},
        {"key": "checkpoint_train_loss", "value": train_loss},
        {"key": "checkpoint_val_loss", "value": val_loss},
        {"key": "n_samples", "value": n_samples},
        {"key": "n_aus", "value": n_aus},
        {"key": "n_threshold_levels", "value": n_thresholds},
        {"key": "target_aus", "value": ", ".join(TARGET_AUS)},
        {"key": "eval_split", "value": eval_split},
    ]

    return pd.DataFrame(rows)


def auto_adjust_columns(writer, sheet_name, df, include_index=False):
    ws = writer.sheets[sheet_name]

    if include_index:
        ws.column_dimensions["A"].width = 12
        start_col = 2
    else:
        start_col = 1

    for col_idx, column in enumerate(df.columns, start=start_col):
        max_len = max(
            len(str(column)),
            *(len(str(v)) for v in df[column].values)
        )
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 35)


def save_to_excel(output_path, run_info_df, global_df, per_au_df, threshold_df, au_subject_df):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        run_info_df.to_excel(writer, sheet_name="run_info", index=False)
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)
        per_au_df.to_excel(writer, sheet_name="per_au_metrics", index=False)
        threshold_df.to_excel(writer, sheet_name="per_au_per_threshold", index=False)

        # index=True, damit AU links als Zeilenname steht.
        if "sum" in au_subject_df.columns:
            au_subject_df = au_subject_df.drop(columns=["sum"])
        au_subject_df.to_excel(writer, sheet_name="au_subject_table", index=True)

        ws = writer.sheets["au_subject_table"]

        n_aus = len(au_subject_df.index) -1
        n_subjects = len(au_subject_df.columns)

        sum_col = n_subjects + 2
        sum_row = n_aus + 2

        ws.cell(row=1, column=sum_col).value = "sum"
        ws.cell(row=sum_row, column=1).value = "sum"

        # Summen rechts pro AU
        for row in range(2, sum_row):
            start_cell = ws.cell(row=row, column=2).coordinate
            end_cell = ws.cell(row=row, column=sum_col - 1).coordinate
            ws.cell(row=row, column=sum_col).value = f"=SUM({start_cell}:{end_cell})"

        # Summen unten pro Subject
        for col in range(2, sum_col + 1):
            start_cell = ws.cell(row=2, column=col).coordinate
            end_cell = ws.cell(row=sum_row - 1, column=col).coordinate
            ws.cell(row=sum_row, column=col).value = f"=SUM({start_cell}:{end_cell})"

        auto_adjust_columns(writer, "run_info", run_info_df, include_index=False)
        auto_adjust_columns(writer, "global_metrics", global_df, include_index=False)
        auto_adjust_columns(writer, "per_au_metrics", per_au_df, include_index=False)
        auto_adjust_columns(writer, "per_au_per_threshold", threshold_df, include_index=False)
        auto_adjust_columns(writer, "au_subject_table", au_subject_df, include_index=True)


def main():
    set_seed(EVAL_SEED)

    model, checkpoint_epoch, train_loss, val_loss = load_model(CHECKPOINT_PATH, DEVICE)

    train_loader, val_loader, test_loader = get_dataloaders(
        IMAGE_DIR,
        LABEL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    eval_loader = val_loader if EVAL_SPLIT == "val" else test_loader

    all_probs, all_preds, all_targets = evaluate_model(
        model=model,
        loader=eval_loader,
        device=DEVICE,
    )

    # Test: für Threshold pro AU
    print("\nWahrscheinlichkeitsstatistiken per AU:")

    for au_idx, au_name in enumerate(TARGET_AUS):
        probs_au = all_probs[:, au_idx, :].reshape(-1)

        print(
            au_name,
            "min:", probs_au.min(),
            "max:", probs_au.max(),
            "mean:", probs_au.mean(),
            "q95:", np.quantile(probs_au, 0.95), # q ist Quantil
            "q99:", np.quantile(probs_au, 0.99),
        )
    #### test ende

    per_au_df = compute_metrics_per_au(all_targets, all_preds, all_probs, TARGET_AUS)
    global_df = compute_global_metrics(all_targets, all_preds, all_probs)
    threshold_df = compute_threshold_level_metrics(all_targets, all_preds, all_probs, TARGET_AUS)

    au_subject_df = compute_au_subject_table(
        loaders=[val_loader, test_loader, train_loader],
        target_aus=TARGET_AUS,
    )

    output_filename = make_output_filename(
        checkpoint_path=CHECKPOINT_PATH,
        seed=EVAL_SEED,
        eval_split=EVAL_SPLIT,
    )

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    run_info_df = build_run_info_df(
        checkpoint_path=CHECKPOINT_PATH,
        output_file=output_path,
        device=DEVICE,
        eval_seed=EVAL_SEED,
        checkpoint_epoch=checkpoint_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        all_targets=all_targets,
        eval_split=EVAL_SPLIT,
    )

    save_to_excel(
        output_path=output_path,
        run_info_df=run_info_df,
        global_df=global_df,
        per_au_df=per_au_df,
        threshold_df=threshold_df,
        au_subject_df=au_subject_df,
    )

    print("\nEvaluation finished.")
    print(f"Excel file saved to: {output_path}")
    print("\nPer-AU metrics:")
    print(per_au_df.to_string(index=False))
    print("\nAU-Subject table:")
    print(au_subject_df.to_string())


if __name__ == "__main__":
    main()