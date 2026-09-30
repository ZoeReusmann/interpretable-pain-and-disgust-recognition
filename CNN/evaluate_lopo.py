import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
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
    DECISION_THRESHOLD_MODE,
    OVERSAMPLING_THRESHOLDS,
)
from constants import TARGET_AUS


"""
LOPO Evaluation: MIt dem DISFA+ Datenset gibt es 9 Probanden, die jeweils als Test-Subject in einem LOPO-Fold dienen können.
Ähnlich zu meiner normalen Eval aufgebaut
Ziel: für jeden Proband muss ein eigener Checkpoint existieren für die finale Evaluation

Excel: 
globalen LOPO-Metriken
Mittelwerten pro AU
allen einzelnen Folds/Subjekten separat
"""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "evaluation_results/lopo_evaluation"


CHECKPOINTS_BY_TEST_SUBJECT = {
    # Benennung pro Proband
    "SN001": "checkpoints/lopo_SN001_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_16-14-48/best_model.pt",
    "SN003": "checkpoints/lopo_SN003_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_17-19-44/best_model.pt",
    "SN004": "checkpoints/lopo_SN004_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_18-29-32/best_model.pt",
    "SN007": "checkpoints/lopo_SN007_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-06-01_13-11-57/best_model.pt",
    "SN009": "checkpoints/lopo_SN009_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_19-40-11/best_model.pt",
    "SN010": "checkpoints/lopo_SN010_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_20-48-05/best_model.pt",
    "SN013": "checkpoints/lopo_SN013_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-05-28_21-57-03/best_model.pt",
    "SN025": "checkpoints/lopo_SN025_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-06-01_14-08-14/best_model.pt",
    "SN027": "checkpoints/lopo_SN027_facenet_vggface2_au_threshold_oversampling_partially_unfrozen_bs8_lr0.0001_oversampling_seed11_2026-06-01_15-03-31/best_model.pt",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path):
    model = AUModel()

    print(f"lade Modell: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def apply_thresholds(probs):
    """
    probs shape: [batch_size, n_aus, n_thresholds]
    Output: binäre preds shape
    """

    if DECISION_THRESHOLD_MODE == "global":
        global_threshold = 0.35
        preds = (probs >= global_threshold).int()

    elif DECISION_THRESHOLD_MODE == "oversampling":
        preds = torch.zeros_like(probs)

        for au_idx, au_name in enumerate(TARGET_AUS):
            threshold = OVERSAMPLING_THRESHOLDS[au_name]
            preds[:, au_idx, :] = (
                probs[:, au_idx, :] >= threshold
            ).int()

    else:
        raise ValueError(
            f"Unknown or unsupported DECISION_THRESHOLD_MODE for this script: "
            f"{DECISION_THRESHOLD_MODE}"
        )

    return preds


def get_predictions(model, loader):
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

            images = batch[0].to(DEVICE)
            targets = batch[1].float().to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds = apply_thresholds(probs)

            # reduce ordinal threshold vector to binary AU activity
            pred_active = (preds.sum(dim=2) > 0).int()

            # if targets are already [batch_size, n_aus]
            if targets.ndim == 2:
                target_active = targets.int()

            # if targets are [batch_size, n_aus, n_thresholds]
            elif targets.ndim == 3:
                target_active = (targets.sum(dim=2) > 0).int()

            else:
                raise ValueError(f"Unexpected target shape: {targets.shape}")

            # probability per AU = maximum probability across threshold levels
            prob_active = probs.max(dim=2).values

            all_probs.append(prob_active.cpu())
            all_preds.append(pred_active.cpu())
            all_targets.append(target_active.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return all_probs, all_preds, all_targets


def compute_metrics(y_true, y_pred, y_prob, test_subject):
    rows = []

    for au_idx, au_name in enumerate(TARGET_AUS):
        true_au = y_true[:, au_idx]
        pred_au = y_pred[:, au_idx]
        prob_au = y_prob[:, au_idx]

        true_au = (np.asarray(true_au) > 0).astype(int)
        pred_au = (np.asarray(pred_au) > 0).astype(int)

        row = {
            "test_subject": test_subject,
            "au": au_name,
            "support_positive": int(true_au.sum()),
            "support_total": int(len(true_au)),
            "positive_rate_true": float(true_au.mean()),
            "positive_rate_pred": float(pred_au.mean()),
            "f1": f1_score(true_au, pred_au, zero_division=0),
            "precision": precision_score(true_au, pred_au, zero_division=0),
            "recall": recall_score(true_au, pred_au, zero_division=0),
            "accuracy": accuracy_score(true_au, pred_au),
        }

        try:
            row["roc_auc"] = roc_auc_score(true_au, prob_au)
        except ValueError:
            row["roc_auc"] = np.nan

        try:
            row["pr_auc"] = average_precision_score(true_au, prob_au)
        except ValueError:
            row["pr_auc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("evaluate_lopo.py startet")

    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not CHECKPOINTS_BY_TEST_SUBJECT:
        raise ValueError(
            "CHECKPOINTS_BY_TEST_SUBJECT ist leer. "
            "Trage zuerst pro Test-Subject den passenden Checkpoint-Pfad ein."
        )

    all_results = []

    for test_subject, checkpoint_path in CHECKPOINTS_BY_TEST_SUBJECT.items():
        print("\n" + "=" * 80)
        print(f"LOPO Evaluation für Test-Subject: {test_subject}")
        print("=" * 80)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint für {test_subject} nicht gefunden: {checkpoint_path}"
            )

        all_subjects = ['SN001', 'SN003', 'SN004', 'SN007', 'SN009', 'SN010', 'SN013', 'SN025', 'SN027']

        test_subjects = [test_subject]

        val_subjects = [s for s in all_subjects if s != test_subject][:1]

        train_subjects = [
            s for s in all_subjects
            if s not in test_subjects + val_subjects
        ]

        _, _, test_loader = get_dataloaders(
            IMAGE_DIR,
            LABEL_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )

        model = load_model(checkpoint_path)

        y_prob, y_pred, y_true = get_predictions(
            model,
            test_loader,
        )

        df_subject = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            test_subject=test_subject,
        )

        all_results.append(df_subject)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_all = pd.concat(all_results, ignore_index=True)

    df_mean_by_au = (
        df_all
        .groupby("au", as_index=False)
        .agg({
            "support_positive": "sum",
            "support_total": "sum",
            "positive_rate_true": "mean",
            "positive_rate_pred": "mean",
            "f1": "mean",
            "precision": "mean",
            "recall": "mean",
            "accuracy": "mean",
            "roc_auc": "mean",
            "pr_auc": "mean",
        })
    )

    df_global = pd.DataFrame([{
        "n_lopo_folds": df_all["test_subject"].nunique(),
        "mean_f1": df_all["f1"].mean(),
        "mean_precision": df_all["precision"].mean(),
        "mean_recall": df_all["recall"].mean(),
        "mean_accuracy": df_all["accuracy"].mean(),
        "mean_roc_auc": df_all["roc_auc"].mean(),
        "mean_pr_auc": df_all["pr_auc"].mean(),
        "decision_threshold_mode": DECISION_THRESHOLD_MODE,
    }])

    output_path = os.path.join(
        OUTPUT_DIR,
        f"lopo_evaluation_{DECISION_THRESHOLD_MODE}_{timestamp}.xlsx",
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="global_summary", index=False)
        df_mean_by_au.to_excel(writer, sheet_name="mean_by_au", index=False)
        df_all.to_excel(writer, sheet_name="all_folds_per_au", index=False)

    print("\nLOPO Evaluation gespeichert unter:")
    print(output_path)

    print("\nGlobal Summary:")
    print(df_global)

    print("\nMean by AU:")
    print(df_mean_by_au)


if __name__ == "__main__":
    main()
