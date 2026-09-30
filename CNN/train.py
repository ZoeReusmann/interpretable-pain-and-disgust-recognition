import os
import random
import torch
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm
from collections import defaultdict

from config import (
    IMAGE_DIR,
    LABEL_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    SEED,
    CHECKPOINT_DIR,
    RUNS_DIR,
    SAVE_EVERY,
    FREEZE_BACKBONE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MODE,
    MODEL_NAME,
    VAL_SUBJECTS,
    TRAIN_SUBJECTS,
    TEST_SUBJECTS,
    TARGET_AUS,
)
from dataloader import get_dataloaders
from model import AUModel

def compute_oversampling_weights(train_subset):
    """
    Berechnet ein Gewicht pro Trainingssample
    Samples mit seltenen aktiven AUs werden häufiger gezogen
    """

    print("Berechne Oversampling-Gewichte...")

    full_dataset = train_subset.dataset
    train_indices = train_subset.indices

    sample_targets = []
    au_positive_counts = torch.zeros(len(TARGET_AUS), dtype=torch.float32)

    for idx in train_indices:
        sample = full_dataset.samples[idx]

        au_labels_raw = full_dataset._load_labels_for_frame(
            sample["subject"],
            sample["trial"],
            sample["frame"]
        )

        binary_aus = (au_labels_raw > 0).float()

        sample_targets.append(binary_aus)
        au_positive_counts += binary_aus

    au_weights = 1.0 / (au_positive_counts + 1e-6)

    sample_weights = []

    for binary_aus in sample_targets:
        active_weights = au_weights[binary_aus == 1]

        if len(active_weights) > 0:
            sample_weight = active_weights.mean()
        else:
            sample_weight = torch.tensor(1.0)

        sample_weights.append(sample_weight.item())

    print("Positive AU counts vor Oversampling:")
    for au_name, count in zip(TARGET_AUS, au_positive_counts):
        print(f"{au_name}: {int(count.item())}")

    return torch.DoubleTensor(sample_weights)

def set_seed(seed): #random seeds für besser Reproduzierbarkeit: GEwichtsinitialisierung, DataLoader SHuffle, Dropout etc.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device): #trainiert das Modell einmal über alle Trainingsdaten, berechnet Loss
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, _, threshold_targets, _ in loop: #Rohlabels werden nicht benötigt, da nur Schwellenwerte trainiert werden
        images = images.to(device)
        threshold_targets = threshold_targets.float().to(device)

        optimizer.zero_grad() # Gradienten soll sich nicht über mehrere Batches aufsummieren
        outputs = model(images)
        loss = criterion(outputs, threshold_targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 👉 Live Update im Progress Bar
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device): #Validiert ohne Gewichtsänderungen
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, _, threshold_targets, _ in loader:
            images = images.to(device)
            threshold_targets = threshold_targets.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, threshold_targets)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def get_config_dict():
    return {
        "IMAGE_DIR": IMAGE_DIR,
        "LABEL_DIR": LABEL_DIR,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "VAL_SUBJECTS": VAL_SUBJECTS,
        "TEST_SUBJECTS": TEST_SUBJECTS,
        "TRAIN_SUBJECTS": TRAIN_SUBJECTS,
        "TARGET_AUS": TARGET_AUS,
        "NUM_WORKERS": NUM_WORKERS,
        "SEED": SEED,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "RUNS_DIR": RUNS_DIR,
        "SAVE_EVERY": SAVE_EVERY,
        "FREEZE_BACKBONE": FREEZE_BACKBONE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_MIN_DELTA": EARLY_STOPPING_MIN_DELTA,
        "SCHEDULER": "ReduceLROnPlateau",
        "SCHEDULER_FACTOR": SCHEDULER_FACTOR,
        "SCHEDULER_PATIENCE": SCHEDULER_PATIENCE,
        "SCHEDULER_MODE": SCHEDULER_MODE,
        "MODEL_NAME": MODEL_NAME,
        "LOSS": "BCEWithLogitsLoss",
        #"MAX_POS_WEIGHT": 10.0,
    }


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, checkpoint_path, pos_weights=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": get_config_dict(),
            #"pos_weights": pos_weights.detach().cpu() if pos_weights is not None else None,
        },
        checkpoint_path
    )

# def compute_pos_weights(train_loader, device, max_weight=10.0):
#     """
#     Berechnet pos_weight pro AU und Threshold-Level.
#     Shape: (N_AUS, N_THRESHOLDS), passend zu outputs/targets.
#     pos_weight = negative_labels / positive_labels
#     """
#     print("Berechne pos_weights aus Trainingsdaten...")

#     positive_counts = None
#     total_count = 0

#     for _, _, threshold_targets, _ in train_loader:
#         threshold_targets = threshold_targets.float()

#         if positive_counts is None:
#             positive_counts = torch.zeros(threshold_targets.shape[1:], dtype=torch.float32)

#         positive_counts += threshold_targets.sum(dim=0)
#         total_count += threshold_targets.shape[0]

#     negative_counts = total_count - positive_counts

#     pos_weights = negative_counts / (positive_counts + 1e-6)  # Verhindert Division durch 0
#     pos_weights = torch.clamp(pos_weights, min=1.0, max=max_weight) # Falls ein Level extrem selten ist, Gewicht begrenzen

#     print("pos_weights:")
#     print(pos_weights)

#     return pos_weights.to(device)

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, _ = get_dataloaders( # _, weil test im training noch nicht benutzt werden soll
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    sample_weights = compute_oversampling_weights(train_loader.dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS, 
        drop_last=True
    )

    print("Oversampling ist aktiv.")

    # für oversampling test
    print(train_loader.dataset.dataset.samples[train_loader.dataset.indices[0]].keys())
    print(train_loader.dataset.dataset.samples[train_loader.dataset.indices[0]])
 
 # für oversampling: wie oft ist jede AU in Trainingsdaten aktiv? (nur threshold level 0, da wir nur das trainieren)
    
    au_counter = defaultdict(int)

    for batch_idx, (_, _, threshold_targets, _) in enumerate(train_loader):

        # threshold level 0
        binary_targets = threshold_targets[:, :, 0]

        for au_idx, au_name in enumerate(TARGET_AUS):
            au_counter[au_name] += binary_targets[:, au_idx].sum().item()

        if batch_idx == 100:
            break

    print("\nOversampled AU counts:")
    for au, count in au_counter.items():
        print(au, int(count))

# test, wieviele Samples in Train und Val vorhanden
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")  
    dataset = train_loader.dataset

    print("Gesamtanzahl Samples:", len(dataset))  
### test ende    

    model = AUModel().to(device)

    if FREEZE_BACKBONE:
        model.freeze_backbone() # nur letzte Layer trainieren, Rest passt so

# für meine Info: wie groß ist mein Modell?
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    #pos_weights = compute_pos_weights(train_loader, device, max_weight=10.0)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW( #adjustiert Gewichte anhand berechneter Gradienten
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    run_name = (
        f"{MODEL_NAME}_"
        # f"weightedLoss_"
        f"partially_unfrozen_"
        f"bs{BATCH_SIZE}_"
        f"lr{LEARNING_RATE}_"
        f"oversampling_"
        f"seed{SEED}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    log_dir = os.path.join(RUNS_DIR, run_name)
    writer = SummaryWriter(log_dir) # Tensorboard (Loss und Lernrate pro Epoche)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss) #Lernrate anpassen, wenn sich Val Loss nicht verbessert

        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path
            )
            print(f"checkpoint saved: {checkpoint_path}")

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA: #bestes Modell speichern
            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                checkpoint_path=best_model_path
            )
            print("best model saved")
        else:
            epochs_without_improvement += 1
            print(f"no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE: # Zeit & Aufwand sparen
            print("early stopping triggered")
            break

    writer.close()
    print("training finished")


if __name__ == "__main__":
    main()