from torch.utils.data import DataLoader, Subset
from dataset import AUDataset
from transforms import train_transform
from config import IMAGE_DIR, LABEL_DIR, BATCH_SIZE, NUM_WORKERS, SEED, TEST_SUBJECTS, VAL_SUBJECTS, TRAIN_SUBJECTS
from torch.utils.data import WeightedRandomSampler, ConcatDataset # für oversampling und für export aller Daten
import torch

"""
Ziel: DataLoader für das AU-Dataset

Rückgabe pro Batch:
- images
- raw_labels
- threshold_targets

ANpassung: Oversampling (seltene AU-positive Frames kommen öfter im Dataset vor & häufige Frame seltener)

Wichtig: Split pro Proband -> aktuelles Projektziel: 70% Probanden für Training, 20% für Val, 10% für Test
"""

def create_weighted_sampler(dataset):
    original_dataset = dataset.dataset
    subset_indices = dataset.indices

    sample_weights = []

    for idx in subset_indices:
        # __getitem__ nutzen, weil dort Labels geladen/erstellt werden
        _, _, targets, _ = original_dataset[idx]

        has_positive_label = targets.sum().item() > 0

        if has_positive_label:
            sample_weights.append(2.0)
        else:
            sample_weights.append(1.0)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler

def get_dataloaders(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, # mit meinem Mac lieber keine parallelen Prozesse, evtl. später erhöhen
   # seed=SEED, bei Probandensplit nich gebraucht, evtl. für später
    train_subjects=None,
    val_subjects=None,
    test_subjects=None,
):
    dataset = AUDataset( # dataset erzeugen
        image_dir=image_dir,
        label_dir=label_dir,
        transform=train_transform
    )

    subjects = sorted(set(sample["subject"] for sample in dataset.samples))
    print("All subjects:", subjects)

    train_subjects = set(TRAIN_SUBJECTS if train_subjects is None else train_subjects)
    val_subjects   = set(VAL_SUBJECTS if val_subjects is None else val_subjects)
    test_subjects  = set(TEST_SUBJECTS if test_subjects is None else test_subjects)

    defined_subjects = train_subjects | val_subjects | test_subjects # check: alle Probanden da?
    missing_subjects = defined_subjects - set(subjects) 

    assert len(missing_subjects) == 0, f"Subjects not found in dataset: {missing_subjects}"
    assert len(train_subjects & val_subjects) == 0 # check: keine Überschneidungen der Probanden in mehreren Sets
    assert len(train_subjects & test_subjects) == 0
    assert len(val_subjects & test_subjects) == 0

    print("Train subjects:", sorted(train_subjects))
    print("Val subjects:", sorted(val_subjects))
    print("Test subjects:", sorted(test_subjects))


# indizes für split pro proband
    train_indices = [
        i for i, sample in enumerate(dataset.samples)
        if sample["subject"] in train_subjects
    ]

    val_indices = [
        i for i, sample in enumerate(dataset.samples)
        if sample["subject"] in val_subjects
    ]

    test_indices = [
    i for i, sample in enumerate(dataset.samples)
    if sample["subject"] in test_subjects
    ]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    export_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Export samples: {len(export_dataset)}")

    # train_loader für oversampling
    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True
    )

    # normal (kein oversampling)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers
    # )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=True
    )



    export_loader = DataLoader(
        export_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, export_loader