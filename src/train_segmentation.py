from pathlib import Path
import time
import os

import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam

from segmentation_model import build_segmentation_model
from segmentation_data_loader import get_segmentation_loaders


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = PROJECT_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "best_unet.pth"

BACKUP_PATH = ARTIFACT_DIR / "best_unet_before_finetuning.pth"

HISTORY_PATH = ARTIFACT_DIR / "segmentation_finetuning_history.csv"

EPOCHS = 10

LEARNING_RATE = 0.00005

THRESHOLD = 0.1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.set_num_threads(
    min(4, os.cpu_count() or 1)
)


# ==========================================
# DICE LOSS
# ==========================================

def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    smooth = 1.0

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


# ==========================================
# COMBINED LOSS
# ==========================================

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        bce_value = self.bce(logits, targets)

        dice_value = dice_loss(logits, targets)

        return bce_value + dice_value


# ==========================================
# DICE METRIC
# ==========================================

def calculate_dice(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= THRESHOLD
    ).float()

    intersection = (
        predictions * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return dice.mean().item()


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch
):

    model.train()

    total_loss = 0.0
    total_dice = 0.0

    for batch_index, (images, masks) in enumerate(loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        batch_dice = calculate_dice(
            outputs.detach(),
            masks
        )

        total_loss += loss.item()
        total_dice += batch_dice

        if (
            (batch_index + 1) % 100 == 0
            or batch_index + 1 == len(loader)
        ):

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Dice: {batch_dice:.4f}",
                flush=True
            )

    return (
        total_loss / len(loader),
        total_dice / len(loader)
    )


# ==========================================
# VALIDATION
# ==========================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for batch_index, (images, masks) in enumerate(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()

            total_dice += calculate_dice(
                outputs,
                masks
            )

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Validation: "
                    f"{batch_index + 1}/{len(loader)} batches",
                    flush=True
                )

    return (
        total_loss / len(loader),
        total_dice / len(loader)
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("U-NET FINE-TUNING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Additional epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Validation threshold:", THRESHOLD)

    # Check existing model

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Existing model not found: {MODEL_PATH}"
        )

    # Preserve original checkpoint

    if not BACKUP_PATH.exists():

        import shutil

        shutil.copy2(
            MODEL_PATH,
            BACKUP_PATH
        )

        print("\nOriginal model backup created.")

    # Load datasets

    train_loader, validation_loader, _ = (
        get_segmentation_loaders()
    )

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # Build model

    model = build_segmentation_model().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    print("\nExisting best U-Net checkpoint loaded.")

    # Loss and optimizer

    criterion = CombinedLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_dice = -1.0

    history = []

    # Fine-tuning

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        print("\n" + "=" * 60)
        print(f"Fine-tuning Epoch {epoch}/{EPOCHS}")

        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch
        )

        validation_loss, validation_dice = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(validation_dice)

        elapsed = time.time() - start_time

        print("\nEPOCH RESULTS")

        print(f"Training Loss   : {train_loss:.4f}")
        print(f"Training Dice   : {train_dice:.4f}")

        print(f"Validation Loss : {validation_loss:.4f}")
        print(f"Validation Dice : {validation_dice:.4f}")

        print(f"Time Taken      : {elapsed / 60:.2f} minutes")

        history.append({
            "epoch": epoch,
            "training_loss": train_loss,
            "training_dice": train_dice,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice
        })

        if validation_dice > best_dice:

            best_dice = validation_dice

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("\nImproved model saved!")

        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False
        )

    print("\n" + "=" * 60)
    print("U-NET FINE-TUNING COMPLETED")
    print("=" * 60)

    print(f"Best validation Dice: {best_dice:.4f}")
    print("Best model:", MODEL_PATH)
    print("Original backup:", BACKUP_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":

    main()