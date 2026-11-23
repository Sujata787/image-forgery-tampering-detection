from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import build_model
from data_loader import get_data_loaders


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "outputs" / "splits"

ARTIFACT_DIR = PROJECT_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "best_resnet18.pth"

HISTORY_PATH = ARTIFACT_DIR / "training_history.csv"

EPOCHS = 5

LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# CLASS WEIGHTS
# ==========================================

def get_class_weights():

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")

    counts = train_df["label"].value_counts()

    authentic_count = int(counts.get(0, 0))
    tampered_count = int(counts.get(1, 0))

    if authentic_count == 0 or tampered_count == 0:
        raise ValueError("Both classes must exist in training data.")

    total = authentic_count + tampered_count

    weights = torch.tensor(
        [
            total / (2.0 * authentic_count),
            total / (2.0 * tampered_count)
        ],
        dtype=torch.float32
    )

    return weights.to(DEVICE)


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# VALIDATE MODEL
# ==========================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE).long()

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    validation_loss = running_loss / total

    validation_accuracy = 100.0 * correct / total

    return validation_loss, validation_accuracy


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("=" * 55)
    print("IMAGE FORGERY DETECTION - MODEL TRAINING")
    print("=" * 55)

    print("\nDevice:", DEVICE)

    # Load datasets
    train_loader, validation_loader, test_loader = (
        get_data_loaders()
    )

    print("\nDataset sizes:")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    # Build model
    model = build_model().to(DEVICE)

    # Calculate class weights
    class_weights = get_class_weights()

    print("\nClass weights:", class_weights)

    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Optimizer
    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_accuracy = 0.0

    history = []

    # Training loop
    for epoch in range(EPOCHS):

        print("\n" + "-" * 55)

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(validation_accuracy)

        print(f"Training Loss      : {train_loss:.4f}")
        print(f"Training Accuracy  : {train_accuracy:.2f}%")

        print(f"Validation Loss    : {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        })

        # Save best model
        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("\nBest model saved:", MODEL_PATH)

    # Save training history
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    print("\n" + "=" * 55)

    print("TRAINING COMPLETED")

    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    print("Model location:", MODEL_PATH)

    print("Training history:", HISTORY_PATH)

    print("=" * 55)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()