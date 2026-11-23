from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from model import build_model
from data_loader import get_data_loaders


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_resnet18.pth"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = build_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    model.eval()

    return model


# ==========================================
# GENERATE PREDICTIONS
# ==========================================

def get_predictions(model, test_loader):

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    return all_labels, all_predictions


# ==========================================
# CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Authentic", "Tampered"],
        zero_division=0,
        output_dict=False
    )

    if not isinstance(report, str):
        raise TypeError(
            "Classification report must be a string."
        )

    print("\nClassification Report:")
    print(report)

    # Save classification report
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # Save evaluation metrics
    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        index=False
    )

    return accuracy, precision, recall, f1


# ==========================================
# CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Authentic",
            "Tampered"
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Image Forgery Detection - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nConfusion matrix saved:", output_path)


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 50)
    print("IMAGE FORGERY DETECTION - MODEL EVALUATION")
    print("=" * 50)

    print("\nDevice:", DEVICE)

    # Load test dataset
    _, _, test_loader = get_data_loaders()

    print("Test batches:", len(test_loader))

    # Load trained model
    model = load_trained_model()

    print("\nTrained model loaded successfully!")

    # Generate predictions
    y_true, y_pred = get_predictions(
        model,
        test_loader
    )

    print("\nTotal test images:", len(y_true))

    # Calculate metrics
    calculate_metrics(
        y_true,
        y_pred
    )

    # Generate confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 50)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()