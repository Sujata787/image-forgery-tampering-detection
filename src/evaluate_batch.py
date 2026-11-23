from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import build_model


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_CSV = PROJECT_DIR / "outputs" / "splits" / "test.csv"

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_resnet18.pth"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "batch_predictions.csv"

DEVICE = torch.device("cpu")

BATCH_SIZE = 32

CLASS_NAMES = ["Authentic", "Tampered"]


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# TEST DATASET
# ==========================================

class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = PROJECT_DIR / row["image_path"]

        label = int(row["label"])

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image = transform(image)

        return image, label, str(image_path)


# ==========================================
# LOAD TEST DATA
# ==========================================

def load_test_data():

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
        )

    dataframe = pd.read_csv(TEST_CSV)

    print("\nTest dataset loaded.")

    print("Total images:", len(dataframe))

    return dataframe


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = build_model(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print("\nResNet18 model loaded successfully.")

    return model


# ==========================================
# BATCH PREDICTION
# ==========================================

def run_predictions(model, test_loader):

    actual_labels = []

    predicted_labels = []

    confidence_scores = []

    image_paths = []

    print("\nStarting batch prediction...\n")

    with torch.no_grad():

        for batch_index, (images, labels, paths) in enumerate(test_loader):

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

            actual_labels.extend(labels.tolist())

            predicted_labels.extend(predictions.cpu().tolist())

            confidence_scores.extend(
                (confidences.cpu().tolist())
            )

            image_paths.extend(paths)

            completed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(test_loader.dataset)
            )

            print(
                f"Processed {completed}/{len(test_loader.dataset)} images"
            )

    return (
        actual_labels,
        predicted_labels,
        confidence_scores,
        image_paths
    )


# ==========================================
# EVALUATION METRICS
# ==========================================

def evaluate_results(
    actual_labels,
    predicted_labels,
    confidence_scores,
    image_paths
):

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    print("\n" + "=" * 60)

    print("BATCH CLASSIFICATION EVALUATION")

    print("=" * 60)

    print("Total test images :", len(actual_labels))

    print(f"Accuracy          : {accuracy * 100:.2f}%")

    print(f"Precision         : {precision:.4f}")

    print(f"Recall            : {recall:.4f}")

    print(f"F1 Score          : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Save individual predictions
    results = pd.DataFrame({

        "image_path": image_paths,

        "actual_label": actual_labels,

        "predicted_label": predicted_labels,

        "actual_class": [
            CLASS_NAMES[label]
            for label in actual_labels
        ],

        "predicted_class": [
            CLASS_NAMES[label]
            for label in predicted_labels
        ],

        "confidence_percent": [
            round(score * 100, 2)
            for score in confidence_scores
        ],

        "correct_prediction": [
            actual == predicted
            for actual, predicted in zip(
                actual_labels,
                predicted_labels
            )
        ]

    })

    results.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\nPrediction results saved:")

    print(RESULT_PATH)

    print("\nBATCH EVALUATION COMPLETED.")


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)

    print("IMAGE FORGERY BATCH EVALUATION")

    print("=" * 60)

    dataframe = load_test_data()

    dataset = TestDataset(dataframe)

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = load_model()

    (
        actual_labels,
        predicted_labels,
        confidence_scores,
        image_paths
    ) = run_predictions(
        model,
        test_loader
    )

    evaluate_results(
        actual_labels,
        predicted_labels,
        confidence_scores,
        image_paths
    )


if __name__ == "__main__":

    main()