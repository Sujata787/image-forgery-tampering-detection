from pathlib import Path

import numpy as np
import torch

from segmentation_model import build_segmentation_model
from segmentation_data_loader import get_segmentation_loaders


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_unet.pth"

OUTPUT_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "best_segmentation_threshold.txt"
)

DEVICE = torch.device("cpu")

THRESHOLDS = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

EPSILON = 1e-7


# ==================================================
# CHECK MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ==================================================
# LOAD VALIDATION DATA
# ==================================================

print("=" * 60)
print("U-NET BEST THRESHOLD SEARCH")
print("=" * 60)

_, validation_loader, _ = get_segmentation_loaders()


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = build_segmentation_model().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()

print("\nModel loaded successfully.")
print("Using validation dataset only.")


# ==================================================
# STORE METRICS
# ==================================================

results = []


# ==================================================
# TEST DIFFERENT THRESHOLDS
# ==================================================

for threshold in THRESHOLDS:

    dice_scores = []
    iou_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= threshold
            ).float()

            # Calculate metrics for each image
            for index in range(images.size(0)):

                predicted = predictions[index]
                actual = masks[index]

                intersection = (
                    predicted * actual
                ).sum().item()

                predicted_pixels = predicted.sum().item()
                actual_pixels = actual.sum().item()

                union = (
                    predicted_pixels
                    + actual_pixels
                    - intersection
                )

                dice = (
                    2 * intersection + EPSILON
                ) / (
                    predicted_pixels
                    + actual_pixels
                    + EPSILON
                )

                iou = (
                    intersection + EPSILON
                ) / (
                    union + EPSILON
                )

                dice_scores.append(dice)
                iou_scores.append(iou)

                total_tp += intersection

                total_fp += (
                    predicted_pixels - intersection
                )

                total_fn += (
                    actual_pixels - intersection
                )

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    precision = total_tp / (
        total_tp + total_fp + EPSILON
    )

    recall = total_tp / (
        total_tp + total_fn + EPSILON
    )

    results.append({
        "threshold": threshold,
        "dice": mean_dice,
        "iou": mean_iou,
        "precision": precision,
        "recall": recall
    })

    print("\nThreshold:", threshold)
    print(f"Mean Dice : {mean_dice:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")


# ==================================================
# FIND BEST THRESHOLD
# ==================================================

best_result = max(
    results,
    key=lambda item: item["dice"]
)


# ==================================================
# SAVE BEST THRESHOLD
# ==================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    file.write(
        f"Best threshold: {best_result['threshold']}\n"
    )

    file.write(
        f"Validation Dice: {best_result['dice']:.4f}\n"
    )

    file.write(
        f"Validation IoU: {best_result['iou']:.4f}\n"
    )


# ==================================================
# FINAL RESULTS
# ==================================================

print("\n" + "=" * 60)
print("BEST THRESHOLD RESULTS")
print("=" * 60)

print(
    "Best threshold:",
    best_result["threshold"]
)

print(
    f"Best validation Dice: "
    f"{best_result['dice']:.4f}"
)

print(
    f"Validation IoU: "
    f"{best_result['iou']:.4f}"
)

print("\nSaved to:", OUTPUT_PATH)

print("=" * 60)
print("THRESHOLD SEARCH COMPLETED")
print("=" * 60)