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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"

THRESHOLD = 0.1

DEVICE = torch.device("cpu")

EPSILON = 1e-7


# ==================================================
# CREATE OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD TEST DATA
# ==================================================

print("=" * 60)
print("FINAL U-NET SEGMENTATION TEST EVALUATION")
print("=" * 60)

_, _, test_loader = get_segmentation_loaders()

print("\nTest dataset loaded successfully.")


# ==================================================
# LOAD MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = build_segmentation_model().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()

print("Best U-Net model loaded successfully.")
print("Device:", DEVICE)
print("Prediction threshold:", THRESHOLD)


# ==================================================
# METRIC STORAGE
# ==================================================

dice_scores = []
iou_scores = []

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

total_actual_positive = 0
total_predicted_positive = 0

evaluated_images = 0


# ==================================================
# EVALUATE TEST DATA
# ==================================================

with torch.no_grad():

    for batch_index, (images, masks) in enumerate(test_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

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

            total_tn += (
                actual.numel()
                - intersection
                - (predicted_pixels - intersection)
                - (actual_pixels - intersection)
            )

            total_actual_positive += actual_pixels
            total_predicted_positive += predicted_pixels

            evaluated_images += 1

        if (batch_index + 1) % 100 == 0:
            print(
                f"Evaluated batches: "
                f"{batch_index + 1}/{len(test_loader)}"
            )


# ==================================================
# CALCULATE FINAL METRICS
# ==================================================

mean_dice = float(np.mean(dice_scores))
mean_iou = float(np.mean(iou_scores))

pixel_precision = total_tp / (
    total_tp + total_fp + EPSILON
)

pixel_recall = total_tp / (
    total_tp + total_fn + EPSILON
)

pixel_accuracy = (
    total_tp + total_tn
) / (
    total_tp + total_fp + total_fn + total_tn + EPSILON
)

actual_positive_percentage = (
    total_actual_positive
    / (evaluated_images * 128 * 128)
) * 100

predicted_positive_percentage = (
    total_predicted_positive
    / (evaluated_images * 128 * 128)
) * 100


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 60)
print("FINAL SEGMENTATION TEST RESULTS")
print("=" * 60)

print("Test images          :", evaluated_images)
print("Prediction threshold :", THRESHOLD)

print(f"Mean Dice Score      : {mean_dice:.4f}")
print(f"Mean IoU             : {mean_iou:.4f}")
print(f"Pixel Precision      : {pixel_precision:.4f}")
print(f"Pixel Recall         : {pixel_recall:.4f}")
print(f"Pixel Accuracy       : {pixel_accuracy:.4f}")

print(
    f"Ground Truth Positive: "
    f"{actual_positive_percentage:.2f}%"
)

print(
    f"Predicted Positive   : "
    f"{predicted_positive_percentage:.2f}%"
)

print("\nConfusion Matrix:")

print("True Positive :", int(total_tp))
print("False Positive:", int(total_fp))
print("False Negative:", int(total_fn))
print("True Negative :", int(total_tn))


# ==================================================
# SAVE RESULTS
# ==================================================

results_path = (
    OUTPUT_DIR / "best_threshold_segmentation_results.txt"
)

with open(results_path, "w", encoding="utf-8") as file:

    file.write("FINAL U-NET SEGMENTATION TEST RESULTS\n")
    file.write("=" * 50 + "\n")

    file.write(f"Threshold: {THRESHOLD}\n")
    file.write(f"Test images: {evaluated_images}\n")
    file.write(f"Mean Dice: {mean_dice:.4f}\n")
    file.write(f"Mean IoU: {mean_iou:.4f}\n")
    file.write(f"Pixel Precision: {pixel_precision:.4f}\n")
    file.write(f"Pixel Recall: {pixel_recall:.4f}\n")
    file.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")

    file.write(
        f"Ground Truth Positive: "
        f"{actual_positive_percentage:.2f}%\n"
    )

    file.write(
        f"Predicted Positive: "
        f"{predicted_positive_percentage:.2f}%\n"
    )

    file.write(f"True Positive: {int(total_tp)}\n")
    file.write(f"False Positive: {int(total_fp)}\n")
    file.write(f"False Negative: {int(total_fn)}\n")
    file.write(f"True Negative: {int(total_tn)}\n")


print("\nResults saved to:")
print(results_path)

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 60)