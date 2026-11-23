from pathlib import Path

import torch
import numpy as np

from segmentation_model import build_segmentation_model
from segmentation_data_loader import get_segmentation_loaders


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_unet.pth"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model():

    print("=" * 60)
    print("U-NET SEGMENTATION MODEL EVALUATION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Load test dataset
    _, _, test_loader = get_segmentation_loaders()

    # Load model
    model = build_segmentation_model().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("\nBest U-Net model loaded successfully.")

    # Metric counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    total_dice = 0.0
    total_iou = 0.0

    total_gt_positive = 0
    total_pred_positive = 0
    total_pixels = 0

    total_images = 0

    # --------------------------------------
    # TEST LOOP
    # --------------------------------------

    with torch.no_grad():

        for batch_index, (images, masks) in enumerate(test_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # Per-image metrics
            for i in range(images.size(0)):

                pred = predictions[i]
                target = masks[i]

                tp = ((pred == 1) & (target == 1)).sum().item()

                fp = ((pred == 1) & (target == 0)).sum().item()

                fn = ((pred == 0) & (target == 1)).sum().item()

                tn = ((pred == 0) & (target == 0)).sum().item()

                intersection = tp

                union = tp + fp + fn

                dice = (
                    2 * tp + 1
                ) / (
                    2 * tp + fp + fn + 1
                )

                iou = (
                    intersection + 1
                ) / (
                    union + 1
                )

                total_dice += dice
                total_iou += iou

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                total_gt_positive += int(target.sum().item())

                total_pred_positive += int(pred.sum().item())

                total_pixels += target.numel()

                total_images += 1

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Evaluated batches: "
                    f"{batch_index + 1}/{len(test_loader)}"
                )

    # --------------------------------------
    # FINAL METRICS
    # --------------------------------------

    mean_dice = total_dice / total_images

    mean_iou = total_iou / total_images

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0
    )

    pixel_accuracy = (
        (total_tp + total_tn) / total_pixels
    )

    gt_percentage = (
        total_gt_positive / total_pixels
    ) * 100

    prediction_percentage = (
        total_pred_positive / total_pixels
    ) * 100

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print("\n" + "=" * 60)
    print("SEGMENTATION TEST RESULTS")
    print("=" * 60)

    print(f"Test images          : {total_images}")

    print(f"Mean Dice Score      : {mean_dice:.4f}")

    print(f"Mean IoU             : {mean_iou:.4f}")

    print(f"Pixel Precision      : {precision:.4f}")

    print(f"Pixel Recall         : {recall:.4f}")

    print(f"Pixel Accuracy       : {pixel_accuracy:.4f}")

    print(f"Ground Truth Positive: {gt_percentage:.2f}%")

    print(f"Predicted Positive   : {prediction_percentage:.2f}%")

    print("\nConfusion Matrix:")

    print(f"True Positive : {total_tp}")

    print(f"False Positive: {total_fp}")

    print(f"False Negative: {total_fn}")

    print(f"True Negative : {total_tn}")

    print("\nEVALUATION COMPLETED.")

    print("=" * 60)


if __name__ == "__main__":

    evaluate_model()