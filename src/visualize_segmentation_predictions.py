from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import modules directly because this script is inside src/
from segmentation_model import build_segmentation_model
from segmentation_data_loader import get_segmentation_loaders


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_unet.pth"

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "segmentation_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 12
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# ==================================================
# CHECK MODEL FILE
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 60)
print("U-NET SEGMENTATION VISUALIZATION")
print("=" * 60)

train_loader, validation_loader, test_loader = (
    get_segmentation_loaders()
)

print("\nTest dataset loaded successfully.")


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

print("Best U-Net model loaded successfully.")
print("Device:", DEVICE)


# ==================================================
# CONVERT NORMALIZED IMAGE TO DISPLAY FORMAT
# ==================================================

def convert_image(image_tensor):

    image = image_tensor.detach().cpu().numpy()

    image = image.transpose(1, 2, 0)

    image = image * STD + MEAN

    image = np.clip(image, 0, 1)

    return image


# ==================================================
# GENERATE PREDICTIONS
# ==================================================

image_count = 0

with torch.no_grad():

    for batch_index, (images, masks) in enumerate(test_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities > THRESHOLD
        ).float()

        for index in range(images.size(0)):

            if image_count >= MAX_IMAGES:
                break

            # Original image
            original_image = convert_image(
                images[index]
            )

            # Actual ground-truth mask
            actual_mask = (
                masks[index, 0].cpu().numpy() > 0.5
            )

            # Predicted mask
            predicted_mask = (
                predictions[index, 0].cpu().numpy() > 0.5
            )

            # ======================================
            # CREATE OVERLAY
            # ======================================

            overlay = original_image.copy()

            # Actual mask = Green
            overlay[actual_mask] = (
                0.5 * original_image[actual_mask]
                + 0.5 * np.array([0, 1, 0])
            )

            # Predicted mask = Red
            overlay[predicted_mask] = (
                0.5 * original_image[predicted_mask]
                + 0.5 * np.array([1, 0, 0])
            )

            # Correct overlap = Yellow
            overlap = actual_mask & predicted_mask

            overlay[overlap] = (
                0.5 * original_image[overlap]
                + 0.5 * np.array([1, 1, 0])
            )

            # ======================================
            # DISPLAY FOUR IMAGES
            # ======================================

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(18, 5)
            )

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")

            axes[1].imshow(
                actual_mask,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[1].set_title("Ground Truth Mask")

            axes[2].imshow(
                predicted_mask,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[2].set_title("Predicted Mask")

            axes[3].imshow(overlay)
            axes[3].set_title(
                "Overlay: Green=Actual, Red=Predicted"
            )

            for axis in axes:
                axis.axis("off")

            plt.tight_layout()

            # ======================================
            # SAVE IMAGE
            # ======================================

            output_path = (
                OUTPUT_DIR
                / f"prediction_{image_count + 1}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            image_count += 1

            print(
                f"Generated visualization "
                f"{image_count}/{MAX_IMAGES}"
            )

        if image_count >= MAX_IMAGES:
            break


# ==================================================
# COMPLETION
# ==================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print("Images generated:", image_count)

print("Saved location:")
print(OUTPUT_DIR)

print("=" * 60)