from pathlib import Path

import numpy as np
import torch

from segmentation_model import build_segmentation_model
from segmentation_data_loader import get_segmentation_loaders


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "artifacts" / "best_unet.pth"

DEVICE = torch.device("cpu")

THRESHOLD = 0.5
MAX_IMAGES = 12


print("=" * 60)
print("SEGMENTATION PREDICTION ANALYSIS")
print("=" * 60)


# Load test data
_, _, test_loader = get_segmentation_loaders()


# Load trained model
model = build_segmentation_model().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("\nModel loaded successfully.")


# Analyze predictions
image_count = 0

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > THRESHOLD).float()

        for index in range(images.size(0)):

            if image_count >= MAX_IMAGES:
                break

            actual_mask = masks[index, 0].numpy()

            predicted_mask = (
                predictions[index, 0].cpu().numpy()
            )

            actual_percentage = np.mean(
                actual_mask > 0.5
            ) * 100

            predicted_percentage = np.mean(
                predicted_mask > 0.5
            ) * 100

            print(f"\nImage {image_count + 1}")

            print(
                f"Actual tampered area: "
                f"{actual_percentage:.2f}%"
            )

            print(
                f"Predicted tampered area: "
                f"{predicted_percentage:.2f}%"
            )

            image_count += 1

        if image_count >= MAX_IMAGES:
            break


print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)