from pathlib import Path

import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from model import build_model
from segmentation_model import build_segmentation_model


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = PROJECT_DIR / "artifacts"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"

CLASSIFICATION_MODEL_PATH = ARTIFACT_DIR / "best_resnet18.pth"

SEGMENTATION_MODEL_PATH = ARTIFACT_DIR / "best_unet.pth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "Authentic",
    1: "Tampered"
}

SEGMENTATION_THRESHOLD = 0.1


# ==========================================
# IMAGE TRANSFORMATIONS
# ==========================================

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]


classification_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])


segmentation_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

def load_models():

    print("\nLoading trained models...")

    if not CLASSIFICATION_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ResNet18 checkpoint not found: {CLASSIFICATION_MODEL_PATH}"
        )

    if not SEGMENTATION_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"U-Net checkpoint not found: {SEGMENTATION_MODEL_PATH}"
        )

    # Load ResNet18
    classification_model = build_model(pretrained=False)

    classification_checkpoint = torch.load(
        CLASSIFICATION_MODEL_PATH,
        map_location=DEVICE
    )

    classification_model.load_state_dict(
        classification_checkpoint
    )

    classification_model.to(DEVICE)

    classification_model.eval()

    # Load U-Net
    segmentation_model = build_segmentation_model()

    segmentation_checkpoint = torch.load(
        SEGMENTATION_MODEL_PATH,
        map_location=DEVICE
    )

    segmentation_model.load_state_dict(
        segmentation_checkpoint
    )

    segmentation_model.to(DEVICE)

    segmentation_model.eval()

    print("ResNet18 loaded successfully.")

    print("U-Net loaded successfully.")

    return classification_model, segmentation_model


# ==========================================
# CLASSIFICATION PREDICTION
# ==========================================

def predict_classification(model, image):

    image_tensor = classification_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    # Convert values to standard Python types
    predicted_class = int(predicted_class.item())

    confidence = float(confidence.item()) * 100

    label = CLASS_NAMES[predicted_class]

    return label, confidence


# ==========================================
# SEGMENTATION PREDICTION
# ==========================================

def predict_segmentation(model, image):

    image_tensor = segmentation_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.sigmoid(outputs)

        predicted_mask = (
            probabilities > SEGMENTATION_THRESHOLD
        ).float()

    mask_array = (
        predicted_mask.squeeze().cpu().numpy() * 255
    ).astype(np.uint8)

    mask_image = Image.fromarray(mask_array)

    # Restore original image dimensions
    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    return mask_image


# ==========================================
# CREATE HIGHLIGHTED OVERLAY
# ==========================================

def create_overlay(original_image, mask_image):

    original_image = original_image.convert("RGBA")

    mask_array = np.array(mask_image)

    red_overlay = Image.new(
        "RGBA",
        original_image.size,
        (255, 0, 0, 0)
    )

    overlay_array = np.array(red_overlay)

    overlay_array[mask_array > 0] = [
        255, 0, 0, 110
    ]

    red_overlay = Image.fromarray(overlay_array)

    result = Image.alpha_composite(
        original_image,
        red_overlay
    )

    return result.convert("RGB")


# ==========================================
# MAIN PREDICTION FUNCTION
# ==========================================

def main():

    print("=" * 60)

    print("IMAGE FORGERY AND TAMPERING DETECTION SYSTEM")

    print("=" * 60)

    try:

        classification_model, segmentation_model = load_models()

    except Exception as error:

        print("\nERROR WHILE LOADING MODELS:")

        print(error)

        return

    image_path_input = input(
        "\nEnter the complete image path: "
    ).strip().strip('"')

    image_path = Path(image_path_input)

    if not image_path.is_file():

        print("\nERROR: Image file not found.")

        return

    try:

        original_image = Image.open(image_path).convert("RGB")

    except Exception as error:

        print("\nERROR: Could not open image.")

        print(error)

        return

    print("\nImage loaded successfully.")

    print("Image dimensions:", original_image.size)

    try:

        # Classification
        label, confidence = predict_classification(
            classification_model,
            original_image
        )

        # Localization
        predicted_mask = predict_segmentation(
            segmentation_model,
            original_image
        )

        # Create overlay
        overlay = create_overlay(
            original_image,
            predicted_mask
        )

        # Save results
        mask_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"

        overlay_path = OUTPUT_DIR / f"{image_path.stem}_overlay.png"

        predicted_mask.save(mask_path)

        overlay.save(overlay_path)

        # Display results
        print("\n" + "=" * 60)

        print("PREDICTION RESULTS")

        print("=" * 60)

        print("Classification :", label)

        print(f"Confidence     : {confidence:.2f}%")

        print("Mask saved     :", mask_path)

        print("Overlay saved  :", overlay_path)

        print("=" * 60)

        print("\nIMPORTANT:")

        print(
            "The highlighted regions are experimental U-Net predictions. "
            "They may not accurately represent the actual tampered area."
        )

    except Exception as error:

        print("\nERROR DURING PREDICTION:")

        print(error)


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()