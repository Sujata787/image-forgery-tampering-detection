from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "localization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(image_path):

    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


# ==========================================
# LOAD GROUND-TRUTH MASK
# ==========================================

def load_mask(mask_path):

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")

        mask_array = np.array(mask)

    # Convert mask into binary format
    binary_mask = mask_array > 0

    return binary_mask


# ==========================================
# CREATE MASK OVERLAY
# ==========================================

def create_overlay(image, mask):

    image_array = np.array(image).copy()

    if image_array.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask dimensions do not match."
        )

    # Highlight tampered pixels in red
    red_overlay = image_array.copy()

    red_overlay[mask] = [255, 0, 0]

    # Blend original image and highlighted mask
    overlay = (
        0.5 * image_array.astype(np.float32)
        + 0.5 * red_overlay.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# ==========================================
# PROCESS ONE SAMPLE
# ==========================================

def process_sample(image_path, mask_path, sample_number):

    print(f"\nProcessing sample {sample_number}")

    image = load_image(image_path)

    mask = load_mask(mask_path)

    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            f"Dimension mismatch: {image_path.name}"
        )

    overlay = create_overlay(image, mask)

    # Create visualization
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Forgery Localization Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"localization_sample_{sample_number}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 55)
    print("IMAGE FORGERY DETECTION - MASK LOCALIZATION")
    print("=" * 55)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_path",
        "mask_path",
        "label"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required_columns - set(df.columns)}"
        )

    # Select tampered images with available masks
    tampered_df = df[
        (df["label"] == 1)
        & df["mask_path"].notna()
        & (df["mask_path"].astype(str).str.strip() != "")
    ]

    if tampered_df.empty:
        raise ValueError(
            "No tampered images with masks found."
        )

    samples = tampered_df.head(NUM_SAMPLES)

    successful = 0

    for index, (_, row) in enumerate(
        samples.iterrows(),
        start=1
    ):

        image_path = Path(str(row["image_path"]))

        mask_path = Path(str(row["mask_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        if not mask_path.is_absolute():
            mask_path = PROJECT_DIR / mask_path

        if not image_path.exists():
            print("Image missing:", image_path)
            continue

        if not mask_path.exists():
            print("Mask missing:", mask_path)
            continue

        process_sample(
            image_path,
            mask_path,
            index
        )

        successful += 1

    print("\n" + "=" * 55)
    print("LOCALIZATION VISUALIZATION COMPLETED")
    print("Successful samples:", successful)
    print("Output directory:", OUTPUT_DIR)
    print("=" * 55)


if __name__ == "__main__":
    main()