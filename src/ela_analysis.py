from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image, ImageChops, ImageEnhance


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "ela"

ELA_QUALITY = 90

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# GENERATE ELA IMAGE
# ==========================================

def generate_ela(
    image_path: Path,
    quality: int = ELA_QUALITY
) -> Image.Image:

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as original:

        original = original.convert("RGB")

        # Compress image into memory
        buffer = BytesIO()

        original.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        with Image.open(buffer) as compressed:

            compressed = compressed.convert("RGB")

            # Calculate image difference
            difference = ImageChops.difference(
                original,
                compressed
            )

            # Convert difference to grayscale
            gray_difference = difference.convert("L")

            # Get pixel intensity range
            extrema = gray_difference.getextrema()

            if not isinstance(extrema, tuple) or len(extrema) != 2:
                raise ValueError(
                    "Unable to calculate ELA pixel intensity."
                )

            max_difference = extrema[1]

            # Ensure maximum difference is numeric
            if not isinstance(max_difference, (int, float)):
                raise TypeError(
                    "Invalid maximum pixel difference."
                )

            # Calculate enhancement scale
            if max_difference == 0:
                scale = 1.0
            else:
                scale = 255.0 / float(max_difference)

            # Enhance difference image
            ela_image = ImageEnhance.Brightness(
                difference
            ).enhance(scale)

            return ela_image


# ==========================================
# MAIN PROCESSING
# ==========================================

def main():

    print("=" * 50)
    print("IMAGE FORGERY DETECTION - ELA ANALYSIS")
    print("=" * 50)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    # Load manifest
    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {"image_path", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Manifest must contain: {required_columns}"
        )

    # Select one authentic and one tampered image
    authentic = df[df["label"] == 0].head(1)

    tampered = df[df["label"] == 1].head(1)

    samples = pd.concat(
        [authentic, tampered],
        ignore_index=True
    )

    if len(samples) != 2:
        raise ValueError(
            "Both authentic and tampered samples are required."
        )

    # Process selected images
    for _, row in samples.iterrows():

        image_path = Path(str(row["image_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        label = int(row["label"])

        category = (
            "Tampered" if label == 1 else "Authentic"
        )

        print(f"\nProcessing: {image_path.name}")
        print(f"Category: {category}")

        ela_image = generate_ela(image_path)

        output_name = (
            f"{category}_{image_path.stem}_ELA.jpg"
        )

        output_path = OUTPUT_DIR / output_name

        ela_image.save(
            output_path,
            format="JPEG"
        )

        print(f"ELA saved: {output_path}")

    print("\nELA generation completed successfully!")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()