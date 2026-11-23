from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

TP_DIR = PROJECT_DIR / "data" / "Tp"
GT_DIR = PROJECT_DIR / "data" / "CASIA 2 Groundtruth"

image_name = "Tp_D_NRD_S_N_cha10002_cha10001_20094.jpg"
mask_name = "Tp_D_NRD_S_N_cha10002_cha10001_20094_gt3.png"

image_path = TP_DIR / image_name
mask_path = GT_DIR / mask_name


def main():

    print("\nUNMATCHED IMAGE AND MASK VERIFICATION")
    print("=" * 55)

    if not image_path.exists():
        print("Image not found!")
        return

    if not mask_path.exists():
        print("Mask not found!")
        return

    with Image.open(image_path) as image:
        image_size = image.size
        image_format = image.format

    with Image.open(mask_path) as mask:
        mask_array = np.array(mask.convert("L"))
        mask_size = mask.size

    unique_values = np.unique(mask_array)

    print("\nImage information:")
    print("Filename:", image_name)
    print("Size:", image_size)
    print("Format:", image_format)

    print("\nMask information:")
    print("Filename:", mask_name)
    print("Size:", mask_size)
    print("Unique pixel values:", unique_values.tolist())

    print("\nVerification:")

    if image_size == mask_size:
        print("Image and mask dimensions match.")
    else:
        print("WARNING: Image and mask dimensions differ.")

    if len(unique_values) <= 10:
        print("Mask appears to have a limited number of pixel values.")
    else:
        print("Mask contains multiple pixel values; inspect it visually.")

    print("\nVerification completed.")


if __name__ == "__main__":
    main()