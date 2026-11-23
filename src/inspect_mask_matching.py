from pathlib import Path
import re
from typing import Optional


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

TP_DIR = PROJECT_DIR / "data" / "Tp"
GT_DIR = PROJECT_DIR / "data" / "CASIA 2 Groundtruth"

IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# --------------------------------------------------
# EXTRACT NUMERIC ID FROM FILENAME
# --------------------------------------------------

def extract_id(filename: str) -> Optional[str]:

    stem = Path(filename).stem

    # Remove ground-truth suffix
    if stem.lower().endswith("_gt"):
        stem = stem[:-3]

    # Extract final numeric ID
    match = re.search(r"_(\d+)$", stem)

    if match:
        return match.group(1)

    return None


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    # Check dataset folders
    if not TP_DIR.exists():
        print(f"ERROR: Tampered image folder not found: {TP_DIR}")
        return

    if not GT_DIR.exists():
        print(f"ERROR: Ground-truth folder not found: {GT_DIR}")
        return

    # Read image files
    images = [
        file for file in TP_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_FORMATS
    ]

    # Read mask files
    masks = [
        file for file in GT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_FORMATS
    ]

    # Create dictionaries using numeric IDs
    image_lookup: dict[str, Path] = {}
    mask_lookup: dict[str, Path] = {}

    images_without_id = []
    masks_without_id = []

    duplicate_image_ids = []
    duplicate_mask_ids = []

    # Process tampered images
    for image in images:

        image_id = extract_id(image.name)

        if image_id is None:
            images_without_id.append(image.name)
            continue

        if image_id in image_lookup:
            duplicate_image_ids.append(image_id)
        else:
            image_lookup[image_id] = image

    # Process ground-truth masks
    for mask in masks:

        mask_id = extract_id(mask.name)

        if mask_id is None:
            masks_without_id.append(mask.name)
            continue

        if mask_id in mask_lookup:
            duplicate_mask_ids.append(mask_id)
        else:
            mask_lookup[mask_id] = mask

    # Get valid IDs
    image_ids: set[str] = set(image_lookup.keys())
    mask_ids: set[str] = set(mask_lookup.keys())

    # Find matching and unmatched IDs
    common_ids = image_ids.intersection(mask_ids)

    missing_mask_ids = image_ids - mask_ids

    unmatched_mask_ids = mask_ids - image_ids

    # --------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("       CASIA 2 MASK MATCHING REPORT")
    print("=" * 60)

    print(f"Tampered images          : {len(images)}")
    print(f"Ground-truth masks       : {len(masks)}")
    print(f"Images with numeric IDs  : {len(image_ids)}")
    print(f"Masks with numeric IDs   : {len(mask_ids)}")
    print(f"Successfully matched IDs : {len(common_ids)}")

    print("\n" + "-" * 60)
    print("IMAGES WITHOUT MATCHING MASKS")
    print("-" * 60)

    if missing_mask_ids:

        for image_id in sorted(missing_mask_ids):
            print(image_lookup[image_id].name)

    else:
        print("None")

    print("\n" + "-" * 60)
    print("MASKS WITHOUT MATCHING IMAGES")
    print("-" * 60)

    if unmatched_mask_ids:

        for mask_id in sorted(unmatched_mask_ids):
            print(mask_lookup[mask_id].name)

    else:
        print("None")

    print("\n" + "-" * 60)
    print("IMAGES WITHOUT NUMERIC IDs")
    print("-" * 60)

    if images_without_id:

        for filename in images_without_id:
            print(filename)

    else:
        print("None")

    print("\n" + "-" * 60)
    print("MASKS WITHOUT NUMERIC IDs")
    print("-" * 60)

    if masks_without_id:

        for filename in masks_without_id:
            print(filename)

    else:
        print("None")

    print("\n" + "-" * 60)
    print("DUPLICATE IMAGE IDs")
    print("-" * 60)

    print(sorted(duplicate_image_ids) if duplicate_image_ids else "None")

    print("\n" + "-" * 60)
    print("DUPLICATE MASK IDs")
    print("-" * 60)

    print(sorted(duplicate_mask_ids) if duplicate_mask_ids else "None")

    print("\n" + "=" * 60)
    print("MASK MATCHING INSPECTION COMPLETED")
    print("=" * 60)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()