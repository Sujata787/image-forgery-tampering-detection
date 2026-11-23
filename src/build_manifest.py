from pathlib import Path
import re
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
AU_DIR = DATA_DIR / "Au"
TP_DIR = DATA_DIR / "Tp"
GT_DIR = DATA_DIR / "CASIA 2 Groundtruth"

MANIFEST_PATH = DATA_DIR / "manifest.csv"

IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# --------------------------------------------------
# EXTRACT IMAGE ID
# --------------------------------------------------

def extract_id(filename: str):

    stem = Path(filename).stem

    # Remove ground-truth suffixes such as _gt and _gt3
    stem = re.sub(r"_gt\d*$", "", stem, flags=re.IGNORECASE)

    match = re.search(r"_(\d+)$", stem)

    return match.group(1) if match else None


# --------------------------------------------------
# GET IMAGE FILES
# --------------------------------------------------

def get_images(folder: Path):

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted([
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_FORMATS
    ])


# --------------------------------------------------
# BUILD MASK LOOKUP
# --------------------------------------------------

def build_mask_lookup():

    masks = get_images(GT_DIR)

    mask_lookup = {}

    for mask in masks:

        mask_id = extract_id(mask.name)

        if mask_id is None:
            continue

        if mask_id in mask_lookup:
            raise ValueError(
                f"Duplicate mask ID {mask_id}: "
                f"{mask_lookup[mask_id].name} and {mask.name}"
            )

        mask_lookup[mask_id] = mask

    return mask_lookup, masks


# --------------------------------------------------
# BUILD MANIFEST
# --------------------------------------------------

def main():

    print("\nBUILDING DATASET MANIFEST")
    print("=" * 55)

    authentic_images = get_images(AU_DIR)
    tampered_images = get_images(TP_DIR)

    mask_lookup, all_masks = build_mask_lookup()

    records = []

    # Authentic images: label 0
    for image in authentic_images:

        records.append({
            "filename": image.name,
            "image_path": str(image.relative_to(PROJECT_DIR)),
            "label": 0,
            "category": "Authentic",
            "mask_path": ""
        })

    # Tampered images: label 1
    matched_count = 0
    missing_count = 0

    for image in tampered_images:

        image_id = extract_id(image.name)

        mask = mask_lookup.get(image_id) if image_id else None

        if mask is not None:
            mask_path = str(mask.relative_to(PROJECT_DIR))
            matched_count += 1
        else:
            mask_path = ""
            missing_count += 1

        records.append({
            "filename": image.name,
            "image_path": str(image.relative_to(PROJECT_DIR)),
            "label": 1,
            "category": "Tampered",
            "mask_path": mask_path
        })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save manifest
    df.to_csv(MANIFEST_PATH, index=False)

    # Report
    print("\nMANIFEST REPORT")
    print("-" * 55)

    print(f"Authentic images       : {len(authentic_images)}")
    print(f"Tampered images        : {len(tampered_images)}")
    print(f"Total images           : {len(df)}")
    print(f"Available ground masks : {len(all_masks)}")
    print(f"Matched masks          : {matched_count}")
    print(f"Missing masks          : {missing_count}")

    print("\nLabel distribution:")
    print(df["category"].value_counts())

    print("\nManifest saved:")
    print(MANIFEST_PATH)

    print("\nManifest creation completed successfully.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()