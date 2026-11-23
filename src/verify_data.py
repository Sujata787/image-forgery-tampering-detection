from pathlib import Path
from PIL import Image, UnidentifiedImageError


# =====================================
# DATASET PATHS
# =====================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "data"

FOLDERS = {
    "Authentic Images (Au)": DATASET_DIR / "Au",
    "Tampered Images (Tp)": DATASET_DIR / "Tp",
    "Ground Truth Masks": DATASET_DIR / "CASIA 2 Groundtruth"
}


# =====================================
# SUPPORTED IMAGE FORMATS
# =====================================

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


# =====================================
# VERIFY DATASET
# =====================================

def verify_folder(folder_path):

    valid = 0
    corrupted = 0
    unsupported = 0

    if not folder_path.is_dir():
        print(f"ERROR: Folder not found: {folder_path}")
        return valid, corrupted, unsupported

    for file_path in folder_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            unsupported += 1
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()

            valid += 1

        except (UnidentifiedImageError, OSError, ValueError):
            corrupted += 1

    return valid, corrupted, unsupported


# =====================================
# MAIN FUNCTION
# =====================================

def main():

    print("\nCASIA2 DATASET VERIFICATION")
    print("=" * 50)

    total_valid = 0
    total_corrupted = 0
    total_unsupported = 0

    for name, folder in FOLDERS.items():

        print(f"\n{name}")
        print("-" * 40)

        valid, corrupted, unsupported = verify_folder(folder)

        print(f"Valid images      : {valid}")
        print(f"Corrupted images  : {corrupted}")
        print(f"Unsupported files : {unsupported}")

        total_valid += valid
        total_corrupted += corrupted
        total_unsupported += unsupported

    print("\n" + "=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)

    print(f"Total valid images     : {total_valid}")
    print(f"Total corrupted images : {total_corrupted}")
    print(f"Total unsupported files: {total_unsupported}")

    print("\nDataset verification completed.")


if __name__ == "__main__":
    main()