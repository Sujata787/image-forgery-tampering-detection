from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "outputs" / "segmentation_splits"

DIAGNOSTIC_FILE = (
    PROJECT_DIR
    / "outputs"
    / "mask_diagnostics"
    / "mismatched_pairs.csv"
)

CLEAN_DIR = (
    PROJECT_DIR
    / "outputs"
    / "segmentation_splits_clean"
)


# ==========================================
# CREATE CLEAN SPLITS
# ==========================================

def create_clean_splits():

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    mismatch_df = pd.read_csv(DIAGNOSTIC_FILE)

    bad_images = {
        str(Path(path).resolve()).lower()
        for path in mismatch_df["image_path"]
    }

    for split_name in [
        "train.csv",
        "validation.csv",
        "test.csv"
    ]:

        input_file = SPLIT_DIR / split_name

        df = pd.read_csv(input_file)

        original_count = len(df)

        image_paths = df["image_path"].apply(
            lambda path: str(Path(path).resolve()).lower()
        )

        clean_df = df.loc[
            ~image_paths.isin(bad_images)
        ].copy()

        removed_count = original_count - len(clean_df)

        output_file = CLEAN_DIR / split_name

        clean_df.to_csv(output_file, index=False)

        print(f"\n{split_name}")
        print(f"Original pairs : {original_count}")
        print(f"Removed pairs  : {removed_count}")
        print(f"Clean pairs    : {len(clean_df)}")

    print("\nCLEAN SEGMENTATION SPLITS CREATED.")
    print(f"Output folder: {CLEAN_DIR}")


if __name__ == "__main__":
    create_clean_splits()