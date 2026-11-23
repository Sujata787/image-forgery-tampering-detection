from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "segmentation_splits"

RANDOM_STATE = 42


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("=" * 60)
    print("SEGMENTATION DATASET PREPARATION")
    print("=" * 60)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_path",
        "mask_path",
        "label"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Select tampered images with masks
    segmentation_df = df[
        (df["label"] == 1)
        & df["mask_path"].notna()
        & (df["mask_path"].astype(str).str.strip() != "")
    ].copy()

    print("\nTotal paired records:", len(segmentation_df))

    # Validate image and mask paths
    valid_records = []

    for _, row in segmentation_df.iterrows():

        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        if not mask_path.is_absolute():
            mask_path = PROJECT_DIR / mask_path

        if image_path.is_file() and mask_path.is_file():

            valid_records.append({
                "image_path": str(image_path),
                "mask_path": str(mask_path)
            })

    valid_df = pd.DataFrame(
        valid_records,
        columns=["image_path", "mask_path"]
    )

    print("Valid image-mask pairs:", len(valid_df))

    if len(valid_df) < 3:
        raise ValueError(
            "Not enough valid image-mask pairs for splitting."
        )

    # 70% training, 15% validation, 15% testing
    train_df, temporary_df = train_test_split(
        valid_df,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # Save CSV files
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nSEGMENTATION DATASET SPLIT")
    print("-" * 40)

    print("Training images  :", len(train_df))
    print("Validation images:", len(validation_df))
    print("Testing images   :", len(test_df))

    print("\nSaved files:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")

    print("\nSEGMENTATION DATA PREPARATION COMPLETED")


if __name__ == "__main__":
    main()