from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nDATASET SPLITTING")
    print("=" * 55)

    # Load manifest
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "filename",
        "image_path",
        "label",
        "category",
        "mask_path"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("Manifest is missing required columns.")

    # First split: 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=TRAIN_SIZE.__class__(1 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Second split: 15% validation, 15% testing
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    # Save datasets
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\nSPLITTING COMPLETED")
    print("-" * 55)

    print(f"Total images      : {len(df)}")
    print(f"Training images   : {len(train_df)}")
    print(f"Validation images : {len(val_df)}")
    print(f"Testing images    : {len(test_df)}")

    print("\nTraining distribution:")
    print(train_df["category"].value_counts())

    print("\nValidation distribution:")
    print(val_df["category"].value_counts())

    print("\nTesting distribution:")
    print(test_df["category"].value_counts())

    print("\nFiles saved:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")

    print("\nDataset splitting successful!")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()