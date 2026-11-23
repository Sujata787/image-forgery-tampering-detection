from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "outputs" / "segmentation_splits_clean"

IMAGE_SIZE = 128
BATCH_SIZE = 2
NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# SEGMENTATION DATASET
# ==================================================

class SegmentationDataset(Dataset):

    def __init__(self, csv_file):

        self.csv_file = Path(csv_file)

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.csv_file}"
            )

        self.data = pd.read_csv(self.csv_file)

        required_columns = ["image_path", "mask_path"]

        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(
                    f"Missing column '{column}' in {self.csv_file.name}"
                )

        self.data = self.data.dropna(
            subset=required_columns
        ).reset_index(drop=True)

        # Verify all image and mask paths.
        for index, row in self.data.iterrows():

            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

        print(
            f"Loaded {len(self.data)} records "
            f"from {self.csv_file.name}"
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        # ------------------------------------------
        # LOAD IMAGE
        # ------------------------------------------

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # ------------------------------------------
        # LOAD MASK
        # ------------------------------------------

        with Image.open(mask_path) as msk:
            mask = msk.convert("L")

        # ------------------------------------------
        # VERIFY ORIGINAL DIMENSIONS
        # ------------------------------------------

        if image.size != mask.size:

            raise ValueError(
                "\nImage-mask dimensions do not match!"
                f"\nImage: {image_path.name}"
                f"\nImage dimensions: {image.size}"
                f"\nMask dimensions: {mask.size}"
                "\nCheck the dataset before training."
            )

        # ------------------------------------------
        # RESIZE IMAGE
        # ------------------------------------------

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # ------------------------------------------
        # RESIZE MASK
        # ------------------------------------------

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        # ------------------------------------------
        # IMAGE TO TENSOR
        # ------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        ).copy()

        image_array = image_array / 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1)

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image_tensor = (image_tensor - mean) / std

        # ------------------------------------------
        # MASK TO TENSOR
        # ------------------------------------------

        mask_array = np.asarray(
            mask,
            dtype=np.uint8
        ).copy()

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0).float()

        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, mask_tensor


# ==================================================
# CREATE DATA LOADERS
# ==================================================

def get_data_loaders():

    train_dataset = SegmentationDataset(
        SPLIT_DIR / "train.csv"
    )

    validation_dataset = SegmentationDataset(
        SPLIT_DIR / "validation.csv"
    )

    test_dataset = SegmentationDataset(
        SPLIT_DIR / "test.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# ==================================================
# COMPATIBILITY FUNCTION
# ==================================================

def get_segmentation_loaders():

    return get_data_loaders()


# ==================================================
# TEST DATA LOADER
# ==================================================

if __name__ == "__main__":

    print("\nSEGMENTATION DATA LOADER TEST")
    print("=" * 50)

    print("Device:", DEVICE)

    train_loader, validation_loader, test_loader = (
        get_data_loaders()
    )

    print("\nDataset batches:")

    print("Training batches  :", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches   :", len(test_loader))

    # ------------------------------------------
    # VERIFY ONE BATCH
    # ------------------------------------------

    images, masks = next(iter(train_loader))

    print("\nBATCH VERIFICATION")
    print("=" * 50)

    print("Image shape:", images.shape)
    print("Mask shape :", masks.shape)

    print("Image dtype:", images.dtype)
    print("Mask dtype :", masks.dtype)

    print("Mask values:", torch.unique(masks))

    assert images.shape == (
        BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE
    )

    assert masks.shape == (
        BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE
    )

    assert torch.all(
        (masks == 0) | (masks == 1)
    )

    print("\nDATA LOADER TEST COMPLETED SUCCESSFULLY.")