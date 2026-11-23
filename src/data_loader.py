from pathlib import Path
from typing import Any

import pandas as pd
import torch

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "outputs" / "splits"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# ==================================================
# IMAGE TRANSFORMATIONS
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# CUSTOM DATASET
# ==================================================

class ForgeryDataset(Dataset):

    def __init__(
        self,
        csv_file: Path,
        transform: Any = None
    ) -> None:

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        required_columns = {"image_path", "label"}

        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                "CSV must contain image_path and label columns."
            )

        if self.data.empty:
            raise ValueError("Dataset is empty.")

        self.image_paths = self.data["image_path"].astype(str).tolist()

        self.labels = self.data["label"].astype(int).tolist()

        # Verify image files
        for relative_path in self.image_paths:

            image_path = PROJECT_DIR / relative_path.replace("\\", "/")

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):

        relative_path = self.image_paths[index]

        image_path = PROJECT_DIR / relative_path.replace("\\", "/")

        label = self.labels[index]

        # Load image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Apply transformation
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Ensure image is a PyTorch tensor
        if not isinstance(image, Tensor):
            raise TypeError("Image transformation did not return a Tensor.")

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label_tensor


# ==================================================
# CREATE DATA LOADERS
# ==================================================

def get_data_loaders():

    train_csv = SPLIT_DIR / "train.csv"
    validation_csv = SPLIT_DIR / "validation.csv"
    test_csv = SPLIT_DIR / "test.csv"

    for csv_file in [train_csv, validation_csv, test_csv]:

        if not csv_file.is_file():
            raise FileNotFoundError(
                f"Split file not found: {csv_file}"
            )

    # Create datasets
    train_dataset = ForgeryDataset(
        train_csv,
        train_transform
    )

    validation_dataset = ForgeryDataset(
        validation_csv,
        validation_transform
    )

    test_dataset = ForgeryDataset(
        test_csv,
        validation_transform
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, validation_loader, test_loader


# ==================================================
# TEST PREPROCESSING
# ==================================================

def main():

    print("\nIMAGE PREPROCESSING VERIFICATION")
    print("=" * 55)

    train_loader, validation_loader, test_loader = get_data_loaders()

    print("\nDataset sizes:")

    train_size = len(pd.read_csv(SPLIT_DIR / "train.csv"))
    validation_size = len(pd.read_csv(SPLIT_DIR / "validation.csv"))
    test_size = len(pd.read_csv(SPLIT_DIR / "test.csv"))

    print("Training   :", train_size)
    print("Validation :", validation_size)
    print("Testing    :", test_size)

    # Load one batch
    images, labels = next(iter(train_loader))

    print("\nTraining batch information:")

    print("Image shape :", images.shape)
    print("Label shape :", labels.shape)

    print("Image dtype :", images.dtype)
    print("Label dtype :", labels.dtype)

    print("\nClass distribution in first batch:")

    print("Authentic (0):", (labels == 0).sum().item())
    print("Tampered (1):", (labels == 1).sum().item())

    print("\nImage preprocessing completed successfully!")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()