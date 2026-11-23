from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "outputs" / "segmentation_splits"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "mask_diagnostics"

PREVIEW_DIR = OUTPUT_DIR / "previews"

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

PREVIEW_SIZE = (420, 420)


# ==========================================
# FIND DIMENSION MISMATCHES
# ==========================================

def inspect_mismatches():

    mismatches = []

    for split_file in SPLIT_FILES:

        csv_path = SPLIT_DIR / split_file

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        print(f"\nChecking {split_file}...")

        for _, row in df.iterrows():

            image_path = Path(str(row["image_path"]))
            mask_path = Path(str(row["mask_path"]))

            if not image_path.is_file():
                print(f"Missing image: {image_path}")
                continue

            if not mask_path.is_file():
                print(f"Missing mask: {mask_path}")
                continue

            try:

                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                    image_size = image.size

                with Image.open(mask_path) as msk:
                    mask = msk.convert("L")
                    mask_size = mask.size

                if image_size != mask_size:

                    mismatches.append({
                        "split": split_file,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "mask_width": mask_size[0],
                        "mask_height": mask_size[1],
                    })

            except Exception as error:

                print(
                    f"Error reading {image_path.name}: {error}"
                )

    return mismatches


# ==========================================
# CREATE VISUAL PREVIEWS
# ==========================================

def create_previews(mismatches, limit=30):

    for index, item in enumerate(mismatches[:limit]):

        image_path = Path(item["image_path"])
        mask_path = Path(item["mask_path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as msk:
            mask = msk.convert("L")

        image_preview = ImageOps.contain(
            image,
            PREVIEW_SIZE
        )

        mask_preview = ImageOps.contain(
            mask.convert("RGB"),
            PREVIEW_SIZE
        )

        canvas = Image.new(
            "RGB",
            (900, 540),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 10),
            f"IMAGE: {image_path.name}",
            fill="black"
        )

        draw.text(
            (460, 10),
            f"MASK: {mask_path.name}",
            fill="black"
        )

        draw.text(
            (10, 30),
            f"Image size: {image.size}",
            fill="black"
        )

        draw.text(
            (460, 30),
            f"Mask size: {mask.size}",
            fill="black"
        )

        canvas.paste(image_preview, (10, 70))

        canvas.paste(mask_preview, (460, 70))

        output_path = PREVIEW_DIR / f"pair_{index + 1}.png"

        canvas.save(output_path)

    print(f"\nCreated previews in: {PREVIEW_DIR}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("IMAGE-MASK DIMENSION DIAGNOSTICS")
    print("=" * 60)

    mismatches = inspect_mismatches()

    if mismatches:

        report_path = OUTPUT_DIR / "mismatched_pairs.csv"

        pd.DataFrame(mismatches).to_csv(
            report_path,
            index=False
        )

        print("\nMISMATCH SUMMARY")
        print("-" * 40)

        print("Total mismatched pairs:", len(mismatches))

        print("Report saved:", report_path)

        create_previews(mismatches)

    else:

        print("\nNo dimension mismatches found.")

    print("\nDIAGNOSTIC COMPLETED.")