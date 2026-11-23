from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "mask_diagnostics"
    / "mismatched_pairs.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "mask_diagnostics"
    / "analysis_report.csv"
)


# ==========================================
# ANALYZE IMAGE-MASK PAIRS
# ==========================================

def analyze_pairs():

    df = pd.read_csv(REPORT_PATH)

    results = []

    for _, row in df.iterrows():

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as msk:
            mask = msk.convert("L")

        image_width, image_height = image.size
        mask_width, mask_height = mask.size

        image_ratio = image_width / image_height
        mask_ratio = mask_width / mask_height

        mask_array = np.asarray(mask)

        unique_values = np.unique(mask_array)

        nonzero_pixels = np.count_nonzero(mask_array)

        total_pixels = mask_array.size

        mask_percentage = (
            nonzero_pixels / total_pixels
        ) * 100

        width_ratio = mask_width / image_width
        height_ratio = mask_height / image_height

        if (
            width_ratio < 0.8
            or width_ratio > 1.2
            or height_ratio < 0.8
            or height_ratio > 1.2
        ):

            status = "LARGE_DIMENSION_DIFFERENCE"

        elif abs(image_ratio - mask_ratio) > 0.03:

            status = "ASPECT_RATIO_DIFFERENCE"

        else:

            status = "SMALL_DIMENSION_DIFFERENCE"

        results.append({

            "image": image_path.name,

            "mask": mask_path.name,

            "image_size": f"{image_width}x{image_height}",

            "mask_size": f"{mask_width}x{mask_height}",

            "image_ratio": round(image_ratio, 4),

            "mask_ratio": round(mask_ratio, 4),

            "mask_unique_values": str(unique_values[:20]),

            "mask_nonzero_percentage": round(
                mask_percentage, 2
            ),

            "status": status

        })

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nMASK ANALYSIS RESULTS")
    print("=" * 60)

    print(report.to_string(index=False))

    print("\nReport saved to:")
    print(OUTPUT_PATH)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    analyze_pairs()