from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"

AU_DIR = DATA_DIR / "Au"
TP_DIR = DATA_DIR / "Tp"
GT_DIR = DATA_DIR / "CASIA 2 Groundtruth"


def inspect_folder(folder, name):

    print(f"\n{name}")
    print("=" * 50)

    if not folder.exists():
        print("Folder not found:", folder)
        return

    files = sorted(
        [f for f in folder.rglob("*") if f.is_file()]
    )

    print("Total files:", len(files))

    print("\nFirst 15 filenames:")

    for file in files[:15]:
        print(file.name)

    unsupported = [
        f for f in files
        if f.suffix.lower() not in {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".tif", ".tiff"
        }
    ]

    if unsupported:
        print("\nUnsupported files:")

        for file in unsupported:
            print(file.name)


def main():

    inspect_folder(AU_DIR, "AUTHENTIC IMAGES")
    inspect_folder(TP_DIR, "TAMPERED IMAGES")
    inspect_folder(GT_DIR, "GROUND TRUTH MASKS")


if __name__ == "__main__":
    main()