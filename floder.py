import os
from pathlib import Path

# ------------------- Configuration -------------------
DATA_DIR = Path("cattle_data/data")   # Original breed folders
OUTPUT_DIR = Path("cattle_data")      # Output folder

def create_folders():
    if not DATA_DIR.exists():
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        return

    class_names = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    print("✅ Folder structure created successfully!")
    print(f"Train directory: {OUTPUT_DIR / 'train'}")
    print(f"Validation directory: {OUTPUT_DIR / 'val'}")
    print(f"Test directory: {OUTPUT_DIR / 'test'}")
    return class_names

if __name__ == "__main__":
    create_folders()
