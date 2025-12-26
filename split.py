import os
import shutil
import random
from pathlib import Path
from floder import create_folders  # Import function from folder.py

# ------------------- Configuration -------------------
DATA_DIR = Path("cattle_data/data")   # Original data
OUTPUT_DIR = Path("cattle_data")      # Output folder
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def split_dataset():
    if not DATA_DIR.exists():
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        return

    class_names = create_folders()
    summary = {}

    for class_name in class_names:
        class_path = DATA_DIR / class_name
        images = [f for f in class_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not images:
            print(f"⚠️ No images found for class '{class_name}'")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = int(TRAIN_RATIO * n_total)
        n_val = int(VAL_RATIO * n_total)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            for img_path in split_images:
                dest = OUTPUT_DIR / split / class_name / img_path.name
                shutil.copy(img_path, dest)

        summary[class_name] = {k: len(v) for k, v in splits.items()}

    # ------------------- Print summary -------------------
    print("\n🎯 Dataset Split Summary:")
    total_train = total_val = total_test = 0
    for cls, counts in summary.items():
        print(f"{cls}: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
        total_train += counts['train']
        total_val += counts['val']
        total_test += counts['test']
    print(f"\nTotal: Train={total_train}, Val={total_val}, Test={total_test}")
    print("\n✅ Dataset successfully split into 'train', 'val', and 'test' folders!")

if __name__ == "__main__":
    split_dataset()
