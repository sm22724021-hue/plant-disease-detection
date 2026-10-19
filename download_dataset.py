"""
Dataset Download and Setup Helper
BCA Third-Year Summer Project: Plant Disease Detection Using AI/ML

This script assists in obtaining and configuring the PlantVillage dataset:
1. Option to download from Kaggle using kagglehub or the Kaggle CLI.
2. Option to download from a direct public archive mirror.
3. Guidance for manual download from Kaggle / Mendeley / GitHub.
4. Dataset integrity verification and folder placement checker.
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import argparse


KAGGLE_DATASET_SLUG = "emmarex/plantdisease"
PLANTVILLAGE_MIRROR_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"


def verify_current_dataset(dataset_dir: str = "dataset") -> bool:
    """Verifies whether dataset/ has valid class folders and images."""
    if not os.path.isdir(dataset_dir):
        print(f"[-] Directory '{dataset_dir}' does not exist.")
        return False

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"[*] Found subdirectories in '{dataset_dir}': {subdirs}")

    # Check if split folders exist
    split_folders = {"train", "validation", "test"}.intersection(set(subdirs))
    if split_folders:
        print(f"[+] Detected prepared split directories: {list(split_folders)}")
        for s in split_folders:
            s_path = os.path.join(dataset_dir, s)
            classes = [d for d in os.listdir(s_path) if os.path.isdir(os.path.join(s_path, d))]
            total_imgs = sum(
                len([f for f in os.listdir(os.path.join(s_path, c)) if not f.startswith(".")])
                for c in classes
            )
            print(f"    - {s}: {len(classes)} classes, {total_imgs} total images.")
        return True

    # Check if raw class folders exist directly inside dataset_dir
    img_count = 0
    for c in subdirs:
        c_path = os.path.join(dataset_dir, c)
        imgs = [f for f in os.listdir(c_path) if not f.startswith(".")]
        img_count += len(imgs)

    if img_count > 0:
        print(f"[+] Detected raw dataset with {len(subdirs)} classes and {img_count} images.")
        print("    Run 'python preprocess_data.py --source_dir dataset' to split into train/val/test.")
        return True

    print("[-] No images found in dataset directory.")
    return False


def download_with_kagglehub(target_dir: str = "raw_dataset"):
    """Attempts downloading PlantVillage dataset using kagglehub."""
    try:
        import kagglehub
        print(f"[*] Downloading '{KAGGLE_DATASET_SLUG}' using kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        print(f"[+] Download completed to cache: {path}")

        # Copy or move to project directory
        os.makedirs(target_dir, exist_ok=True)
        print(f"[*] Copying dataset to '{target_dir}'...")
        shutil.copytree(path, target_dir, dirs_exist_ok=True)
        print(f"[+] Dataset successfully copied to '{target_dir}'.")
        print(f"    Next step: python preprocess_data.py --source_dir {target_dir}")
    except ImportError:
        print("[!] 'kagglehub' is not installed. Install with: pip install kagglehub")
    except Exception as e:
        print(f"[!] Kagglehub download failed: {e}")


def print_manual_instructions():
    """Prints clear, student-friendly instructions for downloading the dataset."""
    print("""
======================================================================
MANUAL DATASET DOWNLOAD GUIDE (PlantVillage Dataset)
======================================================================

For your BCA University Project, you can download the PlantVillage
dataset using any of the following reliable sources:

Option 1: Kaggle (Recommended - Full 54,000+ or Color Subset)
-------------------------------------------------------------
1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
   OR search "PlantVillage" on https://www.kaggle.com
2. Click the 'Download' button (approx. 1.5 GB zip).
3. Unzip the downloaded file.
4. Place the class folders into a directory named 'raw_dataset/':
     plant-disease-detection/
     └── raw_dataset/
         ├── Tomato___Early_blight/
         ├── Tomato___healthy/
         ├── Potato___Late_blight/
         └── ...
5. Run the automated 70/15/15 preprocessor:
     python preprocess_data.py --source_dir raw_dataset

Option 2: Kaggle CLI
--------------------
Run in your terminal:
     kaggle datasets download -d emmarex/plantdisease --unzip -p raw_dataset/
     python preprocess_data.py --source_dir raw_dataset

Option 3: GitHub Official PlantVillage Repository
-------------------------------------------------
Visit: https://github.com/spMohanty/PlantVillage-Dataset
Under 'raw/color', download the color images of the crops you need.

======================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Plant disease dataset setup assistant.")
    parser.add_argument("--verify", action="store_true", help="Verify current dataset setup.")
    parser.add_argument("--kaggle", action="store_true", help="Attempt download via kagglehub.")
    parser.add_argument("--instructions", action="store_true", help="Display manual download steps.")

    args = parser.parse_args()

    print("==================================================")
    print("Plant Disease Detection - Dataset Setup Assistant")
    print("==================================================")

    if args.verify:
        verify_current_dataset()
    elif args.kaggle:
        download_with_kagglehub()
    elif args.instructions:
        print_manual_instructions()
    else:
        # Default workflow: verify first, then print instructions if needed
        is_ready = verify_current_dataset()
        if not is_ready:
            print_manual_instructions()


if __name__ == "__main__":
    main()
