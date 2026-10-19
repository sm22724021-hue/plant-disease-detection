"""
Dataset Preprocessing and Splitting Pipeline
BCA Third-Year Summer Project: Plant Disease Detection Using AI/ML

This script handles:
1. Scanning raw dataset directories containing class subfolders (e.g. PlantVillage).
2. Removing corrupted or zero-byte files.
3. Content-based deduplication (MD5 hash) to prevent data leakage between splits.
4. Stratified splitting into 70% Train, 15% Validation, and 15% Test sets.
5. Displaying class-wise distribution statistics.
6. Generating a visual distribution chart in `results/dataset_distribution.png`.
"""

import os
import shutil
import hashlib
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_hash(file_path: str) -> str:
    """Computes MD5 hash of an image file to detect duplicate images."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(file_path: str) -> bool:
    """Validates that a file is a non-corrupt image that PIL can load."""
    if os.path.getsize(file_path) == 0:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_clean_images(source_dir: str) -> Dict[str, List[str]]:
    """
    Scans the source directory for class subfolders, cleans corrupted images,
    and removes duplicates to prevent data leakage.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    class_images: Dict[str, List[str]] = defaultdict(list)
    seen_hashes = set()
    total_found = 0
    corrupted_count = 0
    duplicate_count = 0

    subdirs = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

    if not subdirs:
        raise ValueError(
            f"No class folders found in '{source_dir}'. "
            "Expected structure: source_dir/class_name/image.jpg"
        )

    print(f"[*] Scanning source directory: {source_dir}")
    print(f"[*] Found {len(subdirs)} class folders.\n")

    for class_name in subdirs:
        class_path = os.path.join(source_dir, class_name)
        files = os.listdir(class_path)

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            file_path = os.path.join(class_path, filename)
            total_found += 1

            # 1. Integrity check
            if not is_valid_image(file_path):
                corrupted_count += 1
                continue

            # 2. Duplicate check (data leakage prevention)
            f_hash = compute_file_hash(file_path)
            if f_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(f_hash)
            class_images[class_name].append(file_path)

    print(f"[+] Total files scanned: {total_found}")
    print(f"[+] Valid unique images retained: {len(seen_hashes)}")
    if corrupted_count > 0:
        print(f"[-] Corrupted files skipped: {corrupted_count}")
    if duplicate_count > 0:
        print(f"[-] Duplicate files removed (leakage prevention): {duplicate_count}")

    return class_images


def split_and_copy_dataset(
    class_images: Dict[str, List[str]],
    target_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, int]]:
    """
    Performs stratified 70/15/15 splitting and copies files into
    dataset/train, dataset/validation, and dataset/test directories.
    """
    import random
    random.seed(seed)

    splits = ["train", "validation", "test"]
    for s in splits:
        os.makedirs(os.path.join(target_dir, s), exist_ok=True)

    split_stats = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0, "total": 0})

    for class_name, file_list in class_images.items():
        # Shuffle deterministically
        shuffled = list(file_list)
        random.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Allocate remainder to test to prevent rounding drops
        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]

        split_mapping = {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }

        for split_name, files in split_mapping.items():
            dest_folder = os.path.join(target_dir, split_name, class_name)
            os.makedirs(dest_folder, exist_ok=True)

            for src_file in files:
                dest_file = os.path.join(dest_folder, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)

            split_stats[class_name][split_name] = len(files)

        split_stats[class_name]["total"] = n

    return split_stats


def generate_distribution_chart(split_stats: Dict[str, Dict[str, int]], output_file: str):
    """
    Generates a clean horizontal bar chart displaying class sample counts across splits.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        classes = list(split_stats.keys())
        train_counts = [split_stats[c]["train"] for c in classes]
        val_counts = [split_stats[c]["validation"] for c in classes]
        test_counts = [split_stats[c]["test"] for c in classes]

        y = np.arange(len(classes))
        height = 0.25

        plt.figure(figsize=(12, max(6, len(classes) * 0.45)))
        plt.barh(y - height, train_counts, height, label="Train (70%)", color="#2d6a4f")
        plt.barh(y, val_counts, height, label="Validation (15%)", color="#52b788")
        plt.barh(y + height, test_counts, height, label="Test (15%)", color="#95d5b2")

        # Clean display labels
        clean_labels = [c.replace("___", " - ").replace("_", " ") for c in classes]
        plt.yticks(y, clean_labels, fontsize=9)
        plt.xlabel("Number of Images", fontsize=11, fontweight="bold")
        plt.title("Plant Disease Dataset Distribution (Train / Val / Test)", fontsize=13, fontweight="bold", pad=15)
        plt.legend(loc="lower right", frameon=True)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(output_file, dpi=200)
        plt.close()
        print(f"[+] Dataset distribution chart saved: {output_file}")
    except ImportError:
        print("[!] Matplotlib not installed; skipping distribution chart generation.")


def print_summary_table(split_stats: Dict[str, Dict[str, int]]):
    """Prints an ASCII summary table of the dataset splits."""
    header = f"{'Class Name':<45} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    separator = "-" * len(header)

    print("\n" + separator)
    print("DATASET SPLIT SUMMARY (70% Train, 15% Validation, 15% Test)")
    print(separator)
    print(header)
    print(separator)

    total_train = 0
    total_val = 0
    total_test = 0
    grand_total = 0

    for class_name, counts in sorted(split_stats.items()):
        total_train += counts["train"]
        total_val += counts["val"] if "val" in counts else counts["validation"]
        total_test += counts["test"]
        grand_total += counts["total"]

        clean_name = (class_name[:42] + "...") if len(class_name) > 45 else class_name
        v_count = counts.get("validation", counts.get("val", 0))
        print(f"{clean_name:<45} | {counts['train']:<8} | {v_count:<8} | {counts['test']:<8} | {counts['total']:<8}")

    print(separator)
    print(f"{'TOTALS':<45} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {grand_total:<8}")
    print(separator + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split plant disease image datasets.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to raw dataset directory with class subfolders.")
    parser.add_argument("--target_dir", type=str, default="dataset", help="Target root directory (creates train, validation, test).")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--chart_path", type=str, default=os.path.join("results", "dataset_distribution.png"), help="Chart output path.")

    args = parser.parse_args()

    assert abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) < 1e-4, "Ratios must sum to 1.0"

    print("==================================================")
    print("Plant Disease Dataset Preprocessing & Splitter")
    print("==================================================")

    class_images = collect_clean_images(args.source_dir)
    stats = split_and_copy_dataset(
        class_images=class_images,
        target_dir=args.target_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print_summary_table(stats)
    generate_distribution_chart(stats, args.chart_path)
    print("[+] Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
