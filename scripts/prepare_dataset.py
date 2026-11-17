"""Prepare a small CPU-friendly subset of the UIEB dataset.

Usage:
    python scripts/prepare_dataset.py --src /path/to/UIEB/raw --train 100 --test 20

The source directory should contain two folders:
    raw-890/          (890 degraded underwater images)
    reference-890/    (890 corresponding reference images)

This script randomly selects --train images for training and --test for
testing, then copies them into the structure expected by the dataloader:

    data/UIEB/
    ├── train/
    │   ├── input/    (degraded)
    │   └── target/   (reference)
    └── test/
        ├── input/
        └── target/
"""

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_image_pairs(src_dir):
    """Discover input/target pairs from the UIEB download layout."""
    input_dir = None
    target_dir = None

    for name in os.listdir(src_dir):
        full = os.path.join(src_dir, name)
        if not os.path.isdir(full):
            continue
        low = name.lower()
        if "raw" in low:
            input_dir = full
        elif "reference" in low or "ref" in low:
            target_dir = full

    if input_dir is None or target_dir is None:
        candidates = [
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
        ]
        print(f"ERROR: Could not find raw-890/ and reference-890/ folders in {src_dir}")
        print(f"  Found subdirectories: {candidates}")
        print("  Expected layout:")
        print("    <src>/raw-890/       (degraded images)")
        print("    <src>/reference-890/ (reference images)")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    inputs = {f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in exts}
    targets = {f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in exts}

    common = sorted(inputs & targets)
    if not common:
        input_stems = {os.path.splitext(f)[0]: f for f in inputs}
        target_stems = {os.path.splitext(f)[0]: f for f in targets}
        shared_stems = sorted(set(input_stems) & set(target_stems))
        pairs = [(input_stems[s], target_stems[s]) for s in shared_stems]
    else:
        pairs = [(f, f) for f in common]

    print(f"Found {len(pairs)} matched image pairs")
    print(f"  Input dir:  {input_dir}")
    print(f"  Target dir: {target_dir}")
    return input_dir, target_dir, pairs


def copy_subset(pairs, input_dir, target_dir, dst_split_dir):
    inp_dst = os.path.join(dst_split_dir, "input")
    tgt_dst = os.path.join(dst_split_dir, "target")
    os.makedirs(inp_dst, exist_ok=True)
    os.makedirs(tgt_dst, exist_ok=True)

    for inp_name, tgt_name in pairs:
        shutil.copy2(os.path.join(input_dir, inp_name), os.path.join(inp_dst, inp_name))
        shutil.copy2(os.path.join(target_dir, tgt_name), os.path.join(tgt_dst, tgt_name))


def main():
    parser = argparse.ArgumentParser(description="Prepare UIEB dataset subset for CPU training")
    parser.add_argument("--src", type=str, required=True,
                        help="Path to downloaded UIEB folder containing raw-890/ and reference-890/")
    parser.add_argument("--dst", type=str, default="data/UIEB",
                        help="Output directory (default: data/UIEB)")
    parser.add_argument("--train", type=int, default=100,
                        help="Number of training images (default: 100)")
    parser.add_argument("--test", type=int, default=20,
                        help="Number of test images (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    input_dir, target_dir, all_pairs = find_image_pairs(args.src)

    total_needed = args.train + args.test
    if total_needed > len(all_pairs):
        print(f"WARNING: Requested {total_needed} images but only {len(all_pairs)} pairs found.")
        print(f"  Adjusting: train={min(args.train, len(all_pairs) - args.test)}, test={args.test}")
        args.train = len(all_pairs) - args.test

    random.seed(args.seed)
    shuffled = random.sample(all_pairs, len(all_pairs))
    test_pairs = shuffled[:args.test]
    train_pairs = shuffled[args.test:args.test + args.train]

    print(f"\nSplitting: {len(train_pairs)} train, {len(test_pairs)} test")

    train_dir = os.path.join(args.dst, "train")
    test_dir = os.path.join(args.dst, "test")

    copy_subset(train_pairs, input_dir, target_dir, train_dir)
    copy_subset(test_pairs, input_dir, target_dir, test_dir)

    print(f"\nDataset prepared at {args.dst}/")
    print(f"  train/input/  — {len(os.listdir(os.path.join(train_dir, 'input')))} images")
    print(f"  train/target/ — {len(os.listdir(os.path.join(train_dir, 'target')))} images")
    print(f"  test/input/   — {len(os.listdir(os.path.join(test_dir, 'input')))} images")
    print(f"  test/target/  — {len(os.listdir(os.path.join(test_dir, 'target')))} images")
    print("\nDone!")


if __name__ == "__main__":
    main()
