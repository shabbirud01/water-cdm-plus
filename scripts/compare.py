"""Generate side-by-side comparison images: Input | GMSRBC | Fused | Ground Truth."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_comparison(input_path, gmsrbc_path, fused_path, target_path, save_path, name):
    inp = Image.open(input_path).convert("RGB")
    gmsrbc = Image.open(gmsrbc_path).convert("RGB")
    fused = Image.open(fused_path).convert("RGB")
    tgt = Image.open(target_path).convert("RGB")

    size = (256, 256)
    inp = inp.resize(size, Image.LANCZOS)
    gmsrbc = gmsrbc.resize(size, Image.LANCZOS)
    fused = fused.resize(size, Image.LANCZOS)
    tgt = tgt.resize(size, Image.LANCZOS)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, img, title in zip(
        axes,
        [inp, gmsrbc, fused, tgt],
        ["Input", "GMSRBC", "Fused (Ours)", "Ground Truth"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(name, fontsize=11, color="gray", y=0.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Dir with restored_*, gmsrbc_*, fused_* files")
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/comparisons")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    input_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))

    print(f"Generating comparisons for {len(input_files)} images...")

    for fname in input_files:
        stem = os.path.splitext(fname)[0]

        input_path = os.path.join(args.input_dir, fname)
        target_path = os.path.join(args.target_dir, fname)

        gmsrbc_path = os.path.join(args.results_dir, f"gmsrbc_{stem}.png")
        fused_path = os.path.join(args.results_dir, f"fused_{stem}.png")

        if not os.path.exists(target_path):
            print(f"  SKIP {fname} — no target found")
            continue
        if not os.path.exists(fused_path):
            print(f"  SKIP {fname} — no fused output found")
            continue

        save_path = os.path.join(args.output_dir, f"comparison_{stem}.png")
        make_comparison(input_path, gmsrbc_path, fused_path, target_path, save_path, stem)

    print(f"\nDone. Comparisons saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
