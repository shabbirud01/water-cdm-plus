import argparse
import os
import sys
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from models.ddim import DDIMSampler
from utils.metrics import compute_psnr, compute_ssim, compute_uiqm, compute_piqe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Test data directory (default: config DATA_DIR/test)")
    args = parser.parse_args()

    config = Config()
    data_dir = args.data_dir or os.path.join(config.DATA_DIR, "test")
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    if not os.path.isdir(input_dir):
        print(f"ERROR: Test input directory not found: {input_dir}")
        sys.exit(1)

    unet = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=config.BASE_CHANNELS,
        channel_mults=tuple(config.CHANNEL_MULTS),
        num_heads=config.NUM_ATTENTION_HEADS,
        num_restormer_blocks=config.NUM_RESTORMER_BLOCKS,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    diffusion = GaussianDiffusion(model=unet, T=config.T)
    sampler = DDIMSampler(diffusion, ddim_steps=config.DDIM_STEPS, eta=config.DDIM_ETA)

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    input_images = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(exts))
    has_targets = os.path.isdir(target_dir)

    results = {"psnr": [], "ssim": [], "uiqm": [], "piqe": []}
    total_time = 0

    print(f"Evaluating {len(input_images)} images (DDIM {config.DDIM_STEPS} steps, {config.IMAGE_SIZE}x{config.IMAGE_SIZE})")
    print("-" * 60)

    for name in input_images:
        img = Image.open(os.path.join(input_dir, name)).convert("RGB")
        x_cond = transform(img).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            restored = sampler.sample(x_cond)
        elapsed = time.time() - t0
        total_time += elapsed

        restored_np = (restored.squeeze(0).clamp(-1, 1) + 1) / 2
        restored_np = restored_np.permute(1, 2, 0).numpy()
        restored_np = (restored_np * 255).astype(np.uint8)

        uiqm = compute_uiqm(restored_np)
        piqe = compute_piqe(restored_np)
        results["uiqm"].append(uiqm)
        results["piqe"].append(piqe)

        line = f"  {name}: UIQM={uiqm:.4f} PIQE={piqe:.2f}"

        if has_targets:
            target_path = os.path.join(target_dir, name)
            if os.path.exists(target_path):
                target = Image.open(target_path).convert("RGB")
                target = target.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
                target_np = np.array(target)
                psnr = compute_psnr(restored_np, target_np)
                ssim = compute_ssim(restored_np, target_np)
                results["psnr"].append(psnr)
                results["ssim"].append(ssim)
                line += f" PSNR={psnr:.2f} SSIM={ssim:.4f}"

        print(f"{line} ({elapsed:.2f}s)")

    print("-" * 60)
    print("AVERAGES:")
    for k, v in results.items():
        if v:
            print(f"  {k.upper()}: {np.mean(v):.4f}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(input_images):.2f}s/image)")


if __name__ == "__main__":
    main()
