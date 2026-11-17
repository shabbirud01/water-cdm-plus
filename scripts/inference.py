"""
Inference script for Upgraded Water-CDM.

Saves three outputs per image:
  restored_<name>.png   — primary diffusion output (DDIM)
  gmsrbc_<name>.png     — GMSRBC traditional enhancement
  fused_<name>.png      — adaptive HSL fusion of primary + GMSRBC
"""

import argparse
import os
import sys
import time

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from models.ddim import DDIMSampler
from utils.gmsrbc import apply_gmsrbc


def rgb_to_hsl(img):
    """Convert RGB float [0,1] image to HSL float image."""
    img = np.clip(img, 0, 1).astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin

    l = (cmax + cmin) / 2.0

    s = np.zeros_like(l)
    mask = delta > 0
    low = mask & (l <= 0.5)
    high = mask & (l > 0.5)
    s[low] = delta[low] / (cmax[low] + cmin[low] + 1e-10)
    s[high] = delta[high] / (2.0 - cmax[high] - cmin[high] + 1e-10)

    h = np.zeros_like(l)
    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g) & ~mask_r
    mask_b = mask & ~mask_r & ~mask_g
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-10)) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-10)) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-10)) + 4)

    return np.stack([h, s, l], axis=2)


def hsl_to_rgb(hsl):
    """Convert HSL float image back to RGB float [0,1]."""
    h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = l - c / 2.0

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    for lo, hi, rv, gv, bv in [
        (0, 60, c, x, 0), (60, 120, x, c, 0), (120, 180, 0, c, x),
        (180, 240, 0, x, c), (240, 300, x, 0, c), (300, 360, c, 0, x),
    ]:
        mask = (h >= lo) & (h < hi)
        r[mask] = rv[mask] if not isinstance(rv, (int, float)) else rv
        g[mask] = gv[mask] if not isinstance(gv, (int, float)) else gv
        b[mask] = bv[mask] if not isinstance(bv, (int, float)) else bv

    rgb = np.stack([r + m, g + m, b + m], axis=2)
    return np.clip(rgb, 0, 1)


def adaptive_hsl_fusion(primary_rgb, gmsrbc_rgb):
    """Fuse primary (diffusion) and GMSRBC outputs in HSL space.

    GMSRBC dominates (80%) since primary is noisy at small model capacity.
    Primary contributes 20% luminance for subtle structural guidance.
    Bilateral filter applied at end to smooth remaining noise.
    """
    primary_f = primary_rgb.astype(np.float64) / 255.0
    gmsrbc_f = gmsrbc_rgb.astype(np.float64) / 255.0

    hsl_primary = rgb_to_hsl(primary_f)
    hsl_gmsrbc = rgb_to_hsl(gmsrbc_f)

    L_primary = hsl_primary[:, :, 2]
    L_gmsrbc = hsl_gmsrbc[:, :, 2]
    eps = 1e-8
    w = 0.2 * L_primary / (L_primary + L_gmsrbc + eps)

    fused_hsl = np.zeros_like(hsl_primary)
    fused_hsl[:, :, 0] = hsl_gmsrbc[:, :, 0]  # hue from GMSRBC
    fused_hsl[:, :, 1] = hsl_gmsrbc[:, :, 1]  # saturation from GMSRBC
    fused_hsl[:, :, 2] = w * L_primary + (1 - w) * L_gmsrbc

    fused_rgb = hsl_to_rgb(fused_hsl)
    fused_uint8 = (fused_rgb * 255).astype(np.uint8)

    fused_uint8 = cv2.bilateralFilter(fused_uint8, d=5, sigmaColor=30, sigmaSpace=30)
    return fused_uint8


def load_model(checkpoint_path, config):
    unet = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=config.BASE_CHANNELS,
        channel_mults=tuple(config.CHANNEL_MULTS),
        num_heads=config.NUM_ATTENTION_HEADS,
        num_restormer_blocks=config.NUM_RESTORMER_BLOCKS,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    if "iteration" in ckpt:
        print(f"  Trained for {ckpt['iteration']} iterations")
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")

    return unet


def restore_image(image_path, unet, config):
    """Run diffusion restoration + GMSRBC + fusion on one image."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    x_cond = transform(img).unsqueeze(0)

    diffusion = GaussianDiffusion(model=unet, T=config.T)
    sampler = DDIMSampler(diffusion, ddim_steps=config.DDIM_STEPS, eta=config.DDIM_ETA)

    t0 = time.time()
    with torch.no_grad():
        restored_tensor = sampler.sample(x_cond)
    elapsed = time.time() - t0

    # Primary output (diffusion)
    restored = (restored_tensor.squeeze(0).clamp(-1, 1) + 1) / 2
    restored = restored.permute(1, 2, 0).numpy()
    restored = (restored * 255).astype(np.uint8)
    primary = Image.fromarray(restored).resize(original_size, Image.LANCZOS)

    # GMSRBC enhancement (traditional, on original image resized to match)
    img_resized = img.resize(original_size, Image.LANCZOS)
    img_np = np.array(img_resized)
    gmsrbc_out = apply_gmsrbc(img_np, sigmas=[15, 80, 250])
    gmsrbc_uint8 = np.clip(gmsrbc_out, 0, 255).astype(np.uint8)
    gmsrbc_img = Image.fromarray(gmsrbc_uint8)

    # Adaptive HSL fusion
    primary_np = np.array(primary)
    fused_np = adaptive_hsl_fusion(primary_np, gmsrbc_uint8)
    fused_img = Image.fromarray(fused_np)

    return primary, gmsrbc_img, fused_img, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    unet = load_model(args.checkpoint, config)
    print(f"DDIM steps: {config.DDIM_STEPS}, image size: {config.IMAGE_SIZE}")

    input_path = args.input or args.input_dir
    if input_path is None:
        input_path = os.path.join(config.DATA_DIR, "test", "input")

    if os.path.isdir(input_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        images = sorted(f for f in os.listdir(input_path) if f.lower().endswith(exts))
        paths = [os.path.join(input_path, f) for f in images]
    else:
        paths = [input_path]

    print(f"Processing {len(paths)} image(s) with 3 outputs each...\n")

    for path in paths:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        ext = ".png"

        primary, gmsrbc_img, fused_img, elapsed = restore_image(path, unet, config)

        p_path = os.path.join(output_dir, f"restored_{stem}{ext}")
        g_path = os.path.join(output_dir, f"gmsrbc_{stem}{ext}")
        f_path = os.path.join(output_dir, f"fused_{stem}{ext}")

        primary.save(p_path)
        gmsrbc_img.save(g_path)
        fused_img.save(f_path)

        print(f"  {name} ({elapsed:.1f}s):")
        print(f"    -> {p_path}")
        print(f"    -> {g_path}")
        print(f"    -> {f_path}")

    print(f"\nDone. {len(paths)} image(s) x 3 outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
