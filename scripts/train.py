import argparse
import os
import sys
import time

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from data.dataset import get_train_loader
from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from utils.losses import CombinedLoss


def train(max_iters=None, quick_test=False):
    config = Config()
    config.ensure_dirs()
    device = torch.device("cpu")
    print(f"Device: {device}")

    if quick_test:
        total_iters = max_iters or 10
    else:
        total_iters = max_iters or config.NUM_ITERS

    unet = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=config.BASE_CHANNELS,
        channel_mults=tuple(config.CHANNEL_MULTS),
        num_heads=config.NUM_ATTENTION_HEADS,
        num_restormer_blocks=config.NUM_RESTORMER_BLOCKS,
    )

    diffusion = GaussianDiffusion(model=unet, T=config.T)
    criterion = CombinedLoss(
        w_mse=config.W_MSE,
        w_perceptual=config.W_PERCEPTUAL,
        w_ssim=config.W_SSIM,
    )
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.LR)

    train_loader = get_train_loader(config)
    writer = SummaryWriter(config.LOG_DIR) if SummaryWriter is not None else None

    param_count = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"Model: {param_count:.2f}M parameters")
    print(f"Training for {total_iters} iterations (quick_test={quick_test})")

    iteration = 0
    start = time.time()
    iter_start = time.time()

    while iteration < total_iters:
        for degraded, clean in train_loader:
            if iteration >= total_iters:
                break

            iter_start = time.time()

            noise_pred, noise, x0_pred, x0 = diffusion.training_losses(clean, degraded)
            losses = criterion(noise_pred, noise, x0_pred, x0)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.5)
            optimizer.step()

            iteration += 1
            iter_time = time.time() - iter_start

            if iteration % config.LOG_EVERY == 0 or iteration <= 10:
                elapsed = time.time() - start
                avg_it_s = iteration / elapsed if elapsed > 0 else 0
                eta = (total_iters - iteration) / avg_it_s if avg_it_s > 0 else 0
                print(
                    f"[{iteration}/{total_iters}] "
                    f"total={losses['total']:.4f} mse={losses['mse']:.4f} "
                    f"perc={losses['perceptual']:.4f} ssim={losses['ssim']:.4f} "
                    f"| {iter_time:.3f}s/it  avg {avg_it_s:.2f}it/s  ETA {eta/60:.1f}min"
                )
                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar(f"loss/{k}", v.item(), iteration)

            if not quick_test and iteration % config.SAVE_EVERY == 0:
                ckpt = {
                    "iteration": iteration,
                    "model": unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                path = os.path.join(config.CHECKPOINT_DIR, f"ckpt_{iteration}.pt")
                torch.save(ckpt, path)
                print(f"Saved {path}")

    elapsed = time.time() - start
    print(f"\nDone. {iteration} iterations in {elapsed:.1f}s ({iteration/elapsed:.2f} it/s)")
    if not quick_test:
        ckpt = {
            "iteration": iteration,
            "model": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        path = os.path.join(config.CHECKPOINT_DIR, "ckpt_final.pt")
        torch.save(ckpt, path)
        print(f"Saved final checkpoint: {path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--quick_test", action="store_true",
                        help="Run only 10 iterations for validation, skip checkpoints")
    args = parser.parse_args()
    train(max_iters=args.max_iters, quick_test=args.quick_test)
