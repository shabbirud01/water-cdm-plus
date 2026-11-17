import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sig_p_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_p_sq
    sig_t_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_t_sq
    sig_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / (
        (mu_p_sq + mu_t_sq + C1) * (sig_p_sq + sig_t_sq + C2)
    )
    return 1.0 - ssim_map.mean()


class LightweightPerceptualLoss(nn.Module):
    """Edge-based perceptual loss — no pretrained weights needed."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))
        self.register_buffer("laplacian", laplacian.repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        filters = [self.sobel_x, self.sobel_y, self.laplacian]
        edges_pred = [F.conv2d(pred, f, padding=1, groups=3) for f in filters]
        edges_target = [F.conv2d(target, f, padding=1, groups=3) for f in filters]
        return sum(F.l1_loss(ep, et) for ep, et in zip(edges_pred, edges_target)) / 3


class CombinedLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_perceptual=0.1, w_ssim=0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.mse = nn.MSELoss()
        self.perceptual = LightweightPerceptualLoss()

    def forward(self, noise_pred, noise_target, x0_pred, x0_target):
        mse = self.mse(noise_pred, noise_target)

        x0_p = (x0_pred.clamp(-1, 1) + 1) / 2
        x0_t = (x0_target.clamp(-1, 1) + 1) / 2
        perceptual = self.perceptual(x0_p, x0_t)
        ssim = ssim_loss(x0_p, x0_t)

        total = self.w_mse * mse + self.w_perceptual * perceptual + self.w_ssim * ssim
        return {
            "total": total,
            "mse": mse,
            "perceptual": perceptual,
            "ssim": ssim,
        }
