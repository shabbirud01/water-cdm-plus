import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import RestormerBlock


def _norm(channels):
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            _norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            _norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_heads=8,
        num_restormer_blocks=2,
    ):
        super().__init__()
        self.base_channels = base_channels
        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder_blocks.append(
                nn.ModuleList([
                    ResBlock(ch, out_ch, time_emb_dim),
                    ResBlock(out_ch, out_ch, time_emb_dim),
                ])
            )
            ch = out_ch
            self.downsamples.append(Downsample(ch))

        # --- Bottleneck with Restormer blocks ---
        self.bottleneck_res1 = ResBlock(ch, ch, time_emb_dim)
        self.bottleneck_restormer = nn.Sequential(
            *[RestormerBlock(ch, num_heads) for _ in range(num_restormer_blocks)]
        )
        self.bottleneck_res2 = ResBlock(ch, ch, time_emb_dim)

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(Upsample(ch))
            self.decoder_blocks.append(
                nn.ModuleList([
                    ResBlock(ch + out_ch, out_ch, time_emb_dim),
                    ResBlock(out_ch, out_ch, time_emb_dim),
                ])
            )
            ch = out_ch

        self.output_conv = nn.Sequential(
            _norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x_t, x_cond, t):
        x = torch.cat([x_t, x_cond], dim=1)

        t_emb = get_timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        x = self.input_conv(x)

        skips = []
        for (res1, res2), down in zip(self.encoder_blocks, self.downsamples):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.bottleneck_res1(x, t_emb)
        x = self.bottleneck_restormer(x)
        x = self.bottleneck_res2(x, t_emb)

        for (res1, res2), up in zip(self.decoder_blocks, self.upsamples):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)

        return self.output_conv(x)
