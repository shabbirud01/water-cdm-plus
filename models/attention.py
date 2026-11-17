import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention from Restormer."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=True
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network from Restormer."""

    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=True
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class RestormerBlock(nn.Module):
    """Single Restormer transformer block: LayerNorm -> MDTA -> LayerNorm -> GDFN."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        b, c, h, w = x.shape

        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.attn(x) + residual

        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.ffn(x) + residual

        return x
