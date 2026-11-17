"""
DDIM Sampler — UPGRADE #2.

Replaces the original DDPM reverse process with DDIM (Song et al., ICLR 2021)
for 20-40x faster inference.

Reference: "Denoising Diffusion Implicit Models" (Song et al., 2021)
"""

import torch
import numpy as np


class DDIMSampler:
    """
    DDIM reverse sampling process.

    Key equation (Eq. 12 from DDIM paper):
    x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
              + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_pred
              + sigma_t * noise

    Setting eta=0 gives deterministic sampling (no added noise).
    """

    def __init__(self, diffusion, ddim_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.timesteps = self._make_timesteps(diffusion.T, ddim_steps)

    def _make_timesteps(self, T, steps):
        """Create evenly spaced DDIM timestep subsequence spanning [0, T-1].

        Uses linspace so the first sampling step is t=T-1, matching
        the pure-noise starting point.
        """
        timesteps = np.linspace(0, T - 1, steps, dtype=np.int64)
        return timesteps[::-1].copy()

    @torch.no_grad()
    def sample(self, x_c, return_intermediates=False):
        device = x_c.device
        shape = x_c.shape

        model = self.diffusion.model
        alpha_bar = self.diffusion.alphas_cumprod

        x_t = torch.randn(shape, device=device)
        intermediates = [x_t] if return_intermediates else None

        for i in range(len(self.timesteps)):
            t_cur = int(self.timesteps[i])
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            noise_pred = model(x_t, x_c, t_batch)

            ab_t = alpha_bar[t_cur]

            x0_pred = (x_t - torch.sqrt(1 - ab_t) * noise_pred) / (torch.sqrt(ab_t) + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(self.timesteps) - 1:
                t_next = int(self.timesteps[i + 1])
                ab_t_next = alpha_bar[t_next]

                sigma = self.eta * torch.sqrt(
                    (1 - ab_t_next) / (1 - ab_t + 1e-8) *
                    (1 - ab_t / (ab_t_next + 1e-8))
                )

                dir_xt = torch.sqrt(
                    torch.clamp(1 - ab_t_next - sigma ** 2, min=0)
                ) * noise_pred

                x_t = torch.sqrt(ab_t_next) * x0_pred + dir_xt

                if self.eta > 0:
                    x_t = x_t + sigma * torch.randn_like(x_t)
            else:
                x_t = x0_pred

            if return_intermediates:
                intermediates.append(x_t)

        x_t = x_t.clamp(-1, 1)

        if return_intermediates:
            return x_t, intermediates
        return x_t
