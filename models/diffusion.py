import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float(),
        )

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def predict_x0_from_noise(self, x_t, t, noise):
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha

    def p_mean_variance(self, x_t, x_cond, t):
        noise_pred = self.model(x_t, x_cond, t)
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_0_pred
            + self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        var = self.posterior_variance[t][:, None, None, None]
        log_var = self.posterior_log_variance_clipped[t][:, None, None, None]
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x_t, x_cond, t):
        mean, _, log_var = self.p_mean_variance(x_t, x_cond, t)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float()[:, None, None, None]
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, x_cond):
        device = x_cond.device
        b = x_cond.shape[0]
        x = torch.randn_like(x_cond)
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, x_cond, t)
        return x

    def training_losses(self, x_0, x_cond, t=None):
        b = x_0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (b,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, x_cond, t)
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        return noise_pred, noise, x_0_pred, x_0
