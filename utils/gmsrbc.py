import cv2
import numpy as np


def _single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1.0) - np.log10(blur + 1.0)
    return retinex


def _multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigmas:
        retinex += _single_scale_retinex(img, sigma)
    return retinex / len(sigmas)


def _per_channel_normalize(msr):
    """Normalize each channel independently to [0, 1] to preserve color structure."""
    out = np.zeros_like(msr)
    for c in range(msr.shape[2]):
        ch = msr[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi - lo > 1e-8:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def _simplest_color_balance(img, low_clip=0.01, high_clip=0.01):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        flat = np.sort(channel.flatten())
        n = len(flat)
        low_val = flat[int(n * low_clip)]
        high_val = flat[int(n * (1.0 - high_clip))]
        channel = np.clip(channel, low_val, high_val)
        if high_val - low_val > 0:
            channel = (channel - low_val) / (high_val - low_val)
        else:
            channel = np.zeros_like(channel)
        out[:, :, c] = channel
    return out


def _clahe_enhance(img):
    """CLAHE + white balance enhancement for images where retinex fails."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Simple white balance
    result = enhanced.astype(np.float64)
    for c in range(3):
        ch = result[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 - p2 > 10:
            result[:, :, c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
    return result


def _quality_score(enhanced, original):
    """Score how reasonable the enhancement is. Lower = worse."""
    e = enhanced.astype(np.float64)
    o = original.astype(np.float64)

    # Check per-channel unique values (posterization = few unique values)
    unique_counts = [len(np.unique(enhanced[:, :, c] // 4)) for c in range(3)]
    min_unique = min(unique_counts)
    if min_unique < 20:
        return 0

    # Check if color distribution shifted too drastically
    o_means = o.mean(axis=(0, 1))
    e_means = e.mean(axis=(0, 1))
    mean_shift = np.abs(e_means - o_means).max()
    if mean_shift > 120:
        return 0

    # Check structural preservation via normalized cross-correlation
    o_gray = np.mean(o, axis=2)
    e_gray = np.mean(e, axis=2)
    o_norm = o_gray - o_gray.mean()
    e_norm = e_gray - e_gray.mean()
    denom = np.sqrt(np.sum(o_norm ** 2) * np.sum(e_norm ** 2))
    if denom > 0:
        ncc = np.sum(o_norm * e_norm) / denom
    else:
        ncc = 0
    if ncc < 0.3:
        return 0

    return 1


def apply_gmsrbc(img, sigmas=None):
    """Gray-world Multi-Scale Retinex with Color Balance.

    Uses per-channel normalization and falls back to CLAHE for images
    where retinex produces poor results (dark, blue-dominant, etc).

    Parameters
    ----------
    img : np.ndarray
        RGB uint8 image of shape (H, W, 3).
    sigmas : list[float] | None
        Gaussian blur scales for MSR.  Defaults to [15, 80, 250].

    Returns
    -------
    np.ndarray
        Enhanced float64 image in [0, 255].
    """
    if sigmas is None:
        sigmas = [15, 80, 250]

    img_f = img.astype(np.float64)

    msr = _multi_scale_retinex(img_f, sigmas)

    # Per-channel normalization (more robust than global min-max)
    normalized = _per_channel_normalize(msr)

    balanced = _simplest_color_balance(normalized, low_clip=0.01, high_clip=0.01)

    # Gray-world adjustment
    mean_per_ch = balanced.mean(axis=(0, 1), keepdims=True)
    global_mean = mean_per_ch.mean()
    if global_mean > 0:
        balanced = balanced * (global_mean / (mean_per_ch + 1e-10))

    result = np.clip(balanced * 255.0, 0, 255)

    if _quality_score(result, img) == 0:
        result = _clahe_enhance(img)

    return result
