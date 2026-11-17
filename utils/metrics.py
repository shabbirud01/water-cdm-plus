import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(img, ref):
    """PSNR between two uint8 images."""
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def compute_ssim(img, ref):
    """Mean SSIM between two uint8 images."""
    return float(structural_similarity(ref, img, channel_axis=2, data_range=255))


# ---------- UIQM ----------

def _uicm(img):
    """Underwater Image Colorfulness Measure."""
    r, g, b = img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)
    rg = r - g
    yb = (r + g) / 2.0 - b
    mu_rg, mu_yb = rg.mean(), yb.mean()
    sig_rg, sig_yb = rg.std(), yb.std()
    return -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(sig_rg ** 2 + sig_yb ** 2)


def _eme(channel, block_size=8):
    """Enhancement Measure Estimation."""
    h, w = channel.shape
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return 0.0
    total = 0.0
    for i in range(bh):
        for j in range(bw):
            block = channel[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmin = block.min()
            bmax = block.max()
            if bmin > 0 and bmax > 0:
                total += np.log(bmax / bmin + 1e-10)
    return 2.0 * total / (bh * bw + 1e-10)


def _uism(img):
    """Underwater Image Sharpness Measure via Sobel + EME."""
    total = 0.0
    for c in range(3):
        ch = img[:, :, c].astype(np.float64)
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sx ** 2 + sy ** 2)
        total += _eme(edge)
    return total


def _uiconm(img, block_size=8):
    """Underwater Image Contrast Measure (logAMEE)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    h, w = gray.shape
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return 0.0
    total = 0.0
    for i in range(bh):
        for j in range(bw):
            block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmin = block.min()
            bmax = block.max()
            ratio = (bmax - bmin) / (bmax + bmin + 1e-10)
            total += ratio * np.log(ratio + 1e-10)
    return total / (bh * bw + 1e-10)


def compute_uiqm(img, c1=0.0282, c2=0.2953, c3=3.5753):
    """Underwater Image Quality Measure."""
    return float(c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img))


# ---------- PIQE ----------

def compute_piqe(img):
    """Simplified Perception-based Image Quality Evaluator (lower is better, 0-100 scale)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = img.astype(np.float64)

    h, w = gray.shape
    block_size = 16
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return 50.0

    variances = []
    for i in range(bh):
        for j in range(bw):
            block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            variances.append(block.var())

    variances = np.array(variances)
    threshold = np.median(variances)

    active_mask = variances > threshold
    if active_mask.sum() == 0:
        return 50.0

    active_vars = variances[active_mask]
    cv = active_vars.std() / (active_vars.mean() + 1e-10)

    score = np.clip(cv * 50.0, 0, 100)
    return float(score)
