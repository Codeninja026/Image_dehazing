"""Full-reference image-quality metrics: PSNR, SSIM, MS-SSIM, and LPIPS.

PSNR/SSIM/MS-SSIM are computed in TensorFlow (no extra dependency). LPIPS is
optional and uses the ``lpips`` PyTorch package if it is installed; if not, the
LPIPS helpers return ``None`` so callers can skip it gracefully.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(tf.image.psnr(pred[None], gt[None], max_val=1.0).numpy()[0])


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(tf.image.ssim(pred[None], gt[None], max_val=1.0).numpy()[0])


def ms_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Multi-scale SSIM. Falls back to single-scale SSIM for images < 176 px."""
    h, w = pred.shape[:2]
    if min(h, w) < 176:  # MS-SSIM needs >= 2^4 * 11 px across 5 scales
        return ssim(pred, gt)
    return float(tf.image.ssim_multiscale(pred[None], gt[None], max_val=1.0).numpy()[0])


# ---- LPIPS (optional) ----
_LPIPS_MODEL = None


def _get_lpips(net: str = "alex"):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips  # type: ignore
            import torch  # noqa: F401
            _LPIPS_MODEL = lpips.LPIPS(net=net)
            _LPIPS_MODEL.eval()
        except Exception:
            _LPIPS_MODEL = False  # mark unavailable
    return _LPIPS_MODEL


def lpips_available() -> bool:
    return bool(_get_lpips())


def lpips_distance(pred: np.ndarray, gt: np.ndarray, net: str = "alex"):
    """LPIPS perceptual distance (lower is better), or None if lpips is absent."""
    model = _get_lpips(net)
    if not model:
        return None
    import torch

    def to_t(x):  # HWC [0,1] -> 1x3xHxW in [-1,1]
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        return t * 2.0 - 1.0

    with torch.no_grad():
        d = model(to_t(pred), to_t(gt))
    return float(d.reshape(-1)[0].item())
