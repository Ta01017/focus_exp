from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@lru_cache(maxsize=2)
def _lpips_model(net: str = "alex"):
    import torch
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requested. Install it with: pip install lpips") from exc
    model = lpips.LPIPS(net=net)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _to_tensor01(image: np.ndarray) -> torch.Tensor:
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def compute_gt_metrics(
    fused: np.ndarray,
    gt: np.ndarray,
    metrics: Iterable[str],
    lpips_net: str = "alex",
) -> Dict[str, float]:
    requested = set(metrics)
    fused_f = fused.astype(np.float64) / 255.0
    gt_f = gt.astype(np.float64) / 255.0
    values: Dict[str, float] = {}

    if "mse" in requested or "psnr" in requested:
        mse = float(np.mean((fused_f - gt_f) ** 2))
        if "mse" in requested:
            values["mse"] = mse
        if "psnr" in requested:
            values["psnr"] = float(peak_signal_noise_ratio(gt_f, fused_f, data_range=1.0))

    if "mae" in requested:
        values["mae"] = float(np.mean(np.abs(fused_f - gt_f)))

    if "ssim" in requested:
        win = min(7, fused_f.shape[0], fused_f.shape[1])
        if win % 2 == 0:
            win -= 1
        if win < 3:
            raise ValueError(f"SSIM requires at least 3x3 images; got {fused_f.shape[:2]}")
        values["ssim"] = float(
            structural_similarity(gt_f, fused_f, channel_axis=2, data_range=1.0, win_size=win)
        )

    if "lpips" in requested:
        model = _lpips_model(lpips_net)
        device = next(model.parameters()).device
        x = _to_tensor01(fused).to(device) * 2.0 - 1.0
        y = _to_tensor01(gt).to(device) * 2.0 - 1.0
        with torch.inference_mode():
            values["lpips"] = float(model(x, y).reshape(-1)[0].item())

    if "ms_ssim_gt" in requested:
        try:
            from pytorch_msssim import ms_ssim
        except ImportError as exc:
            raise RuntimeError(
                "MS-SSIM requested. Install it with: pip install pytorch-msssim"
            ) from exc
        x = _to_tensor01(fused)
        y = _to_tensor01(gt)
        if min(x.shape[-2:]) < 161:
            raise ValueError(
                "pytorch-msssim requires sufficiently large images for five scales; "
                f"got {tuple(x.shape[-2:])}. Resize only if your protocol explicitly allows it."
            )
        with torch.inference_mode():
            values["ms_ssim_gt"] = float(ms_ssim(x, y, data_range=1.0).item())

    return values
