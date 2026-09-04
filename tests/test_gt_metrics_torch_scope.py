from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("skimage")

from mfif_eval_toolkit.mfif_eval import gt_metrics


class _FakeLPIPS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).abs().mean().reshape(1, 1, 1, 1)


def test_lpips_metric_has_module_level_torch_scope(monkeypatch) -> None:
    monkeypatch.setattr(gt_metrics, "_lpips_model", lambda _net: _FakeLPIPS())
    fused = np.zeros((8, 8, 3), dtype=np.uint8)
    gt = np.full((8, 8, 3), 255, dtype=np.uint8)

    values = gt_metrics.compute_gt_metrics(fused, gt, ["lpips"])

    assert values["lpips"] == 2.0


def test_tensor_conversion_copies_read_only_numpy_input() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.flags.writeable = False

    tensor = gt_metrics._to_tensor01(image)

    assert tensor.shape == (1, 3, 4, 4)
