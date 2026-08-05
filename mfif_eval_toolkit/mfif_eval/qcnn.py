from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class QCNNMetric:
    """Adapter for the official TPAMI 2024 QCNN model and checkpoint.

    This imports model.py and resnet34.pth from the official MFIF-Metrics
    repository. The aggregation is a vectorized rewrite of the official
    test_color.py/test_grayscale.py evaluation logic.
    """

    def __init__(self, repo_root: Path, device: str = "auto") -> None:
        qroot = repo_root / "QCNN-metric"
        model_file = qroot / "model.py"
        checkpoint = qroot / "resnet34.pth"
        if not model_file.exists() or not checkpoint.exists():
            raise FileNotFoundError(
                "QCNN files not found. Run prepare_backends.sh or pass --tpami-root. "
                f"Expected {model_file} and {checkpoint}."
            )

        spec = importlib.util.spec_from_file_location("mfif_qcnn_official_model", model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = module.resnet34(num_classes=1000).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _patches(feature: torch.Tensor, kernel: int = 13) -> torch.Tensor:
        h, w = feature.shape[-2:]
        padding = 0 if h >= kernel and w >= kernel else (kernel - min(h, w)) // 2 + 1
        patches = F.unfold(feature, kernel_size=kernel, padding=padding)
        return patches.squeeze(0).transpose(0, 1).contiguous()

    @staticmethod
    def _global_standardize(x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x)
        return (x - torch.mean(x)) / torch.clamp(std, min=1e-12)

    @staticmethod
    def _level_score(a: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        a = QCNNMetric._global_standardize(QCNNMetric._patches(a))
        b = QCNNMetric._global_standardize(QCNNMetric._patches(b))
        f = QCNNMetric._global_standardize(QCNNMetric._patches(f))

        var_a = torch.var(a, dim=1, unbiased=True)
        var_b = torch.var(b, dim=1, unbiased=True)
        lam = var_a / torch.clamp(var_a + var_b, min=1e-12)
        weight = torch.maximum(var_a, var_b)

        sim_af = F.cosine_similarity(a, f, dim=1, eps=1e-12)
        sim_bf = F.cosine_similarity(b, f, dim=1, eps=1e-12)
        local = lam * sim_af + (1.0 - lam) * sim_bf
        weight = weight / torch.clamp(weight.sum(), min=1e-12)
        return torch.sum(weight * local)

    def __call__(self, source_a: Path, source_b: Path, fused: Path) -> float:
        a = self._tensor(source_a).to(self.device)
        b = self._tensor(source_b).to(self.device)
        f = self._tensor(fused).to(self.device)
        if a.shape != b.shape or a.shape != f.shape:
            raise ValueError(f"QCNN requires equal shapes: {a.shape}, {b.shape}, {f.shape}")
        with torch.inference_mode():
            out_a = self.model(a)
            out_b = self.model(b)
            out_f = self.model(f)
            features_a: List[torch.Tensor] = list(out_a[1:5])
            features_b: List[torch.Tensor] = list(out_b[1:5])
            features_f: List[torch.Tensor] = list(out_f[1:5])
            scores = [
                self._level_score(x, y, z)
                for x, y, z in zip(features_a, features_b, features_f)
            ]
            return float(torch.stack(scores).mean().item())
