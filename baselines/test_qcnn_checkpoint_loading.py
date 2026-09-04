from __future__ import annotations

from pathlib import Path

import torch

from mfif_eval_toolkit.mfif_eval import qcnn


def test_qcnn_legacy_checkpoint_is_loaded_on_cpu_before_model_move(
    tmp_path: Path, monkeypatch
) -> None:
    qroot = tmp_path / "QCNN-metric"
    qroot.mkdir()
    (qroot / "model.py").write_text(
        "import torch\n"
        "def resnet34(num_classes=1000):\n"
        "    return torch.nn.Linear(2, 2)\n",
        encoding="utf-8",
    )
    checkpoint = qroot / "resnet34.pth"
    checkpoint.touch()
    expected = torch.nn.Linear(2, 2).state_dict()
    load_args = {}

    def fake_load(path, *, map_location, weights_only):
        load_args.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {f"module.{key}": value for key, value in expected.items()}

    monkeypatch.setattr(qcnn.torch, "load", fake_load)

    metric = qcnn.QCNNMetric(tmp_path, device="cpu")

    assert load_args["path"] == checkpoint
    assert load_args["map_location"] == torch.device("cpu")
    assert load_args["weights_only"] is True
    assert next(metric.model.parameters()).device == torch.device("cpu")
