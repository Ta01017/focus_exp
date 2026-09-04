from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("skimage")

from mfif_eval_toolkit.mfif_eval.evaluator import evaluate


def test_missing_qcnn_backend_is_recorded_as_skip_under_fail_fast(tmp_path: Path) -> None:
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    fused = tmp_path / "f.png"
    for path in (source_a, source_b, fused):
        path.touch()
    frame = pd.DataFrame(
        [
            {
                "row_id": 0,
                "dataset": "smoke",
                "sample_id": "sample",
                "mode": "no_gt",
                "method": "method",
                "source_a": str(source_a),
                "source_b": str(source_b),
                "gt": "",
                "fused": str(fused),
            }
        ]
    )

    result = evaluate(
        frame,
        ["qcnn"],
        toolkit_root=tmp_path,
        tpami_root=tmp_path / "missing_backend",
        objective_root=tmp_path,
        continue_on_error=False,
    )

    assert result.at[0, "error"] == ""
    assert "qcnn:backend_unavailable:FileNotFoundError" in result.at[0, "skipped_metrics"]
    runtime_skips = result.attrs["runtime_skip_report"]
    assert runtime_skips.at[0, "metric"] == "qcnn"
    assert runtime_skips.at[0, "reason"].startswith("backend_unavailable:FileNotFoundError")
