import csv
import json
from pathlib import Path

import pytest

from archive_real_scene_results import METHODS, REQUIRED_METRICS, publish


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_run(root: Path, tag: str = "tag", dataset: str = "Data") -> None:
    for method, layout in METHODS.items():
        infer = root / "infer" / layout.infer_name
        infer.mkdir(parents=True)
        prediction = infer / f"{method}_pred.png"
        prediction.write_bytes(b"png")
        rows = [{"sample_id": method, "prediction": str(prediction), "success": "True"}]
        write_csv(infer / "inference_manifest.csv", rows)
        (infer / "inference_manifest.json").write_text(json.dumps(rows))
        (infer / "errors.jsonl").write_text("")
        (infer / "run_config.json").write_text("{}")
        eval_root = root / layout.eval_root.format(tag=tag)
        write_csv(eval_root / "manifests" / f"{dataset}_{method}.csv", [{"sample_id": method}])
        result = eval_root / "results" / dataset
        for name in REQUIRED_METRICS:
            if name.endswith(".csv"):
                write_csv(result / name, [{"sample_id": method}])
            else:
                (result / name).parent.mkdir(parents=True, exist_ok=True)
                (result / name).write_text("{}")


def test_publish_replaces_only_selected_methods(tmp_path):
    output, archive = tmp_path / "output", tmp_path / "archive"
    fake_run(output)
    untouched = archive / "FULX2.0_ORIGIN" / "predictions"
    untouched.mkdir(parents=True)
    (untouched / "old.png").write_bytes(b"old")
    stale = archive / "DSIFT" / "predictions"
    stale.mkdir(parents=True)
    (stale / "stale.png").write_bytes(b"old")
    counts = publish(output, archive, "tag", "Data")
    assert counts == {method: 1 for method in METHODS}
    assert not (stale / "stale.png").exists()
    assert (archive / "DSIFT" / "predictions" / "DSIFT_pred.png").is_file()
    assert (untouched / "old.png").read_bytes() == b"old"


@pytest.mark.parametrize("failure", ["prediction", "failed_manifest", "manifest", "metrics"])
def test_failure_does_not_replace_existing_archive(tmp_path, failure):
    output, archive = tmp_path / "output", tmp_path / "archive"
    fake_run(output)
    existing = archive / "DSIFT" / "predictions"
    existing.mkdir(parents=True)
    marker = existing / "keep.png"
    marker.write_bytes(b"keep")
    if failure == "prediction":
        (output / "infer" / "DSIFT" / "DSIFT_pred.png").unlink()
    elif failure == "failed_manifest":
        write_csv(output / "infer" / "DSIFT" / "inference_manifest.csv", [
            {"sample_id": "DSIFT", "prediction": "missing.png", "success": "False"}
        ])
    elif failure == "manifest":
        (output / "eval" / "DSIFT" / "manifests" / "Data_DSIFT.csv").unlink()
    else:
        (output / "eval" / "DSIFT" / "results" / "Data" / "summary.csv").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        publish(output, archive, "tag", "Data")
    assert marker.read_bytes() == b"keep"
