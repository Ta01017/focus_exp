import csv
import json

from PIL import Image

from build_region_manifest import build


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_uses_edit_image_focus_maps_and_inference_index(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png", "gt.png", "fa.png", "fb.png", "pred.png"):
        Image.new("RGB", (8, 8), "white").save(images / name)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{
        "image": "images/gt.png",
        "edit_image": ["images/a.png", "images/b.png", "images/fa.png", "images/fb.png"],
    }]))
    inference = tmp_path / "inference.csv"
    write_csv(inference, [{
        "index": 0, "sample_id": "one", "source_a": str(images / "a.png"),
        "source_b": str(images / "b.png"), "gt": str(images / "gt.png"),
        "prediction": str(images / "pred.png"), "success": "True",
    }])
    output = tmp_path / "region.csv"
    assert build(metadata, inference, output, "RealSceneVal68", "DSIFT") == 1
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["m_a"] == str((images / "fa.png").resolve())
    assert row["m_b"] == str((images / "fb.png").resolve())
    assert row["sample_id"] == "one"
