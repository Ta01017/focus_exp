import csv
import json

from PIL import Image

from build_region_manifest import build
from check_route_v3_metadata import has_three_routes


def test_route_v3_schema_detection_rejects_two_focus_maps():
    assert has_three_routes({"m_a": "a", "m_b": "b", "m_g": "g"})
    assert has_three_routes({"edit_image": ["a", "b", "ma", "mb", "mg"]})
    assert not has_three_routes({"edit_image": ["a", "b", "focus_a", "focus_b"]})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_uses_three_route_maps_and_inference_index(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png", "gt.png", "pred.png"):
        Image.new("RGB", (8, 8), "white").save(images / name)
    Image.new("L", (8, 8), 255).save(images / "ma.png")
    Image.new("L", (8, 8), 0).save(images / "mb.png")
    Image.new("L", (8, 8), 0).save(images / "mg.png")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{
        "image": "images/gt.png",
        "edit_image": ["images/a.png", "images/b.png"],
        "m_a": "images/ma.png", "m_b": "images/mb.png", "m_g": "images/mg.png",
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
    assert row["m_a"] == str((images / "ma.png").resolve())
    assert row["m_b"] == str((images / "mb.png").resolve())
    assert row["m_g"] == str((images / "mg.png").resolve())
    assert row["sample_id"] == "one"
