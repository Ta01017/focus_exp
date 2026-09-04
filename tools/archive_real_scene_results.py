#!/usr/bin/env python3
"""Validate and atomically publish RealSceneVal68 comparison results."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodLayout:
    archive_name: str
    infer_name: str
    eval_root: str


METHODS = {
    "DSIFT": MethodLayout("DSIFT", "DSIFT", "eval/DSIFT"),
    "IFCNN": MethodLayout("IFCNN", "IFCNN", "eval/IFCNN"),
    "SwinFusion": MethodLayout("SwinFusion", "SwinFusion-metadata-y", "eval/SwinFusion/{tag}"),
    "ZMFF": MethodLayout("ZMFF", "ZMFF", "eval/ZMFF"),
    "ReDiffuse": MethodLayout("ReDiffuse_ORIGIN", "ReDiffuse-official-y", "eval/ReDiffuse/{tag}"),
    "Flux2": MethodLayout("Flux2", "Flux2", "eval/Flux2/{tag}"),
}
DEFAULT_METHODS = ("DSIFT", "IFCNN", "SwinFusion", "ZMFF")
REQUIRED_METRICS = ("per_image.csv", "summary.csv", "skipped_metrics.csv", "run_metadata.json")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required file missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required file missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_method(output_root: Path, stage_root: Path, method: str, tag: str,
                 dataset: str, require_region: bool = False,
                 region_dataset: str = "RealSceneVal68") -> tuple[int, Path]:
    layout = METHODS[method]
    infer_dir = output_root / "infer" / layout.infer_name
    eval_root = output_root / layout.eval_root.format(tag=tag)
    result_dir = eval_root / "results" / dataset
    infer_manifest = infer_dir / "inference_manifest.csv"
    rows = read_csv(infer_manifest)
    successful = [row for row in rows if truthy(row.get("success", ""))]
    if not successful:
        raise ValueError(f"{method}: inference manifest has no successful rows")

    method_stage = stage_root / layout.archive_name
    predictions = method_stage / "predictions"
    manifests = method_stage / "manifest"
    metrics = method_stage / "metrics"
    predictions.mkdir(parents=True)
    manifests.mkdir()
    metrics.mkdir()

    seen_names: set[str] = set()
    for row in successful:
        prediction = Path(row.get("prediction", ""))
        if not prediction.is_absolute():
            prediction = (infer_dir / prediction).resolve()
        if not prediction.is_file():
            raise FileNotFoundError(f"{method}: successful prediction missing: {prediction}")
        if prediction.name in seen_names:
            raise ValueError(f"{method}: duplicate prediction filename: {prediction.name}")
        seen_names.add(prediction.name)
        copy_file(prediction, predictions / prediction.name)

    for name in ("inference_manifest.csv", "inference_manifest.json", "errors.jsonl", "run_config.json"):
        copy_file(infer_dir / name, manifests / name)
    evaluation_manifest = eval_root / "manifests" / f"{dataset}_{method}.csv"
    copy_file(evaluation_manifest, manifests / "evaluation_manifest.csv")
    eval_rows = read_csv(evaluation_manifest)
    if len(eval_rows) != len(successful):
        raise ValueError(
            f"{method}: evaluation rows={len(eval_rows)} do not match successful inference rows={len(successful)}"
        )

    for name in REQUIRED_METRICS:
        copy_file(result_dir / name, metrics / name)
    per_image = read_csv(result_dir / "per_image.csv")
    if len(per_image) != len(successful):
        raise ValueError(
            f"{method}: metric rows={len(per_image)} do not match successful inference rows={len(successful)}"
        )
    for source in result_dir.iterdir():
        if source.is_file() and not (metrics / source.name).exists():
            copy_file(source, metrics / source.name)
    if require_region:
        region_root = output_root / "region_eval"
        region_manifest = (
            region_root / "manifests" / region_dataset / layout.archive_name / "region_manifest_route_v3.csv"
        )
        region_metrics = region_root / "metrics" / region_dataset / layout.archive_name
        copy_file(region_manifest, manifests / "region_manifest_route_v3.csv")
        region_rows = read_csv(region_manifest)
        if len(region_rows) != len(successful):
            raise ValueError(
                f"{method}: region rows={len(region_rows)} do not match successful inference rows={len(successful)}"
            )
        for name in ("route_metrics_per_image.csv", "route_metrics_summary.csv", "eval.log"):
            target_name = "route_v3_eval.log" if name == "eval.log" else name
            copy_file(region_metrics / name, metrics / target_name)
        if len(read_csv(region_metrics / "route_metrics_per_image.csv")) != len(successful):
            raise ValueError(f"{method}: region metric row count does not match successful inference rows")
    else:
        region_skip = output_root / "region_eval" / "SKIPPED.txt"
        if region_skip.is_file():
            copy_file(region_skip, metrics / "region_v3_SKIPPED.txt")
    if len(list(predictions.glob("*_pred.png"))) != len(successful):
        raise ValueError(f"{method}: staged prediction count validation failed")
    return len(successful), method_stage


def publish(output_root: Path, archive_root: Path, tag: str, dataset: str,
            require_region: bool = False,
            region_dataset: str = "RealSceneVal68",
            methods: tuple[str, ...] = DEFAULT_METHODS) -> dict[str, int]:
    output_root = output_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    stage_root = archive_root / f".archive-stage-{uuid.uuid4().hex}"
    backup_root = stage_root / "_previous"
    counts: dict[str, int] = {}
    replaced: list[tuple[Path, Path | None]] = []
    try:
        stage_root.mkdir()
        for method in methods:
            counts[method], _ = stage_method(
                output_root, stage_root, method, tag, dataset,
                require_region=require_region, region_dataset=region_dataset,
            )
        for method in methods:
            layout = METHODS[method]
            target_method = archive_root / layout.archive_name
            target_method.mkdir(parents=True, exist_ok=True)
            for kind in ("manifest", "metrics", "predictions"):
                source = stage_root / layout.archive_name / kind
                target = target_method / kind
                backup = None
                if target.exists():
                    backup = backup_root / layout.archive_name / kind
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                try:
                    os.replace(source, target)
                except Exception:
                    if backup is not None and backup.exists():
                        os.replace(backup, target)
                    raise
                replaced.append((target, backup))
        return counts
    except Exception:
        for target, backup in reversed(replaced):
            if target.exists():
                shutil.rmtree(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--tag", default="swinfusion_mix_v1_y")
    parser.add_argument("--dataset", default="RealMFIFZeddV4")
    parser.add_argument("--require-region", action="store_true")
    parser.add_argument("--region-dataset", default="RealSceneVal68")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(DEFAULT_METHODS))
    args = parser.parse_args()
    counts = publish(
        args.output_root, args.archive_root, args.tag, args.dataset,
        require_region=args.require_region, region_dataset=args.region_dataset,
        methods=tuple(args.methods),
    )
    for method, count in counts.items():
        print(f"[ARCHIVED] method={method} predictions={count} target={args.archive_root / method}")
    print(f"[DONE] archive_root={args.archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
