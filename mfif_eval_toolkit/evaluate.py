#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mfif_eval.evaluator import (
    build_skip_report,
    evaluate,
    filter_manifest,
    load_manifest,
    summarize,
    write_metadata,
)
from mfif_eval.registry import METRIC_SETS, METRICS, expand_metrics


def csv_list(value: str):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Unified GT/no-GT evaluator for multi-focus image fusion."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--metrics", default="all",
        help="Comma-separated metric IDs or sets. Sets: " + ", ".join(sorted(METRIC_SETS)),
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or all")
    parser.add_argument("--methods", default="all", help="Comma-separated method names or all")
    parser.add_argument("--mode", choices=["all", "gt", "no_gt"], default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tpami-root", type=Path, default=root / "third_party" / "MFIF-Metrics")
    parser.add_argument(
        "--objective-root", type=Path,
        default=root / "third_party" / "Objective-evaluation-for-image-fusion",
    )
    parser.add_argument("--matlab-command", default="matlab")
    parser.add_argument("--qcnn-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument(
        "--source-metrics-on-gt", action="store_true",
        help="Also compute source-based fusion metrics on rows that have GT.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-metrics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_metrics:
        for name, spec in METRICS.items():
            print(f"{name:16s} {spec.mode:6s} {spec.backend:7s} {spec.display:20s} {spec.source}")
        print("\nMetric sets:")
        for name, members in METRIC_SETS.items():
            print(f"{name:16s} {','.join(members)}")
        return 0

    requested_metrics = expand_metrics(csv_list(args.metrics))
    frame = load_manifest(args.manifest)
    datasets = None if args.datasets == "all" else csv_list(args.datasets)
    methods = None if args.methods == "all" else csv_list(args.methods)
    frame = filter_manifest(frame, datasets=datasets, methods=methods, mode=args.mode)
    if frame.empty:
        raise RuntimeError("No manifest rows remain after filtering")

    # Silently skip metrics that cannot apply to any selected row; retain mixed mode behavior.
    selected_modes = set(frame["mode"])
    active = [
        m for m in requested_metrics
        if METRICS[m].mode in selected_modes
        or METRICS[m].mode == "both"
        or (args.source_metrics_on_gt and "gt" in selected_modes and METRICS[m].mode == "no_gt")
    ]
    skipped = [m for m in requested_metrics if m not in active]
    if skipped:
        print(f"[INFO] Skipping metrics not applicable to selected modes: {','.join(skipped)}")

    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        frame,
        active,
        toolkit_root=root,
        tpami_root=args.tpami_root.resolve(),
        objective_root=args.objective_root.resolve(),
        matlab_command=args.matlab_command,
        qcnn_device=args.qcnn_device,
        lpips_net=args.lpips_net,
        continue_on_error=not args.fail_fast,
        source_metrics_on_gt=args.source_metrics_on_gt,
    )
    summary = summarize(result, active)
    skip_report = build_skip_report(
        frame,
        requested_metrics,
        source_metrics_on_gt=args.source_metrics_on_gt,
    )
    if not skip_report.empty:
        grouped_skips = skip_report.groupby("row_id", sort=False).apply(
            lambda rows: "; ".join(f"{r.metric}:{r.reason}" for r in rows.itertuples())
        )
        for row_id, message in grouped_skips.items():
            result.loc[result["row_id"] == row_id, "skipped_metrics"] = message
    result.to_csv(args.output_dir / "per_image.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    skip_report.to_csv(args.output_dir / "skipped_metrics.csv", index=False)
    write_metadata(args.output_dir / "run_metadata.json", active, vars(args))

    print(f"[DONE] rows={len(result)}")
    print(f"[PER_IMAGE] {args.output_dir / 'per_image.csv'}")
    print(f"[SUMMARY]   {args.output_dir / 'summary.csv'}")
    print(f"[SKIPPED]   {args.output_dir / 'skipped_metrics.csv'}")
    print(summary.to_string(index=False))
    failures = int((result["error"].astype(str).str.len() > 0).sum())
    if failures:
        print(f"[WARNING] {failures} rows have errors; inspect per_image.csv")
        return 2 if args.fail_fast else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
