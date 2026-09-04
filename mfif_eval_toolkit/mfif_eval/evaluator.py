from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .gt_metrics import compute_gt_metrics
from .io_utils import ensure_same_shape, load_rgb, resolve_path
from .registry import METRICS
from .source_metrics import PYTHON_SOURCE_METRICS_VERSION
from .source_metrics.backend import MATLAB_MAPPING, PARITY_STATUS, run_python_source_metrics

REQUIRED_COLUMNS = {"dataset", "sample_id", "mode", "method", "source_a", "source_b", "fused"}


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "gt" not in frame.columns:
        frame["gt"] = ""
    frame = frame.copy()
    frame["mode"] = frame["mode"].astype(str).str.lower().str.strip()
    invalid = sorted(set(frame["mode"]) - {"gt", "no_gt"})
    if invalid:
        raise ValueError(f"Invalid mode values: {invalid}; expected gt or no_gt")
    base = path.parent.resolve()
    for column in ["source_a", "source_b", "gt", "fused"]:
        frame[column] = [
            "" if (p := resolve_path(v, base)) is None else str(p)
            for v in frame[column]
        ]
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    return frame


def filter_manifest(
    frame: pd.DataFrame,
    datasets: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    mode: str = "all",
) -> pd.DataFrame:
    out = frame
    if datasets:
        wanted = set(datasets)
        out = out[out["dataset"].isin(wanted)]
    if methods:
        wanted = set(methods)
        out = out[out["method"].isin(wanted)]
    if mode != "all":
        out = out[out["mode"] == mode]
    return out.reset_index(drop=True)


def validate_rows(frame: pd.DataFrame) -> None:
    problems: List[str] = []
    for _, row in frame.iterrows():
        required = ["source_a", "source_b", "fused"]
        if row["mode"] == "gt":
            required.append("gt")
        for col in required:
            value = str(row[col]).strip()
            if not value:
                problems.append(f"row {row['row_id']}: empty {col}")
            elif not Path(value).exists():
                problems.append(f"row {row['row_id']}: missing {col}={value}")
    if problems:
        preview = "\n".join(problems[:50])
        suffix = "" if len(problems) <= 50 else f"\n... and {len(problems)-50} more"
        raise FileNotFoundError(f"Manifest validation failed:\n{preview}{suffix}")


def metric_skip_reason(row: pd.Series, metric: str, source_metrics_on_gt: bool = False) -> str:
    spec = METRICS[metric]
    mode = str(row["mode"])
    has_gt = bool(str(row.get("gt", "")).strip())
    if spec.mode == "both":
        return ""
    if spec.mode == "gt" and (mode != "gt" or not has_gt):
        return "requires_gt"
    if spec.mode == "no_gt" and mode == "gt":
        return "" if source_metrics_on_gt else "source_metric_disabled_on_gt"
    if spec.mode not in {mode, "both"}:
        return f"requires_{spec.mode}"
    return ""


def build_skip_report(
    frame: pd.DataFrame,
    metrics: List[str],
    source_metrics_on_gt: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        for metric in metrics:
            reason = metric_skip_reason(row, metric, source_metrics_on_gt=source_metrics_on_gt)
            if reason:
                rows.append(
                    {
                        "row_id": row["row_id"],
                        "dataset": row["dataset"],
                        "sample_id": row["sample_id"],
                        "mode": row["mode"],
                        "method": row["method"],
                        "metric": metric,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["row_id", "dataset", "sample_id", "mode", "method", "metric", "reason"],
    )


def evaluate(
    frame: pd.DataFrame,
    metrics: List[str],
    toolkit_root: Path,
    tpami_root: Path,
    objective_root: Path,
    matlab_command: str = "matlab",
    qcnn_device: str = "auto",
    lpips_net: str = "alex",
    continue_on_error: bool = True,
    source_metrics_on_gt: bool = False,
) -> pd.DataFrame:
    validate_rows(frame)
    output = frame.copy()
    errors: Dict[int, List[str]] = {int(row_id): [] for row_id in output["row_id"]}

    gt_metrics = [m for m in metrics if METRICS[m].backend == "python_gt"]
    source_metrics = [m for m in metrics if METRICS[m].backend == "python_source"]
    need_qcnn = "qcnn" in metrics

    for metric in metrics:
        output[metric] = np.nan
    output["skipped_metrics"] = ""
    runtime_skip_rows: List[Dict[str, object]] = []

    skip_report = build_skip_report(
        output,
        metrics,
        source_metrics_on_gt=source_metrics_on_gt,
    )
    if not skip_report.empty:
        grouped_skips = skip_report.groupby("row_id", sort=False).apply(
            lambda rows: "; ".join(f"{r.metric}:{r.reason}" for r in rows.itertuples())
        )
        for row_id, message in grouped_skips.items():
            output.loc[output["row_id"] == row_id, "skipped_metrics"] = message

    if gt_metrics:
        for idx, row in tqdm(output.iterrows(), total=len(output), desc="GT/Python metrics"):
            applicable = [
                m for m in gt_metrics
                if not metric_skip_reason(row, m, source_metrics_on_gt=source_metrics_on_gt)
            ]
            if not applicable:
                continue
            try:
                if row["mode"] != "gt":
                    continue
                fused = load_rgb(Path(row["fused"]))
                gt = load_rgb(Path(row["gt"]))
                ensure_same_shape(fused, gt)
                for metric in applicable:
                    try:
                        values = compute_gt_metrics(fused, gt, [metric], lpips_net=lpips_net)
                        if metric in values:
                            output.at[idx, metric] = values[metric]
                    except Exception as exc:
                        errors[int(row["row_id"])].append(
                            f"{metric}: {type(exc).__name__}: {exc}"
                        )
                        if not continue_on_error:
                            raise
            except Exception as exc:
                errors[int(row["row_id"])].append(f"python: {type(exc).__name__}: {exc}")
                if not continue_on_error:
                    raise

    if source_metrics:
        source_mask = output["mode"] == "no_gt"
        if source_metrics_on_gt:
            source_mask = source_mask | (output["mode"] == "gt")
        subset = output[source_mask].copy()
        if not subset.empty:
            source_values = run_python_source_metrics(subset, source_metrics)
            index_by_row = {int(rid): idx for idx, rid in enumerate(output["row_id"].tolist())}
            for row_id, values in source_values.iterrows():
                idx = index_by_row[int(row_id)]
                for metric in source_metrics:
                    if metric in values:
                        output.at[idx, metric] = values[metric]
                message = str(values.get("source_error", "")).strip()
                if message and message.lower() != "nan":
                    errors[int(row_id)].append(message)
                    if not continue_on_error:
                        raise RuntimeError(message)

    if need_qcnn:
        qcnn = None
        try:
            from .qcnn import QCNNMetric

            qcnn = QCNNMetric(tpami_root, device=qcnn_device)
        except Exception as exc:
            # An optional metric backend not being installed is not an image
            # evaluation failure, even under --fail-fast.  Keep every other
            # metric usable and record an explicit skip for QCNN instead.
            qcnn_mask = output["mode"] == "no_gt"
            if source_metrics_on_gt:
                qcnn_mask = qcnn_mask | (output["mode"] == "gt")
            reason = f"backend_unavailable:{type(exc).__name__}: {exc}"
            for idx, row in output.loc[qcnn_mask].iterrows():
                previous = str(output.at[idx, "skipped_metrics"]).strip()
                message = f"qcnn:{reason}"
                output.at[idx, "skipped_metrics"] = f"{previous}; {message}" if previous else message
                runtime_skip_rows.append(
                    {
                        "row_id": row["row_id"],
                        "dataset": row["dataset"],
                        "sample_id": row["sample_id"],
                        "mode": row["mode"],
                        "method": row["method"],
                        "metric": "qcnn",
                        "reason": reason,
                    }
                )
        if qcnn is not None:
            for idx, row in tqdm(output.iterrows(), total=len(output), desc="QCNN"):
                if row["mode"] != "no_gt" and not (source_metrics_on_gt and row["mode"] == "gt"):
                    continue
                try:
                    output.at[idx, "qcnn"] = qcnn(
                        Path(row["source_a"]), Path(row["source_b"]), Path(row["fused"])
                    )
                except Exception as exc:
                    errors[int(row["row_id"])].append(f"qcnn: {type(exc).__name__}: {exc}")
                    if not continue_on_error:
                        raise

    output["error"] = [" | ".join(errors[int(row_id)]) for row_id in output["row_id"]]
    output.attrs["runtime_skip_report"] = pd.DataFrame(
        runtime_skip_rows,
        columns=["row_id", "dataset", "sample_id", "mode", "method", "metric", "reason"],
    )
    return output


def summarize(per_image: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    present = [metric for metric in metrics if metric in per_image.columns]
    grouped = per_image.groupby(["dataset", "mode", "method"], dropna=False)
    mean = grouped[present].mean(numeric_only=True).add_suffix("_mean")
    std = grouped[present].std(numeric_only=True, ddof=1).add_suffix("_std")
    count = grouped.size().rename("n")
    failures = grouped["error"].apply(lambda s: int((s.astype(str).str.len() > 0).sum())).rename("failures")
    return pd.concat([count, failures, mean, std], axis=1).reset_index()


def write_metadata(path: Path, metrics: List[str], args: Dict) -> None:
    payload = {
        "metrics": [
            {
                "id": metric,
                "display": METRICS[metric].display,
                "mode": METRICS[metric].mode,
                "backend": METRICS[metric].backend,
                "higher_is_better": METRICS[metric].higher_is_better,
                "source": METRICS[metric].source,
            }
            for metric in metrics
        ],
        "python_source_metrics": {
            "version": PYTHON_SOURCE_METRICS_VERSION,
            "matlab_mapping": {metric: MATLAB_MAPPING.get(metric) for metric in metrics if metric in MATLAB_MAPPING},
            "parity_status": {metric: PARITY_STATUS.get(metric) for metric in metrics if metric in PARITY_STATUS},
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in args.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
