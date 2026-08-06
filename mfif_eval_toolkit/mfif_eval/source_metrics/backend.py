from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from ..io_utils import load_rgb
from .common import as_gray_double255, ensure_same_shape, nanmean_scalar
from .qabf import qabf, qabf_analysis
from .qcb import qcb
from .qg import qg
from .qmi import qmi
from .qncie import qncie
from .qp import qp
from .qs import qe, qs
from .qsf import qsf
from .qviff import qviff
from .ms_ssim_src import ms_ssim_src

PYTHON_SOURCE_METRICS_VERSION = "python_source_port_v1"

SOURCE_METRIC_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "qmi": qmi,
    "qsf": qsf,
    "qs": qs,
    "qcb": qcb,
    "qabf": qabf,
    "qabf_analysis": qabf_analysis,
    "qncie": qncie,
    "qg": qg,
    "qp": qp,
    "qe": qe,
    "qviff": qviff,
    "ms_ssim_src": ms_ssim_src,
}

MATLAB_MAPPING = {
    "qmi": "metricMI(A, B, F, 1)",
    "qsf": "metricZheng(A, B, F)",
    "qs": "metricPeilla(A, B, F, 1)",
    "qcb": "metricChenBlum(A, B, F)",
    "qabf": "Qabf(A, B, F)",
    "qabf_analysis": "analysis_Qabf(A, B, F)",
    "qncie": "metricWang(A, B, F)",
    "qg": "metricXydeas(A, B, F)",
    "qp": "metricZhao(A, B, F)",
    "qe": "metricPeilla(A, B, F, 3)",
    "qviff": "VIFF_Public(A, B, F)",
    "ms_ssim_src": "analysis_MSSSIM(A, B, F)",
}

PARITY_STATUS = {name: "pending" for name in SOURCE_METRIC_FUNCTIONS}


def compute_source_metric(
    name: str,
    source_a: np.ndarray,
    source_b: np.ndarray,
    fused: np.ndarray,
) -> float:
    if name not in SOURCE_METRIC_FUNCTIONS:
        raise KeyError(f"unknown source metric: {name}")
    ensure_same_shape(source_a, source_b, fused)
    a = as_gray_double255(source_a)
    b = as_gray_double255(source_b)
    f = as_gray_double255(fused)
    value = SOURCE_METRIC_FUNCTIONS[name](a, b, f)
    return nanmean_scalar(value)


def run_python_source_metrics(
    frame: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    metric_list = list(metrics)
    rows = []
    for _, row in frame.iterrows():
        values = {"row_id": int(row["row_id"])}
        errors = []
        try:
            a = load_rgb(Path(row["source_a"]))
            b = load_rgb(Path(row["source_b"]))
            f = load_rgb(Path(row["fused"]))
            ensure_same_shape(a, b, f)
        except Exception as exc:
            for metric in metric_list:
                values[metric] = np.nan
            values["source_error"] = f"source load: {type(exc).__name__}: {exc}"
            rows.append(values)
            continue
        for metric in metric_list:
            try:
                values[metric] = compute_source_metric(metric, a, b, f)
            except Exception as exc:
                values[metric] = np.nan
                errors.append(f"{metric}: {type(exc).__name__}: {exc}")
        values["source_error"] = " | ".join(errors)
        rows.append(values)
    return pd.DataFrame(rows).set_index("row_id")
