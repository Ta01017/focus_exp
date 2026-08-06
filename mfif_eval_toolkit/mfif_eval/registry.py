from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MetricSpec:
    name: str
    display: str
    mode: str  # gt, no_gt, both
    backend: str  # python_gt, python_source, qcnn
    higher_is_better: bool
    source: str


METRICS: Dict[str, MetricSpec] = {
    # Full-reference metrics.
    "psnr": MetricSpec("psnr", "PSNR", "gt", "python_gt", True, "scikit-image"),
    "ssim": MetricSpec("ssim", "SSIM", "gt", "python_gt", True, "scikit-image"),
    "lpips": MetricSpec("lpips", "LPIPS", "gt", "python_gt", False, "lpips"),
    "mae": MetricSpec("mae", "MAE", "gt", "python_gt", False, "native"),
    "mse": MetricSpec("mse", "MSE", "gt", "python_gt", False, "native"),
    "ms_ssim_gt": MetricSpec(
        "ms_ssim_gt", "MS-SSIM(GT)", "gt", "python_gt", True, "pytorch-msssim"
    ),
    # No-GT fusion metrics. Python ports keep the original MATLAB function mapping.
    "qmi": MetricSpec("qmi", "QMI", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: metricMI(...,1)"),
    "qsf": MetricSpec("qsf", "QSF", "no_gt", "python_source", True, "Objective-evaluation: metricZheng"),
    "qs": MetricSpec("qs", "QS", "no_gt", "python_source", True, "metricPeilla(...,1)"),
    "qcb": MetricSpec("qcb", "QCB", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: metricChenBlum"),
    "qabf": MetricSpec("qabf", "QAB/F", "no_gt", "python_source", True, "Objective-evaluation: Qabf"),
    "qabf_analysis": MetricSpec(
        "qabf_analysis", "Qabf-analysis", "no_gt", "python_source", True,
        "Objective-evaluation: analysis_Qabf (alternate Xydeas-Petrovic implementation)"
    ),
    "qncie": MetricSpec("qncie", "QNCIE", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: metricWang"),
    "qg": MetricSpec("qg", "QG", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: metricXydeas"),
    "qp": MetricSpec("qp", "QP", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: metricZhao"),
    "qe": MetricSpec("qe", "QE", "no_gt", "python_source", True, "metricPeilla(...,3)"),
    "qviff": MetricSpec("qviff", "QVIFF", "no_gt", "python_source", True, "TPAMI MFIF-Metrics: VIFF_Public"),
    "ms_ssim_src": MetricSpec(
        "ms_ssim_src", "MS-SSIM(source)", "no_gt", "python_source", True,
        "Objective-evaluation: analysis_MSSSIM / analysis_ms_ssim"
    ),
    "qcnn": MetricSpec("qcnn", "QCNN", "no_gt", "qcnn", True, "TPAMI 2024 official QCNN"),
}


METRIC_SETS: Dict[str, List[str]] = {
    "gt_main": ["psnr", "ssim", "lpips"],
    "gt_all": ["psnr", "ssim", "lpips", "mae", "mse", "ms_ssim_gt"],
    "no_gt_main": ["qmi", "qabf", "qcb", "qviff", "qcnn"],
    "ips": ["qmi", "qsf", "qs", "qcb", "qabf", "qncie"],
    "rediffuse": ["qabf", "qmi", "qg", "qp", "qe", "ms_ssim_src"],
    "all_no_gt": [
        "qmi", "qsf", "qs", "qcb", "qabf", "qabf_analysis", "qncie",
        "qg", "qp", "qe", "qviff", "ms_ssim_src", "qcnn",
    ],
}
METRIC_SETS["all"] = METRIC_SETS["gt_all"] + METRIC_SETS["all_no_gt"]


def expand_metrics(tokens: List[str]) -> List[str]:
    result: List[str] = []
    for token in tokens:
        key = token.strip().lower()
        if not key:
            continue
        if key in METRIC_SETS:
            candidates = METRIC_SETS[key]
        elif key in METRICS:
            candidates = [key]
        else:
            valid = sorted(list(METRICS) + list(METRIC_SETS))
            raise ValueError(f"Unknown metric or metric set: {token}. Valid: {', '.join(valid)}")
        for metric in candidates:
            if metric not in result:
                result.append(metric)
    return result
