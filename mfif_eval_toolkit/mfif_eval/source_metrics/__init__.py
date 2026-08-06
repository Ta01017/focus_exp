"""Python-only source-image fusion metrics.

Faithful Python port; MATLAB numerical parity pending per metric.
"""
from .backend import PYTHON_SOURCE_METRICS_VERSION, compute_source_metric, run_python_source_metrics

__all__ = [
    "PYTHON_SOURCE_METRICS_VERSION",
    "compute_source_metric",
    "run_python_source_metrics",
]
