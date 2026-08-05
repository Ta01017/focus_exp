from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def _matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def run_matlab_metrics(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    toolkit_root: Path,
    tpami_root: Path,
    objective_root: Path,
    matlab_command: str = "matlab",
) -> pd.DataFrame:
    metrics = list(metrics)
    if not metrics:
        return pd.DataFrame(index=frame.index)
    if shutil.which(matlab_command) is None:
        raise RuntimeError(
            f"MATLAB executable '{matlab_command}' not found. Legacy metrics requested: {metrics}. "
            "Install MATLAB or run only Python/QCNN metrics."
        )
    if not (tpami_root / "fusion-metrics").exists():
        raise FileNotFoundError(f"TPAMI fusion-metrics not found: {tpami_root}")
    if not objective_root.exists():
        raise FileNotFoundError(f"Objective-evaluation repo not found: {objective_root}")

    with tempfile.TemporaryDirectory(prefix="mfif_metrics_") as tmp:
        tmp_dir = Path(tmp)
        job_csv = tmp_dir / "jobs.csv"
        out_csv = tmp_dir / "legacy_results.csv"
        jobs = frame[["row_id", "source_a", "source_b", "fused"]].copy()
        jobs.to_csv(job_csv, index=False, quoting=csv.QUOTE_MINIMAL)

        call = (
            f"addpath('{_matlab_quote(toolkit_root / 'matlab')}');"
            f"run_legacy_batch('{_matlab_quote(job_csv)}','{_matlab_quote(out_csv)}',"
            f"'{_matlab_quote(tpami_root)}','{_matlab_quote(objective_root)}',"
            f"'{_matlab_quote(','.join(metrics))}');"
        )
        cmd = [matlab_command, "-batch", call]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "MATLAB legacy metric batch failed.\n"
                f"COMMAND: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not out_csv.exists():
            raise RuntimeError(f"MATLAB completed without writing {out_csv}\n{proc.stdout}")
        result = pd.read_csv(out_csv)
        result["row_id"] = result["row_id"].astype(int)
        return result.set_index("row_id")
