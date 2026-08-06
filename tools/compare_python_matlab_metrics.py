#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optional developer parity tool for Python source metrics vs MATLAB references."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", default="all_no_gt")
    parser.add_argument("--python-output", type=Path, default=Path("/tmp/python_metric_parity"))
    parser.add_argument("--matlab-command", default="matlab")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if shutil.which(args.matlab_command) is None:
        print(f"[SKIP] {args.matlab_command} not found; MATLAB parity not run.")
        return 0
    root = Path(__file__).resolve().parents[1]
    python_cmd = [
        "python",
        str(root / "mfif_eval_toolkit" / "evaluate.py"),
        "--manifest",
        str(args.manifest),
        "--metrics",
        args.metrics,
        "--output-dir",
        str(args.python_output),
    ]
    subprocess.run(python_cmd, check=True)
    print("[TODO] MATLAB reference batch comparison hook. Python results written to:")
    print(args.python_output)
    print(f"Default tolerance: {args.tolerance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
