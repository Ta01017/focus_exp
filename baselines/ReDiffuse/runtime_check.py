import platform
import sys
from pathlib import Path


def require_official_b_conv():
    package = Path(__file__).resolve().parent / "Condition_Noise_Predictor"
    source = package / "__pycache__" / "B_Conv.cpython-38.pyc"
    runtime = package / "B_Conv.pyc"
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 8):
        raise RuntimeError(
            f"ReDiffuse official B_Conv bytecode requires CPython 3.8; current={sys.version}")
    if not source.is_file():
        raise RuntimeError(f"Official B_Conv CPython 3.8 bytecode is missing: {source}")
    if not runtime.is_file():
        raise RuntimeError(
            f"Prepared runtime bytecode is missing: {runtime}. "
            "Run prepare_official_bytecode.py first.")
    return runtime
