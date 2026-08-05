from __future__ import annotations

import hashlib
import importlib
import importlib.util
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    package_dir = repo_dir / "Condition_Noise_Predictor"
    source_pyc = package_dir / "__pycache__" / "B_Conv.cpython-38.pyc"
    runtime_pyc = package_dir / "B_Conv.pyc"
    if platform.python_implementation() != "CPython":
        raise RuntimeError("Official B_Conv bytecode requires CPython.")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError(
            "Official B_Conv bytecode requires CPython 3.8. "
            f"Current interpreter: {sys.version}")
    if not source_pyc.is_file():
        raise FileNotFoundError(f"Official B_Conv bytecode is missing: {source_pyc}")
    payload = source_pyc.read_bytes()
    if len(payload) < 16:
        raise RuntimeError(f"Invalid or truncated pyc file: {source_pyc}")
    file_magic = payload[:4]
    runtime_magic = importlib.util.MAGIC_NUMBER
    if file_magic != runtime_magic:
        raise RuntimeError(
            "B_Conv bytecode magic does not match the current CPython interpreter. "
            f"file={file_magic.hex()} runtime={runtime_magic.hex()}")
    sha256 = hashlib.sha256(payload).hexdigest()
    shutil.copy2(source_pyc, runtime_pyc)
    importlib.invalidate_caches()
    module = importlib.import_module("Condition_Noise_Predictor.B_Conv")
    print(f"[B_CONV] source={source_pyc}")
    print(f"[B_CONV] runtime={runtime_pyc}")
    print(f"[B_CONV] sha256={sha256}")
    print(f"[B_CONV] python={sys.version}")
    print(f"[B_CONV] magic={file_magic.hex()}")
    print(f"[B_CONV] imported={module.__file__}")


if __name__ == "__main__":
    main()
