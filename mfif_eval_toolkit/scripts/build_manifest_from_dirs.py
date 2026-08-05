#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_STRIP = r"(?i)(?:[_\-](?:a|b|gt|f|fused|output|result|src|ref|target|1|2))$"


def files(directory: Path, include_pattern: Optional[str] = None):
    """List image files, optionally filtering by a regex matched against the filename stem."""
    include_re = re.compile(include_pattern) if include_pattern else None
    result = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if include_re is not None and include_re.search(path.stem) is None:
            continue
        result.append(path)
    return sorted(result)


def key(path: Path, pattern: str) -> str:
    return re.sub(pattern, "", path.stem).lower()


def index(
    directory: Path,
    strip_pattern: str,
    include_pattern: Optional[str],
    role: str,
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    selected = files(directory, include_pattern)
    if not selected:
        suffix = f" include={include_pattern!r}" if include_pattern else ""
        raise RuntimeError(f"No image files selected for {role} in {directory}.{suffix}")

    for path in selected:
        k = key(path, strip_pattern)
        if k in mapping:
            raise ValueError(
                f"Duplicate normalized key '{k}' for {role} in {directory}: "
                f"{mapping[k]} and {path}. Use a stricter --include-{role} regex "
                "or adjust the matching --strip-* regex."
            )
        mapping[k] = path.resolve()
    return mapping


def method_arg(value: str) -> Tuple[str, Path, Optional[str]]:
    """
    Parse METHOD=/path or METHOD=/path::INCLUDE_REGEX.

    The optional regex is matched against filename stems. It is useful when outputs
    from several methods are stored in the same directory.
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected METHOD=/path/to/fused_dir or METHOD=/path::INCLUDE_REGEX"
        )
    name, payload = value.split("=", 1)
    if "::" in payload:
        directory, include_pattern = payload.split("::", 1)
        include_pattern = include_pattern or None
    else:
        directory, include_pattern = payload, None
    return name, Path(directory), include_pattern


def main():
    p = argparse.ArgumentParser(
        description=(
            "Build evaluator manifest by matching normalized filename stems. "
            "A/B/GT may be stored in separate directories or together in one directory."
        )
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=["gt", "no_gt"], required=True)
    p.add_argument("--source-a-dir", type=Path, required=True)
    p.add_argument("--source-b-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path)
    p.add_argument(
        "--fused",
        action="append",
        type=method_arg,
        required=True,
        help="METHOD=/dir or METHOD=/dir::REGEX when a result directory is mixed",
    )

    p.add_argument("--include-a", help="Regex matched against A filename stems")
    p.add_argument("--include-b", help="Regex matched against B filename stems")
    p.add_argument("--include-gt", help="Regex matched against GT filename stems")
    p.add_argument(
        "--include-fused",
        help="Default regex for fused filename stems; per-method METHOD=/dir::REGEX overrides it",
    )

    p.add_argument("--strip-a", default=DEFAULT_STRIP)
    p.add_argument("--strip-b", default=DEFAULT_STRIP)
    p.add_argument("--strip-gt", default=DEFAULT_STRIP)
    p.add_argument("--strip-fused", default=DEFAULT_STRIP)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if args.mode == "gt" and args.gt_dir is None:
        p.error("--gt-dir is required for mode=gt")

    a = index(args.source_a_dir, args.strip_a, args.include_a, "a")
    b = index(args.source_b_dir, args.strip_b, args.include_b, "b")
    gt = (
        index(args.gt_dir, args.strip_gt, args.include_gt, "gt")
        if args.gt_dir
        else {}
    )

    fused_maps = []
    for name, directory, method_include in args.fused:
        include_pattern = method_include or args.include_fused
        mapping = index(
            directory,
            args.strip_fused,
            include_pattern,
            f"fused[{name}]",
        )
        fused_maps.append((name, mapping))

    common = set(a) & set(b)
    if args.mode == "gt":
        common &= set(gt)
    for _, mapping in fused_maps:
        common &= set(mapping)
    if not common:
        raise RuntimeError(
            "No common normalized sample keys found. Check --include-* and --strip-* patterns."
        )

    rows = []
    for sample_id in sorted(common):
        for method, mapping in fused_maps:
            rows.append(
                {
                    "dataset": args.dataset,
                    "sample_id": sample_id,
                    "mode": args.mode,
                    "method": method,
                    "source_a": str(a[sample_id]),
                    "source_b": str(b[sample_id]),
                    "gt": str(gt[sample_id]) if args.mode == "gt" else "",
                    "fused": str(mapping[sample_id]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(
        f"[DONE] {args.output} rows={len(rows)} samples={len(common)} "
        f"methods={len(fused_maps)}"
    )
    print(
        f"[SELECTED] A={len(a)} B={len(b)} "
        f"GT={len(gt) if args.mode == 'gt' else 0}"
    )


if __name__ == "__main__":
    main()
