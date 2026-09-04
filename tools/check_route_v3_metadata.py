#!/usr/bin/env python3
"""Return success only when every metadata row declares three route-v3 maps."""
import argparse
import json
from pathlib import Path

from build_region_manifest import ROUTE_A_KEYS, ROUTE_B_KEYS, ROUTE_G_KEYS, first


def has_three_routes(item):
    if all(first(item, keys) is not None for keys in (ROUTE_A_KEYS, ROUTE_B_KEYS, ROUTE_G_KEYS)):
        return True
    route_set = item.get("route_maps") or item.get("routes") or item.get("masks")
    if isinstance(route_set, (list, tuple)) and len(route_set) >= 3:
        return True
    edits = item.get("edit_image") or []
    return isinstance(edits, list) and len(edits) >= 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    items = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or not items:
        raise ValueError("metadata must be a non-empty list")
    missing = [index for index, item in enumerate(items)
               if not isinstance(item, dict) or not has_three_routes(item)]
    if missing:
        print(f"route_v3=no rows={len(items)} missing={len(missing)} first_missing={missing[:5]}")
        return 1
    print(f"route_v3=yes rows={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
