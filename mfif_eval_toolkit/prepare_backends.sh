#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/third_party"
REFERENCE_SOURCES=0
if [[ "${1:-}" == "--reference-sources" ]]; then
  REFERENCE_SOURCES=1
fi

clone_or_update() {
  local url="$1"
  local dst="$2"
  if [[ -d "$dst/.git" ]]; then
    echo "[UPDATE] $dst"
    git -C "$dst" pull --ff-only
  else
    echo "[CLONE] $url -> $dst"
    git clone --depth 1 "$url" "$dst"
  fi
}

clone_or_update \
  https://github.com/yuliu316316/MFIF-Metrics.git \
  "$ROOT/third_party/MFIF-Metrics"

if [[ "$REFERENCE_SOURCES" == 1 ]]; then
  clone_or_update \
    https://github.com/thfylsty/Objective-evaluation-for-image-fusion.git \
    "$ROOT/third_party/Objective-evaluation-for-image-fusion"
fi

cat <<MSG
[DONE]
TPAMI metrics: $ROOT/third_party/MFIF-Metrics

Install Python dependencies with:
  python -m pip install -r "$ROOT/requirements.txt"

Python source metrics are shipped in this repository.
QCNN uses the official model.py and resnet34.pth in MFIF-Metrics/QCNN-metric.
Use --reference-sources only to download MATLAB reference sources for parity work.
MSG
