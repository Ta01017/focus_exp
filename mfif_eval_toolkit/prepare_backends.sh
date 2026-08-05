#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/third_party"

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

clone_or_update \
  https://github.com/thfylsty/Objective-evaluation-for-image-fusion.git \
  "$ROOT/third_party/Objective-evaluation-for-image-fusion"

cat <<MSG
[DONE]
TPAMI metrics: $ROOT/third_party/MFIF-Metrics
Legacy extras: $ROOT/third_party/Objective-evaluation-for-image-fusion

Install Python dependencies with:
  python -m pip install -r "$ROOT/requirements.txt"

Legacy metrics require MATLAB with Image Processing Toolbox.
QCNN uses the official model.py and resnet34.pth in MFIF-Metrics/QCNN-metric.
MSG
