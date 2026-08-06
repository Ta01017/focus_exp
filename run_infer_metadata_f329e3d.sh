#!/usr/bin/env bash
set -Eeuo pipefail

# Safe metadata inference launcher for commit:
# f329e3d6488984121256f48bd2d369e38df3c5f1
#
# Usage:
#   METADATA=/path/metadata.json METHOD=all \
#   bash run_infer_metadata_f329e3d.sh
#
# Optional:
#   OUTPUT_ROOT=/path/outputs
#   CUDA_VISIBLE_GPU=0
#   FUSIONDIFF_CKPT=/path/model.pt
#   IFCNN_CKPT=/path/IFCNN-MAX.pth
#   SWINFUSION_CKPT=/path/10000_E.pth
#   ZMFF_ITERATIONS=1300
#   SAMPLING_STEPS=2000
#   MAX_SAMPLES=-1
#   STRICT=0  # all-mode skips unavailable methods; STRICT=1 fails instead

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
METHOD="${METHOD:-all}"
METADATA="${METADATA:?Set METADATA=/absolute/path/metadata.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
CUDA_VISIBLE_GPU="${CUDA_VISIBLE_GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_INPUTS="${SAVE_INPUTS:-0}"
SIZE_POLICY="${SIZE_POLICY:-error}"
SEED="${SEED:-17}"
ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}"
SAMPLING_STEPS="${SAMPLING_STEPS:-2000}"
STRICT="${STRICT:-0}"

mkdir -p "$OUTPUT_ROOT"
[[ -f "$METADATA" ]] || { echo "[ERROR] missing metadata: $METADATA" >&2; exit 2; }

failed=0
skipped=0

missing() {
  local message="$1"
  if [[ "$STRICT" == "1" || "${METHOD,,}" != "all" ]]; then
    echo "[ERROR] $message" >&2
    exit 3
  fi
  echo "[SKIP] $message"
  skipped=$((skipped + 1))
}

run_python_method() {
  local name="$1"; shift
  echo "============================================================"
  echo "[INFER] $name"
  echo "============================================================"
  if ! CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_GPU" "$@"; then
    echo "[FAIL] $name" >&2
    failed=$((failed + 1))
    [[ "$STRICT" == "1" ]] && exit 4
  fi
}

run_ifcnn() {
  local repo="$ROOT/baselines/IFCNN"
  local ckpt="${IFCNN_CKPT:-$repo/Code/snapshots/IFCNN-MAX.pth}"
  [[ -f "$ckpt" ]] || { missing "IFCNN checkpoint not found: $ckpt"; return; }
  run_python_method IFCNN "$PYTHON" "$repo/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/IFCNN" \
    --checkpoint "$ckpt" \
    --device "$DEVICE" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY"
}

run_swinfusion() {
  local repo="$ROOT/baselines/SwinFusion"
  local ckpt="${SWINFUSION_CKPT:-$repo/Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models/10000_E.pth}"
  [[ -f "$ckpt" ]] || { missing "SwinFusion checkpoint not found: $ckpt"; return; }
  run_python_method SwinFusion "$PYTHON" "$repo/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/SwinFusion" \
    --checkpoint "$ckpt" \
    --device "$DEVICE" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY"
}

run_zmff() {
  local repo="$ROOT/baselines/ZMFF"
  run_python_method ZMFF "$PYTHON" "$repo/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/ZMFF" \
    --device "$DEVICE" \
    --iterations "$ZMFF_ITERATIONS" \
    --seed "$SEED" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY"
}

run_fusiondiff() {
  local repo="$ROOT/baselines/ImageFusion/FusionDiff"
  local ckpt="${FUSIONDIFF_CKPT:-}"
  [[ -n "$ckpt" && -f "$ckpt" ]] || {
    missing "FusionDiff requires FUSIONDIFF_CKPT=/path/to/trained/model.pt"
    return
  }
  run_python_method FusionDiff "$PYTHON" "$repo/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/FusionDiff" \
    --checkpoint "$ckpt" \
    --device "$DEVICE" \
    --sampling-steps "$SAMPLING_STEPS" \
    --seed "$SEED" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY"
}

run_rediffuse() {
  local repo="$ROOT/baselines/ReDiffuse"
  local source="$repo/Condition_Noise_Predictor/B_Conv.py"
  local ckpt="${REDIFFUSE_CKPT:-$repo/weights/model.pt}"
  [[ -f "$source" ]] || {
    missing "ReDiffuse is blocked: missing source file $source (a CPython 3.8 pyc is not a reliable replacement)"
    return
  }
  [[ -f "$ckpt" ]] || { missing "ReDiffuse checkpoint not found: $ckpt"; return; }
  run_python_method ReDiffuse "$PYTHON" "$repo/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/ReDiffuse" \
    --checkpoint "$ckpt" \
    --device "$DEVICE" \
    --sampling-steps "$SAMPLING_STEPS" \
    --seed "$SEED" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY"
}

run_dsift() {
  echo "============================================================"
  echo "[INFER] DSIFT"
  echo "============================================================"
  if ! "$PYTHON" "$ROOT/baselines/DSIFT-MFIF/infer_metadata.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_ROOT/DSIFT" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite "$OVERWRITE" \
    --save-inputs "$SAVE_INPUTS" \
    --size-policy "$SIZE_POLICY" \
    --device "${DSIFT_DEVICE:-auto}" \
    --chunk-rows "${DSIFT_CHUNK_ROWS:-0}" \
    --scale "${DSIFT_SCALE:-48}" \
    --block-size "${DSIFT_BLOCK_SIZE:-8}" \
    --matching "${DSIFT_MATCHING:-1}"; then
    echo "[FAIL] DSIFT" >&2
    failed=$((failed + 1))
    [[ "$STRICT" == "1" ]] && exit 4
  fi
}

case "${METHOD,,}" in
  ifcnn) run_ifcnn ;;
  swinfusion|swin) run_swinfusion ;;
  zmff) run_zmff ;;
  fusiondiff|fd) run_fusiondiff ;;
  rediffuse) run_rediffuse ;;
  dsift) run_dsift ;;
  all)
    run_dsift
    run_ifcnn
    run_swinfusion
    run_zmff
    run_fusiondiff
    run_rediffuse
    ;;
  *)
    echo "[ERROR] Unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

echo "============================================================"
echo "[DONE] failed=$failed skipped=$skipped output=$OUTPUT_ROOT"
echo "============================================================"
[[ "$failed" -eq 0 ]]
