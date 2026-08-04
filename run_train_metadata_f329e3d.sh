#!/usr/bin/env bash
set -Eeuo pipefail

# Safe metadata training launcher for commit:
# f329e3d6488984121256f48bd2d369e38df3c5f1
#
# Supported at this commit:
#   METHOD=swinfusion
#   METHOD=fusiondiff
#   METHOD=all          # sequentially runs the two methods above
#
# Not supported:
#   DSIFT / ZMFF       : no supervised training by design
#   IFCNN              : the bundled official repository has no complete training entry
#   ReDiffuse          : blocked at this commit; see CODEX_FIX_F329E3D.md
#
# Usage:
#   TRAIN_META=/path/train.json VAL_META=/path/val.json \
#   METHOD=swinfusion bash run_train_metadata_f329e3d.sh
#
# Smoke:
#   MAX_TRAIN_STEPS=1 MAX_SAMPLES=2 NUM_WORKERS=0 ...

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
METHOD="${METHOD:-all}"
TRAIN_META="${TRAIN_META:?Set TRAIN_META=/absolute/path/train_metadata.json}"
VAL_META="${VAL_META:?Set VAL_META=/absolute/path/val_metadata.json}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-17}"
START_INDEX="${START_INDEX:-0}"
RESUME="${RESUME:-}"

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }
}
require_file "$TRAIN_META"
require_file "$VAL_META"

run_swinfusion() {
  local repo="$ROOT/baselines/SwinFusion"
  local gpus="${SWIN_VISIBLE_GPUS:-0,1,2}"
  require_file "$repo/main_train_swinfusion.py"
  require_file "$repo/options/swinir/train_swinfusion_mff.json"

  if [[ -n "$RESUME" ]]; then
    echo "[WARN] SwinFusion does not take --resume directly."
    echo "[WARN] Put G/E/optimizerG checkpoints in the configured models directory;"
    echo "[WARN] the original entry auto-detects the latest matching iteration."
  fi

  echo "============================================================"
  echo "[TRAIN] SwinFusion"
  echo "[GPUS]  $gpus"
  echo "============================================================"
  (
    cd "$repo"
    CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" main_train_swinfusion.py \
      --opt options/swinir/train_swinfusion_mff.json \
      --train-metadata "$TRAIN_META" \
      --val-metadata "$VAL_META" \
      --start-index "$START_INDEX" \
      --max-samples "$MAX_SAMPLES" \
      --max-train-steps "$MAX_TRAIN_STEPS" \
      --num-workers "$NUM_WORKERS" \
      --seed "$SEED"
  )
}

run_fusiondiff() {
  local repo="$ROOT/baselines/ImageFusion/FusionDiff"
  local gpu="${DIFF_VISIBLE_GPU:-0}"
  require_file "$repo/train.py"
  require_file "$repo/config.json"

  local resume_args=()
  if [[ -n "$RESUME" ]]; then
    require_file "$RESUME"
    resume_args=(--resume "$RESUME")
  fi

  echo "============================================================"
  echo "[TRAIN] FusionDiff"
  echo "[GPU]   $gpu"
  echo "============================================================"
  (
    cd "$repo"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py \
      --config config.json \
      --dataset-format metadata \
      --train-metadata "$TRAIN_META" \
      --val-metadata "$VAL_META" \
      --start-index "$START_INDEX" \
      --max-samples "$MAX_SAMPLES" \
      --max-train-steps "$MAX_TRAIN_STEPS" \
      --num-workers "$NUM_WORKERS" \
      --seed "$SEED" \
      "${resume_args[@]}"
  )
}

case "${METHOD,,}" in
  swinfusion|swin)
    run_swinfusion
    ;;
  fusiondiff|fd)
    run_fusiondiff
    ;;
  all)
    run_swinfusion
    run_fusiondiff
    echo "[SKIP] ReDiffuse: commit is blocked by missing B_Conv.py and preprocessing mismatch."
    echo "[SKIP] IFCNN: no complete official supervised training entry."
    echo "[SKIP] DSIFT/ZMFF: training is not applicable."
    ;;
  rediffuse)
    echo "[ERROR] ReDiffuse training is intentionally blocked for commit f329e3d." >&2
    echo "[ERROR] Apply CODEX_FIX_F329E3D.md first." >&2
    exit 3
    ;;
  ifcnn|dsift|zmff)
    echo "[ERROR] METHOD=$METHOD has no reliable supervised training entry in this repository." >&2
    exit 3
    ;;
  *)
    echo "[ERROR] Unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac
