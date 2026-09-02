#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
TRAIN_META="${TRAIN_META:?Set TRAIN_META=/path/train.json}"
VAL_META="${VAL_META:?Set VAL_META=/path/val.json}"
METHOD="${METHOD:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/train}"
TAG="${TAG:-metadata_v2}"
GPUS="${GPUS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-17}"
INIT_MODE="${INIT_MODE:-scratch}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-}"
RESUME_DIR="${RESUME_DIR:-}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
STRICT="${STRICT:-0}"
FAIL_ON_SPLIT_OVERLAP="${FAIL_ON_SPLIT_OVERLAP:-0}"

[[ -f "$TRAIN_META" ]] || { echo "[ERROR] missing TRAIN_META=$TRAIN_META" >&2; exit 2; }
[[ -f "$VAL_META" ]] || { echo "[ERROR] missing VAL_META=$VAL_META" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
echo "[CONFIG] ROOT=$ROOT METHOD=$METHOD TRAIN_META=$TRAIN_META VAL_META=$VAL_META OUTPUT_ROOT=$OUTPUT_ROOT TAG=$TAG GPUS=$GPUS NUM_WORKERS=$NUM_WORKERS MAX_SAMPLES=$MAX_SAMPLES MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS SEED=$SEED INIT_MODE=$INIT_MODE"

ensure_output() {
  local path="$1"
  if [[ -d "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" && "$OVERWRITE_OUTPUT" != "1" ]]; then
    echo "[ERROR] output is non-empty: $path" >&2; return 2
  fi
}

run_swinfusion() {
  local repo="$ROOT/baselines/SwinFusion" output="$OUTPUT_ROOT/swinfusion/$TAG"
  [[ "$INIT_MODE" == "resume" ]] || ensure_output "$output"
  local init_args=(--init-mode "$INIT_MODE" --output-dir "$output")
  [[ "$INIT_MODE" != "official" ]] || init_args+=(--init-checkpoint-dir "$INIT_CHECKPOINT_DIR")
  [[ "$INIT_MODE" != "resume" ]] || init_args+=(--resume-dir "$RESUME_DIR")
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" "$repo/main_train_swinfusion.py" \
    --opt "$repo/options/swinir/train_swinfusion_mff.json" \
    --train-metadata "$TRAIN_META" --val-metadata "$VAL_META" \
    --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" \
    --max-train-steps "$MAX_TRAIN_STEPS" --num-workers "$NUM_WORKERS" --seed "$SEED" \
    --overwrite-output "$OVERWRITE_OUTPUT" --fail-on-split-overlap "$FAIL_ON_SPLIT_OVERLAP" "${init_args[@]}"
}

run_fusiondiff() {
  local repo="$ROOT/baselines/ImageFusion/FusionDiff" output="$OUTPUT_ROOT/fusiondiff/$TAG"
  ensure_output "$output"; mkdir -p "$output"
  local resume_args=(); [[ -z "$RESUME_DIR" ]] || resume_args=(--resume "$RESUME_DIR")
  (cd "$output" && CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" "$repo/train.py" \
    --config "$repo/config.json" --dataset-format metadata \
    --train-metadata "$TRAIN_META" --val-metadata "$VAL_META" \
    --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" \
    --max-train-steps "$MAX_TRAIN_STEPS" --num-workers "$NUM_WORKERS" --seed "$SEED" \
    --fail-on-split-overlap "$FAIL_ON_SPLIT_OVERLAP" \
    "${resume_args[@]}")
}

rediffuse_blocked() {
  local source="$ROOT/baselines/ReDiffuse/Condition_Noise_Predictor/B_Conv.py"
  [[ -f "$source" ]] && { echo "[ERROR] ReDiffuse source exists but strict checkpoint/forward approval has not been recorded." >&2; return 3; }
  echo "[BLOCKED] ReDiffuse missing verified official source: $source"
  return 3
}

case "${METHOD,,}" in
  swinfusion|swin) run_swinfusion ;;
  fusiondiff|fd) run_fusiondiff ;;
  rediffuse) rediffuse_blocked ;;
  all)
    run_swinfusion
    run_fusiondiff
    if ! rediffuse_blocked; then [[ "$STRICT" == "1" ]] && exit 3; fi
    ;;
  *) echo "[ERROR] supervised METHOD must be swinfusion, fusiondiff, or all" >&2; exit 2 ;;
esac
