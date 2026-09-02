#!/usr/bin/env bash
# Train/infer/evaluate the non-diffusion baselines for the mixed MFIF dataset.
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
TRAIN_META="${TRAIN_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/mfif_train_mix_v1/mfif_train_mix_v1/metadata_train_mix_v1_balanced.json}"
VAL_META="${VAL_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/real_mfif_zedd_selfshot_v4_0901/metadata_val_final.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/mfif_mix_v1}"
TAG="${TAG:-swinfusion_mix_v1_y}"
GPUS="${GPUS:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
INFER_MAX_SAMPLES="${INFER_MAX_SAMPLES:--1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
TRAIN_CROP_SIZE="${TRAIN_CROP_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-9}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
SEED="${SEED:-17}"
EVAL_METRICS="${EVAL_METRICS:-auto}"
RUN_ARCHIVE="${RUN_ARCHIVE:-1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827/RealSceneVal68}"

[[ -f "$TRAIN_META" ]] || { echo "[ERROR] TRAIN_META not found: $TRAIN_META" >&2; exit 2; }
[[ -f "$VAL_META" ]] || { echo "[ERROR] VAL_META not found: $VAL_META" >&2; exit 2; }

if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  fi
fi
[[ -n "$GPUS" ]] || { echo '[ERROR] No GPU found. Set GPUS=0 or GPUS=0,1,2.' >&2; exit 2; }
IFS=',' read -r -a GPU_LIST <<<"$GPUS"
(( ${#GPU_LIST[@]} >= 1 )) || { echo '[ERROR] GPUS is empty' >&2; exit 2; }

echo "[PREFLIGHT] checking training metadata"
"$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$TRAIN_META" --require-gt
echo "[PREFLIGHT] checking validation metadata"
"$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$VAL_META"
VAL_MODE="$("$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$VAL_META" --print-eval-mode)"
if [[ "$EVAL_METRICS" == auto ]]; then
  if [[ "$VAL_MODE" == gt ]]; then EVAL_METRICS=all; else EVAL_METRICS=all_no_gt; fi
fi
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
[[ "$VAL_MODE" != gt ]] || EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS} --source-metrics-on-gt"

echo "[CONFIG] GPUs=$GPUS train=$TRAIN_META val=$VAL_META val_mode=$VAL_MODE output=$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[TRAIN] SwinFusion on visible GPUs $GPUS (DataParallel)"
  RUN_TRAIN=1 RUN_INFER=0 RUN_EVAL=0 CUDA_VISIBLE_GPU="$GPUS" \
    TRAIN_META="$TRAIN_META" VAL_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" \
    MAX_SAMPLES="$TRAIN_MAX_SAMPLES" MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
    TRAIN_CROP_SIZE="$TRAIN_CROP_SIZE" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" SEED="$SEED" PYTHON="$PYTHON" \
    bash "$ROOT/scripts/pipelines/swinfusion.sh" 2>&1 | tee "$OUTPUT_ROOT/logs/swinfusion_train.log"
fi

if [[ "$RUN_INFER" != 1 && "$RUN_EVAL" != 1 ]]; then
  echo '[DONE] training stage complete'
  exit 0
fi

methods=(swinfusion ifcnn zmff dsift)
run_method() {
  local method="$1" gpu="$2" out_subdir
  case "$method" in
    swinfusion) out_subdir=SwinFusion-metadata-y ;;
    ifcnn) out_subdir=IFCNN ;;
    zmff) out_subdir=ZMFF ;;
    dsift) out_subdir=DSIFT ;;
  esac
  local spec="RealMFIFZeddV4|$VAL_MODE|$VAL_META|$out_subdir|$EVAL_METRICS"
  echo "[JOB] method=$method physical_gpu=$gpu"
  RUN_TRAIN=0 RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" CUDA_VISIBLE_GPU="$gpu" \
    TEST_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" MAX_SAMPLES="$INFER_MAX_SAMPLES" \
    OVERWRITE="$OVERWRITE" SEED="$SEED" PYTHON="$PYTHON" EVAL_SPECS="$spec" \
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" IFCNN_CKPT="${IFCNN_CKPT:-$ROOT/baselines/IFCNN/Code/snapshots/IFCNN-MAX.pth}" \
    SWINFUSION_CKPT="${SWINFUSION_CKPT:-}" SWINFUSION_CHECKPOINT_MODE=metadata-y \
    ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}" bash "$ROOT/scripts/pipelines/$method.sh"
}

pids=()
for slot in "${!GPU_LIST[@]}"; do
  (
    for index in "${!methods[@]}"; do
      (( index % ${#GPU_LIST[@]} == slot )) || continue
      run_method "${methods[$index]}" "${GPU_LIST[$slot]}" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${methods[$index]}_infer_eval.log"
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "[DONE] worker_failures=$failed outputs=$OUTPUT_ROOT"
(( failed == 0 )) || exit 1

if [[ "$RUN_ARCHIVE" == 1 ]]; then
  [[ "$RUN_INFER" == 1 && "$RUN_EVAL" == 1 ]] || {
    echo '[ERROR] RUN_ARCHIVE=1 requires RUN_INFER=1 and RUN_EVAL=1' >&2
    exit 2
  }
  echo "[ARCHIVE] target=$ARCHIVE_ROOT"
  "$PYTHON" "$ROOT/tools/archive_real_scene_results.py" \
    --output-root "$OUTPUT_ROOT" --archive-root "$ARCHIVE_ROOT" \
    --tag "$TAG" --dataset RealMFIFZeddV4
fi
