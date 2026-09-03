#!/usr/bin/env bash
# Train/infer/evaluate the non-diffusion baselines for the mixed MFIF dataset.
set -Eeuo pipefail

PROJECT_ROOT="${FOCUS_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Pipelines historically consume ROOT. Always point it at this repository;
# never inherit a generic shell ROOT that may mean the dataset storage root.
ROOT="$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
TRAIN_META="${TRAIN_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/mfif_train_mix_v1/metadata_train_mix_v1_balanced.json}"
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
PREFLIGHT_TRAIN_MAX_CHECK="${PREFLIGHT_TRAIN_MAX_CHECK:-32}"
PREFLIGHT_VAL_MAX_CHECK="${PREFLIGHT_VAL_MAX_CHECK:-16}"
PREFLIGHT_WORKERS="${PREFLIGHT_WORKERS:-8}"
RUN_ARCHIVE="${RUN_ARCHIVE:-1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827/RealSceneVal68}"
RUN_REGION_EVAL="${RUN_REGION_EVAL:-1}"
REGION_PYTHON="${REGION_PYTHON:-}"
REGION_EVAL="${REGION_EVAL:-$PROJECT_ROOT/route3/region_eval_route_v3.py}"
REGION_DATASET="${REGION_DATASET:-RealSceneVal68}"
if [[ -z "$REGION_PYTHON" ]]; then
  if [[ -x /root/miniconda3/envs/p312/bin/python ]]; then
    REGION_PYTHON=/root/miniconda3/envs/p312/bin/python
  else
    REGION_PYTHON="$PYTHON"
  fi
fi

[[ -f "$TRAIN_META" ]] || { echo "[ERROR] TRAIN_META not found: $TRAIN_META" >&2; exit 2; }
[[ -f "$VAL_META" ]] || { echo "[ERROR] VAL_META not found: $VAL_META" >&2; exit 2; }
if [[ "$RUN_REGION_EVAL" == 1 ]]; then
  [[ -f "$REGION_EVAL" ]] || { echo "[ERROR] REGION_EVAL not found: $REGION_EVAL" >&2; exit 2; }
  [[ "$RUN_INFER" == 1 ]] || { echo '[ERROR] RUN_REGION_EVAL=1 requires RUN_INFER=1' >&2; exit 2; }
fi

if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  fi
fi
[[ -n "$GPUS" ]] || { echo '[ERROR] No GPU found. Set GPUS=0 or GPUS=0,1,2.' >&2; exit 2; }
IFS=',' read -r -a GPU_LIST <<<"$GPUS"
(( ${#GPU_LIST[@]} >= 1 )) || { echo '[ERROR] GPUS is empty' >&2; exit 2; }
TRAIN_GPU="${TRAIN_GPU:-${GPU_LIST[0]}}"
if [[ ! ",$GPUS," == *",$TRAIN_GPU,"* ]]; then
  echo "[ERROR] TRAIN_GPU=$TRAIN_GPU is not included in GPUS=$GPUS" >&2
  exit 2
fi

echo "[PREFLIGHT] checking training metadata"
if [[ "$PREFLIGHT_TRAIN_MAX_CHECK" != 0 ]]; then
  "$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$TRAIN_META" --require-gt \
    --max-check "$PREFLIGHT_TRAIN_MAX_CHECK" --workers "$PREFLIGHT_WORKERS"
else
  echo '[PREFLIGHT] training image check skipped'
fi
echo "[PREFLIGHT] checking validation metadata"
"$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$VAL_META" \
  --max-check "$PREFLIGHT_VAL_MAX_CHECK" --workers "$PREFLIGHT_WORKERS"
VAL_MODE="$("$PYTHON" "$ROOT/tools/check_mfif_metadata.py" --metadata "$VAL_META" \
  --max-check "$PREFLIGHT_VAL_MAX_CHECK" --print-eval-mode)"
if [[ "$EVAL_METRICS" == auto ]]; then
  if [[ "$VAL_MODE" == gt ]]; then EVAL_METRICS=all; else EVAL_METRICS=all_no_gt; fi
fi
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
[[ "$VAL_MODE" != gt ]] || EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS} --source-metrics-on-gt"

echo "[CONFIG] project_root=$PROJECT_ROOT GPUs=$GPUS train_gpu=$TRAIN_GPU train=$TRAIN_META val=$VAL_META val_mode=$VAL_MODE output=$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[TRAIN] SwinFusion single-GPU training on physical GPU $TRAIN_GPU"
  RUN_TRAIN=1 RUN_INFER=0 RUN_EVAL=0 CUDA_VISIBLE_GPU="$TRAIN_GPU" \
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
  local method="$1" gpu="$2" out_subdir method_label
  case "$method" in
    swinfusion) out_subdir=SwinFusion-metadata-y; method_label=SwinFusion ;;
    ifcnn) out_subdir=IFCNN; method_label=IFCNN ;;
    zmff) out_subdir=ZMFF; method_label=ZMFF ;;
    dsift) out_subdir=DSIFT; method_label=DSIFT ;;
  esac
  local spec="RealMFIFZeddV4|$VAL_MODE|$VAL_META|$out_subdir|$EVAL_METRICS"
  echo "[JOB] method=$method physical_gpu=$gpu"
  RUN_TRAIN=0 RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" CUDA_VISIBLE_GPU="$gpu" \
    TEST_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" MAX_SAMPLES="$INFER_MAX_SAMPLES" \
    OVERWRITE="$OVERWRITE" SEED="$SEED" PYTHON="$PYTHON" EVAL_SPECS="$spec" \
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" IFCNN_CKPT="${IFCNN_CKPT:-$ROOT/baselines/IFCNN/Code/snapshots/IFCNN-MAX.pth}" \
    SWINFUSION_CKPT="${SWINFUSION_CKPT:-}" SWINFUSION_CHECKPOINT_MODE=metadata-y \
    ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}" bash "$ROOT/scripts/pipelines/$method.sh"
  if [[ "$RUN_REGION_EVAL" == 1 ]]; then
    local region_root="$OUTPUT_ROOT/region_eval"
    local region_manifest="$region_root/manifests/$REGION_DATASET/$method_label/region_manifest_route_v3.csv"
    local region_metrics="$region_root/metrics/$REGION_DATASET/$method_label"
    "$PYTHON" "$ROOT/tools/build_region_manifest.py" \
      --metadata "$VAL_META" \
      --inference-manifest "$OUTPUT_ROOT/infer/$out_subdir/inference_manifest.csv" \
      --output "$region_manifest" --dataset "$REGION_DATASET" --method "$method_label" \
      --route-sum-tolerance 0.05
    mkdir -p "$region_metrics"
    CUDA_VISIBLE_DEVICES="$gpu" "$REGION_PYTHON" "$REGION_EVAL" \
      --manifest "$region_manifest" --output-dir "$region_metrics" \
      --device cuda:0 --lpips-net alex \
      --route-confidence 0 --route-sum-tolerance 0.05 \
      --patch-size 64 --patch-stride 32 \
      --g-patch-min-coverage 0.80 --g-rsr-psnr-margin 0.20 \
      2>&1 | tee "$region_metrics/eval.log"
  fi
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
  archive_args=(--output-root "$OUTPUT_ROOT" --archive-root "$ARCHIVE_ROOT"
                --tag "$TAG" --dataset RealMFIFZeddV4
                --region-dataset "$REGION_DATASET")
  [[ "$RUN_REGION_EVAL" != 1 ]] || archive_args+=(--require-region)
  "$PYTHON" "$ROOT/tools/archive_real_scene_results.py" "${archive_args[@]}"
fi
