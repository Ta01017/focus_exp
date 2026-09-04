#!/usr/bin/env bash
# Train/infer/evaluate the non-diffusion baselines for the mixed MFIF dataset.
set -Eeuo pipefail

PROJECT_ROOT="${FOCUS_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Pipelines historically consume ROOT. Always point it at this repository;
# never inherit a generic shell ROOT that may mean the dataset storage root.
ROOT="$PROJECT_ROOT"
OUTPUT_ROOT_WAS_SET="${OUTPUT_ROOT+x}"
ARCHIVE_ROOT_WAS_SET="${ARCHIVE_ROOT+x}"
PYTHON="${PYTHON:-python3}"
TRAIN_META="${TRAIN_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/mfif_train_mix_v1/metadata_train_mix_v1_balanced.json}"
VAL_META="${VAL_META:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/real_mfif_zedd_selfshot_v4_0901/metadata_val_final.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/mfif_mix_v1}"
TAG="${TAG:-swinfusion_mix_v1_y}"
GPUS="${GPUS:-}"
SMOKE="${SMOKE:-0}"
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
RUN_REDIFFUSE="${RUN_REDIFFUSE:-1}"
REDIFFUSE_PYTHON="${REDIFFUSE_PYTHON:-/root/miniconda3/envs/rediffuse38_0806/bin/python}"
REDIFFUSE_OUTPUT_ROOT="${REDIFFUSE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
REGION_PYTHON="${REGION_PYTHON:-}"
REGION_EVAL="${REGION_EVAL:-$PROJECT_ROOT/route3/region_eval_route_v3.py}"
REGION_DATASET="${REGION_DATASET:-RealSceneVal68}"
ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}"

if [[ "$SMOKE" == 1 ]]; then
  MAX_TRAIN_STEPS=2
  TRAIN_MAX_SAMPLES=4
  INFER_MAX_SAMPLES=1
  TRAIN_BATCH_SIZE=1
  NUM_WORKERS=0
  ZMFF_ITERATIONS=2
  PREFLIGHT_TRAIN_MAX_CHECK=4
  PREFLIGHT_VAL_MAX_CHECK=1
  if [[ -z "$OUTPUT_ROOT_WAS_SET" ]]; then
    OUTPUT_ROOT="$ROOT/outputs/smoke_$(date +%Y%m%d_%H%M%S)"
    REDIFFUSE_OUTPUT_ROOT="$OUTPUT_ROOT"
  fi
  # Exercise the archive code without ever replacing formal comparison data.
  if [[ -z "$ARCHIVE_ROOT_WAS_SET" ]]; then
    ARCHIVE_ROOT="$OUTPUT_ROOT/smoke_archive/RealSceneVal68"
  fi
  echo '[SMOKE] 2 SwinFusion steps, 1 inference/evaluation sample per method, temporary archive'
fi
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
REDIFFUSE_GPU="${REDIFFUSE_GPU:-${GPU_LIST[${#GPU_LIST[@]}-1]}}"
WORKER_GPU_LIST=()
for gpu in "${GPU_LIST[@]}"; do
  if [[ "$RUN_REDIFFUSE" == 1 && ${#GPU_LIST[@]} -gt 1 && "$gpu" == "$REDIFFUSE_GPU" ]]; then continue; fi
  WORKER_GPU_LIST+=("$gpu")
done
(( ${#WORKER_GPU_LIST[@]} >= 1 )) || { echo '[ERROR] no GPU remains for the main methods' >&2; exit 2; }
TRAIN_GPU="${TRAIN_GPU:-${WORKER_GPU_LIST[0]}}"
worker_csv="$(IFS=,; echo "${WORKER_GPU_LIST[*]}")"
if [[ ! ",$worker_csv," == *",$TRAIN_GPU,"* ]]; then
  echo "[ERROR] TRAIN_GPU=$TRAIN_GPU is not included in main worker GPUs=$worker_csv" >&2
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

echo "[CONFIG] smoke=$SMOKE project_root=$PROJECT_ROOT GPUs=$GPUS main_gpus=$worker_csv train_gpu=$TRAIN_GPU rediffuse=$RUN_REDIFFUSE rediffuse_gpu=$REDIFFUSE_GPU"
echo "[CONFIG] train=$TRAIN_META val=$VAL_META val_mode=$VAL_MODE output=$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"
STATUS_DIR="$(mktemp -d "$OUTPUT_ROOT/.task-status.XXXXXX")"

run_rediffuse_job() {
  REDIFFUSE_GPU="$REDIFFUSE_GPU" VAL_META="$VAL_META" OUTPUT_ROOT="$REDIFFUSE_OUTPUT_ROOT" \
    ARCHIVE_ROOT="$ARCHIVE_ROOT" REDIFFUSE_PYTHON="$REDIFFUSE_PYTHON" \
    EVAL_PYTHON="$REGION_PYTHON" MAX_SAMPLES="$INFER_MAX_SAMPLES" OVERWRITE="$OVERWRITE" \
    SEED="$SEED" RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" \
    RUN_REGION_EVAL="$RUN_REGION_EVAL" RUN_ARCHIVE="$RUN_ARCHIVE" \
    bash "$ROOT/run_rediffuse_real_v1.sh"
}

rediffuse_pid=""
if [[ "$RUN_REDIFFUSE" == 1 && "$RUN_INFER" == 1 && ${#GPU_LIST[@]} -gt 1 ]]; then
  echo "[LAUNCH] ReDiffuse_ORIGIN dedicated physical GPU $REDIFFUSE_GPU"
  (
    if run_rediffuse_job 2>&1 | tee "$OUTPUT_ROOT/logs/rediffuse_origin.log"; then
      touch "$STATUS_DIR/rediffuse.ok"
    else
      touch "$STATUS_DIR/rediffuse.failed"
      echo '[FAILED] ReDiffuse_ORIGIN; other tasks continue' >&2
    fi
  ) &
  rediffuse_pid="$!"
elif [[ "$RUN_REDIFFUSE" == 1 && "$RUN_INFER" == 1 ]]; then
  echo "[WARN] only one GPU supplied; ReDiffuse_ORIGIN will run after the main methods on GPU $REDIFFUSE_GPU"
fi

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
  if ! RUN_TRAIN=0 RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" CUDA_VISIBLE_GPU="$gpu" \
    TEST_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" MAX_SAMPLES="$INFER_MAX_SAMPLES" \
    OVERWRITE="$OVERWRITE" SEED="$SEED" PYTHON="$PYTHON" EVAL_SPECS="$spec" \
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" IFCNN_CKPT="${IFCNN_CKPT:-$ROOT/baselines/IFCNN/Code/snapshots/IFCNN-MAX.pth}" \
    SWINFUSION_CKPT="${SWINFUSION_CKPT:-}" SWINFUSION_CHECKPOINT_MODE=metadata-y \
    ZMFF_ITERATIONS="$ZMFF_ITERATIONS" bash "$ROOT/scripts/pipelines/$method.sh"; then
    echo "[FAILED] $method inference/full-image evaluation" >&2
    return 1
  fi
  if [[ "$RUN_REGION_EVAL" == 1 ]]; then
    local region_root="$OUTPUT_ROOT/region_eval"
    local region_manifest="$region_root/manifests/$REGION_DATASET/$method_label/region_manifest_route_v3.csv"
    local region_metrics="$region_root/metrics/$REGION_DATASET/$method_label"
    if ! "$PYTHON" "$ROOT/tools/build_region_manifest.py" \
      --metadata "$VAL_META" \
      --inference-manifest "$OUTPUT_ROOT/infer/$out_subdir/inference_manifest.csv" \
      --output "$region_manifest" --dataset "$REGION_DATASET" --method "$method_label" \
      --route-sum-tolerance 0.05; then
      echo "[FAILED] $method route3 manifest" >&2
      return 1
    fi
    mkdir -p "$region_metrics"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$REGION_PYTHON" "$REGION_EVAL" \
      --manifest "$region_manifest" --output-dir "$region_metrics" \
      --device cuda:0 --lpips-net alex \
      --route-confidence 0 --route-sum-tolerance 0.05 \
      --patch-size 64 --patch-stride 32 \
      --g-patch-min-coverage 0.80 --g-rsr-psnr-margin 0.20 \
      2>&1 | tee "$region_metrics/eval.log"; then
      echo "[FAILED] $method route3 evaluation" >&2
      return 1
    fi
  fi
}

pids=()
launch_method_queue() {
  local gpu="$1"
  shift
  (
    for method in "$@"; do
      if run_method "$method" "$gpu" \
          2>&1 | tee "$OUTPUT_ROOT/logs/${method}_infer_eval.log"; then
        touch "$STATUS_DIR/$method.ok"
      else
        touch "$STATUS_DIR/$method.failed"
        echo "[FAILED] $method; next task on this GPU continues" >&2
      fi
    done
  ) &
  pids+=("$!")
}

training_pid=""
if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[LAUNCH] SwinFusion training on physical GPU $TRAIN_GPU"
  (
    if RUN_TRAIN=1 RUN_INFER=0 RUN_EVAL=0 CUDA_VISIBLE_GPU="$TRAIN_GPU" \
      TRAIN_META="$TRAIN_META" VAL_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" \
      MAX_SAMPLES="$TRAIN_MAX_SAMPLES" MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
      TRAIN_CROP_SIZE="$TRAIN_CROP_SIZE" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" SEED="$SEED" PYTHON="$PYTHON" \
      bash "$ROOT/scripts/pipelines/swinfusion.sh" 2>&1 | tee "$OUTPUT_ROOT/logs/swinfusion_train.log"; then
      touch "$STATUS_DIR/swinfusion_train.ok"
    else
      touch "$STATUS_DIR/swinfusion_train.failed"
      echo '[FAILED] SwinFusion training; other GPU tasks continue' >&2
    fi
  ) &
  training_pid="$!"
fi

if [[ "$RUN_INFER" == 1 || "$RUN_EVAL" == 1 ]]; then
  if [[ "$RUN_TRAIN" == 1 && ${#WORKER_GPU_LIST[@]} -gt 1 ]]; then
    # Keep TRAIN_GPU exclusive. Independent methods immediately occupy every
    # other main GPU while SwinFusion is training.
    EARLY_GPU_LIST=()
    for gpu in "${WORKER_GPU_LIST[@]}"; do [[ "$gpu" == "$TRAIN_GPU" ]] || EARLY_GPU_LIST+=("$gpu"); done
    early_methods=(ifcnn zmff dsift)
    for slot in "${!EARLY_GPU_LIST[@]}"; do
      queue=()
      for index in "${!early_methods[@]}"; do
        (( index % ${#EARLY_GPU_LIST[@]} == slot )) && queue+=("${early_methods[$index]}")
      done
      launch_method_queue "${EARLY_GPU_LIST[$slot]}" "${queue[@]}"
    done
    # SwinFusion inference needs the new checkpoint, but does not need to wait
    # for IFCNN/ZMFF/DSIFT on the other cards.
    wait "$training_pid" || true
    training_pid=""
    launch_method_queue "$TRAIN_GPU" swinfusion
  else
    all_methods=(swinfusion ifcnn zmff dsift)
    for slot in "${!WORKER_GPU_LIST[@]}"; do
      queue=()
      for index in "${!all_methods[@]}"; do
        (( index % ${#WORKER_GPU_LIST[@]} == slot )) && queue+=("${all_methods[$index]}")
      done
      launch_method_queue "${WORKER_GPU_LIST[$slot]}" "${queue[@]}"
    done
  fi
fi

[[ -z "$training_pid" ]] || wait "$training_pid" || true
for pid in "${pids[@]}"; do wait "$pid" || true; done

if [[ "$RUN_ARCHIVE" == 1 ]]; then
  if [[ "$RUN_INFER" != 1 || "$RUN_EVAL" != 1 ]]; then
    touch "$STATUS_DIR/main_archive.failed"
    echo '[FAILED] RUN_ARCHIVE=1 requires RUN_INFER=1 and RUN_EVAL=1; remaining tasks continue' >&2
  else
    echo "[ARCHIVE] target=$ARCHIVE_ROOT"
    for archive_method in SwinFusion IFCNN ZMFF DSIFT; do
      archive_args=(--output-root "$OUTPUT_ROOT" --archive-root "$ARCHIVE_ROOT"
                    --tag "$TAG" --dataset RealMFIFZeddV4 --methods "$archive_method"
                    --region-dataset "$REGION_DATASET")
      [[ "$RUN_REGION_EVAL" != 1 ]] || archive_args+=(--require-region)
      if "$PYTHON" "$ROOT/tools/archive_real_scene_results.py" "${archive_args[@]}"; then
        touch "$STATUS_DIR/archive_${archive_method}.ok"
      else
        touch "$STATUS_DIR/archive_${archive_method}.failed"
        echo "[FAILED] $archive_method archive; remaining archives continue" >&2
      fi
    done
  fi
fi

if [[ "$RUN_REDIFFUSE" == 1 && "$RUN_INFER" == 1 && -z "$rediffuse_pid" ]]; then
  if run_rediffuse_job 2>&1 | tee "$OUTPUT_ROOT/logs/rediffuse_origin.log"; then
    touch "$STATUS_DIR/rediffuse.ok"
  else
    touch "$STATUS_DIR/rediffuse.failed"
  fi
elif [[ -n "$rediffuse_pid" ]]; then
  wait "$rediffuse_pid" || true
fi

failed_files=("$STATUS_DIR"/*.failed)
if [[ -e "${failed_files[0]}" ]]; then
  echo "[DONE WITH FAILURES] outputs=$OUTPUT_ROOT"
  for failure in "${failed_files[@]}"; do echo "  - $(basename "$failure" .failed)"; done
  exit 1
fi
echo "[DONE] all requested tasks succeeded; outputs=$OUTPUT_ROOT"
