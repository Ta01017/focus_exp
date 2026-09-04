#!/usr/bin/env bash
# Flux2 is intentionally run as the final single-GPU train/infer/eval job.
set -Eeuo pipefail

ROOT="${FOCUS_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}"
TRAIN_META="${TRAIN_META:-$DATA_ROOT/dataset/mfif_train_mix_v1/metadata_train_mix_v1_balanced.json}"
VAL_META="${VAL_META:-$DATA_ROOT/dataset/real_mfif_zedd_selfshot_v4_0901/metadata_val_final.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/focus/runs/mfif_mix_v1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$DATA_ROOT/focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827/RealSceneVal68}"
FLUX2_GPU="${FLUX2_GPU:-0}"
FLUX2_GPUS="${FLUX2_GPUS:-$FLUX2_GPU}"
FLUX2_TAG="${FLUX2_TAG:-flux2_klein_4b_focus_lora}"
FLUX2_PYTHON="${FLUX2_PYTHON:-/root/miniconda3/envs/p312/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/root/miniconda3/envs/p312/bin/python}"
ROUTE_SUITE="${ROUTE_SUITE:-$DATA_ROOT/focus/pixrestore_mfif_paper_suite_v7_20260831}"
REGION_EVAL="${REGION_EVAL:-$ROOT/route3/region_eval_route_v3.py}"
REGION_DATASET="${REGION_DATASET:-RealSceneVal68}"
RUN_TRAIN="${RUN_TRAIN:-1}"; RUN_INFER="${RUN_INFER:-1}"; RUN_EVAL="${RUN_EVAL:-1}"
RUN_REGION_EVAL="${RUN_REGION_EVAL:-1}"; RUN_ARCHIVE="${RUN_ARCHIVE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"; OVERWRITE="${OVERWRITE:-0}"; SEED="${SEED:-17}"

spec="RealMFIFZeddV4|gt|$VAL_META|Flux2|all"
ROOT="$ROOT" PYTHON="$EVAL_PYTHON" FLUX2_PYTHON="$FLUX2_PYTHON" \
  CUDA_VISIBLE_GPU="$FLUX2_GPU" FLUX2_GPUS="$FLUX2_GPUS" TRAIN_META="$TRAIN_META" VAL_META="$VAL_META" \
  OUTPUT_ROOT="$OUTPUT_ROOT" FLUX2_TAG="$FLUX2_TAG" RUN_TRAIN="$RUN_TRAIN" \
  RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" MAX_SAMPLES="$MAX_SAMPLES" \
  OVERWRITE="$OVERWRITE" SEED="$SEED" SMOKE="${SMOKE:-0}" EVAL_SPECS="$spec" \
  EVAL_EXTRA_ARGS='--source-metrics-on-gt' \
  bash "$ROOT/scripts/pipelines/flux2.sh"

if [[ "$RUN_REGION_EVAL" == 1 ]]; then
  route_root="$OUTPUT_ROOT/region_eval"
  route_dir="$route_root/routes/$REGION_DATASET/Flux2"
  route_metadata="$route_dir/metadata_route_v3.json"
  region_manifest="$route_root/manifests/$REGION_DATASET/Flux2/region_manifest_route_v3.csv"
  region_metrics="$route_root/metrics/$REGION_DATASET/Flux2"
  "$EVAL_PYTHON" "$ROOT/tools/prepare_route_v3_metadata.py" \
    --metadata "$VAL_META" --inference-manifest "$OUTPUT_ROOT/infer/Flux2/inference_manifest.csv" \
    --suite "$ROUTE_SUITE" --converter "$ROOT/route3/make_routes_from_focus_ab.py" \
    --output-dir "$route_dir" --output-metadata "$route_metadata"
  "$EVAL_PYTHON" "$ROOT/tools/build_region_manifest.py" --metadata "$route_metadata" \
    --inference-manifest "$OUTPUT_ROOT/infer/Flux2/inference_manifest.csv" \
    --output "$region_manifest" --dataset "$REGION_DATASET" --method Flux2 --route-sum-tolerance 0.05
  mkdir -p "$region_metrics"
  CUDA_VISIBLE_DEVICES="$FLUX2_GPU" "$EVAL_PYTHON" "$REGION_EVAL" \
    --manifest "$region_manifest" --output-dir "$region_metrics" --device cuda:0 --lpips-net alex \
    --route-confidence 0 --route-sum-tolerance 0.05 --patch-size 64 --patch-stride 32 \
    --g-patch-min-coverage 0.80 --g-rsr-psnr-margin 0.20 2>&1 | tee "$region_metrics/eval.log"
fi

if [[ "$RUN_ARCHIVE" == 1 ]]; then
  args=(--output-root "$OUTPUT_ROOT" --archive-root "$ARCHIVE_ROOT" --tag "$FLUX2_TAG"
        --dataset RealMFIFZeddV4 --methods Flux2 --region-dataset "$REGION_DATASET")
  [[ "$RUN_REGION_EVAL" != 1 ]] || args+=(--require-region)
  "$EVAL_PYTHON" "$ROOT/tools/archive_real_scene_results.py" "${args[@]}"
fi
