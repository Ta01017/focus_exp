#!/usr/bin/env bash
# Long-running official ReDiffuse inference + full-image and route3 evaluation.
set -Eeuo pipefail

PROJECT_ROOT="${FOCUS_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="$PROJECT_ROOT"
DATA_ROOT="${DATA_ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}"
VAL_META="${VAL_META:-$DATA_ROOT/dataset/real_mfif_zedd_selfshot_v4_0901/metadata_val_final.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/focus/runs/rediffuse_real_v1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$DATA_ROOT/focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827/RealSceneVal68}"
REDIFFUSE_CKPT="${REDIFFUSE_CKPT:-$ROOT/baselines/ReDiffuse/weights/model.pt}"
REDIFFUSE_PYTHON="${REDIFFUSE_PYTHON:-/root/miniconda3/envs/rediffuse38_0806/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/root/miniconda3/envs/p312/bin/python}"
REDIFFUSE_GPU="${REDIFFUSE_GPU:-0}"
TAG="${TAG:-rediffuse_official_y}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-2000}"
OVERWRITE="${OVERWRITE:-0}"
SEED="${SEED:-17}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_REGION_EVAL="${RUN_REGION_EVAL:-1}"
RUN_ARCHIVE="${RUN_ARCHIVE:-1}"
REGION_EVAL="${REGION_EVAL:-$ROOT/route3/region_eval_route_v3.py}"
REGION_DATASET="${REGION_DATASET:-RealSceneVal68}"
ROUTE_SUITE="${ROUTE_SUITE:-$DATA_ROOT/focus/pixrestore_mfif_paper_suite_v7_20260831}"

for required in "$VAL_META" "$REDIFFUSE_CKPT" "$REDIFFUSE_PYTHON" "$EVAL_PYTHON" "$REGION_EVAL"; do
  [[ -e "$required" ]] || { echo "[ERROR] required path not found: $required" >&2; exit 2; }
done
[[ "$RUN_INFER" == 1 || "$RUN_EVAL" != 1 ]] || { echo '[ERROR] RUN_EVAL=1 requires RUN_INFER=1' >&2; exit 2; }
[[ "$RUN_INFER" == 1 || "$RUN_REGION_EVAL" != 1 ]] || { echo '[ERROR] RUN_REGION_EVAL=1 requires RUN_INFER=1' >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"

EFFECTIVE_RUN_REGION_EVAL="$RUN_REGION_EVAL"

echo "[CONFIG] ReDiffuse_ORIGIN gpu=$REDIFFUSE_GPU infer_python=$REDIFFUSE_PYTHON eval_python=$EVAL_PYTHON"
echo "[CONFIG] metadata=$VAL_META output=$OUTPUT_ROOT archive=$ARCHIVE_ROOT"

SPEC="RealMFIFZeddV4|gt|$VAL_META|ReDiffuse-official-y|all"
RUN_TRAIN=0 RUN_INFER="$RUN_INFER" RUN_EVAL="$RUN_EVAL" \
  ROOT="$ROOT" PYTHON="$EVAL_PYTHON" REDIFFUSE_PYTHON="$REDIFFUSE_PYTHON" \
  TEST_META="$VAL_META" OUTPUT_ROOT="$OUTPUT_ROOT" TAG="$TAG" \
  REDIFFUSE_CKPT="$REDIFFUSE_CKPT" REDIFFUSE_CHECKPOINT_MODE=official-y \
  CUDA_VISIBLE_GPU="$REDIFFUSE_GPU" MAX_SAMPLES="$MAX_SAMPLES" \
  SAMPLING_STEPS="$SAMPLING_STEPS" OVERWRITE="$OVERWRITE" SEED="$SEED" \
  EVAL_SPECS="$SPEC" EVAL_EXTRA_ARGS='--source-metrics-on-gt' \
  bash "$ROOT/scripts/pipelines/rediffuse.sh" 2>&1 | tee "$OUTPUT_ROOT/logs/rediffuse_infer_full_eval.log"

if [[ "$EFFECTIVE_RUN_REGION_EVAL" == 1 ]]; then
  region_manifest="$OUTPUT_ROOT/region_eval/manifests/$REGION_DATASET/ReDiffuse_ORIGIN/region_manifest_route_v3.csv"
  region_metrics="$OUTPUT_ROOT/region_eval/metrics/$REGION_DATASET/ReDiffuse_ORIGIN"
  route_dir="$OUTPUT_ROOT/region_eval/routes/$REGION_DATASET/ReDiffuse_ORIGIN"
  route_metadata="$route_dir/metadata_route_v3.json"
  "$EVAL_PYTHON" "$ROOT/tools/prepare_route_v3_metadata.py" \
    --metadata "$VAL_META" \
    --inference-manifest "$OUTPUT_ROOT/infer/ReDiffuse-official-y/inference_manifest.csv" \
    --suite "$ROUTE_SUITE" --converter "$ROOT/route3/make_routes_from_focus_ab.py" \
    --output-dir "$route_dir" --output-metadata "$route_metadata"
  "$EVAL_PYTHON" "$ROOT/tools/build_region_manifest.py" \
    --metadata "$route_metadata" \
    --inference-manifest "$OUTPUT_ROOT/infer/ReDiffuse-official-y/inference_manifest.csv" \
    --output "$region_manifest" --dataset "$REGION_DATASET" --method ReDiffuse_ORIGIN \
    --route-sum-tolerance 0.05
  mkdir -p "$region_metrics"
  CUDA_VISIBLE_DEVICES="$REDIFFUSE_GPU" "$EVAL_PYTHON" "$REGION_EVAL" \
    --manifest "$region_manifest" --output-dir "$region_metrics" \
    --device cuda:0 --lpips-net alex \
    --route-confidence 0 --route-sum-tolerance 0.05 \
    --patch-size 64 --patch-stride 32 \
    --g-patch-min-coverage 0.80 --g-rsr-psnr-margin 0.20 \
    2>&1 | tee "$region_metrics/eval.log"
fi

if [[ "$RUN_ARCHIVE" == 1 ]]; then
  [[ "$RUN_INFER" == 1 && "$RUN_EVAL" == 1 ]] || { echo '[ERROR] archive requires inference and full evaluation' >&2; exit 2; }
  archive_args=(--output-root "$OUTPUT_ROOT" --archive-root "$ARCHIVE_ROOT"
                --tag "$TAG" --dataset RealMFIFZeddV4 --methods ReDiffuse
                --region-dataset "$REGION_DATASET")
  [[ "$EFFECTIVE_RUN_REGION_EVAL" != 1 ]] || archive_args+=(--require-region)
  "$EVAL_PYTHON" "$ROOT/tools/archive_real_scene_results.py" "${archive_args[@]}"
fi

echo "[DONE] ReDiffuse_ORIGIN output=$OUTPUT_ROOT archive=$ARCHIVE_ROOT/ReDiffuse_ORIGIN"
