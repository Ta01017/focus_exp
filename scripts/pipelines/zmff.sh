#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD_NAME="ZMFF"
TEST_META="${TEST_META:-}"
INFER_OUTPUT="$OUTPUT_ROOT/infer"
EVAL_OUTPUT="$OUTPUT_ROOT/eval/$METHOD_NAME"

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[TRAIN] ZMFF is zero-shot per-sample optimization; skip supervised training."
fi

if [[ "$RUN_INFER" == 1 ]]; then
  require_file "$TEST_META" TEST_META
  METHOD=zmff METADATA="$TEST_META" OUTPUT_ROOT="$INFER_OUTPUT" PYTHON="$PYTHON" \
    CUDA_VISIBLE_GPU="$CUDA_VISIBLE_GPU" SEED="$SEED" ZMFF_ITERATIONS="$ZMFF_ITERATIONS" \
    ZMFF_MAX_SIDE="${ZMFF_MAX_SIDE:-1024}" \
    START_INDEX="$START_INDEX" MAX_SAMPLES="$MAX_SAMPLES" OVERWRITE="$OVERWRITE" \
    bash "$ROOT/run_infer_metadata_v3.sh"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  run_eval_for_specs "$METHOD_NAME" "$INFER_OUTPUT" "$EVAL_OUTPUT" "$EVAL_SPECS"
fi
