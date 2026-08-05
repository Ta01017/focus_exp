#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD_NAME="DSIFT"
TEST_META="${TEST_META:-}"
INFER_OUTPUT="$OUTPUT_ROOT/infer"
EVAL_OUTPUT="$OUTPUT_ROOT/eval/$METHOD_NAME"

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[TRAIN] DSIFT-MFIF is a non-learning MATLAB method; skip training."
fi

if [[ "$RUN_INFER" == 1 ]]; then
  require_file "$TEST_META" TEST_META
  require_cmd matlab
  METHOD=dsift METADATA="$TEST_META" OUTPUT_ROOT="$INFER_OUTPUT" PYTHON="$PYTHON" \
    CUDA_VISIBLE_GPU="$CUDA_VISIBLE_GPU" START_INDEX="$START_INDEX" MAX_SAMPLES="$MAX_SAMPLES" \
    OVERWRITE="$OVERWRITE" bash "$ROOT/run_infer_metadata_v3.sh"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  run_eval_for_specs "$METHOD_NAME" "$INFER_OUTPUT" "$EVAL_OUTPUT" "$EVAL_SPECS"
fi
