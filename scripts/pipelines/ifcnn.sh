#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD_NAME="IFCNN"
TEST_META="${TEST_META:-}"
INFER_OUTPUT="$OUTPUT_ROOT/infer"
EVAL_OUTPUT="$OUTPUT_ROOT/eval/$METHOD_NAME"
IFCNN_CKPT="${IFCNN_CKPT:-}"

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[TRAIN] IFCNN has no reliable supervised metadata training entry in this repo; skip."
fi

if [[ "$RUN_INFER" == 1 ]]; then
  require_file "$TEST_META" TEST_META
  require_file "$IFCNN_CKPT" IFCNN_CKPT
  METHOD=ifcnn METADATA="$TEST_META" OUTPUT_ROOT="$INFER_OUTPUT" PYTHON="$PYTHON" \
    IFCNN_CKPT="$IFCNN_CKPT" CUDA_VISIBLE_GPU="$CUDA_VISIBLE_GPU" START_INDEX="$START_INDEX" \
    MAX_SAMPLES="$MAX_SAMPLES" OVERWRITE="$OVERWRITE" bash "$ROOT/run_infer_metadata_v3.sh"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  run_eval_for_specs "$METHOD_NAME" "$INFER_OUTPUT" "$EVAL_OUTPUT" "$EVAL_SPECS"
fi
