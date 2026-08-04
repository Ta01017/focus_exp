#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
METADATA="${METADATA:?Set METADATA=/path/test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/infer}"
METHOD="${METHOD:-all}"
CUDA_VISIBLE_GPU="${CUDA_VISIBLE_GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
START_INDEX="${START_INDEX:-0}"; MAX_SAMPLES="${MAX_SAMPLES:--1}"
OVERWRITE="${OVERWRITE:-0}"; SEED="${SEED:-17}"; STRICT="${STRICT:-0}"
ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}"
[[ -f "$METADATA" ]] || { echo "[ERROR] missing METADATA=$METADATA" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
failed=0; skipped=0

unavailable() {
  if [[ "$STRICT" == "1" || "${METHOD,,}" != "all" ]]; then echo "[ERROR] $1" >&2; return 3; fi
  echo "[SKIP] $1"; skipped=$((skipped+1)); return 0
}
run_py() { local name="$1"; shift; echo "[INFER] $name $*"; CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_GPU" "$@" || { failed=$((failed+1)); [[ "$STRICT" != "1" ]]; }; }

run_dsift() {
  command -v matlab >/dev/null || { unavailable "DSIFT requires MATLAB"; return; }
  (cd "$ROOT/baselines/DSIFT-MFIF" && matlab -batch "infer_metadata('metadata','$METADATA','output_dir','$OUTPUT_ROOT/DSIFT','start_index',$START_INDEX,'max_samples',$MAX_SAMPLES,'overwrite',$OVERWRITE)")
}
run_ifcnn() {
  local ckpt="${IFCNN_CKPT:-$ROOT/baselines/IFCNN/Code/snapshots/IFCNN-MAX.pth}"
  [[ -f "$ckpt" ]] || { unavailable "IFCNN checkpoint missing: $ckpt"; return; }
  run_py IFCNN "$PYTHON" "$ROOT/baselines/IFCNN/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/IFCNN" --checkpoint "$ckpt" --device "$DEVICE" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"
}
run_swinfusion() {
  local ckpt="${SWINFUSION_CKPT:-}"
  [[ -n "$ckpt" && -f "$ckpt" ]] || { unavailable "SwinFusion requires explicit Multi-Focus SWINFUSION_CKPT"; return; }
  [[ "$ckpt" == *Multi_Focus* || "${ALLOW_EXPLICIT_MFIF_CKPT:-0}" == "1" ]] || { unavailable "SwinFusion checkpoint path is not identifiable as Multi_Focus: $ckpt"; return; }
  echo "[CHECKPOINT] SwinFusion MFIF=$ckpt"
  run_py SwinFusion "$PYTHON" "$ROOT/baselines/SwinFusion/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/SwinFusion" --checkpoint "$ckpt" --device "$DEVICE" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"
}
run_zmff() { run_py ZMFF "$PYTHON" "$ROOT/baselines/ZMFF/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/ZMFF" --device "$DEVICE" --iterations "$ZMFF_ITERATIONS" --seed "$SEED" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_fusiondiff() {
  local ckpt="${FUSIONDIFF_CKPT:-}"; [[ -n "$ckpt" && -f "$ckpt" ]] || { unavailable "FusionDiff requires FUSIONDIFF_CKPT"; return; }
  run_py FusionDiff "$PYTHON" "$ROOT/baselines/ImageFusion/FusionDiff/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/FusionDiff" --checkpoint "$ckpt" --device "$DEVICE" --sampling-steps 2000 --seed "$SEED" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"
}
run_rediffuse() {
  local source="$ROOT/baselines/ReDiffuse/Condition_Noise_Predictor/B_Conv.py" ckpt="${REDIFFUSE_CKPT:-$ROOT/baselines/ReDiffuse/weights/model.pt}"
  [[ -f "$source" ]] || { unavailable "ReDiffuse blocked: missing verified $source"; return; }
  [[ -f "$ckpt" ]] || { unavailable "ReDiffuse checkpoint missing: $ckpt"; return; }
  run_py ReDiffuse "$PYTHON" "$ROOT/baselines/ReDiffuse/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/ReDiffuse" --checkpoint "$ckpt" --device "$DEVICE" --sampling-steps 2000 --seed "$SEED" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"
}

case "${METHOD,,}" in
 dsift) run_dsift;; ifcnn) run_ifcnn;; swinfusion|swin) run_swinfusion;; zmff) run_zmff;; fusiondiff|fd) run_fusiondiff;; rediffuse) run_rediffuse;;
 all) run_dsift; run_ifcnn; run_swinfusion; run_zmff; run_fusiondiff; run_rediffuse;;
 *) echo "[ERROR] unknown METHOD=$METHOD" >&2; exit 2;;
esac
echo "[DONE] failed=$failed skipped=$skipped output=$OUTPUT_ROOT"
[[ "$failed" -eq 0 ]]
