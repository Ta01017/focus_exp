#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; PYTHON="${PYTHON:-python3}"; METHOD="${METHOD:-all}"
METADATA="${METADATA:?Set METADATA}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/infer}"; CUDA_VISIBLE_GPU="${CUDA_VISIBLE_GPU:-0}"
START_INDEX="${START_INDEX:-0}"; MAX_SAMPLES="${MAX_SAMPLES:--1}"; OVERWRITE="${OVERWRITE:-0}"; SEED="${SEED:-17}"; STRICT="${STRICT:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-2000}"; ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}"; REDIFFUSE_PYTHON="${REDIFFUSE_PYTHON:-python3.8}"
[[ -f "$METADATA" ]] || { echo '[ERROR] metadata missing' >&2; exit 2; }; mkdir -p "$OUTPUT_ROOT"; failed=0
skip() { [[ "$STRICT" != 1 && "${METHOD,,}" == all ]] && { echo "[SKIP] $1"; return 0; }; echo "[ERROR] $1" >&2; return 3; }
run_ifcnn(){ local c="${IFCNN_CKPT:-}"; [[ -f "$c" ]]||{ skip 'IFCNN_CKPT missing'; return; }; "$PYTHON" "$ROOT/baselines/IFCNN/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/IFCNN" --checkpoint "$c" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_swin(){ local c="${SWINFUSION_CKPT:-}"; [[ -f "$c" ]]||{ skip 'SWINFUSION_CKPT missing'; return; }; "$PYTHON" "$ROOT/baselines/SwinFusion/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/SwinFusion" --checkpoint "$c" --checkpoint-mode metadata-rgb --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_fd(){ local c="${FUSIONDIFF_CKPT:-}"; [[ -f "$c" ]]||{ skip 'FUSIONDIFF_CKPT missing'; return; }; "$PYTHON" "$ROOT/baselines/ImageFusion/FusionDiff/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/FusionDiff" --checkpoint "$c" --sampling-steps "$SAMPLING_STEPS" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_rd(){ local c="${REDIFFUSE_CKPT:-}"; [[ -f "$c" ]]||{ skip 'REDIFFUSE_CKPT missing'; return; }; "$REDIFFUSE_PYTHON" "$ROOT/baselines/ReDiffuse/prepare_official_bytecode.py"; "$REDIFFUSE_PYTHON" "$ROOT/baselines/ReDiffuse/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/ReDiffuse" --checkpoint "$c" --sampling-steps "$SAMPLING_STEPS" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_zmff(){ "$PYTHON" "$ROOT/baselines/ZMFF/infer_metadata.py" --metadata "$METADATA" --output-dir "$OUTPUT_ROOT/ZMFF" --iterations "$ZMFF_ITERATIONS" --seed "$SEED" --start-index "$START_INDEX" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"; }
run_dsift(){ command -v matlab >/dev/null||{ skip 'MATLAB missing'; return; }; (cd "$ROOT/baselines/DSIFT-MFIF"&&matlab -batch "infer_metadata('metadata','$METADATA','output_dir','$OUTPUT_ROOT/DSIFT','start_index',$START_INDEX,'max_samples',$MAX_SAMPLES)"); }
one(){ case "$1" in dsift)run_dsift;;ifcnn)run_ifcnn;;swinfusion)run_swin;;zmff)run_zmff;;fusiondiff)run_fd;;rediffuse)run_rd;;esac; }
if [[ "${METHOD,,}" == all ]]; then for m in dsift ifcnn swinfusion zmff fusiondiff rediffuse; do one "$m"||failed=$((failed+1)); done; else one "${METHOD,,}"; fi
echo "[DONE] failed=$failed"; [[ "$failed" == 0 ]]
