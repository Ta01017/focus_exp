#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"; PYTHON="${PYTHON:-python3}"
METHOD="${METHOD:-swinfusion}"; TRAIN_META="${TRAIN_META:?Set TRAIN_META}"; VAL_META="${VAL_META:?Set VAL_META}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/train}"; TAG="${TAG:-metadata_rgb}"; GPUS="${GPUS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"; MAX_SAMPLES="${MAX_SAMPLES:--1}"; MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:--1}"; SEED="${SEED:-17}"
INIT_MODE="${INIT_MODE:-scratch}"; INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"; INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-}"
RESUME="${RESUME:-}"; RESUME_DIR="${RESUME_DIR:-}"; VALIDATION_MODE="${VALIDATION_MODE:-loss}"
SAMPLE_VAL_EVERY="${SAMPLE_VAL_EVERY:-10}"; SAMPLE_VAL_COUNT="${SAMPLE_VAL_COUNT:-4}"; REDIFFUSE_PYTHON="${REDIFFUSE_PYTHON:-python3.8}"
[[ -f "$TRAIN_META" && -f "$VAL_META" ]] || { echo '[ERROR] metadata file missing' >&2; exit 2; }
echo "[CONFIG] METHOD=$METHOD TRAIN_META=$TRAIN_META VAL_META=$VAL_META OUTPUT_ROOT=$OUTPUT_ROOT TAG=$TAG GPUS=$GPUS WORKERS=$NUM_WORKERS SEED=$SEED INIT_MODE=$INIT_MODE VALIDATION_MODE=$VALIDATION_MODE"
empty_output() { [[ ! -d "$1" || -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "[ERROR] non-empty output: $1" >&2; return 2; }; }
run_swin() { local out="$OUTPUT_ROOT/SwinFusion/$TAG"; [[ "$INIT_MODE" == resume ]] || empty_output "$out"; local a=(--init-mode "$INIT_MODE" --output-dir "$out"); [[ "$INIT_MODE" != official ]] || a+=(--init-checkpoint-dir "$INIT_CHECKPOINT_DIR"); [[ "$INIT_MODE" != resume ]] || a+=(--resume-dir "$RESUME_DIR"); CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" "$ROOT/baselines/SwinFusion/main_train_swinfusion.py" --opt "$ROOT/baselines/SwinFusion/options/swinir/train_swinfusion_mff.json" --train-metadata "$TRAIN_META" --val-metadata "$VAL_META" --num-workers "$NUM_WORKERS" --max-samples "$MAX_SAMPLES" --max-train-steps "$MAX_TRAIN_STEPS" --seed "$SEED" "${a[@]}"; }
run_diff() { local method="$1" py="$2" repo="$3" out="$OUTPUT_ROOT/$method/$TAG"; empty_output "$out"; mkdir -p "$out"; local a=(); [[ -z "$INIT_CHECKPOINT" ]] || a+=(--init-checkpoint "$INIT_CHECKPOINT"); [[ -z "$RESUME" ]] || a+=(--resume "$RESUME"); (cd "$out" && CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$py" "$repo/train.py" --config "$repo/config.json" --dataset-format metadata --train-metadata "$TRAIN_META" --val-metadata "$VAL_META" --num-workers "$NUM_WORKERS" --max-samples "$MAX_SAMPLES" --max-train-steps "$MAX_TRAIN_STEPS" --seed "$SEED" --validation-mode "$VALIDATION_MODE" --sample-val-every "$SAMPLE_VAL_EVERY" --sample-val-count "$SAMPLE_VAL_COUNT" "${a[@]}"); }
run_rediffuse() { "$REDIFFUSE_PYTHON" "$ROOT/baselines/ReDiffuse/prepare_official_bytecode.py"; run_diff ReDiffuse "$REDIFFUSE_PYTHON" "$ROOT/baselines/ReDiffuse"; }
case "${METHOD,,}" in swinfusion|swin) run_swin;; fusiondiff) run_diff FusionDiff "$PYTHON" "$ROOT/baselines/ImageFusion/FusionDiff";; rediffuse) run_rediffuse;; all) run_swin; run_diff FusionDiff "$PYTHON" "$ROOT/baselines/ImageFusion/FusionDiff"; run_rediffuse;; *) echo '[ERROR] unsupported supervised method' >&2; exit 2;; esac
