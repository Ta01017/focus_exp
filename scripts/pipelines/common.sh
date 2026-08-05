#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/pipeline}"
CUDA_VISIBLE_GPU="${CUDA_VISIBLE_GPU:-0}"
SEED="${SEED:-17}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
START_INDEX="${START_INDEX:-0}"
OVERWRITE="${OVERWRITE:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-2000}"
ZMFF_ITERATIONS="${ZMFF_ITERATIONS:-1300}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SPECS="${EVAL_SPECS:-}"
EVAL_METRICS_GT="${EVAL_METRICS_GT:-gt_main}"
EVAL_METRICS_NO_GT="${EVAL_METRICS_NO_GT:-no_gt_main}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

require_file() {
  local path="$1" label="$2"
  [[ -f "$path" ]] || { echo "[ERROR] $label not found: $path" >&2; exit 2; }
}

require_cmd() {
  command -v "$1" >/dev/null || { echo "[ERROR] command not found: $1" >&2; exit 2; }
}

run_eval_for_specs() {
  local method="$1"
  local infer_root="$2"
  local eval_root="$3"
  local specs="$4"
  [[ -n "$specs" ]] || { echo "[EVAL] EVAL_SPECS empty; skip evaluation"; return 0; }
  mkdir -p "$eval_root/manifests" "$eval_root/results"
  local merged=()
  local IFS=$'\n'
  for spec in $specs; do
    [[ -n "$spec" ]] || continue
    IFS='|' read -r dataset mode metadata out_subdir metrics <<<"$spec"
    [[ -n "$dataset" && -n "$mode" && -n "$out_subdir" ]] || {
      echo "[ERROR] Bad EVAL_SPECS row: $spec" >&2
      exit 2
    }
    if [[ -z "${metrics:-}" ]]; then
      if [[ "$mode" == gt ]]; then metrics="$EVAL_METRICS_GT"; else metrics="$EVAL_METRICS_NO_GT"; fi
    fi
    local inference_manifest="$infer_root/$out_subdir/inference_manifest.csv"
    local eval_manifest="$eval_root/manifests/${dataset}_${method}.csv"
    require_file "$inference_manifest" "inference manifest for $dataset"
    "$PYTHON" "$ROOT/mfif_eval_toolkit/scripts/manifest_from_inference.py" \
      --inference-manifest "$inference_manifest" \
      --dataset "$dataset" \
      --mode "$mode" \
      --method "$method" \
      --output "$eval_manifest"
    "$PYTHON" "$ROOT/mfif_eval_toolkit/evaluate.py" \
      --manifest "$eval_manifest" \
      --metrics "$metrics" \
      --output-dir "$eval_root/results/$dataset" \
      $EVAL_EXTRA_ARGS
    merged+=("$eval_manifest")
  done
  if [[ "${#merged[@]}" -gt 1 ]]; then
    "$PYTHON" "$ROOT/mfif_eval_toolkit/scripts/merge_manifests.py" \
      "${merged[@]}" \
      --output "$eval_root/manifests/all_${method}.csv"
  fi
}
