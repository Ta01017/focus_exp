#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880}"
PY="${PY:-/root/miniconda3/envs/p312/bin/python}"
STAMP="${STAMP:-20260831_095326}"
GPU_COMMON="${GPU_COMMON:-1}"
GPU_REAL="${GPU_REAL:-3}"
INCLUDE_EXTRA_REAL="${INCLUDE_EXTRA_REAL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER="${BUILDER:-$SCRIPT_DIR/build_all_region_route_v3_manifests.py}"
EVAL="${EVAL:-$SCRIPT_DIR/region_eval_route_v3.py}"
COLLECT="${COLLECT:-$SCRIPT_DIR/collect_region_route_v3.py}"
COMPARE_ROOT="${COMPARE_ROOT:-$ROOT/focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827}"
OUT_ROOT="${OUT_ROOT:-$ROOT/focus/models/COMPARE_RESULTS_REGION_V3}"

# Optional explicit overrides. Leave empty to use defaults/auto-detection.
CB_REFINED_CACHE="${CB_REFINED_CACHE:-}"
REAL_REFINED_CACHE="${REAL_REFINED_CACHE:-}"
REAL_CONTROL_CACHE="${REAL_CONTROL_CACHE:-}"
REAL_SEVERE_CACHE="${REAL_SEVERE_CACHE:-}"
CB_FULLGEN_CACHE="${CB_FULLGEN_CACHE:-}"
REAL_FULLGEN_CACHE="${REAL_FULLGEN_CACHE:-}"

for required in "$PY" "$BUILDER" "$EVAL" "$COLLECT"; do
    if [[ ! -f "$required" ]]; then
        echo "[FATAL] missing required file: $required" >&2
        exit 1
    fi
done

BUILD_ARGS=(
    --root "$ROOT"
    --stamp "$STAMP"
    --compare-root "$COMPARE_ROOT"
    --output-root "$OUT_ROOT"
)

[[ -n "$CB_REFINED_CACHE" ]] && BUILD_ARGS+=(--cb-refined-cache "$CB_REFINED_CACHE")
[[ -n "$REAL_REFINED_CACHE" ]] && BUILD_ARGS+=(--real-refined-cache "$REAL_REFINED_CACHE")
[[ -n "$CB_FULLGEN_CACHE" ]] && BUILD_ARGS+=(--cb-fullgen-cache "$CB_FULLGEN_CACHE")
[[ -n "$REAL_FULLGEN_CACHE" ]] && BUILD_ARGS+=(--real-fullgen-cache "$REAL_FULLGEN_CACHE")

if [[ "$INCLUDE_EXTRA_REAL" == "1" ]]; then
    BUILD_ARGS+=(--include-extra-real)
    [[ -n "$REAL_CONTROL_CACHE" ]] && BUILD_ARGS+=(--real-control-cache "$REAL_CONTROL_CACHE")
    [[ -n "$REAL_SEVERE_CACHE" ]] && BUILD_ARGS+=(--real-severe-cache "$REAL_SEVERE_CACHE")
fi

echo "[1/3] preflight and build all manifests"
"$PY" "$BUILDER" "${BUILD_ARGS[@]}"

COMMON_METHODS=(
    DSIFT FULX2.0_ORIGIN IFCNN FusionDiff ReDiffuse_ORIGIN SwinFusion ZMFF
    AvgBlend FullGen G_Diagnostic wo_Generation wo_Refiner Ours
)

REAL_METHODS=("${COMMON_METHODS[@]}")
if [[ "$INCLUDE_EXTRA_REAL" == "1" ]]; then
    REAL_METHODS+=(plus5k_Control plus5k_Severe)
fi

launch_dataset() {
    local gpu="$1"
    local dataset="$2"
    shift 2
    local methods=("$@")
    local session="routev3_${dataset}"
    local done_file="$OUT_ROOT/.done_${dataset}"
    local methods_q=""
    printf -v methods_q '%q ' "${methods[@]}"

    tmux kill-session -t "$session" 2>/dev/null || true
    rm -f "$done_file"

    tmux new-session -d -s "$session" bash -lc "
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES='$gpu'
for METHOD in $methods_q; do
    MANIFEST='$OUT_ROOT/manifests/$dataset/'\"\$METHOD\"'/region_manifest_route_v3.csv'
    METRIC_OUT='$OUT_ROOT/metrics/$dataset/'\"\$METHOD\"
    mkdir -p \"\$METRIC_OUT\"
    echo '[RUN] $dataset' \"\$METHOD\"
    '$PY' '$EVAL' \\
        --manifest \"\$MANIFEST\" \\
        --output-dir \"\$METRIC_OUT\" \\
        --device cuda:0 \\
        --lpips-net alex \\
        --route-confidence 0 \\
        --route-sum-tolerance 0.05 \\
        --patch-size 64 \\
        --patch-stride 32 \\
        --g-patch-min-coverage 0.80 \\
        --g-rsr-psnr-margin 0.20 \\
        2>&1 | tee \"\$METRIC_OUT/eval.log\"
done
touch '$done_file'
echo '[DATASET DONE] $dataset'
"
    echo "[LAUNCH] $dataset GPU=$gpu tmux=$session"
}

echo "[2/3] launch evaluations"
launch_dataset "$GPU_COMMON" CommonBlurGeometryVal200 "${COMMON_METHODS[@]}"
launch_dataset "$GPU_REAL" RealMFFAlignedVal110 "${REAL_METHODS[@]}"

COLLECT_SESSION="routev3_collect"
tmux kill-session -t "$COLLECT_SESSION" 2>/dev/null || true
tmux new-session -d -s "$COLLECT_SESSION" bash -lc "
set -Eeuo pipefail
while [[ ! -f '$OUT_ROOT/.done_CommonBlurGeometryVal200' || ! -f '$OUT_ROOT/.done_RealMFFAlignedVal110' ]]; do
    sleep 10
done
ARGS=(--output-root '$OUT_ROOT')
if [[ '$INCLUDE_EXTRA_REAL' == '1' ]]; then ARGS+=(--include-extra-real); fi
'$PY' '$COLLECT' \"\${ARGS[@]}\" 2>&1 | tee '$OUT_ROOT/collect.log'
echo '[ALL DONE]'
"

echo "[3/3] launched collector"
echo "tmux attach -t routev3_CommonBlurGeometryVal200"
echo "tmux attach -t routev3_RealMFFAlignedVal110"
echo "tmux attach -t routev3_collect"
echo "final summary: $OUT_ROOT/REGION_ROUTE_V3_ALL_SUMMARY.csv"
