#!/usr/bin/env bash
set -Eeuo pipefail


ROOT=/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880

PY=/root/miniconda3/envs/p312/bin/python

REGION_EVAL=$ROOT/focus/pixrestore_mfif_paper_suite_v7_20260831/tools/region_eval_v2.py


BASE=$ROOT/focus/models/COMPARE_RESULTS_REGION_V2


GPU=${GPU:-0}


run(){

DATASET=$1
METHOD=$2
GPU=$3


MANIFEST=$BASE/manifests/$DATASET/$METHOD/region_manifest_v2.csv

OUT=$BASE/results/$DATASET/$METHOD


if [ ! -f "$MANIFEST" ];then
    echo "skip missing $MANIFEST"
    return
fi


echo "RUN $DATASET $METHOD GPU=$GPU"


CUDA_VISIBLE_DEVICES=$GPU \
$PY $REGION_EVAL \
 --manifest "$MANIFEST" \
 --output-dir "$OUT" \
 --device cuda:0 \
 --lpips-net alex \
 --sharp-threshold 0.70 \
 --blur-threshold 0.30 \
 --patch-size 64 \
 --patch-stride 32 \
 --g-patch-min-coverage 0.80 \
 --g-rsr-psnr-margin 0.20 \
 > "$OUT/eval.log" 2>&1


}


TASKS=(

"CommonBlurGeometryVal200 DSIFT"
"CommonBlurGeometryVal200 FULX2.0_ORIGIN"
"CommonBlurGeometryVal200 IFCNN"
"CommonBlurGeometryVal200 FusionDiff"
"CommonBlurGeometryVal200 ReDiffuse_ORIGIN"
"CommonBlurGeometryVal200 SwinFusion"
"CommonBlurGeometryVal200 ZMFF"


"RealMFFAlignedVal110 DSIFT"
"RealMFFAlignedVal110 FULX2.0_ORIGIN"
"RealMFFAlignedVal110 IFCNN"
"RealMFFAlignedVal110 FusionDiff"
"RealMFFAlignedVal110 ReDiffuse_ORIGIN"
"RealMFFAlignedVal110 SwinFusion"
"RealMFFAlignedVal110 ZMFF"

)



GPU=0

for t in "${TASKS[@]}"
do

set -- $t

run "$1" "$2" "$GPU" &

GPU=$((GPU+1))

if [ $GPU -eq 4 ];then
    wait
    GPU=0
fi

done


wait

echo DONE