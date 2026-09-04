#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHOD_NAME=Flux2
TRAIN_META="${TRAIN_META:?Set TRAIN_META}"
VAL_META="${VAL_META:?Set VAL_META}"
TAG="${FLUX2_TAG:-flux2_klein_4b_focus_lora}"
TRAIN_OUTPUT="$OUTPUT_ROOT/train/$METHOD_NAME/$TAG"
INFER_OUTPUT="$OUTPUT_ROOT/infer/Flux2"
EVAL_OUTPUT="$OUTPUT_ROOT/eval/$METHOD_NAME/$TAG"
FLUX2_PYTHON="${FLUX2_PYTHON:-/root/miniconda3/envs/p312/bin/python}"
FLUX2_GPUS="${FLUX2_GPUS:-${CUDA_VISIBLE_GPU:-0}}"
IFS=',' read -r -a flux2_gpu_list <<<"$FLUX2_GPUS"
FLUX2_NUM_PROCESSES="${FLUX2_NUM_PROCESSES:-${#flux2_gpu_list[@]}}"
FLUX2_TRAIN_SCRIPT="${FLUX2_TRAIN_SCRIPT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/train_flux2.py}"
FLUX2_INFER_SCRIPT="${FLUX2_INFER_SCRIPT:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/focus/infer_flux2_focus_fusion_batch_4B_4.py}"
FLUX2_EPOCHS="${FLUX2_EPOCHS:-5}"
FLUX2_MAX_ITEMS="${FLUX2_MAX_ITEMS:-}"
FLUX2_MAX_PIXELS="${FLUX2_MAX_PIXELS:-4194304}"
FLUX2_INFERENCE_STEPS="${FLUX2_INFERENCE_STEPS:-4}"
FLUX2_LORA="${FLUX2_LORA:-}"

[[ "${SMOKE:-0}" != 1 ]] || { FLUX2_EPOCHS=1; FLUX2_MAX_ITEMS=1; FLUX2_MAX_PIXELS=262144; }
require_file "$FLUX2_TRAIN_SCRIPT" FLUX2_TRAIN_SCRIPT
require_file "$FLUX2_INFER_SCRIPT" FLUX2_INFER_SCRIPT

if [[ "$RUN_TRAIN" == 1 ]]; then
  mkdir -p "$TRAIN_OUTPUT"
  extra=(); [[ -z "$FLUX2_MAX_ITEMS" ]] || extra+=(--max_data_items "$FLUX2_MAX_ITEMS")
  lora_targets='to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj'
  for block in {0..19}; do lora_targets+=",single_transformer_blocks.$block.attn.to_out"; done
  CUDA_VISIBLE_DEVICES="$FLUX2_GPUS" "$FLUX2_PYTHON" -m accelerate.commands.launch \
    --num_processes "$FLUX2_NUM_PROCESSES" --num_machines 1 --mixed_precision bf16 \
    "$FLUX2_TRAIN_SCRIPT" \
    --dataset_base_path "$(dirname "$TRAIN_META")" --dataset_metadata_path "$TRAIN_META" \
    --data_file_keys image,edit_image --extra_inputs edit_image \
    --max_pixels "$FLUX2_MAX_PIXELS" --dataset_repeat 10 \
    --model_id_with_origin_paths 'black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors' \
    --tokenizer_path 'black-forest-labs/FLUX.2-klein-4B:tokenizer/' \
    --learning_rate 1e-4 --num_epochs "$FLUX2_EPOCHS" \
    --remove_prefix_in_ckpt pipe.dit. --output_path "$TRAIN_OUTPUT" \
    --lora_base_model dit --lora_target_modules "$lora_targets" \
    --lora_rank 32 --use_gradient_checkpointing --dataset_num_workers "${NUM_WORKERS:-8}" \
    --find_unused_parameters "${extra[@]}"
fi

if [[ -z "$FLUX2_LORA" ]]; then
  FLUX2_LORA="$(find "$TRAIN_OUTPUT" -maxdepth 1 -name 'epoch-*.safetensors' -type f -printf '%p\n' | sort -V | tail -1)"
fi
require_file "$FLUX2_LORA" FLUX2_LORA

if [[ "$RUN_INFER" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_GPU" "$FLUX2_PYTHON" "$ROOT/baselines/Flux2/infer_metadata.py" \
    --metadata "$VAL_META" --output-dir "$INFER_OUTPUT" --external-script "$FLUX2_INFER_SCRIPT" \
    --lora-path "$FLUX2_LORA" --device cuda --dtype bf16 \
    --num-inference-steps "$FLUX2_INFERENCE_STEPS" --max-pixels "$FLUX2_MAX_PIXELS" \
    --seed "$SEED" --max-samples "$MAX_SAMPLES" --overwrite "$OVERWRITE"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  run_eval_for_specs "$METHOD_NAME" "$OUTPUT_ROOT/infer" "$EVAL_OUTPUT" "$EVAL_SPECS"
fi
