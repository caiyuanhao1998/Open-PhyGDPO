#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
data_dir=${PHYGDPO_DATA_DIR:-data}
if [[ -z "${PHYGDPO_DATA_DIR:-}" && ! -d data && -d ../PhyAugPipe/T2V_data/VIDGEN-1M_unzip ]]; then
  data_dir=../PhyAugPipe
fi
score_dir="manifests/original-14b/train_part_0_14B_seed_0/final.csv"
default_score="../PhyAugPipe/videophy/evaluated_score/train_part_0_14B_seed_0/final.csv"
if [[ -f "$data_dir/$score_dir" ]]; then default_score="$data_dir/$score_dir"; fi
score_csv=${PHYGDPO_SCORE_CSV-$default_score}
pairs_csv=${PHYGDPO_PAIRS_CSV:-}
if [[ -z "$score_csv" && -z "$pairs_csv" && -f "$data_dir/manifests/original-14b-unscored/pairs.csv" ]]; then
  pairs_csv="$data_dir/manifests/original-14b-unscored/pairs.csv"
fi
pair_args=()
if [[ -n "$pairs_csv" ]]; then pair_args=(--dataset_pairs_manifest "$pairs_csv" --dataset_pairs_root "$data_dir"); fi

accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train_PhyGDPO.py \
  --dataset_base_path "$data_dir/T2V_data/VIDGEN-1M_unzip" \
  --height 480 \
  --width 832 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --total_training_steps 3000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "models/train/Wan2.1-T2V-14B_PhyGDPO" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 48 \
  --skip_download \
  --dataset_base_path_positive "$data_dir/T2V_data/VIDGEN-1M_unzip" \
  --dataset_base_path_negative "$data_dir/DPO_data/Wan14B" \
  --save_every_steps 500 \
  --log_every_steps 10 \
  --gradient_accumulation_steps 1 \
  --dpo_beta 1000 \
  --use_gradient_checkpointing_offload \
  --dataset_metadata_path "$score_csv" \
  --alpha_min 0.5 \
  --k_alpha 5.0 \
  --b_alpha 0.5 \
  --k_gamma 3.0 \
  --b_gamma 0.5 \
  --lambda_gamma 0.5 \
  "${pair_args[@]}" "$@"
