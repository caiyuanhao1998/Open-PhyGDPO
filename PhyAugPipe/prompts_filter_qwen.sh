#!/bin/bash

IDX=$1

python3 prompts_filter_qwen.py \
  --prompts_json processed_json/vidgen_filter_deepseek_part_${IDX}.json \
  --cot_md prompts_filter_qwen.md \
  --video_dir T2V_data/VIDGEN-1M_unzip/ \
  --model_dir models/Qwen2.5-VL-72B-Instruct \
  --output_folder processed_json \
  --out processed_json/vidgen_filter_deepseek_qwen_part_${IDX}.jsonl \
  --tp 8 \
  --dtype bfloat16 \
  --batch_size 1 \
  --num_frames 4 \
  --physics_score_threshold 0.7


python3 jsonl_to_json.py \
  --jsonl_path processed_json/vidgen_filter_deepseek_qwen_part_${IDX}.jsonl \
  --json_path  processed_json/vidgen_filter_deepseek_qwen_part_${IDX}.json