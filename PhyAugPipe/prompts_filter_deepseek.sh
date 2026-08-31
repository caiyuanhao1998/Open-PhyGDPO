#!/bin/bash

IDX=$1

python3 prompt_filter_deepseek.py \
  --prompts_json T2V_data/VIDGEN-1M/VidGen_1M_video_caption_${IDX}.json \
  --md_file     prompt_filter_deepseek.md \
  --model_dir   models/DeepSeek-R1-Distill-Llama-70B/ \
  --output_folder processed_json \
  --out         processed_json/vidgen_filter_deepseek_part_${IDX}.jsonl \
  --max_len     1000000

python3 jsonl_to_json.py \
  --jsonl_path processed_json/vidgen_filter_deepseek_part_${IDX}.jsonl \
  --json_path  processed_json/vidgen_filter_deepseek_part_${IDX}.json