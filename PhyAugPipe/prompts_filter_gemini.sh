#!/usr/bin/env bash
set -euo pipefail

IDX=${1:?Usage: GOOGLE_API_KEY=... $0 PART_INDEX}
: "${GOOGLE_API_KEY:?Export GOOGLE_API_KEY before running this script}"

python3 prompts_filter_gemini.py \
  --video_dir T2V_data/VIDGEN-1M_unzip/ \
  --prompts_json processed_json/vidgen_filter_deepseek_part_${IDX}.json \
  --instruction_md prompts_filter_gemini.md \
  --out processed_json/vidgen_filter_deepseek_gemini_part_${IDX}.jsonl \
  --retry 5 \
  --model gemini-2.5-pro \
  --filter_prompt "Is the style of this video unrealistic, such as animation, cartoon, anime, pixel art, video game, 3-D game engine, Lego stop-motion, and so on?" \
  --max_num 1000000

python3 jsonl_to_json.py \
  --jsonl_path processed_json/vidgen_filter_deepseek_gemini_part_${IDX}.jsonl \
  --json_path  processed_json/vidgen_filter_deepseek_gemini_part_${IDX}.json
