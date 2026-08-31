#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-../models/videophy_2_auto}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES="$GPU" python3 inference.py --input_csv examples/sa_pc.csv --checkpoint "$CHECKPOINT" --output_csv examples/output_sa.csv --task sa
CUDA_VISIBLE_DEVICES="$GPU" python3 inference.py --input_csv examples/sa_pc.csv --checkpoint "$CHECKPOINT" --output_csv examples/output_pc.csv --task pc
CUDA_VISIBLE_DEVICES="$GPU" python3 inference.py --input_csv examples/rule.csv --checkpoint "$CHECKPOINT" --output_csv examples/output_rule.csv --task rule
CUDA_VISIBLE_DEVICES="$GPU" python3 eval_all_score.py --input_csv examples/sa_pc.csv --checkpoint "$CHECKPOINT" --output_csv examples/eval_all_score.csv
