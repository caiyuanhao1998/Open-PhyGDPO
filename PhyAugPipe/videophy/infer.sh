#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage="Usage: bash infer.sh CHECKPOINT SA_PC_CSV OUTPUT_DIR [RULE_CSV]"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    printf '%s\n' "$usage"
    exit 0
fi
if (( $# > 4 )); then
    printf '%s\n' "$usage" >&2
    exit 2
fi

checkpoint=${1:-"$script_dir/../models/videophy_2_auto"}
input_csv=${2:-"$script_dir/examples/sa_pc.csv"}
output_dir=${3:-"$script_dir/examples"}
rule_csv=${4:-"$script_dir/examples/rule.csv"}

if [[ ! -f "$input_csv" ]]; then
    printf 'SA/PC input CSV not found: %s\n%s\n' "$input_csv" "$usage" >&2
    exit 2
fi
if (( $# >= 4 )) && [[ ! -f "$rule_csv" ]]; then
    printf 'Rule input CSV not found: %s\n' "$rule_csv" >&2
    exit 2
fi

if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
num_gpus=${PHYGDPO_WORLD_SIZE:-$(python3 -c 'import torch; print(min(8, torch.cuda.device_count()))')}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    printf 'VideoPhy2 requires at least one visible CUDA GPU.\n' >&2
    exit 2
fi
launch=(python3 -m torch.distributed.run --standalone --nproc-per-node="$num_gpus")
"${launch[@]}" "$script_dir/VIDEOPHY2/inference.py" --input_csv "$input_csv" \
    --checkpoint "$checkpoint" --output_csv "$output_dir/eval_all_score.csv" \
    --task joint --task_output_dir "$output_dir"

if [[ -f "$rule_csv" ]]; then
    "${launch[@]}" "$script_dir/VIDEOPHY2/inference.py" --input_csv "$rule_csv" \
        --checkpoint "$checkpoint" --output_csv "$output_dir/output_rule.csv" --task rule
fi
