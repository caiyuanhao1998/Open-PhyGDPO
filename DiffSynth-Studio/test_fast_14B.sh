#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# > 0 )) && [[ $# != 1 || $1 != --plan ]]; then
  echo "Usage: bash test_fast_14B.sh [--plan]" >&2
  exit 2
fi

recipe=examples/wanvideo/fast_i2v_training
scene=(
  --image "$recipe/evaluation/inputs/gymnastics.png"
  --prompt "A gymnast swings forward on the parallel bars, releases both hands, and lands on both feet on the mat."
  --seed 0
)

python "$recipe/generate.py" video "${scene[@]}" --base --output outputs/gymnastics_fast_14b_base.mp4 "$@"
python "$recipe/generate.py" video "${scene[@]}" --output outputs/gymnastics_fast_14b.mp4 "$@"
