#!/usr/bin/env bash
set -euo pipefail
studio=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recipe="$studio/examples/wanvideo/fast_i2v_training"

options=()
while (( $# )); do
  case "$1" in
    --negative-source)
      case "${2:-}" in
        released|generated) export PHYGDPO_NEGATIVE_SOURCE="$2" ;;
        *) echo "Use --negative-source released or generated." >&2; exit 2 ;;
      esac
      shift 2 ;;
    *) options+=("$1"); shift ;;
  esac
done

# The DPO stage starts from the step-8 adapter produced by this SFT run.
unset PHYGDPO_INIT_ADAPTER
python "$recipe/run.py" sft --full-epoch --skip-eval --require-pairs "${options[@]}"
python "$recipe/run.py" dpo --skip-eval "${options[@]}"
