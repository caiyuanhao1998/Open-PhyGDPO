#!/usr/bin/env bash
set -euo pipefail
studio=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PHYGDPO_RUN_DIR="${PHYGDPO_RUN_DIR:-$studio/runs/wan-fast}"
exec python "$studio/examples/wanvideo/fast_i2v_training/generate.py" negatives --seed-manifest "$@"
