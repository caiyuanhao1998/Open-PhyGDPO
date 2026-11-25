#!/usr/bin/env bash
set -euo pipefail

studio=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python -m pip install \
  -r "$studio/examples/wanvideo/fast_i2v_training/requirements.txt" \
  "torchvision==0.28.0" \
  "gradio==6.17.3" \
  "xfuser==0.4.5" \
  "lightning==2.6.5" \
  "sentence-transformers==3.4.1"

# pip install vllm
# pip install -q -U google-genai
# pip install openai
