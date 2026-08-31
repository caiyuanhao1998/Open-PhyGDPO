#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  "xfuser==0.4.5" \
  "lightning==2.6.5" \
  "peft==0.20.0" \
  "sentence-transformers==3.4.1"

# pip install vllm
# pip install -q -U google-genai
# pip install openai
