#!/usr/bin/env bash
set -euo pipefail
studio=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python "$studio/examples/wanvideo/launch_t2v.py" negative 14B "$@"
