#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]; then
  # The official ctypes wheel advertises py3-none, but its internal tag says cp36.
  # Keep its bundled FFmpeg and rebuild only the wheel metadata.
  work=$(mktemp -d)
  trap 'rm -rf -- "$work"' EXIT
  wheel=decord-0.6.0-py3-none-manylinux2010_x86_64.whl
  printf '%s\n' 'decord==0.6.0 --hash=sha256:51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976' |
    python -m pip download --only-binary=:all: --no-deps --require-hashes -r /dev/stdin -d "$work"
  python -m zipfile -e "$work/$wheel" "$work/unpacked"
  sed -i 's/^Tag: cp36-cp36m-/Tag: py3-none-/' "$work/unpacked/decord-0.6.0.dist-info/WHEEL"
  mkdir "$work/fixed"
  python -m wheel pack "$work/unpacked" -d "$work/fixed"
  python -m pip install --no-index --find-links="$work/fixed" --force-reinstall --no-deps decord==0.6.0
fi

python -m pip install -r requirements.txt
