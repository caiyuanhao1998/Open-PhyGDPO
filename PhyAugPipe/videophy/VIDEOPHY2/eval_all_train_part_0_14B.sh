#!/usr/bin/env bash

# Keep shell options and traps local even when sourced from the README.
(
  set -euo pipefail
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  video_root=${VIDEOPHY_VIDEO_ROOT:-"$script_dir/../../DPO_data/Wan14B"}
  score_root=${VIDEOPHY_SCORE_ROOT:-"$script_dir/../evaluated_score"}
  checkpoint=${VIDEOPHY_CHECKPOINT:-"$script_dir/../../models/videophy_2_auto"}

  score_seed() (
    seed=$1
    output="$score_root/train_14B_seed_$seed"
    videos="$video_root/seed_$seed"
    mkdir -p -- "$output" || return 1
    exec 9>"$output/.scoring.lock" || return 1
    flock -x 9 || return 1
    stage=$(mktemp -d "$output/.scoring.XXXXXXXX") || return 1
    trap 'rm -rf -- "$stage"' EXIT

    pids=()
    for gpu in {0..7}; do
      CUDA_VISIBLE_DEVICES="$gpu" python3 "$script_dir/eval_all_score_ddp_videophy2_trainset_score.py" \
        -G "$gpu" -k "$((gpu + 1))" -N 8 --batch_size 2 \
        --video_dir "$videos" --prompt_dir "$videos" --checkpoint "$checkpoint" --output_dir "$stage" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    if (( failed )); then
      echo "VideoPhy2 worker failed for seed $seed. See the worker output above." >&2
      return 1
    fi
    python3 "$script_dir/eval_all_score_ddp_merge_trainset_score.py" --slice_dir "$stage" \
      --expected-slices 8 --video-dir "$videos" --outfile "$stage/final.csv" || return 1

    # Expose final.csv only after the complete, validated run is ready.
    mv -- "$stage"/slice*.csv "$stage/joint_score.txt" "$output/" || return 1
    mv -- "$stage/final.csv" "$output/final.csv"
  )

  for seed in 3 1 0; do
    score_seed "$seed" || exit 1
  done
)
