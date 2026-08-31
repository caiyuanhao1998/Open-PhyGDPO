# 1. evaluating gemini_part_0_14B - seed_3
OUTPUT_DIR="../evaluated_score/train_14B_seed_3"

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python3 eval_all_score_ddp_videophy2_trainset_score.py \
      -G $i -k $((i+1)) -N 8 \
      --video_dir  ../../DPO_data/Wan14B/seed_3 \
      --prompt_dir ../../DPO_data/Wan14B/seed_3 \
      --checkpoint ../../models/videophy_2_auto \
      --output_dir "$OUTPUT_DIR" \
      --batch_size 2 &
done
wait

# 合并切片
python3 eval_all_score_ddp_merge_trainset_score.py \
        --slice_dir "$OUTPUT_DIR" \
        --outfile   "$OUTPUT_DIR/final.csv"



# 2. evaluating gemini_part_0_14B - seed_1
OUTPUT_DIR="../evaluated_score/train_14B_seed_1"

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python3 eval_all_score_ddp_videophy2_trainset_score.py \
      -G $i -k $((i+1)) -N 8 \
      --video_dir  ../../DPO_data/Wan14B/seed_1 \
      --prompt_dir ../../DPO_data/Wan14B/seed_1 \
      --checkpoint ../../models/videophy_2_auto \
      --output_dir "$OUTPUT_DIR" \
      --batch_size 2 &
done
wait

# 合并切片
python3 eval_all_score_ddp_merge_trainset_score.py \
        --slice_dir "$OUTPUT_DIR" \
        --outfile   "$OUTPUT_DIR/final.csv"



# 3. evaluating gemini_part_0_14B - seed_0
OUTPUT_DIR="../evaluated_score/train_14B_seed_0"

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python3 eval_all_score_ddp_videophy2_trainset_score.py \
      -G $i -k $((i+1)) -N 8 \
      --video_dir  ../../DPO_data/Wan14B/seed_0 \
      --prompt_dir ../../DPO_data/Wan14B/seed_0 \
      --checkpoint ../../models/videophy_2_auto \
      --output_dir "$OUTPUT_DIR" \
      --batch_size 2 &
done
wait

# 合并切片
python3 eval_all_score_ddp_merge_trainset_score.py \
        --slice_dir "$OUTPUT_DIR" \
        --outfile   "$OUTPUT_DIR/final.csv"
