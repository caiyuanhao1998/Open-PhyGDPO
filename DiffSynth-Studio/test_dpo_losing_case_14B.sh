# Seed_0
for i in $(seq 0 15); do
  torchrun --standalone --nproc_per_node=8 examples/wanvideo/wan_14B_tensor_parallel_DPO_losing_case.py \
    --prompt_json_file json_file/vidgen_filter_gemini_part_0_dpo_${i}.json \
    --output_dir ../PhyAugPipe/DPO_data/Wan14B/seed_0 \
    --seed 0
done

# Seed_1
for i in $(seq 0 15); do
  torchrun --standalone --nproc_per_node=8 examples/wanvideo/wan_14B_tensor_parallel_DPO_losing_case.py \
    --prompt_json_file json_file/vidgen_filter_gemini_part_0_dpo_${i}.json \
    --output_dir ../PhyAugPipe/DPO_data/Wan14B/seed_0 \
    --seed 1
done

# Seed_3
for i in $(seq 0 15); do
  torchrun --standalone --nproc_per_node=8 examples/wanvideo/wan_14B_tensor_parallel_DPO_losing_case.py \
    --prompt_json_file json_file/vidgen_filter_gemini_part_0_dpo_${i}.json \
    --output_dir ../PhyAugPipe/DPO_data/Wan14B/seed_0 \
    --seed 3
done