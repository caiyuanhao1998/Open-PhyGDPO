# export PATH=$HOME/.local/bin:$PATH
  # accelerate launch \
  # --num_processes 8 \
  # --num_machines 1 \
  # --mixed_precision bf16 \
  # examples/wanvideo/model_training/train.py \
  # --dataset_base_path data/example_video_dataset \
  # --dataset_metadata_path data/example_video_dataset/metadata.csv \
  # --height 480 \
  # --width 832 \
  # --dataset_repeat 100 \
  # --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  # --learning_rate 1e-4 \
  # --num_epochs 5 \
  # --remove_prefix_in_ckpt "pipe.dit." \
  # --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  # --lora_base_model "dit" \
  # --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  # --lora_rank 32 \
  # --skip_download

# 单卡调试
  # CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 examples/wanvideo/model_training/train.py \
  # --dataset_base_path data/example_video_dataset \
  # --height 480 \
  # --width 832 \
  # --dataset_repeat 100 \
  # --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  # --learning_rate 1e-4 \
  # --num_epochs 5 \
  # --remove_prefix_in_ckpt "pipe.dit." \
  # --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  # --lora_base_model "dit" \
  # --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  # --lora_rank 32 \
  # --skip_download \
  # --dataset_metadata_path data/example_video_dataset/metadata.csv


# 多卡 physics training lora - extended prompt - gemini
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train_physics.py \
  --dataset_base_path /data/home/yuanhaoc/T2V_data/VIDGEN-1M_unzip/ \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_physics_gemini_extend" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --skip_download \
  --extend \
  --dataset_metadata_path train_json/vidgen_filter_gemini_part_0.json



# 多卡 physics training lora - not extended prompt - gemini
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train_physics.py \
  --dataset_base_path /data/home/yuanhaoc/T2V_data/VIDGEN-1M_unzip/ \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_physics_gemini" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --skip_download \
  --dataset_metadata_path train_json/vidgen_filter_gemini_part_0.json


# 多卡 physics training lora - extended prompt - qwen
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train_physics.py \
  --dataset_base_path /data/home/yuanhaoc/T2V_data/VIDGEN-1M_unzip/ \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_physics_qwen_extend" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --skip_download \
  --extend \
  --dataset_metadata_path train_json/vidgen_filter_deepseek_qwen_part_0.json



# 多卡 physics training lora - not extended prompt
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train_physics.py \
  --dataset_base_path /data/home/yuanhaoc/T2V_data/VIDGEN-1M_unzip/ \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_physics_qwen" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --skip_download \
  --dataset_metadata_path train_json/vidgen_filter_deepseek_qwen_part_0.json