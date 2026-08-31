torchrun --standalone --nproc_per_node=8 examples/wanvideo/wan_14B_tensor_parallel.py

torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-14B_tensor_parallel_PhyGDPO.py --lora_path models/pretrained/Wan2.1-T2V-14B_PhyGDPO/step-0001000.safetensors