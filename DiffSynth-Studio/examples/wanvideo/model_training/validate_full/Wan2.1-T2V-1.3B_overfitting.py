import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
import os

# 写一个解析器，输入一个 model_path，一个 output_folder
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
args = parser.parse_args()


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_gemini_physics_dpo_v7_minuslose_overfitting/step-0000100.safetensors")
# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_gemini_physics_dpo_v7_overfitting/step-0000100.safetensors")
# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_gemini_physics_dpo_v7_nolose_overfitting/step-0000100.safetensors")
# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_gemini_physics_dpo_v7_parallel_minuslose_overfitting/step-0000100.safetensors")
# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_gemini_physics_dpo_v7_parallel_overfitting/step-0000100.safetensors")
state_dict = load_state_dict(args.model_path)
pipe.dit.load_state_dict(state_dict)
pipe.enable_vram_management()

os.makedirs(args.output_folder, exist_ok=True)

video_1 = pipe(
    prompt="A lively kitten sprinting swiftly across a lush green lawn, in a vivid documentary-style shot. High quality, ultrarealistic detail and breath-taking movie-like camera capture.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1, tiled=True
)
save_video(video_1, os.path.join(args.output_folder, "cat_lawn.mp4"), fps=15, quality=5)

video_2 = pipe(
    prompt="A lively puppy sprinting swiftly across a lush green lawn, in a vivid documentary-style shot. High quality, ultrarealistic detail and breath-taking movie-like camera capture.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1, tiled=True
)
save_video(video_2, os.path.join(args.output_folder, "dog_lawn.mp4"), fps=15, quality=5)

video_3 = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1, tiled=True
)
save_video(video_3, os.path.join(args.output_folder, "video_Wan2.1-T2V-1.3B.mp4"), fps=15, quality=5)
