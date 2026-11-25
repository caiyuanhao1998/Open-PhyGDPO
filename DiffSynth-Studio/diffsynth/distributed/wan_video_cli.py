import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm

from .wan_video_outputs import denoising_progress, generation_record, publish, skip_on_all_ranks
from .wan_video import (
    EXAMPLE_PROMPTS, GPU_COUNTS_BY_MODEL, NEGATIVE_PROMPT, ParallelWanVideo,
    WanVideoCheckpoint, initialize_cfg_groups,
)


def generate_videos(checkpoint, jobs, steps, seed, num_gpus=None, width=832, height=480, num_frames=81):
    worker_count = int(os.environ["LOCAL_WORLD_SIZE"])
    if worker_count not in GPU_COUNTS_BY_MODEL[checkpoint.model_size]:
        raise ValueError(f"Unsupported worker count {worker_count} for {checkpoint.model_size}")
    if num_gpus is not None and num_gpus != worker_count:
        raise ValueError(f"Requested {num_gpus} GPUs but torchrun launched {worker_count} workers")
    if worker_count > torch.cuda.device_count():
        raise ValueError(f"{worker_count} workers exceed the number of visible GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        pipeline = None
        for prompt, output, skip_existing in jobs:
            task = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, seed=seed, num_inference_steps=steps, tiled=False,
                        width=width, height=height, num_frames=num_frames)
            record = (generation_record(checkpoint.model_size, task, "sharded",
                                        directory=checkpoint.dit[0].parent, adapter=checkpoint.lora)
                      if dist.get_rank() == 0 else None)
            if skip_existing and skip_on_all_ranks(output, record, device):
                continue
            if pipeline is None:
                pipeline = ParallelWanVideo(checkpoint, device, initialize_cfg_groups(checkpoint.model_size))
            pipeline.pipe.dit.to(device)
            with tqdm(total=steps, desc="Denoising", disable=dist.get_rank() != 0) as bar:
                def progress(step, stage):
                    bar.update(step - bar.n)
                    bar.set_description_str(stage)

                frames = pipeline.generate(prompt, NEGATIVE_PROMPT, width, height, num_frames, steps, seed, progress)
            if dist.get_rank() == 0:
                publish(frames, output, record, caption=skip_existing)
    finally:
        dist.destroy_process_group()



@torch.inference_mode()
def generate_offloaded(checkpoint, prompt, output, args):
    from .. import ModelManager, WanVideoPipeline

    large = checkpoint.model_size == "14B"
    task = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, seed=args.seed, width=args.width,
                height=args.height, num_frames=args.frames, num_inference_steps=args.num_inference_steps, tiled=large)
    record = generation_record(checkpoint.model_size, task, "cpu-offload" if large else "native",
                               directory=checkpoint.dit[0].parent, adapter=checkpoint.lora)
    manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)
    dit_files = list(map(str, checkpoint.dit)) if large else str(checkpoint.dit[0])
    manager.load_models([dit_files, str(checkpoint.text_encoder), str(checkpoint.vae)])
    if checkpoint.lora:
        if large:
            manager.load_lora(str(checkpoint.lora), lora_alpha=1.0)
        else:
            # Match the Space's BF16 LoRA merge on CPU before moving the DiT to CUDA.
            from ..models.lora import GeneralLoRAFromPeft
            from ..models.utils import load_state_dict

            dit = manager.fetch_model("wan_video_dit")
            adapter = load_state_dict(str(checkpoint.lora))
            loader = GeneralLoRAFromPeft()
            pairs = loader.get_name_dict(adapter)
            if not pairs or loader.match(dit, adapter) is None or any(down not in adapter for _, down in pairs.values()):
                raise ValueError(f"LoRA checkpoint {checkpoint.lora} does not match the 1.3B DiT.")
            for name, (up, down) in pairs.items():
                dit.get_parameter(name).add_(adapter[up].bfloat16() @ adapter[down].bfloat16())
    pipe = WanVideoPipeline.from_model_manager(manager, device="cuda", torch_dtype=torch.bfloat16)
    if large:
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
    else:
        pipe.enable_cpu_offload()
    frames = pipe(**task, progress_bar_cmd=denoising_progress)
    publish(frames, output, record)


def main():
    parser = argparse.ArgumentParser(description="Generate a BF16 video with the low-memory Wan T2V backend.")
    parser.add_argument("mode", choices=("base", "lora", "negative"))
    parser.add_argument("size", choices=tuple(GPU_COUNTS_BY_MODEL))
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Run the native pipeline on one GPU with CPU offload (layer-wise with tiled VAE for 14B)")
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--lora_path", type=Path)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt")
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--output_folder", type=Path, default=Path("."))
    parser.add_argument("--prompt_json_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--use_extended", action="store_true")
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    size, mode = args.size, args.mode
    counts = GPU_COUNTS_BY_MODEL[size]
    if args.cpu_offload:
        if mode == "negative" or args.num_gpus not in (None, 1) or int(os.getenv("WORLD_SIZE", "1")) != 1:
            parser.error("--cpu-offload supports a single-GPU base or LoRA example")
        args.num_gpus = 1
    if args.num_gpus is not None and args.num_gpus not in counts:
        parser.error(f"{size} supports {counts} GPUs")
    if min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        parser.error("width and height must be positive multiples of 16")
    if args.frames < 1 or args.frames % 4 != 1:
        parser.error("frame count must be 4n + 1")
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive")
    if mode == "negative" and (not args.prompt_json_file or not args.output_dir):
        parser.error("negative requires --prompt_json_file and --output_dir")
    if mode != "lora" and args.lora_path:
        parser.error("--lora_path applies to lora mode")
    if "LOCAL_RANK" not in os.environ:
        count = args.num_gpus or counts[-1]
        if count > torch.cuda.device_count():
            parser.error(f"{count} GPUs requested, {torch.cuda.device_count()} visible")
        os.execv(sys.executable, [sys.executable, "-m", "torch.distributed.run", "--standalone",
                 f"--nproc-per-node={count}", "--module", "diffsynth.distributed.wan_video_cli", *sys.argv[1:]])

    adapter = args.lora_path
    if mode == "negative":
        entries = json.loads(args.prompt_json_file.read_text(encoding="utf-8"))[args.bias:]
        if args.reverse:
            entries.reverse()
        field = "extended" if args.use_extended else "original"
        jobs = [(entry[field], args.output_dir / f"{entry['vid']}.mp4", True) for entry in entries]
    else:
        suffix = "_PhyGDPO" if mode == "lora" else ""
        sport = "weightlifting" if size == "1.3B" else ("gymnastics" if suffix else "gymnastic")
        if mode == "lora" and adapter is None:
            filename = "epoch-0.safetensors" if size == "1.3B" else "step-0001000.safetensors"
            adapter = Path(f"models/pretrained/Wan2.1-T2V-{size}_PhyGDPO") / filename
        output = args.output_path or args.output_folder / f"{sport}_{size}{suffix}.mp4"
        jobs = [(args.prompt if args.prompt is not None else EXAMPLE_PROMPTS[size], output, False)]
    model_dir = args.model_dir or Path(f"models/Wan-AI/Wan2.1-T2V-{size}")
    checkpoint = WanVideoCheckpoint.from_directory(model_dir, size, adapter)
    if args.cpu_offload:
        generate_offloaded(checkpoint, jobs[0][0], jobs[0][1], args)
        return
    generate_videos(checkpoint, jobs, args.num_inference_steps, args.seed, args.num_gpus, args.width, args.height, args.frames)


if __name__ == "__main__":
    main()
