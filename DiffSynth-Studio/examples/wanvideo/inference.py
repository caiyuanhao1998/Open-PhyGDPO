"""Generate a video with a Wan base model or the corresponding PhyGDPO weights."""
import argparse
import os
from pathlib import Path
import shlex
import sys

from launch_t2v import COUNTS

STUDIO = Path(__file__).resolve().parents[2]
GPU_COUNTS = {"t2v-1.3b": COUNTS["1.3B"], "fast-14b": (1, 2, 4, 8), "t2v-14b": COUNTS["14B"]}


def gpu_count(model, visible, requested=None):
    supported = GPU_COUNTS[model]
    if requested is None:
        if visible < 1:
            raise ValueError("Video generation requires an NVIDIA CUDA GPU.")
        return max(count for count in supported if count <= visible)
    count = int(requested)
    if count not in supported or count > visible:
        raise ValueError(f"{model} supports {supported} GPUs. Requested {count} with {visible} visible.")
    return count


def t2v_worker(model, weights, count, steps, output, *, reference_small=False):
    import torch

    large = model == "t2v-14b"
    size = "14B" if large else "1.3B"
    prefix = "PHYGDPO_T2V_14B" if large else "PHYGDPO_T2V"
    filename = "step-0001000.safetensors" if large else "epoch-0.safetensors"
    base = os.getenv(f"{prefix}_MODEL_DIR", f"models/Wan-AI/Wan2.1-T2V-{size}")
    options = ["--model_dir", base, "--num_inference_steps", str(steps), "--output_path", str(output)]
    if weights == "phygdpo":
        adapter = os.getenv(f"{prefix}_LORA", f"models/pretrained/Wan2.1-T2V-{size}_PhyGDPO/{filename}")
        options.extend(["--lora_path", adapter])
    h100_reference = large and count == 8 and all("H100" in torch.cuda.get_device_name(i) for i in range(8))
    if h100_reference or (reference_small and not large and count == 4):
        script = STUDIO / f"examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-{size}_tensor_parallel_PhyGDPO.py"
        if weights == "base":
            options.append("--base")
        if h100_reference:
            options.append("--report-progress")
        return ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={count}", str(script), *options]
    if large and count == 1:
        options.append("--cpu-offload")
    mode = "lora" if weights == "phygdpo" else "base"
    return ["-m", "diffsynth.distributed.wan_video_cli", mode, size, "--num-gpus", str(count), *options]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=GPU_COUNTS, required=True)
    parser.add_argument("--weights", choices=("base", "phygdpo"), default="phygdpo")
    parser.add_argument("--num-gpus", type=int, help="Defaults to the largest supported visible GPU count")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path, help="Starting frame for Fast-14B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, help="Defaults to 4 for Fast-14B and 50 for T2V")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Print the worker command without generating a video")
    args = parser.parse_args()
    fast = args.model == "fast-14b"
    if fast != (args.image is not None):
        parser.error("Supply --image for Fast-14B only.")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Starting frame does not exist: {args.image}")
    if not args.prompt.strip() or args.output.suffix.lower() != ".mp4":
        parser.error("Supply a nonempty prompt and an .mp4 output path.")
    steps = args.steps if args.steps is not None else (4 if fast else 50)
    if steps < 1 or min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        parser.error("Steps must be positive and dimensions must be positive multiples of 16.")
    if args.frames < 1 or args.frames % 4 != 1:
        parser.error("Frame count must be 4n + 1.")
    import torch

    try:
        count = gpu_count(args.model, torch.cuda.device_count(), args.num_gpus)
    except ValueError as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(STUDIO), env.get("PYTHONPATH"))))
    output = args.output.resolve()
    if fast:
        env["PHYGDPO_WORLD_SIZE"] = str(count)
        command = [str(Path(__file__).resolve().parent / "fast_i2v_training/generate.py"), "video",
                   "--image", str(args.image.resolve()), "--steps", str(steps), "--output", str(output)]
        if args.weights == "base":
            command.append("--base")
    else:
        command = t2v_worker(args.model, args.weights, count, steps, output, reference_small=True)
    command += ["--prompt", args.prompt, "--seed", str(args.seed), "--width", str(args.width),
                "--height", str(args.height), "--frames", str(args.frames)]
    print((f"PHYGDPO_WORLD_SIZE={count} " if fast else "") + shlex.join([sys.executable, *command]), flush=True)
    if not args.plan:
        os.chdir(STUDIO)
        os.execve(sys.executable, [sys.executable, *command], env)


if __name__ == "__main__":
    main()
