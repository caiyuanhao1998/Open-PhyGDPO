"""Run the original Wan T2V examples or generate seed-separated losing cases."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


STUDIO = Path(__file__).resolve().parents[2]
EXAMPLES = Path("examples/wanvideo")
COUNTS = {"1.3B": (1, 2, 3, 4), "14B": (1, 2, 3, 4, 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("examples", "negative"))
    parser.add_argument("size", choices=COUNTS)
    parser.add_argument("--num-gpus", type=int, default=os.getenv("PHYGDPO_WORLD_SIZE"))
    parser.add_argument(
        "--backend", choices=("auto", "native", "sharded"), default=os.getenv("PHYGDPO_BACKEND", "auto"),
        help="Auto uses sharded loading for three GPUs or for multi-GPU 14B negative generation. "
             "Sharded loading slices DiT weights before GPU transfer. Native uses the original tensor-parallel scripts.",
    )
    parser.add_argument("--prompt-json", type=Path, action="append", help="One prompt shard; repeat to use several")
    parser.add_argument("--prompt-dir", type=Path, default=os.getenv("PHYGDPO_PROMPT_DIR"))
    data_root = STUDIO / "data"
    legacy_root = STUDIO.parent / "PhyAugPipe"
    if not data_root.is_dir() and (legacy_root / "T2V_data/VIDGEN-1M_unzip").is_dir():
        data_root = legacy_root
    parser.add_argument("--data-dir", type=Path, default=os.getenv("PHYGDPO_DATA_DIR") or data_root)
    parser.add_argument("--plan", action="store_true", help="Print the exact commands without reserving GPUs")
    args = parser.parse_args()
    size = args.size
    n = args.num_gpus if args.num_gpus is not None else COUNTS[size][-1]
    if n not in COUNTS[size]:
        parser.error(f"{size} supports {COUNTS[size]} GPUs")
    backend = args.backend
    if backend == "auto":
        # Keep the original native kernels for the eight-GPU 14B reference example.
        shard_negatives = args.mode == "negative" and size == "14B" and n > 1
        backend = "sharded" if n == 3 or shard_negatives else "native"
    if backend == "native" and n == 3:
        parser.error("Three GPUs require --backend sharded")
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(STUDIO), env.get("PYTHONPATH"))))
    data = args.data_dir.resolve()

    def run(mode, script, *options):
        entry = ["--module", "diffsynth.distributed.wan_video_cli", mode, size] if backend == "sharded" else [script]
        cmd = [str(x) for x in [*launch, *entry, *options]]
        print(shlex.join(cmd), flush=True)
        if not args.plan:
            subprocess.run(cmd, cwd=STUDIO, env=env, check=True)

    if args.mode == "examples":
        if args.prompt_json or args.prompt_dir:
            parser.error("--prompt-json and --prompt-dir apply only to negative generation")
        run("base", EXAMPLES / f"wan_{size}_tensor_parallel.py")
        lora = "epoch-0.safetensors" if size == "1.3B" else "step-0001000.safetensors"
        run("lora", EXAMPLES / f"model_training/validate_lora/Wan2.1-T2V-{size}_tensor_parallel_PhyGDPO.py",
            "--lora_path", Path(f"models/pretrained/Wan2.1-T2V-{size}_PhyGDPO") / lora)
        return
    prepared = data / "manifests" / f"original-{size.lower()}-positives-only-unscored" / "prompts.json"
    if args.prompt_json:
        prompts = args.prompt_json
    elif args.prompt_dir is None and prepared.is_file():
        prompts = [prepared]
    else:
        directory = args.prompt_dir
        if directory is None:
            candidates = (data / "json_file/gemini", STUDIO / "json_file/gemini",
                          STUDIO / "json_file", legacy_root / "json_file/gemini")
            first_shard = "vidgen_filter_gemini_part_0_dpo_0.json"
            directory = next((d for d in candidates if (d / first_shard).is_file()), candidates[0])
        prompts = [directory / f"vidgen_filter_gemini_part_0_dpo_{i}.json" for i in range(16)]
    prompts = [path.resolve() for path in prompts]
    for path in prompts:
        if not args.plan and not path.is_file():
            parser.error(f"Missing {path}. Supply --prompt-json or --prompt-dir (see DiffSynth-Studio/downloads/README.md).")
    for seed in (0, 1, 3):
        directory = data / f"DPO_data/Wan{size}/seed_{seed}"
        for path in prompts:
            run("negative", EXAMPLES / f"wan_{size}_tensor_parallel_DPO_losing_case.py",
                "--prompt_json_file", path, "--output_dir", directory, "--seed", seed)


if __name__ == "__main__":
    main()
