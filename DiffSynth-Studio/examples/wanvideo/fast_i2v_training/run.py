"""Encode, train and evaluate checkpoint boundaries in the same order on restart."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from settings import (
    BATCH,
    DPO,
    DPO_SAVE_EVERY,
    GPUS,
    INIT_ADAPTER,
    SAVE_EVERY,
    SFT,
    SAMPLES,
    TRAIN_DIR,
    UPDATES,
    validate_manifest,
    select_cases,
)

HERE = Path(__file__).resolve().parent


def launch(worker, *args):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={GPUS}",
        "--",
        str(HERE / worker),
        *map(str, args),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, check=True, env=env)


def checkpoint_path(stage, step):
    return (SFT if stage == "sft" else DPO) / "checkpoints" / f"step-{step:04d}"


def run(args, boundaries):
    from prepare_model import verify_base
    from settings import MODEL

    paired = args.stage == "dpo" or args.require_pairs
    validate_manifest(paired=paired)
    if paired:
        from settings import PAIRS
        print(f"Using negative pairs from {PAIRS}", flush=True)
    verify_base(MODEL)
    if args.stage == "dpo" and not INIT_ADAPTER.is_file():
        raise FileNotFoundError(f"Missing SFT initializer {INIT_ADAPTER}")
    from train_sft import checkpoint_configuration_matches
    if args.stage == "sft":
        from train_sft import configuration
    else:
        from train_dpo import configuration
    expected_config = configuration()
    for stage in (
        ["text", "positive"]
        if args.stage == "sft"
        else ["text", "positive", "negative"]
    ):
        launch("encode.py", stage)

    def evaluate(*options):
        cases = [arg for name in (args.eval_case or []) for arg in ("--case", name)]
        launch("evaluate.py", *options, *cases)

    if not args.skip_eval:
        evaluate("features")
    for step in boundaries:
        folder = checkpoint_path(args.stage, step)
        if not folder.exists():
            # Both trainers save intermediate checkpoints during a continuous run.
            stop = boundaries[-1] if args.skip_eval else step
            launch("train_" + args.stage + ".py", "--through-step", stop)
        info = json.loads((folder / "training.json").read_text())
        adapter = folder / "pytorch_lora_weights.safetensors"
        if (
            not checkpoint_configuration_matches(info["configuration"], expected_config)
            or info["step"] != step
            or info["samples_consumed"] != min(BATCH * step, SAMPLES)
            or not adapter.is_file()
            or any(
                not (folder / f"optimizer-rank{rank}.pt").is_file()
                for rank in range(GPUS)
            )
        ):
            raise ValueError(f"Incomplete or incompatible checkpoint {folder}")
        if not args.skip_eval:
            if args.stage == "dpo" and step == boundaries[0]:
                evaluate(
                    "generate",
                    "--label",
                    "dpo-initializer",
                    "--adapter",
                    INIT_ADAPTER,
                )
            evaluate(
                "generate",
                "--label",
                f"{args.stage}-{step:04d}",
                "--adapter",
                adapter,
            )
            if args.stage == "sft" and step == boundaries[0]:
                evaluate("generate", "--label", "base")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sft", "dpo"))
    p.add_argument("--require-pairs", action="store_true",
                   help="Verify the DPO training pairs before beginning the combined training run")
    stop_at = p.add_mutually_exclusive_group()
    stop_at.add_argument("--full-epoch", action="store_true", help="Process every row in the selected manifest")
    stop_at.add_argument(
        "--through-step",
        type=int,
        help="Final checkpoint boundary (default SFT step 8, DPO one epoch)",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train without generating the 64 matched cases",
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help="Print schedule without allocating GPUs or writing files",
    )
    p.add_argument("--eval-case", action="append", help="Evaluate one fixed case ID; repeat to select several")
    args = p.parse_args()
    if args.skip_eval and args.eval_case:
        p.error("--skip-eval and --eval-case cannot be combined")
    case_count = 0
    if not args.skip_eval:
        scenes = json.loads((HERE / "evaluation/scenes.json").read_text())["scenes"]
        try:
            case_count = len(select_cases(scenes, args.eval_case))
        except ValueError as error:
            p.error(str(error))
    interval = SAVE_EVERY if args.stage == "sft" else DPO_SAVE_EVERY
    all_steps = [*range(interval, UPDATES, interval), UPDATES]
    stop = (
        args.through_step
        if args.through_step is not None
        else (min(SAVE_EVERY, UPDATES) if args.stage == "sft" and not args.full_epoch else UPDATES)
    )
    if stop not in all_steps:
        p.error(f"--through-step must be one of {all_steps}")
    boundaries = [step for step in all_steps if step <= stop]
    print(
        json.dumps(
            dict(
                stage=args.stage,
                rows=SAMPLES,
                workers=GPUS,
                effective_batch=BATCH,
                checkpoint_dir=str(checkpoint_path(args.stage, stop).parent),
                checkpoints={i: min(i * BATCH, SAMPLES) for i in boundaries},
                last_update_size=SAMPLES - (UPDATES - 1) * BATCH,
                evaluation_cases_per_checkpoint=case_count,
            ),
            indent=2,
        ),
        flush=True,
    )
    if args.plan:
        return
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with (TRAIN_DIR / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, boundaries)


if __name__ == "__main__":
    main()
