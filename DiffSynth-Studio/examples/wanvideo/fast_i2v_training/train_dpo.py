"""Score-free preference training with the step-8 LoRA as the fixed reference."""
import argparse
from datetime import timedelta
from functools import partial
import json
import os
import time

from settings import (
    ACCUMULATION,
    BATCH,
    BETA,
    BOUNDARIES,
    GPUS,
    INIT_ADAPTER,
    PAIRS,
    DPO as RUN,
    SAMPLES,
    SEED,
    UPDATES,
    negative_features,
)
from settings import (
    digest,
    is_wan14b_negative,
    validate_manifest,
    read_csv,
    write_json,
    text_features,
    video_features,
)
from training_utils import install_adapter
from train_sft import (
    LEARNING_RATE,
    configuration as sft_configuration,
    count_samples,
    make_model,
    restore,
    save_checkpoint,
    trim_metrics,
)
from training_utils import preference_coefficient

CHECKPOINTS = RUN / "checkpoints"
METRICS = RUN / "logs/metrics.jsonl"

from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F


def configuration():
    config = sft_configuration()
    config.pop("positive_examples")
    original_negatives = all(is_wan14b_negative(row["negative_video"]) for row in read_csv(PAIRS))
    config.update(
        manifest_sha256=digest(PAIRS),
        preference_pairs=SAMPLES,
        checkpoint_every_optimizer_steps=4,
        lora_init={"filename": INIT_ADAPTER.name, "sha256": digest(INIT_ADAPTER)},
        reference="frozen copy of the initializer on the same CausVid-fused base",
        optimizer="fresh AdamW state",
        objective=("pairwise diffusion DPO with one fixed Wan14B negative per positive" if original_negatives
                   else "pairwise diffusion DPO with one fixed negative per positive"),
        alpha=1.0,
        gamma=1.0,
        beta=BETA,
        conditioning="same caption, positive first frame and denoising timestep for both branches",
    )
    return config


def sample(row, index, config, device):
    positive = load_file(str(video_features(row)))
    path, key = negative_features(row)
    with safe_open(path, framework="pt") as source:
        negative = source.get_tensor(key)
    text = load_file(str(text_features(row)))["text"]
    text = F.pad(text, (0, 0, 0, 512 - text.shape[0])).unsqueeze(0).to(device)
    image = positive["image"].unsqueeze(0).to(device)
    condition = positive["condition"].unsqueeze(0).to(device)
    _, _, frames, height, width = condition.shape
    mask = condition.new_zeros(1, 4, frames, height, width)
    mask[:, :, 0] = 1
    condition = torch.cat((mask, condition), dim=1)

    slot = index % len(config["scheduler_timesteps"])
    timestep = torch.tensor([config["scheduler_timesteps"][slot]], device=device)
    sigma = config["scheduler_sigmas"][slot]
    rng = torch.Generator(device=device).manual_seed(SEED + index)
    branches = []
    for value in (positive["video"], negative):
        clean = value.unsqueeze(0).to(device, dtype=torch.float32)
        noise = torch.randn(clean.shape, generator=rng, device=device)
        noised = torch.lerp(clean, noise, sigma).to(torch.bfloat16)
        branches.append((torch.cat((noised, condition), dim=1), noise - clean))
    return text, image, timestep, branches


def branch_mse(model, timestep, text, image, branch):
    x, target = branch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            x, timestep, text, encoder_hidden_states_image=image, return_dict=False
        )[0]
    return F.mse_loss(prediction.float(), target)


@torch.no_grad()
def reference_values(parameters, reference, error, branches):
    policy = {
        name: param.to_local().detach().clone() for name, param in parameters.items()
    }
    try:
        for name, param in parameters.items():
            param.to_local().copy_(reference[name])
        return error(branches[0]), error(branches[1])
    finally:
        for name, param in parameters.items():
            param.to_local().copy_(policy[name])


def train(stop):
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != GPUS or stop not in BOUNDARIES:
        raise ValueError("Unexpected DPO world size or checkpoint boundary")
    rows = validate_manifest(paired=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) != SAMPLES:
        raise ValueError("Number of pairs differs from the SFT-positive manifest")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        config = configuration()
        if rank == 0:
            write_json(RUN / "training-config.json", config)
            print(json.dumps(config), flush=True)

        mesh = init_device_mesh("cuda", (world,))
        net, parameters = make_model(mesh)
        install_adapter(INIT_ADAPTER, parameters, mesh)
        reference = {
            name: param.to_local().detach().clone()
            for name, param in parameters.items()
        }
        optimizer = torch.optim.AdamW(
            parameters.values(), lr=LEARNING_RATE, foreach=False
        )
        start = restore(parameters, optimizer, config, rank, checkpoint_dir=CHECKPOINTS)
        if rank == 0:
            trim_metrics(start, step_log=METRICS)
            print(
                f"DPO ready at update {start}/{UPDATES}; initialized and froze reference from "
                f"{INIT_ADAPTER}; {ACCUMULATION} microbatches per {BATCH}-pair update",
                flush=True,
            )
        dist.barrier()
        if start >= stop:
            raise ValueError(f"Checkpoint already at {start}, requested through {stop}")

        for step in range(start + 1, stop + 1):
            started = time.monotonic()
            lower, upper = count_samples(step - 1), count_samples(step)
            count = upper - lower
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            totals = torch.zeros(5, dtype=torch.float32, device=device)

            for batch, begin in enumerate(range(lower, upper, world), 1):
                tick = time.monotonic()
                ordinal = begin + rank
                real = ordinal < upper
                index = ordinal if real else begin
                text, image, timestep, branches = sample(
                    rows[index], index, config, device
                )

                error = partial(branch_mse, net, timestep, text, image)

                ref_win, ref_lose = reference_values(
                    parameters, reference, error, branches
                )
                with torch.no_grad():
                    lose = error(branches[1])
                win_graph = error(branches[0])
                win = win_graph.detach()
                loss, coefficient = preference_coefficient(
                    win, lose, ref_win, ref_lose, BETA
                )
                coefficient = coefficient * (world / count if real else 0.0)
                win_graph.backward(coefficient)
                del win_graph
                error(branches[1]).backward(-coefficient)

                values = torch.stack((loss, win, lose, ref_win, ref_lose)) * int(real)
                dist.all_reduce(values)
                if not torch.isfinite(values).all().item():
                    raise FloatingPointError(
                        f"Non-finite pair loss at step {step}, microbatch {batch}"
                    )
                totals += values
                if rank == 0:
                    real_count = min(world, upper - begin)
                    print(
                        json.dumps(
                            {
                                "optimizer_step": step,
                                "micro_batch": batch,
                                "pairs_seen": min(begin + world, upper),
                                "micro_batch_dpo": values[0].item() / real_count,
                                "micro_batch_winner_mse": values[1].item() / real_count,
                                "micro_batch_rejected_mse": values[2].item()
                                / real_count,
                                "micro_batch_reference_winner_mse": values[3].item()
                                / real_count,
                                "micro_batch_reference_rejected_mse": values[4].item()
                                / real_count,
                                "elapsed_seconds": time.monotonic() - tick,
                            }
                        ),
                        flush=True,
                    )
                del error, text, image, timestep, branches, values

            norm2 = torch.stack(
                [
                    param.grad.to_local().float().square().sum()
                    for param in parameters.values()
                    if param.grad is not None
                ]
            ).sum()
            dist.all_reduce(norm2)
            if not torch.isfinite(norm2).item() or norm2.item() <= 0:
                raise FloatingPointError(
                    f"Invalid LoRA gradient at step {step}: {norm2.item()}"
                )
            optimizer.step()
            peak = torch.tensor(
                torch.cuda.max_memory_allocated() / 2**30, device=device
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                loss, win, lose, ref_win, ref_lose = (totals / count).tolist()
                record = {
                    "step": step,
                    "total_steps": UPDATES,
                    "pairs_in_update": count,
                    "pairs_consumed": upper,
                    "dpo_loss": loss,
                    "winner_mse": win,
                    "rejected_mse": lose,
                    "reference_winner_mse": ref_win,
                    "reference_rejected_mse": ref_lose,
                    "grad_norm": norm2.sqrt().item(),
                    "max_allocated_gib": peak.item(),
                    "seconds": time.monotonic() - started,
                }
                with METRICS.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step in BOUNDARIES:
                save_checkpoint(
                    step,
                    parameters,
                    optimizer,
                    config,
                    rank,
                    checkpoint_dir=CHECKPOINTS,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-step", type=int, choices=BOUNDARIES, required=True)
    train(parser.parse_args().through_step)
