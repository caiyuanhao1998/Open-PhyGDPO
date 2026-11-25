"""Eight-GPU supervised flow matching for the fused Wan2.1 Fast I2V model."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import math
import os
import re
import time

from accelerate import init_empty_weights
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.nn import functional as F

from settings import (
    ACCUMULATION,
    DATASET,
    FPS,
    FRAMES,
    GPUS,
    HEIGHT,
    MODEL,
    SFT as RUN,
    SAMPLES,
    SAVE_EVERY,
    SEED,
    WIDTH,
    validate_manifest,
    text_features,
    video_features,
)

from training_utils import GROUP, BlockGroup, group_checkpoint, reduce_block_memory


LORA_NAME = "sports_sft"
LORA_RANK = 32
LEARNING_RATE = 1e-5
BATCH_SIZE = GPUS * ACCUMULATION
NUM_UPDATES = math.ceil(SAMPLES / BATCH_SIZE)
CHECKPOINTS = RUN / "checkpoints"
STEP_LOG = RUN / "logs/metrics.jsonl"
PROJECTIONS = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)


def adapter_key(name):
    grouped = re.fullmatch(r"blocks\.(\d+)\.layers\.(\d+)\.(.+)", name)
    if grouped:
        block = int(grouped[1]) * GROUP + int(grouped[2])
        name = f"blocks.{block}.{grouped[3]}"
    return "transformer." + name.replace(f".{LORA_NAME}.weight", ".weight")


def make_model(mesh, *, cpu_offload=False):
    model_dir = MODEL / "transformer"
    configuration = WanTransformer3DModel.load_config(model_dir)
    if configuration["num_layers"] != 40 or configuration["in_channels"] != 36:
        raise ValueError("Expected the Wan2.1 I2V 14B transformer")
    targets = [
        f"blocks.{i}.{suffix}"
        for i in range(configuration["num_layers"])
        for suffix in PROJECTIONS
    ]
    index = json.loads(
        (model_dir / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    shard_options = dict(
        mesh=mesh, mp_policy=precision, reshard_after_forward=True,
        offload_policy=CPUOffloadPolicy(pin_memory=False) if cpu_offload else OffloadPolicy(),
    )
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    with init_empty_weights():
        net = WanTransformer3DModel.from_config(configuration)
        net.requires_grad_(False)
        net.add_adapter(
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_RANK,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
            adapter_name=LORA_NAME,
        )
    used = set()
    with ExitStack() as sources:
        shards = {
            name: sources.enter_context(
                safe_open(str(model_dir / name), framework="pt")
            )
            for name in set(index.values())
        }

        def materialize(module, prefix, recurse=True):
            for short_name, parameter in list(module.named_parameters(recurse=recurse)):
                name = prefix + short_name
                if ".lora_A." in name:
                    bound = 1 / math.sqrt(parameter.shape[1])
                    tensor = torch.empty(parameter.shape, dtype=torch.float32)
                    tensor.uniform_(-bound, bound, generator=rng)
                elif ".lora_B." in name:
                    tensor = torch.zeros(parameter.shape, dtype=torch.float32)
                else:
                    key = name.replace(".base_layer.", ".")
                    tensor = shards[index[key]].get_tensor(key)
                    used.add(key)
                if tensor.shape != parameter.shape or parameter.requires_grad != (
                    ".lora_" in name
                ):
                    raise ValueError(f"Unexpected transformer parameter: {name}")
                parent, _, leaf = short_name.rpartition(".")
                module.get_submodule(parent).register_parameter(
                    leaf, nn.Parameter(tensor, requires_grad=parameter.requires_grad)
                )

        for i, block in enumerate(net.blocks):
            materialize(block, f"blocks.{i}.")
            reduce_block_memory(block)
            fully_shard(block, **shard_options)
            if dist.get_rank() == 0 and (i + 1) % 10 == 0:
                print(f"Loaded and sharded {i + 1}/40 DiT blocks", flush=True)
        for name, child in net.named_children():
            if name != "blocks":
                materialize(child, name + ".")
        materialize(net, "", recurse=False)
    if used != set(index):
        raise ValueError(
            f"Unloaded fused model tensors: {sorted(set(index) - used)[:10]}"
        )
    blocks = list(net.blocks)
    net.blocks = nn.ModuleList(
        BlockGroup(blocks[i : i + GROUP]) for i in range(0, len(blocks), GROUP)
    )
    fully_shard(net, **shard_options)
    net.enable_gradient_checkpointing(gradient_checkpointing_func=group_checkpoint)
    net.train()
    parameters = {
        adapter_key(name): p for name, p in net.named_parameters() if p.requires_grad
    }
    if len(parameters) != 2 * len(targets):
        raise ValueError(f"Unexpected number of new LoRA tensors: {len(parameters)}")
    return net, parameters


def configuration():
    fusion = json.loads((MODEL / "fusion.json").read_text())
    if fusion["causvid_weight"] != 0.95 or fusion["adapted_linear_layers"] != 486:
        raise ValueError("The local fused base differs from the audited Space fusion")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL / "scheduler", local_files_only=True
    )
    if (
        scheduler.config.flow_shift != 8.0
        or scheduler.config.prediction_type != "flow_prediction"
    ):
        raise ValueError("Expected the Space flow-prediction scheduler (shift 8)")
    scheduler.set_timesteps(4)
    return {
        "manifest_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "positive_examples": SAMPLES,
        "data_order_seed": SEED,
        "num_gpus": GPUS,
        "micro_batch_size": GPUS,
        "gradient_accumulation_steps": ACCUMULATION,
        "batch_size": BATCH_SIZE,
        "checkpoint_every_optimizer_steps": SAVE_EVERY,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "source_fps": FPS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "lora_init": "kaiming_uniform_A_zero_B",
        "learning_rate": LEARNING_RATE,
        "objective": "supervised flow velocity MSE on real videos",
        "attention": "bidirectional Wan2.1 I2V",
        "conditioning": "same real video's first frame and original caption",
        "scheduler_timesteps": scheduler.timesteps.tolist(),
        "scheduler_sigmas": scheduler.sigmas[:-1].tolist(),
        "fused_base": fusion,
    }


def sample(row, sample_index, config, device):
    features = load_file(str(video_features(row)))
    text = load_file(str(text_features(row)))["text"]
    text = F.pad(text, (0, 0, 0, 512 - text.shape[0])).unsqueeze(0).to(device)
    image = features["image"].unsqueeze(0).to(device)
    clean = features["video"].unsqueeze(0).to(device, dtype=torch.float32)
    condition = features["condition"].unsqueeze(0).to(device)
    _, _, frames, height, width = condition.shape
    mask = condition.new_zeros(1, 4, frames, height, width)
    mask[:, :, 0] = 1

    # The shuffled videos are balanced across the four sampling timesteps.
    slot = sample_index % len(config["scheduler_timesteps"])
    sigma = config["scheduler_sigmas"][slot]
    timestep = torch.tensor([config["scheduler_timesteps"][slot]], device=device)
    rng = torch.Generator(device=device).manual_seed(SEED + sample_index)
    noise = torch.randn(clean.shape, generator=rng, device=device)
    noisy = torch.lerp(clean, noise, sigma).to(torch.bfloat16)
    return (
        torch.cat((noisy, mask, condition), dim=1),
        timestep,
        text,
        image,
        noise - clean,
    )


def count_samples(step):
    return min(step * BATCH_SIZE, SAMPLES)


@torch.no_grad()
def save_checkpoint(
    step, parameters, optimizer, config, rank, checkpoint_dir=CHECKPOINTS
):
    stage = checkpoint_dir / f".step-{step:04d}.partial"
    final = checkpoint_dir / f"step-{step:04d}"
    if rank == 0:
        if final.exists():
            raise FileExistsError(final)
        stage.mkdir(exist_ok=True)
    dist.barrier()
    shard, adapter = {}, {}
    for name, p in parameters.items():
        state = optimizer.state[p]
        shard[name] = {
            "parameter": p.to_local().detach().cpu(),
            "step": state["step"].cpu(),
            "exp_avg": state["exp_avg"].to_local().cpu(),
            "exp_avg_sq": state["exp_avg_sq"].to_local().cpu(),
        }
        value = p.detach().full_tensor()
        if rank == 0:
            adapter[name] = value.to(device="cpu", dtype=torch.bfloat16).contiguous()
    torch.save(shard, stage / f"optimizer-rank{rank}.pt")
    if rank == 0:
        if not all(torch.isfinite(value).all() for value in adapter.values()):
            raise FloatingPointError("Non-finite exported LoRA")
        nonzero_b = sum(
            bool(value.count_nonzero())
            for name, value in adapter.items()
            if name.endswith(".lora_B.weight")
        )
        if nonzero_b == 0:
            raise RuntimeError("The new LoRA B matrices did not update")
        save_file(
            adapter,
            str(stage / "pytorch_lora_weights.safetensors"),
            metadata={"format": "pt", "rank": str(LORA_RANK), "alpha": str(LORA_RANK)},
        )
        info = {
            "step": step,
            "samples_consumed": count_samples(step),
            "nonzero_lora_B_layers": nonzero_b,
            "configuration": config,
        }
        (stage / "training.json").write_text(json.dumps(info, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        stage.rename(final)
        print(
            f"Checkpoint saved at {final} after {count_samples(step)} samples",
            flush=True,
        )
    dist.barrier()


# Original experiment CSV -> the byte-exact equivalent produced by the data downloader.
# Each pair has the same ordered rows and media. Only the layout of the CSV and paths changed.
_RELEASED_MANIFEST_HASHES = {
    "ddde2f0547fb731d969e5d1692fdd036eb357f5c1c002fb8c85823a9adc52bb5":
        "b1bc92c3419f326c02708ada22228e18edb44cfeac3abe27d5907332843c2e8b",
    "4db32e9ab93c273fea2f2e20a60505643f7e1b8a30efc7238b10c6392971a6dc":
        "7183a8527a107c2b827ac760218933e5cb4e875d5eb570d126c5078347a41261",
}


def checkpoint_configuration_matches(saved, current):
    if not isinstance(saved, dict) or not {"manifest_sha256", "lora_init"} <= saved.keys():
        return False

    def identity(config):
        result = dict(config)
        manifest = config["manifest_sha256"]
        result["manifest_sha256"] = _RELEASED_MANIFEST_HASHES.get(manifest, manifest)
        adapter = config["lora_init"]
        if isinstance(adapter, dict) and adapter.get("sha256"):
            # The full SHA identifies the weights even when an exported LoRA is moved or renamed.
            result["lora_init"] = {k: v for k, v in adapter.items() if k not in ("path", "filename")}
        return result

    return identity(saved) == identity(current)


@torch.no_grad()
def restore(parameters, optimizer, config, rank, checkpoint_dir=CHECKPOINTS):
    previous = sorted(
        path
        for path in checkpoint_dir.glob("step-*")
        if (path / "training.json").is_file()
    )
    if not previous:
        return 0
    folder = previous[-1]
    metadata = json.loads((folder / "training.json").read_text())
    if (
        not checkpoint_configuration_matches(metadata["configuration"], config)
        or count_samples(metadata["step"]) != metadata["samples_consumed"]
    ):
        raise ValueError(f"Resume configuration does not match {folder}")
    values = torch.load(
        folder / f"optimizer-rank{rank}.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    if values.keys() != parameters.keys():
        raise ValueError(f"Unmatched optimizer state in {folder}")
    for name, p in parameters.items():
        state = values[name]
        p.to_local().copy_(state["parameter"])
        optimizer.state[p]["step"] = state["step"]
        for field in ("exp_avg", "exp_avg_sq"):
            moment = torch.empty_like(p)
            moment.to_local().copy_(state[field])
            optimizer.state[p][field] = moment
    return metadata["step"]


def trim_metrics(start, step_log=STEP_LOG):
    rows = []
    if step_log.exists():
        for line in step_log.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry["step"] <= start:
                rows.append(entry)
    if [entry["step"] for entry in rows] != list(range(1, start + 1)):
        raise ValueError("The logs do not match the last completed checkpoint")
    temporary = step_log.with_suffix(".partial")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(step_log)


def train(stop):
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if size != GPUS or stop not in range(1, NUM_UPDATES + 1):
        raise ValueError("Invalid world size or target optimizer step")
    rows = validate_manifest()
    if len(rows) != SAMPLES:
        raise ValueError("Unexpected positive sample count")
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        config = configuration()
        if rank == 0:
            (RUN / "training-config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
            print(json.dumps(config), flush=True)
        mesh = init_device_mesh("cuda", (size,))
        model, parameters = make_model(mesh)
        optimizer = torch.optim.AdamW(
            list(parameters.values()), lr=LEARNING_RATE, foreach=False
        )
        start = restore(parameters, optimizer, config, rank)
        if rank == 0:
            trim_metrics(start)
            print(
                f"Training ready at update {start}/{NUM_UPDATES}; "
                f"{len(parameters)} LoRA tensors",
                flush=True,
            )
        dist.barrier()

        for step in range(start + 1, stop + 1):
            started = time.monotonic()
            lower, upper = count_samples(step - 1), count_samples(step)
            count = upper - lower
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            for batch, begin in enumerate(range(lower, upper, size), 1):
                tick = time.monotonic()
                ordinal = begin + rank
                real = ordinal < upper
                index = ordinal if real else begin
                x, t, text, image, target = sample(rows[index], index, config, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(
                        x, t, text, encoder_hidden_states_image=image, return_dict=False
                    )[0]
                loss = F.mse_loss(prediction.float(), target)
                # FSDP averages the reduced gradients across ranks. Summing these
                # contributions yields the mean over actual examples in this update.
                (loss * (size / count if real else 0.0)).backward()
                value = loss.detach() * int(real)
                dist.all_reduce(value)
                total_loss += value
                if not torch.isfinite(value).item():
                    raise FloatingPointError(
                        f"Non-finite SFT loss at step {step}, batch {batch}"
                    )
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "optimizer_step": step,
                                "micro_batch": batch,
                                "samples_seen": min(begin + size, upper),
                                "micro_batch_loss": value.item()
                                / min(size, upper - begin),
                                "elapsed_seconds": time.monotonic() - tick,
                            }
                        ),
                        flush=True,
                    )
                del x, t, text, image, target, prediction, loss

            grad_sq = torch.stack(
                [
                    p.grad.to_local().float().square().sum()
                    for p in parameters.values()
                    if p.grad is not None
                ]
            ).sum()
            dist.all_reduce(grad_sq)
            if not torch.isfinite(grad_sq).item() or grad_sq.item() <= 0:
                raise FloatingPointError(
                    f"Invalid LoRA gradient at step {step}: {grad_sq.item()}"
                )
            optimizer.step()
            peak = torch.tensor(
                torch.cuda.max_memory_allocated() / 2**30, device=device
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                entry = {
                    "step": step,
                    "total_steps": NUM_UPDATES,
                    "samples_in_update": count,
                    "samples_consumed": upper,
                    "loss": total_loss.item() / count,
                    "grad_norm": grad_sq.sqrt().item(),
                    "max_allocated_gib": peak.item(),
                    "seconds": time.monotonic() - started,
                }
                with STEP_LOG.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
            if step % SAVE_EVERY == 0 or step == NUM_UPDATES:
                save_checkpoint(step, parameters, optimizer, config, rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--through-step",
        type=int,
        choices=[*range(SAVE_EVERY, NUM_UPDATES, SAVE_EVERY), NUM_UPDATES],
        default=NUM_UPDATES,
    )
    train(parser.parse_args().through_step)
