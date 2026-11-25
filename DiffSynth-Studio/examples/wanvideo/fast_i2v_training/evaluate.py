"""Data-parallel evaluation with a Wan 14B transformer sharded over the selected GPUs."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import json
from importlib.metadata import version
from collections import defaultdict

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from safetensors import safe_open, SafetensorError
from safetensors.torch import load_file
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import prompt_clean
from diffusers.video_processor import VideoProcessor
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    T5TokenizerFast,
    UMT5EncoderModel,
)


from encode import atomic_save, raw_video
from cache_identity import component_hash, fingerprint, recipe_hash
from settings import CACHE, DATASET, GPUS, MODEL, ROOT, digest, read_csv, select_cases, write_json
from train_sft import make_model
from training_utils import install_adapter

ASSETS = Path(__file__).resolve().parent / "evaluation"


def read_plan():
    path = ASSETS / "scenes.json"
    plan = json.loads(path.read_text())
    by_sport = defaultdict(list)
    for row in plan["scenes"]:
        by_sport[row["scene"]].append(row)
        if (
            row["case_id"] != f"{row['scene']}-p{row['variant']}"
            or row["seed"] != 0
            or digest(ASSETS / row["image"]) != row["image_sha256"]
        ):
            raise ValueError(f"Changed comparison input {row['case_id']}")
    if len(by_sport) != 16 or any(
        {r["variant"] for r in group} != {1, 2, 3, 4}
        or len(group) != 4
        or any(
            len({r[key] for r in group}) != 4
            for key in ("image_sha256", "prompt", "source_video_id")
        )
        for group in by_sport.values()
    ):
        raise ValueError("Expected four distinct scenes and prompts per sport")
    ids = {r["source_video_id"] for r in plan["scenes"]}
    if len(ids) != len(plan["scenes"]):
        raise ValueError("Duplicate source video across evaluation scenes")
    if DATASET.is_file() and ids.intersection(r["vid"] for r in read_csv(DATASET)):
        raise ValueError("Evaluation source IDs overlap the training set")
    return plan, digest(path)


def feature_path(row, features_hash, root):
    return root / f"{row['case_id']}-{features_hash[:16]}.safetensors"


def load_vae(device):
    return (
        AutoencoderKLWan.from_pretrained(
            MODEL / "vae", torch_dtype=torch.float32, local_files_only=True
        )
        .eval()
        .to(device)
    )


def latent_statistics(vae, device):
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def valid_features(path, features_hash, settings):
    if not path.is_file():
        return False
    shapes = {
        "text": [1, 512, 4096],
        "image": [1, 257, 1280],
        "condition": [1, 20, (settings["frames"] - 1) // 4 + 1, settings["height"] // 8, settings["width"] // 8],
    }
    try:
        with safe_open(path, framework="pt") as saved:
            return ((saved.metadata() or {}).get("features_sha256") == features_hash
                    and set(saved.keys()) == shapes.keys()
                    and all(saved.get_slice(k).get_shape() == shape and saved.get_slice(k).get_dtype() == "BF16"
                            for k, shape in shapes.items()))
    except (OSError, SafetensorError, ValueError):
        return False


@torch.no_grad()
def encode_inputs(rows, features_hash, settings, rank, device, root, text_batch_size=None):
    rows = [r for r in rows[rank::GPUS] if not valid_features(feature_path(r, features_hash, root), features_hash, settings)]
    if not rows:
        return
    tokenizer = T5TokenizerFast.from_pretrained(
        MODEL / "tokenizer", local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL / "text_encoder",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    text = []
    batch_size = text_batch_size or len(rows)
    for start in range(0, len(rows), batch_size):
        tokens = tokenizer(
            [prompt_clean(row["prompt"]) for row in rows[start:start + batch_size]],
            max_length=512, padding="max_length", truncation=True, return_tensors="pt",
        ).to(device)
        states = text_encoder(tokens.input_ids, tokens.attention_mask).last_hidden_state
        states = states.masked_fill(~tokens.attention_mask.bool().unsqueeze(-1), 0)
        text.extend(states.bfloat16().cpu().split(1))
    del text_encoder, states, tokens
    torch.cuda.empty_cache()

    vae = load_vae(device)
    clip = (
        CLIPVisionModel.from_pretrained(
            MODEL / "image_encoder", torch_dtype=torch.float32, local_files_only=True
        )
        .eval()
        .to(device)
    )
    processor = CLIPImageProcessor.from_pretrained(
        MODEL / "image_processor", local_files_only=True
    )
    video_processor = VideoProcessor(vae_scale_factor=8)
    mean, std = latent_statistics(vae, device)
    width, height, frames = (settings[key] for key in ("width", "height", "frames"))
    frame_video = torch.zeros(
        (1, 3, frames, height, width), dtype=torch.float32, device=device
    )
    for i, row in enumerate(rows):
        if "positive_video" in row:
            # Use the same time sampling, scaling and center crop as the SFT encoder.
            reference = Image.fromarray(raw_video(row["positive_video"], width, height, 1)[0])
        else:
            with Image.open(ASSETS / row["image"]) as image:
                reference = ImageOps.exif_transpose(image).convert("RGB")
            if reference.size != (width, height):
                reference = ImageOps.fit(reference, (width, height), method=Image.Resampling.LANCZOS)
        pixels = video_processor.preprocess(reference, height=height, width=width)
        frame_video[:, :, 0] = pixels.to(device=device, dtype=torch.float32)
        image_embed = clip(
            **processor(images=reference, return_tensors="pt").to(device),
            output_hidden_states=True,
        ).hidden_states[-2]
        condition = vae.encode(frame_video).latent_dist.mode()
        condition = (condition - mean) * std.reciprocal()
        mask = condition.new_zeros((1, 4, *condition.shape[2:]))
        mask[:, :, 0] = 1
        dest = feature_path(row, features_hash, root)
        atomic_save(dest, {
            "text": text[i],
            "image": image_embed.bfloat16(),
            "condition": torch.cat((mask, condition), dim=1).bfloat16(),
        }, {"features_sha256": features_hash})
        print(f"rank {rank}: prepared shared inputs for {row['case_id']}", flush=True)


@torch.no_grad()
def denoise(model, row, features, settings, device):
    rng = torch.Generator(device="cpu").manual_seed(row["seed"])
    condition = features["condition"].to(device)
    shape = (condition.shape[0], 16, *condition.shape[2:])
    if "noise" in row:
        latent = load_file(row["noise"])["noise"].to(device)
        if latent.shape != shape or latent.dtype != torch.float32:
            raise ValueError("Unexpected Space preset noise shape or dtype")
    else:
        latent = torch.randn(shape, generator=rng, dtype=torch.float32).to(device)
    text = features["text"].to(device)
    image = features["image"].to(device)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL / "scheduler", local_files_only=True
    )
    scheduler.set_timesteps(settings["steps"], device=device)
    for index, timepoint in enumerate(scheduler.timesteps, 1):
        if dist.get_rank() == 0:
            print(f"Denoising · step {index} of {settings['steps']}", flush=True)
        x = torch.cat((latent, condition), dim=1).bfloat16()
        with torch.autocast("cuda", dtype=torch.bfloat16), model.cache_context("cond"):
            prediction = model(
                x,
                timepoint.expand(latent.shape[0]),
                text,
                encoder_hidden_states_image=image,
                return_dict=False,
            )[0]
        latent = scheduler.step(prediction, timepoint, latent, return_dict=False)[0]
    if not torch.isfinite(latent).all().item():
        raise FloatingPointError(f"Non-finite generated latent for {row['scene']}")
    return latent


@torch.no_grad()
def write_video(vae, latent, path, settings, device):
    mean, std = latent_statistics(vae, device)
    pixels = vae.decode(latent.float() * std + mean, return_dict=False)[0]
    frames = VideoProcessor(vae_scale_factor=8).postprocess_video(
        pixels, output_type="np"
    )[0]
    if not np.isfinite(frames).all():
        raise FloatingPointError(f"Non-finite decoded frames: {path}")
    frames = np.ascontiguousarray(np.clip(frames * 255.0, 0, 255), dtype=np.uint8)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (settings["width"], settings["height"]),
        fps=settings["fps"],
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        quality=None,
        output_params=["-movflags", "+faststart", "-crf", "17", "-threads", "2"],
    )
    try:
        writer.send(None)
        for frame in frames:
            writer.send(frame)
    finally:
        writer.close()
    verify_video(temporary, settings)
    temporary.replace(path)


def verify_video(path, settings):
    import imageio_ffmpeg

    with _video_reader(path, imageio_ffmpeg) as stream:
        metadata = next(stream)
        if (
            tuple(metadata["size"]) != (settings["width"], settings["height"])
            or abs(metadata["fps"] - settings["fps"]) > 0.05
        ):
            raise ValueError(f"Wrong video dimensions or FPS: {path}")
        frames = sum(1 for _ in stream)
    if frames != settings["frames"]:
        raise ValueError(
            f"Expected {settings['frames']} frames in {path}, got {frames}"
        )


def _video_reader(path, reader):
    from contextlib import closing

    return closing(
        reader.read_frames(str(path), pix_fmt="rgb24", input_params=["-threads", "1"])
    )


def completed(video, expected):
    receipt = video.with_suffix(".json")
    if not video.is_file() or not receipt.is_file():
        return False
    try:
        record = json.loads(receipt.read_text())
        return all(record.get(k) == v for k, v in expected.items()) and record.get(
            "video_sha256"
        ) == digest(video)
    except (OSError, ValueError):
        return False


@torch.no_grad()
def generate(output_dir, adapter, plan, plan_hash, features_hash, device, rank, feature_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = plan["settings"]
    if settings["guidance_scale"] != 1:
        raise ValueError("Expected the single-pass Fast recipe")
    # Rank zero decides the work list so all peers participate in identical FSDP collectives.
    shared = [None]
    if rank == 0:
        model_hash = digest(adapter) if adapter else None
        base_hash = component_hash("transformer")
        scheduler_hash = component_hash("scheduler")
        runtime = dict(
            world_size=GPUS,
            device=torch.cuda.get_device_name(device),
            torch=str(torch.__version__),
            diffusers=version("diffusers"),
            transformer_dtype="bfloat16",
            latent_dtype="float32",
            vae_dtype="float32",
            noise_generator="cpu",
            base_fusion_sha256=digest(MODEL / "fusion.json"),
        )
        pending = []
        for row in plan["scenes"]:
            expected = dict(
                plan_sha256=plan_hash,
                features_sha256=features_hash,
                adapter_sha256=model_hash,
                base_sha256=base_hash,
                scheduler_sha256=scheduler_hash,
                runtime=runtime,
                prompt=row["prompt"],
                seed=row["seed"],
                settings=settings,
                **({"positive_video_sha256": row["positive_video_sha256"]} if "positive_video" in row
                   else {"image_sha256": row["image_sha256"]}),
            )
            if "conditioning" in row:
                expected.update(conditioning_sha256=row["conditioning_sha256"], noise_sha256=row["noise_sha256"])
            if not completed(output_dir / (row["case_id"] + ".mp4"), expected):
                pending.append((row, expected))
        shared[0] = pending
    dist.broadcast_object_list(shared, src=0, device=device)
    pending = shared[0]
    if not pending:
        return
    mesh = init_device_mesh("cuda", (GPUS,))
    cpu_offload = GPUS == 1
    if GPUS == 2:
        # Keep room for denoising and FP32 VAE decoding on pairs of 24 GB cards.
        free_bytes = torch.tensor(torch.cuda.mem_get_info(device)[0], device=device)
        dist.all_reduce(free_bytes, op=dist.ReduceOp.MIN)
        cpu_offload = free_bytes.item() < 32 * 1024**3
    model, parameters = make_model(mesh, cpu_offload=cpu_offload)
    if adapter is None:
        model.disable_adapters()
    else:
        install_adapter(adapter, parameters, mesh)
    model.eval()
    model.disable_gradient_checkpointing()
    vae = None
    for offset in range(0, len(pending), GPUS):
        batch = pending[offset : offset + GPUS]
        row, expected = batch[min(rank, len(batch) - 1)]
        feature_file = row.get("conditioning") or feature_path(row, features_hash, feature_dir)
        features = load_file(str(feature_file))
        latent = denoise(model, row, features, settings, device)
        if rank < len(batch):
            torch.cuda.empty_cache()
            if rank == 0:
                print("Decoding video", flush=True)
            if vae is None:
                vae = load_vae(device)
            output = output_dir / (row["case_id"] + ".mp4")
            write_video(vae, latent, output, settings, device)
            write_json(
                output.with_suffix(".json"), dict(expected, video_sha256=digest(output))
            )
            print(output, flush=True)
        del features, latent
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "features", "generate"))
    parser.add_argument("--label", default="base")
    parser.add_argument("--case", action="append", help="Limit to one case ID; repeat to select several")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--feature-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.jobs) != bool(args.output_dir):
        parser.error("--jobs and --output-dir must be supplied together")
    if Path(args.label).name != args.label or args.label in {"", ".", ".."}:
        parser.error("label must be a single directory name")
    if args.jobs:
        plan = json.loads(args.jobs.read_text())
        plan_hash = fingerprint(plan)
    else:
        plan, plan_hash = read_plan()
        plan["scenes"] = select_cases(plan["scenes"], args.case)
    if args.stage == "check":
        print(f"Verified {len(plan['scenes'])} matched inputs, plan {plan_hash}")
        return
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if size != GPUS:
        raise ValueError(f"Expected {GPUS} workers")
    torch.set_num_threads(2)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if args.jobs and all("conditioning" in row for row in plan["scenes"]):
        features_hash = fingerprint([row["conditioning_sha256"] for row in plan["scenes"]])
    else:
        features_hash = fingerprint([plan_hash, recipe_hash("evaluation")])
    feature_dir = args.feature_dir or CACHE / "evaluation"
    feature_dir.mkdir(exist_ok=True, parents=True)
    if args.stage == "features":
        encode_inputs(
            plan["scenes"], features_hash, plan["settings"], rank, device, feature_dir,
            text_batch_size=4 if args.jobs else None,
        )
        return
    dist.init_process_group("nccl", timeout=timedelta(minutes=25))
    try:
        output_dir = args.output_dir or ROOT / "evaluation" / args.label
        generate(output_dir, args.adapter, plan, plan_hash, features_hash, device, rank, feature_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
