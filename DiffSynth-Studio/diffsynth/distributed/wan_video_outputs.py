"""Publish generated videos and their provenance only after the encoder finishes."""
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import imageio_ffmpeg
import torch
import torch.distributed as dist


def local_devices():
    count = os.getenv("LOCAL_WORLD_SIZE")
    return int(count) if count else torch.cuda.device_count()


def denoising_progress(timesteps):
    for step, timestep in enumerate(timesteps, 1):
        print(f"Denoising · step {step} of {len(timesteps)}", flush=True)
        yield timestep
    print("Decoding video", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_digest(path):
    real = path.resolve(strict=True)
    if real.parent.name == "blobs" and len(real.name) == 64:
        return real.name
    stat = real.stat()
    identity = [str(real), stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "phygdpo/weights"
    try:
        root.mkdir(parents=True, exist_ok=True)
        receipt = root / key
        if receipt.is_file():
            value = receipt.read_text().strip()
            if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                return value
    except OSError:
        return sha256(real)
    value = sha256(real)
    try:
        with NamedTemporaryFile(mode="w", dir=root, delete=False) as tmp:
            tmp.write(value)
        Path(tmp.name).replace(receipt)
    except OSError:
        pass
    return value


@lru_cache(maxsize=32)
def weight_files(size, directory, adapter):
    root = Path(directory)
    shards = (["diffusion_pytorch_model.safetensors"] if size == "1.3B" else
              [f"diffusion_pytorch_model-{i:05d}-of-00006.safetensors" for i in range(1, 7)])
    paths = [root / name for name in [*shards, "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]]
    if adapter:
        paths.append(Path(adapter))
    return [{"name": path.name, "bytes": path.stat().st_size, "sha256": weight_digest(path)} for path in paths]


def generation_record(size, task, backend, *, directory=None, adapter=None):
    from ..models import wan_video_dit

    world = dist.get_world_size() if dist.is_initialized() else 1
    groups = 2 if backend == "sharded" and size == "14B" and world >= 4 else 1
    operators = (("FLASH_ATTN_3_AVAILABLE", "flash-attn-3"),
                 ("FLASH_ATTN_2_AVAILABLE", "flash-attn-2"),
                 ("SAGE_ATTN_AVAILABLE", "sageattention"))
    attention = next((name for flag, name in operators if getattr(wan_video_dit, flag, False)), "torch-sdpa")
    return {
        "model": "Wan2.1-T2V-" + size, "backend": backend, "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "attention": attention, "tp": world // groups, "cfg_groups": groups,
        "gpu_models": [torch.cuda.get_device_name(i) for i in range(local_devices())],
        "weights": weight_files(size, directory or f"models/Wan-AI/Wan2.1-T2V-{size}", adapter),
        "width": 832, "height": 480, "num_frames": 81, "fps": 15,
        "cfg_scale": 5.0, "sigma_shift": 5.0, "tiled": True,
        **{k: v for k, v in task.items() if k not in ("progress_bar_cmd", "output_path")},
    }


@contextmanager
def output_lock(directory, mode):
    with (directory / ".wan-output.lock").open("a+") as lock:
        fcntl.flock(lock, mode)
        yield


def completed(path, record):
    path = Path(path)
    if not path.is_file():
        return False
    with output_lock(path.parent, fcntl.LOCK_SH):
        try:
            saved = json.loads(path.with_suffix(".json").read_text())
            return (saved["generation"] == record
                    and path.with_suffix(".txt").read_text() == record["prompt"]
                    and saved["video_sha256"] == sha256(path))
        except (OSError, KeyError, ValueError):
            return False


def skip_on_all_ranks(path, record, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    skip = rank == 0 and completed(path, record)
    if dist.is_initialized():
        decision = torch.tensor(skip, dtype=torch.uint8, device=device)
        dist.broadcast(decision, src=0)
        return bool(decision.item())
    return skip


def publish(frames, path, record, *, caption=False):
    from .. import save_video

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".wan-") as tmp:
        tmp = Path(tmp)
        video = tmp / "video.mp4"
        save_video(frames, str(video), fps=record["fps"], quality=5)
        # Decode in ffmpeg and transfer only the frame records across the pipe.
        probe = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror", "-threads", "1",
             "-i", str(video), "-map", "0:v:0", "-an", "-vsync", "0", "-c:v", "rawvideo", "-threads", "1",
             "-f", "framecrc", "pipe:1"], capture_output=True, check=True, text=True,
        ).stdout.splitlines()
        if (f"#dimensions 0: {record['width']}x{record['height']}" not in probe
                or f"#tb 0: 1/{record['fps']}" not in probe
                or sum(line.startswith("0,") for line in probe) != record["num_frames"]):
            raise ValueError(f"Encoded video has unexpected dimensions, frame rate or frame count: {path}")
        receipt = {"generation": record, "video_sha256": sha256(video)}
        (tmp / "record.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if caption:
            (tmp / "caption.txt").write_text(record["prompt"], encoding="utf-8")
        with output_lock(path.parent, fcntl.LOCK_EX):
            video.replace(path)
            if caption:
                (tmp / "caption.txt").replace(path.with_suffix(".txt"))
            (tmp / "record.json").replace(path.with_suffix(".json"))
