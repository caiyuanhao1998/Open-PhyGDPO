"""Identify the exact encoder files, preprocessing and media used by a feature cache."""
import fcntl
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import imageio_ffmpeg

from settings import CACHE, FPS, FRAMES, HEIGHT, MODEL, WIDTH, digest, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=None)
def component_hash(component):
    directory = MODEL / component
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"Empty model component: {directory}")
    stats = {}
    for path in files:
        st = path.stat()
        stats[str(path.relative_to(directory))] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    # Workers share the digest of immutable files. Restoring/replacing a file invalidates its entry.
    root = CACHE / ".fingerprints"
    root.mkdir(parents=True, exist_ok=True)
    key = fingerprint([str(directory.resolve()), component])
    receipt = root / (key + ".json")
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            previous = json.loads(receipt.read_text())
        except (FileNotFoundError, ValueError):
            previous = {}
        if previous.get("stats") == stats:
            return previous["sha256"]
        sha = fingerprint({str(p.relative_to(directory)): digest(p) for p in files})
        write_json(receipt, dict(stats=stats, sha256=sha))
        return sha


@lru_cache(maxsize=None)
def recipe_hash(stage):
    components = {
        "positive": ("vae", "image_encoder", "image_processor"),
        "negative": ("vae",),
        "text": ("tokenizer", "text_encoder"),
        "evaluation": ("vae", "image_encoder", "image_processor", "text_encoder", "tokenizer"),
    }[stage]
    # Hash the code that selects frames, normalizes pixels/latents and tokenizes captions.
    implementation = "evaluate.py" if stage == "evaluation" else "encode.py"
    return fingerprint({
        "stage": stage, "implementation": digest(Path(__file__).with_name(implementation)),
        "components": {name: component_hash(name) for name in components},
        "preprocessing": [WIDTH, HEIGHT, FRAMES, FPS],
        "ffmpeg": imageio_ffmpeg.get_ffmpeg_version() if stage in ("positive", "negative") else None,
        "software": {name: version(name) for name in ("torch", "diffusers", "transformers", "imageio-ffmpeg", "Pillow", "numpy", "ftfy", "beautifulsoup4")},
    })
