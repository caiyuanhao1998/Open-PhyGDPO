"""Portable locations and hyperparameters of the Fast I2V training recipe."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from model_path import FOLDER, model_directory


def location(variable, default):
    return Path(os.getenv(variable) or default).expanduser().resolve()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def find_manifest(data, *names):
    candidates = [data / "manifests" / name for name in names]
    return next((path for path in candidates if path.is_file()), candidates[0])


RELEASED_PAIRS = ("fast-14b/pairs.csv", "fast-dpo/pairs.csv")


def pair_manifest(data, *, validate=False):
    source = os.getenv("PHYGDPO_NEGATIVE_SOURCE")
    if source not in (None, "released", "generated"):
        raise ValueError("PHYGDPO_NEGATIVE_SOURCE must be released or generated")
    directory = data / "manifests/fast-generated"
    explicit = os.getenv("PHYGDPO_PAIRS_CSV")
    if explicit:
        selected = Path(explicit).expanduser().resolve()
        released_paths = {(data / "manifests" / name).resolve() for name in RELEASED_PAIRS}
        selected_source = None
        if selected in released_paths:
            selected_source = "released"
        elif selected.is_relative_to(directory):
            selected_source = "generated"
        # Ignore a stale selection from the opposite source. A custom CSV is checked against its rows.
        if source is None or selected_source in (None, source):
            return selected
    if source != "generated":
        return find_manifest(data, *RELEASED_PAIRS)
    candidates = sorted(directory.glob("seed_*/pairs.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if validate and len(candidates) > 1:
        raise ValueError("Multiple generated pair manifests found. Select one with PHYGDPO_PAIRS_CSV.")
    return directory / "seed_0/pairs.csv"


def is_wan14b_negative(path):
    return Path(path).parent.parent.name == "Wan14B"


def positive_manifest(data):
    return find_manifest(
        data, "fast-14b/sft.csv", "fast-14b-positives-only/positives.csv",
        "fast-sft/sft.csv", "fast-dpo/sft.csv", "fast-dpo-positives-only/positives.csv",
    )


STUDIO = Path(__file__).resolve().parents[3]
DEFAULT_RUN = STUDIO / "runs/wan-fast"
ROOT = location("PHYGDPO_RUN_DIR", DEFAULT_RUN)
shared_data = STUDIO / "data"
# A resumed run may have its own dataset from before the shared layout.
run_data = ROOT / "data"
if (run_data / "manifests").is_dir() and (ROOT != DEFAULT_RUN or not shared_data.is_dir()):
    shared_data = run_data
DATA = location("PHYGDPO_DATA_DIR", shared_data)
MODEL = model_directory(ROOT)
PAIRS = pair_manifest(DATA)
generated_manifests = DATA / "manifests/fast-generated"
run_suffix = Path()
if PAIRS.is_relative_to(generated_manifests):
    run_suffix = Path("generated") / PAIRS.parent.relative_to(generated_manifests)
elif os.getenv("PHYGDPO_PAIRS_CSV") and PAIRS.is_file():
    # A custom --pairs-out file also needs its own weights and feature cache.
    if os.getenv("PHYGDPO_NEGATIVE_SOURCE") == "generated" or any(
        not is_wan14b_negative(row.get("negative_video", "")) for row in read_csv(PAIRS)
    ):
        run_suffix = Path("generated") / digest(PAIRS)[:16]
CACHE = location("PHYGDPO_CACHE_DIR", ROOT / "features" / run_suffix)
TRAIN_DIR = location("PHYGDPO_TRAIN_DIR", STUDIO / "models" / "train" / FOLDER / run_suffix)
SFT = TRAIN_DIR / "sft"
DPO = TRAIN_DIR / "dpo"
DATASET = location("PHYGDPO_POSITIVE_CSV", os.getenv("PHYGDPO_SFT_CSV", positive_manifest(DATA)))
WIDTH, HEIGHT, FRAMES, FPS = 832, 480, 81, 15
GPUS = int(os.getenv("PHYGDPO_WORLD_SIZE", "8"))
BATCH = 64
if GPUS <= 0 or BATCH % GPUS:
    raise ValueError("World size must divide the effective batch of 64")
ACCUMULATION = BATCH // GPUS
SAVE_EVERY = 8
DPO_SAVE_EVERY = 4
SEED = 20261121
BETA = 1000.0


SAMPLES = len(read_csv(DATASET)) if DATASET.is_file() else 1403
UPDATES = math.ceil(SAMPLES / BATCH)
INIT_ADAPTER = location(
    "PHYGDPO_INIT_ADAPTER",
    SFT
    / "checkpoints"
    / f"step-{min(SAVE_EVERY,UPDATES):04d}"
    / "pytorch_lora_weights.safetensors",
)
BOUNDARIES = list(range(DPO_SAVE_EVERY, UPDATES, DPO_SAVE_EVERY)) + [UPDATES]


def video_features(row):
    return CACHE / "positive" / f"{row['vid']}.safetensors"


def text_features(row):
    prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
    return CACHE / "text" / f"{row['vid']}-{prompt_hash}.safetensors"


def negative_features(row):
    return (
        CACHE / "negative" / f"{row['vid']}-s{row['negative_seed']}.safetensors",
        "negative",
    )


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_manifest(paired=False):
    rows = read_csv(DATASET)
    if not rows or len({r["vid"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate positive IDs")
    for r in rows:
        path = DATA / r["video"]
        if path.stem != r["vid"] or not path.is_file() or not r["prompt"].strip():
            raise ValueError(f"Missing video or prompt for {r['vid']}")
        if (
            int(r["width"]) * int(r["height"]) != int(r["pixels"])
            or int(r["pixels"]) < WIDTH * HEIGHT
        ):
            raise ValueError(f"Invalid source resolution for {r['vid']}")
    if paired:
        pairs_path = pair_manifest(DATA, validate=True)
        if not pairs_path.is_file():
            raise FileNotFoundError(
                f"Missing negative pair manifest: {pairs_path}. For the default run, download the released "
                "Wan2.1-T2V-14B negatives with 'python downloads/download_training_data.py fast-14b "
                "--negatives-only'. To use your own pairs, pass --negative-source generated or set PHYGDPO_PAIRS_CSV."
            )
        if pairs_path != PAIRS:
            raise ValueError("The selected pair manifest changed during startup. Select it with PHYGDPO_PAIRS_CSV.")
        selection = os.getenv("PHYGDPO_NEGATIVE_SOURCE")
        released = selection == "released" or (selection is None and not os.getenv("PHYGDPO_PAIRS_CSV"))
        pairs = read_csv(pairs_path)
        if len(pairs) != len(rows):
            raise ValueError("SFT and DPO lengths differ")
        for r, p in zip(rows, pairs, strict=True):
            if any(p[k] != v for k, v in r.items()):
                raise ValueError(f"Mismatched SFT and DPO row {r['vid']}")
            path = DATA / p["negative_video"]
            if selection == "generated" and is_wan14b_negative(path):
                raise ValueError(f"Generated pairs requested but {pairs_path} points to a Wan14B negative for {r['vid']}")
            if released and (not is_wan14b_negative(path) or p["negative_seed"] not in ("0", "1", "3")):
                raise ValueError(f"Expected a released Wan2.1-T2V-14B negative for {r['vid']} in {pairs_path}")
            if (
                path.stem != p["vid"]
                or path.parent.name != f"seed_{p['negative_seed']}"
                or not path.is_file()
            ):
                raise ValueError(f"Missing matching negative for {r['vid']}")
        return pairs
    return rows

def select_cases(scenes, requested):
    if not requested:
        return scenes
    missing = set(requested) - {row["case_id"] for row in scenes}
    if missing:
        raise ValueError(f"Unknown evaluation cases: {sorted(missing)}")
    return [row for row in scenes if row["case_id"] in requested]
