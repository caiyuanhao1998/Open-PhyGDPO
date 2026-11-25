"""Fetch the pinned CausVid-fused I2V base used for training."""
import argparse
import json
from pathlib import Path

from model_path import FOLDER
from presets import SPACE
from settings import INIT_ADAPTER, MODEL, ROOT, digest

REPO = "CaiYuanhao/PhyGDPO"
REVISION = "64029230b488e0ab5c32a44144562c7e31969f00"
ADAPTER = "epoch-0.safetensors"
ADAPTER_HASH = "d445132ac0aa057755805fc00919b9cb0bbfd5efdc27d8c296156b33f04b3a3f"
BASE_FILES = json.loads(Path(__file__).with_name("model-files.json").read_text())


def demo_inputs(case_ids=None, verify_only=False):
    """Fetch and verify the released conditioning and noise for the selected Space presets."""
    files = {key: SPACE["scenes"][key]["conditioning_sha256"] for key in
             (case_ids if case_ids is not None else SPACE["scenes"])}
    files["seed-0"] = SPACE["noise_sha256"]
    directory = ROOT / "models/space-inputs"
    paths = {}
    for name, expected in files.items():
        filename = f"templates/fast14b/conditioning/{name}.safetensors"
        path = directory / filename
        if not path.is_file() and not verify_only:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download("CaiYuanhao/PhyGDPO", filename, repo_type="space",
                                       revision=SPACE["revision"], local_dir=directory))
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or incorrect Space input: {path}")
        paths[name] = path
    return paths


def verify_base(directory):
    for name, size in BASE_FILES.items():
        path = directory / name
        if not path.is_file() or (
            name != "transformer/config.json" and path.stat().st_size != size
        ):
            raise ValueError(f"Missing or incomplete model file: {path}")
    fusion = json.loads((directory / "fusion.json").read_text())
    if (
        fusion["base_commit"],
        fusion["causvid_commit"],
        fusion["causvid_weight"],
        fusion["adapted_linear_layers"],
    ) != (
        "b184e23a8a16b20f108f727c902e769e873ffc73",
        "8260d429d19fd7a72304cad059160b95d843913f",
        0.95,
        486,
    ):
        raise ValueError("Unexpected base or CausVid fusion")
    cfg = json.loads((directory / "transformer/config.json").read_text())
    if (cfg["num_layers"], cfg["in_channels"]) != (40, 36):
        raise ValueError("Expected the 40-block I2V transformer with 36 input channels")
    index = json.loads(
        (
            directory / "transformer/diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )
    shards = set(index["weight_map"].values())
    if len(shards) != 12 or not all(("transformer/" + x) in BASE_FILES for x in shards):
        raise ValueError("Unexpected transformer shard map")
    return len(shards)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files without network access",
    )
    p.add_argument(
        "--published-sft",
        action="store_true",
        help="Also fetch/verify the released step-8 adapter for use as PHYGDPO_INIT_ADAPTER",
    )
    p.add_argument("--demo-inputs", action="store_true", help="Also prepare the pinned Space example conditioning and noise")
    a = p.parse_args()
    hub = ROOT / "models/hub"
    path = hub / FOLDER
    try:
        count = verify_base(MODEL)
    except (ValueError, KeyError, FileNotFoundError):
        if a.verify_only or (MODEL.exists() and MODEL.resolve() != path.resolve()):
            raise
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=REPO,
            revision=REVISION,
            allow_patterns=[f"{FOLDER}/{n}" for n in BASE_FILES],
            local_dir=hub,
        )
        count = verify_base(path)
        if MODEL != path and not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            MODEL.symlink_to(path, target_is_directory=True)
    print(
        f"Verified {count} fused transformer shards and pipeline components under {MODEL}"
    )
    if a.published_sft:
        candidates = [MODEL / ADAPTER, hub / FOLDER / ADAPTER]
        if a.verify_only:
            candidates.append(INIT_ADAPTER)
        adapter = next((p for p in candidates if p.is_file() and digest(p) == ADAPTER_HASH), None)
        if adapter is None and not a.verify_only:
            from huggingface_hub import hf_hub_download

            adapter = Path(
                hf_hub_download(
                    repo_id=REPO,
                    revision=REVISION,
                    filename=f"{FOLDER}/{ADAPTER}",
                    local_dir=hub,
                )
            )
        if adapter is None or digest(adapter) != ADAPTER_HASH:
            raise ValueError("Released step-8 adapter SHA-256 mismatch or missing file")
        print(f"PHYGDPO_INIT_ADAPTER={adapter}")
    if a.demo_inputs:
        print(f"Verified {len(demo_inputs(verify_only=a.verify_only))} Space conditioning and noise files")


if __name__ == "__main__":
    main()
