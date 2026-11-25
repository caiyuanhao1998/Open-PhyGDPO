"""Rebuild the Space base from original, pinned local Wan and CausVid weights."""
import argparse
import hashlib
from tempfile import TemporaryDirectory
import json
import os
from pathlib import Path
import torch
from diffusers import (
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from settings import MODEL


def verify_inputs(base, causvid):
    records = json.loads(Path(__file__).with_name("fusion-inputs.json").read_text())
    paths = [(base / name, oid) for name, oid in records["wan"].items()]
    for path, expected in [*paths, (causvid, records["causvid"])]:
        if len(expected) == 40:  # Ordinary Git blob, rather than LFS.
            digest = hashlib.sha1(f"blob {path.stat().st_size}".encode() + bytes([0]))
        else:
            digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"File differs from the pinned fusion source: {path}")


def fuse(base, causvid, out, device):
    if out.exists():
        raise FileExistsError(out)
    verify_inputs(base, causvid)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as staging:
        partial = Path(staging) / out.name
        partial.mkdir()
        fuse_into(base, causvid, partial, device, destination=out)
        partial.rename(out)


@torch.no_grad()
def fuse_into(base, causvid, partial, device, destination):
    # Preserve the pipeline's mixed storage (time/norm parameters that Diffusers keeps in FP32).
    transformer = WanTransformer3DModel.from_pretrained(
        str(base / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    if transformer.config.in_channels != 36 or transformer.config.image_dim is None:
        raise ValueError("Expected Wan 2.1 I2V-14B")
    state, metadata = WanImageToVideoPipeline.lora_state_dict(
        str(causvid), return_lora_metadata=True
    )
    state = WanImageToVideoPipeline._maybe_expand_t2v_lora_for_i2v(transformer, state)
    WanImageToVideoPipeline.load_lora_into_transformer(
        state, transformer=transformer, adapter_name="causvid_lora", metadata=metadata
    )
    transformer.set_adapters("causvid_lora", weights=0.95)
    adapted = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, BaseTunerLayer)
    ]
    audit = {
        "space_commit": "18146426e964a4cd009f596ec17da34793df48c3",
        "base_commit": "b184e23a8a16b20f108f727c902e769e873ffc73",
        "causvid_commit": "8260d429d19fd7a72304cad059160b95d843913f",
        "causvid_weight": 0.95,
        "adapted_linear_layers": len(adapted),
        "converted_tensors_after_i2v_expansion": len(state),
        "changed_probe_values": {},
    }
    del state
    if len(adapted) < 400:
        raise RuntimeError(audit)
    for name, module in adapted:
        module.to(device)
        if abs(float(module.scaling["causvid_lora"]) - 0.95) > 1e-6:
            raise ValueError((name, module.scaling))
        before = module.get_base_layer().weight[:32, :64].clone()
        module.merge(safe_merge=True, adapter_names=["causvid_lora"])
        if name in (
            "blocks.0.attn1.to_q",
            "blocks.39.ffn.net.2",
            "blocks.0.attn2.add_k_proj",
        ):
            audit["changed_probe_values"][name] = int(
                (before != module.get_base_layer().weight[:32, :64]).sum()
            )
        module.to("cpu")
        parent, _, leaf = name.rpartition(".")
        setattr(transformer.get_submodule(parent), leaf, module.get_base_layer())
    if (
        audit["changed_probe_values"]["blocks.0.attn1.to_q"] == 0
        or audit["changed_probe_values"]["blocks.0.attn2.add_k_proj"] != 0
    ):
        raise RuntimeError(audit)
    if any("lora_" in n for n in transformer.state_dict()):
        raise RuntimeError("Adapter modules remain")
    if hasattr(transformer, "peft_config"):
        del transformer.peft_config
    transformer._hf_peft_config_loaded = False
    transformer.save_pretrained(
        str(partial / "transformer"), safe_serialization=True, max_shard_size="3GB"
    )
    config_file = partial / "transformer/config.json"
    config = json.loads(config_file.read_text())
    config.pop("_name_or_path", None)
    config_file.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for name in (
        "vae",
        "tokenizer",
        "text_encoder",
        "image_encoder",
        "image_processor",
    ):
        if not (partial / name).exists():
            (partial / name).symlink_to(
                os.path.relpath(base / name, destination), target_is_directory=True
            )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(base / "scheduler"), flow_shift=8.0
    )
    scheduler.save_pretrained(str(partial / "scheduler"))
    config = json.loads((base / "model_index.json").read_text())
    config["image_encoder"] = ["transformers", "CLIPVisionModel"]
    (partial / "model_index.json").write_text(json.dumps(config, indent=2) + "\n")
    (partial / "fusion.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers at b184e23",
    )
    parser.add_argument(
        "--causvid",
        type=Path,
        required=True,
        help="Kijai/WanVideo_comfy rank-32 safetensors at 8260d42",
    )
    parser.add_argument("--out", type=Path, default=MODEL)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fuse(args.base.resolve(), args.causvid.resolve(), args.out.resolve(), args.device)
