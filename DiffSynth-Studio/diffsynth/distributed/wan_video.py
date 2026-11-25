from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .. import ModelManager, WanVideoPipeline
from ..models.lora import GeneralLoRAFromPeft
from ..models.utils import init_weights_on_device, load_state_dict
from ..models.wan_video_dit import RMSNorm, WanModel
from ..pipelines.wan_video import model_fn_wan_video


GPU_COUNTS_BY_MODEL = {
    "1.3B": (1, 2, 3, 4),
    "14B": (1, 2, 3, 4, 8),
}
GIB = 1024**3
EXAMPLE_PROMPTS = {
    "1.3B": "A weightlifter successfully completes a snatch with a 25kg barbell, holding it momentarily overhead.",
    "14B": "A gymnast drops from the parallel bars and lands safely on the mat below.",
}
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)


@dataclass(frozen=True)
class WanVideoCheckpoint:
    model_size: str
    dit: tuple[Path, ...]
    text_encoder: Path
    vae: Path
    lora: Path | None = None

    @classmethod
    def from_directory(cls, directory, model_size, lora=None):
        directory = Path(directory)
        if model_size == "1.3B":
            names = ("diffusion_pytorch_model.safetensors",)
        else:
            names = tuple(f"diffusion_pytorch_model-{shard:05d}-of-00006.safetensors" for shard in range(1, 7))
        return cls(
            model_size=model_size,
            dit=tuple(directory / name for name in names),
            text_encoder=directory / "models_t5_umt5-xxl-enc-bf16.pth",
            vae=directory / "Wan2.1_VAE.pth",
            lora=Path(lora) if lora else None,
        )


class RowParallelLinear(nn.Linear):
    def forward(self, x):
        output = F.linear(x, self.weight)
        dist.all_reduce(output, group=self.group)
        if self.bias is not None:
            output.add_(self.bias)
        return output


class ShardedRMSNorm(RMSNorm):
    def norm(self, x):
        sum_of_squares = x.square().sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_of_squares, group=self.group)
        return x * torch.rsqrt(sum_of_squares / self.full_dim + self.eps)


def read_wan_dit(shards):
    weights = {}
    for path in shards:
        weights.update(load_state_dict(str(path)))
    weights, config = WanModel.state_dict_converter().from_civitai(weights)
    if not config or config["has_image_input"]:
        raise ValueError("Expected a Wan2.1 text-to-video DiT checkpoint.")
    return weights, config


@torch.no_grad()
def load_sharded_dit(checkpoint, device, dtype, group):
    weights, config = read_wan_dit(checkpoint.dit)
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    num_heads = config["num_heads"]
    if size > num_heads:
        raise ValueError(f"Wan2.1 supports at most {num_heads} tensor-parallel ranks for this checkpoint.")

    # Leave space for activations, including on a GPU shared with another job.
    free = torch.tensor([torch.cuda.mem_get_info(device)[0]], device=device, dtype=torch.int64)
    available = torch.empty(size, device=device, dtype=torch.int64)
    dist.all_gather_into_tensor(available, free, group=group)
    bytes_per_head = sum(weight.numel() for weight in weights.values()) * dtype.itemsize / num_heads
    capacity = [int((memory - 4 * GIB) / bytes_per_head) for memory in available.tolist()]
    if min(capacity) < 1 or sum(capacity) < num_heads:
        raise torch.cuda.OutOfMemoryError("There is insufficient free GPU memory to shard the DiT across these GPUs.")
    head_counts = [1] * size
    for _ in range(num_heads - size):
        target = min((i for i in range(size) if head_counts[i] < capacity[i]), key=head_counts.__getitem__)
        head_counts[target] += 1

    dim = config["dim"]
    head_dim = dim // num_heads
    head_start = sum(head_counts[:rank])
    local_heads = head_counts[rank]
    head_slice = slice(head_start * head_dim, (head_start + local_heads) * head_dim)
    ffn_dim = config["ffn_dim"]
    ffn_slice = slice(head_start * ffn_dim // num_heads, (head_start + local_heads) * ffn_dim // num_heads)
    ffn_width = ffn_slice.stop - ffn_slice.start
    shards = {}

    with init_weights_on_device():
        dit = WanModel(**config).eval()
        for i, block in enumerate(dit.blocks):
            for name in ("self_attn", "cross_attn"):
                attn = getattr(block, name)
                attn.num_heads = local_heads
                attn.attn.num_heads = local_heads
                width = local_heads * head_dim
                prefix = f"blocks.{i}.{name}"
                for projection in ("q", "k", "v"):
                    setattr(attn, projection, nn.Linear(dim, width, device="meta"))
                    shards[f"{prefix}.{projection}.weight"] = (0, head_slice)
                    shards[f"{prefix}.{projection}.bias"] = (0, head_slice)
                for normalization in ("norm_q", "norm_k"):
                    norm = ShardedRMSNorm(width, eps=getattr(attn, normalization).eps)
                    norm.group = group
                    norm.full_dim = dim
                    setattr(attn, normalization, norm)
                    shards[f"{prefix}.{normalization}.weight"] = (0, head_slice)
                attn.o = RowParallelLinear(width, dim, device="meta")
                attn.o.group = group
                shards[f"{prefix}.o.weight"] = (1, head_slice)

            block.ffn[0] = nn.Linear(dim, ffn_width, device="meta")
            block.ffn[2] = RowParallelLinear(ffn_width, dim, device="meta")
            block.ffn[2].group = group
            shards[f"blocks.{i}.ffn.0.weight"] = (0, ffn_slice)
            shards[f"blocks.{i}.ffn.0.bias"] = (0, ffn_slice)
            shards[f"blocks.{i}.ffn.2.weight"] = (1, ffn_slice)

    # The safetensors are memory mapped. Slice before copying or converting to BF16.
    for name, (axis, section) in shards.items():
        weights[name] = weights[name].narrow(axis, section.start, section.stop - section.start)
    dit.load_state_dict(weights, assign=True, strict=True)
    dit.to(device=device, dtype=dtype).requires_grad_(False)
    if checkpoint.lora is not None:
        adapter = load_state_dict(str(checkpoint.lora))
        loader = GeneralLoRAFromPeft()
        pairs = loader.get_name_dict(adapter)
        if not pairs or loader.match(dit, adapter) is None or any(down not in adapter for _, down in pairs.values()):
            raise ValueError(f"LoRA checkpoint {checkpoint.lora} does not match the selected base model.")
        for name, (up, down) in pairs.items():
            weight_up, weight_down = adapter[up], adapter[down]
            if name in shards:
                axis, section = shards[name]
                if axis == 0:
                    weight_up = weight_up[section]
                else:
                    weight_down = weight_down[:, section]
            # Merge one local delta at a time to keep memory bounded during loading.
            weight_up = weight_up.to(device=device, dtype=dtype)
            weight_down = weight_down.to(device=device, dtype=dtype)
            dit.get_parameter(name).add_(weight_up @ weight_down)
    return dit


def initialize_cfg_groups(model_size):
    world_size = dist.get_world_size()
    if model_size == "1.3B" and world_size not in GPU_COUNTS_BY_MODEL[model_size]:
        raise ValueError("Wan2.1 1.3B supports 1, 2, 3, or 4 GPUs.")
    if model_size == "1.3B" or world_size < 4:
        return dist.group.WORLD, None
    negative_rank = world_size // 2
    positive_group = dist.new_group(list(range(negative_rank)))
    negative_group = dist.new_group(list(range(negative_rank, world_size)))
    return (positive_group if dist.get_rank() < negative_rank else negative_group), negative_rank


class ParallelWanVideo:
    def __init__(self, checkpoint, device, groups):
        self.rank = dist.get_rank()
        self.device = device
        self.dtype = torch.bfloat16
        group, self.negative_rank = groups
        self.pipe = WanVideoPipeline(device=device, torch_dtype=self.dtype)
        if self.rank == 0:
            manager = ModelManager(device="cpu", torch_dtype=self.dtype)
            manager.load_models([str(checkpoint.text_encoder), str(checkpoint.vae)])
            self.pipe.fetch_models(manager)
            if self.pipe.text_encoder is None or self.pipe.vae is None:
                raise ValueError("The checkpoint is missing a recognized text encoder or VAE.")
        self.pipe.dit = load_sharded_dit(checkpoint, device, self.dtype, group)

    def _encode_prompts(self, prompt, negative_prompt):
        if self.rank == 0:
            text_memory = sum(p.numel() * p.element_size() for p in self.pipe.text_encoder.parameters())
            offload_dit = torch.cuda.mem_get_info(self.device)[0] < text_memory + 2 * GIB
            if offload_dit:
                self.pipe.dit.cpu()
            try:
                self.pipe.text_encoder.to(self.device)
                pos = self.pipe.encode_prompt(prompt, positive=True)["context"].contiguous()
                neg = self.pipe.encode_prompt(negative_prompt, positive=False)["context"].contiguous()
            finally:
                self.pipe.text_encoder.cpu()
                torch.cuda.empty_cache()
            if offload_dit:
                self.pipe.dit.to(self.device)
        else:
            shape = (1, self.pipe.prompter.text_len, self.pipe.dit.text_embedding[0].in_features)
            pos = torch.empty(shape, device=self.device, dtype=self.dtype)
            neg = torch.empty_like(pos)
        dist.broadcast(pos, src=0)
        dist.broadcast(neg, src=0)
        return pos, neg

    @torch.inference_mode()
    def generate(self, prompt, negative_prompt, width, height, frames, steps, seed, progress):
        pipe = self.pipe
        height, width = pipe.check_resize_height_width(height, width)
        pipe.scheduler.set_timesteps(steps, shift=5.0)
        latents = pipe.generate_noise((1, 16, (frames - 1) // 4 + 1, height // 8, width // 8),
                                      seed=seed, dtype=torch.float32).to(device=self.device, dtype=self.dtype)
        pos, neg = self._encode_prompts(prompt, negative_prompt)
        for step, timestep in enumerate(pipe.scheduler.timesteps):
            if self.rank == 0:
                progress(step, f"Denoising · step {step + 1} of {steps}")
            model_timestep = timestep.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            if self.negative_rank is None:
                positive = model_fn_wan_video(pipe.dit, x=latents, timestep=model_timestep, context=pos)
                negative = model_fn_wan_video(pipe.dit, x=latents, timestep=model_timestep, context=neg)
            else:
                # The conditional and unconditional predictions are independent within a timestep.
                is_positive = self.rank < self.negative_rank
                prediction = model_fn_wan_video(pipe.dit, x=latents, timestep=model_timestep,
                                                context=pos if is_positive else neg)
                scratch = torch.empty_like(prediction)
                positive, negative = (prediction, scratch) if is_positive else (scratch, prediction)
                dist.broadcast(positive, src=0)
                dist.broadcast(negative, src=self.negative_rank)
            latents = pipe.scheduler.step(negative + 5.0 * (positive - negative), timestep, latents)

        if self.rank == 0:
            progress(steps, "Decoding video")
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info(self.device)[0] < 8 * GIB:
                pipe.dit.cpu()
                torch.cuda.empty_cache()
            try:
                pipe.vae.to(self.device)
                return pipe.tensor2video(pipe.decode_video(latents, tiled=False)[0])
            finally:
                pipe.vae.cpu()
                torch.cuda.empty_cache()
