import torch
import lightning as pl
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard
from lightning.pytorch.strategies import ModelParallelStrategy
from diffsynth import ModelManager, WanVideoPipeline, save_video
from tqdm import tqdm
import argparse
import os

# -----------------------------------------------------------------------------
# Minimal dataset wrapper ------------------------------------------------------
# -----------------------------------------------------------------------------


class ToyDataset(torch.utils.data.Dataset):
    """Wrap a list of inference tasks so that we can feed it to a DataLoader."""

    def __init__(self, tasks):
        self.tasks = tasks

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)


# -----------------------------------------------------------------------------
# LightningModule with Tensor Parallel DiT ------------------------------------
# -----------------------------------------------------------------------------

class LitWan13B(pl.LightningModule):
    """Tensor‑parallel version of Wan‑1.3B text‑to‑video pipeline (DiT backbone).
    The TP plan is identical to the 14 B script – only the checkpoint paths and
    model size differ.
    """

    def __init__(self, model_dir, lora_path):
        super().__init__()

        # ── 1. Load checkpoints (single shard for 1.3 B) ──────────────────────
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
                os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(model_dir, "Wan2.1_VAE.pth"),
            ],
            torch_dtype=torch.bfloat16,
        )

        model_manager.load_lora(lora_path, lora_alpha=1.0)

        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )

    # ---------------------------------------------------------------------
    # 2. Define tensor‑parallel partition plan (identical to 14 B) ---------
    # ---------------------------------------------------------------------

    def configure_model(self):
        """Executed by ModelParallelStrategy after device mesh is ready."""
        tp_mesh = self.device_mesh["tensor_parallel"]

        # Top‑level splits ---------------------------------------------------
        top_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "head": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
                use_local_output=True,
            ),
        }
        self.pipe.dit = parallelize_module(self.pipe.dit, tp_mesh, top_plan)

        # Per‑layer plan (self‑attention, cross‑attention, FFN, norms) -------
        layer_plan = {
            # Self‑Attention -------------------------------------------------
            "self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Shard(1), Shard(0)),
            ),
            "self_attn.q": SequenceParallel(),
            "self_attn.k": SequenceParallel(),
            "self_attn.v": SequenceParallel(),
            "self_attn.norm_q": SequenceParallel(),
            "self_attn.norm_k": SequenceParallel(),
            "self_attn.attn": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),
                desired_input_layouts=(Shard(2), Shard(2), Shard(2)),
            ),
            "self_attn.o": RowwiseParallel(
                input_layouts=Shard(2), output_layouts=Replicate()
            ),
            # Cross‑Attention ----------------------------------------------
            "cross_attn": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Shard(1), Replicate()),
            ),
            "cross_attn.q": SequenceParallel(),
            "cross_attn.k": SequenceParallel(),
            "cross_attn.v": SequenceParallel(),
            "cross_attn.norm_q": SequenceParallel(),
            "cross_attn.norm_k": SequenceParallel(),
            "cross_attn.attn": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),
                desired_input_layouts=(Shard(2), Shard(2), Shard(2)),
            ),
            "cross_attn.o": RowwiseParallel(
                input_layouts=Shard(2), output_layouts=Replicate(), use_local_output=False
            ),
            # Feed‑Forward ---------------------------------------------------
            "ffn.0": ColwiseParallel(input_layouts=Shard(1)),
            "ffn.2": RowwiseParallel(output_layouts=Replicate()),
            # Norms & Gating -----------------------------------------------
            "norm1": SequenceParallel(use_local_output=True),
            "norm2": SequenceParallel(use_local_output=True),
            "norm3": SequenceParallel(use_local_output=True),
            "gate": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), Replicate()),
                desired_input_layouts=(Replicate(), Replicate(), Replicate()),
            ),
        }

        # Apply identical plan to every transformer block -------------------
        for blk in self.pipe.dit.blocks:
            parallelize_module(blk, tp_mesh, layer_plan)

    # ---------------------------------------------------------------------
    # 3. Inference step ----------------------------------------------------
    # ---------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        task = batch[0]
        task["progress_bar_cmd"] = tqdm if self.local_rank == 0 else (
            lambda x: x)
        out_path = task.pop("output_path")
        with torch.no_grad(), torch.inference_mode(False):
            video = self.pipe(**task)
        if self.local_rank == 0:
            save_video(video, out_path, fps=15, quality=5)


# -----------------------------------------------------------------------------
# Entry point -----------------------------------------------------------------
# -----------------------------------------------------------------------------


# ---------------- main ------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lora_path", type=str, default="models/train/Wan2.1-T2V-1.3B_PhyGDPO/epoch-0.safetensors")
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="A weightlifter successfully completes a snatch with a 25kg barbell, holding it momentarily overhead.")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size < 1 or 1536 % world_size or 8960 % world_size:
        raise ValueError(
            f"Unsupported tensor-parallel size {world_size}; use 1, 2, 4, or 8 visible GPUs."
        )

    tasks = [
        {
            "prompt": args.prompt,
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "tiled": False,
            "output_path": os.path.join(args.output_folder, "weightlifting_1.3B_PhyGDPO.mp4"),
        },
    ]

    dataloader = torch.utils.data.DataLoader(
        ToyDataset(tasks), collate_fn=lambda x: x
    )

    model = LitWan13B(model_dir=args.model_dir, lora_path=args.lora_path)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=ModelParallelStrategy(),
    )
    trainer.test(model, dataloader)
