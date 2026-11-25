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
from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.distributed.wan_video import EXAMPLE_PROMPTS, NEGATIVE_PROMPT, WanVideoCheckpoint
from diffsynth.distributed.wan_video_outputs import denoising_progress, generation_record, local_devices, publish
from tqdm import tqdm
import argparse

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

class LitWan(pl.LightningModule):
    """Tensor-parallel Wan2.1-T2V-14B with the PhyGDPO adapter."""

    def __init__(self, lora_path, model_dir="models/Wan-AI/Wan2.1-T2V-14B", report_progress=False):
        super().__init__()
        self.adapter_path = lora_path

        self.model_dir = model_dir
        self.report_progress = report_progress
        checkpoint = WanVideoCheckpoint.from_directory(model_dir, "14B", lora_path)
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [list(map(str, checkpoint.dit)), str(checkpoint.text_encoder), str(checkpoint.vae)],
            torch_dtype=torch.bfloat16,
        )

        if lora_path is not None:
            model_manager.load_lora(lora_path, lora_alpha=1.0)

        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )

    # ---------------------------------------------------------------------
    # 2. Define tensor‑parallel partition plan (identical to 14 B) ---------
    # ---------------------------------------------------------------------

    def configure_model(self):
        tp_mesh = self.device_mesh["tensor_parallel"]
        plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "head": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
                use_local_output=True,
            )
        }
        self.pipe.dit = parallelize_module(self.pipe.dit, tp_mesh, plan)
        for block_id, block in enumerate(self.pipe.dit.blocks):
            layer_tp_plan = {
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
                "self_attn.o": RowwiseParallel(input_layouts=Shard(2), output_layouts=Replicate()),

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
                "cross_attn.o": RowwiseParallel(input_layouts=Shard(2), output_layouts=Replicate(), use_local_output=False),

                "ffn.0": ColwiseParallel(input_layouts=Shard(1)),
                "ffn.2": RowwiseParallel(output_layouts=Replicate()),

                "norm1": SequenceParallel(use_local_output=True),
                "norm2": SequenceParallel(use_local_output=True),
                "norm3": SequenceParallel(use_local_output=True),
                "gate": PrepareModuleInput(
                    input_layouts=(Shard(1), Replicate(), Replicate()),
                    desired_input_layouts=(
                        Replicate(), Replicate(), Replicate()),
                )
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

    def test_step(self, batch):
        data = batch[0]
        progress = denoising_progress if self.report_progress else tqdm
        data["progress_bar_cmd"] = progress if self.local_rank == 0 else lambda x: x
        output_path = data.pop("output_path")
        with torch.no_grad(), torch.inference_mode(False):
            video = self.pipe(**data)
        if self.global_rank == 0:
            record = generation_record("14B", data, "native", directory=self.model_dir, adapter=self.adapter_path)
            publish(video, output_path, record)


# -----------------------------------------------------------------------------
# Entry point -----------------------------------------------------------------
# -----------------------------------------------------------------------------


# ---------------- main ------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--base", action="store_true", help="Use the base model without an adapter")
    weights.add_argument("--lora_path", type=str, default="models/train/Wan2.1-T2V-14B_PhyGDPO/step-0001000.safetensors")
    parser.add_argument("--model_dir", default="models/Wan-AI/Wan2.1-T2V-14B")
    parser.add_argument("--prompt", default=EXAMPLE_PROMPTS["14B"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output_path", default="gymnastics_14B_PhyGDPO.mp4")
    parser.add_argument("--report-progress", action="store_true")
    args = parser.parse_args()
    if args.base:
        args.lora_path = None

    tasks = [dict(prompt=args.prompt, negative_prompt=NEGATIVE_PROMPT, seed=args.seed,
                  width=args.width, height=args.height, num_frames=args.frames,
                  num_inference_steps=args.num_inference_steps, tiled=False, output_path=args.output_path)]

    dataloader = torch.utils.data.DataLoader(
        ToyDataset(tasks), collate_fn=lambda x: x
    )

    model = LitWan(lora_path=args.lora_path, model_dir=args.model_dir, report_progress=args.report_progress)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=local_devices(),
        strategy=ModelParallelStrategy(),
    )
    trainer.test(model, dataloader)
