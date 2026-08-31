import torch
import lightning as pl
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput, PrepareModuleOutput
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module
from lightning.pytorch.strategies import ModelParallelStrategy
from diffsynth import ModelManager, WanVideoPipeline, save_video
from tqdm import tqdm
from modelscope import snapshot_download
import json
import os
from pdb import set_trace as stx
import random
import argparse
import torch.distributed as dist


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, tasks=[]):
        self.tasks = tasks

    def __getitem__(self, data_id):
        return self.tasks[data_id]

    def __len__(self):
        return len(self.tasks)

# 只让 rank0 写入，不然多进程会同时写入一个文件，会报错


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
                ],
                "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16,
        )
        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda")

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
        data["progress_bar_cmd"] = tqdm if self.local_rank == 0 else lambda x: x
        output_path = data.pop("output_path")
        # 若视频不存在，则生成视频；若存在，则跳过
        if not os.path.exists(output_path):
            with torch.no_grad(), torch.inference_mode(False):
                video = self.pipe(**data)
            if self.local_rank == 0:
                save_video(video, output_path, fps=15, quality=5)


if __name__ == "__main__":
    # 读取 PhyGenBench 的 prompt, 构造新的 ToyDataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_json_file', type=str,
                        required=True, help='Path to the prompt JSON file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated videos.')
    parser.add_argument('--use_extended', action='store_true',
                        help='Use extended prompt if set, otherwise use original prompt.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the random number generator.')
    parser.add_argument('--bias', type=int, default=0,
                        help='Bias to start from.')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse the prompt order.')
    args = parser.parse_args()
    prompt_json_file = args.prompt_json_file
    with open(prompt_json_file, 'r') as f:
        prompts = json.load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_dict_list = []
    prompt_id = args.bias
    if args.reverse:
        prompts = prompts[args.bias:][::-1]
    else:
        prompts = prompts[args.bias:]
    for prompt in prompts:
        cur_prompt = prompt["extended"] if args.use_extended else prompt["original"]
        data_dict_list.append({
            "prompt": cur_prompt,
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "num_inference_steps": 50,
            "seed": args.seed,
            "tiled": False,
            "output_path": f"{output_dir}/{prompt['vid']}.mp4",
        })
        # write down the prompt
        # stx()
        if is_rank0():
            with open(f"{output_dir}/{prompt['vid']}.txt", "w") as f:
                f.write(cur_prompt)
        prompt_id += 1

    dataloader = torch.utils.data.DataLoader(
        ToyDataset(data_dict_list),
        collate_fn=lambda x: x
    )
    model = LitModel()
    trainer = pl.Trainer(accelerator="gpu", devices=torch.cuda.device_count(
    ), strategy=ModelParallelStrategy())
    trainer.test(model, dataloader)
