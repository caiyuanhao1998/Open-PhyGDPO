import re
import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from template import PROMPT_SA, PROMPT_PHYSICS, PROMPT_RULE
from utils import decoded_responses, parse_score

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'csv')
parser.add_argument('--task', type=str, default='sa', choices=['sa', 'pc', 'rule', 'joint'], help='task')
parser.add_argument('--checkpoint', type = str, required = True, help = 'checkpoint')
parser.add_argument('--lora_checkpoint', default = None, type = str, help = 'lora trained ckpt')
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--num_frames', type = int, default = 32)
parser.add_argument('--output_csv', type = str, required = True, help = 'csv')
parser.add_argument('--task_output_dir', type=Path, help='Also save the individual SA and PC CSVs in joint mode')

args = parser.parse_args()
if args.batch_size < 1:
    parser.error('--batch_size must be positive')

generate_kwargs = {
    'do_sample': False,
    'top_k': 1,
    'temperature': 0.001,
    'max_length': 256,
}

def inference(args, model, df, processor, tokenizer, task):
    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), args.batch_size), desc=task.upper()):
            batch = df.iloc[start : start + args.batch_size]
            if task == 'sa':
                prompts = [PROMPT_SA.format(caption=caption) for caption in batch['caption']]
            elif task == 'pc':
                prompts = [PROMPT_PHYSICS] * len(batch)
            else:
                prompts = [PROMPT_RULE.format(rule=rule) for rule in batch['rule']]
            inputs = processor(
                text=prompts, videos=batch['videopath'].tolist(),
                num_frames=args.num_frames, return_tensors='pt',
            )
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, **generate_kwargs)
            for output in decoded_responses(outputs, inputs, tokenizer):
                print(output.lower().strip())
                scores.append(parse_score(output, task))
    return scores


def modify_keys(state_dict):
    new_state_dict = defaultdict()

    pattern = re.compile(r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight')

    for key, value in state_dict.items():
        if pattern.match(key):
            key = key.split('.')
            key.insert(-1, 'base_layer')
            key = '.'.join(key)
        new_state_dict[key] = value

    return new_state_dict

def load_model(device):
    checkpoint = args.checkpoint

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )
    print('Model Loaded')
    model.eval()

    lora_checkpoint = args.lora_checkpoint
    if lora_checkpoint:
        peft_config = LoraConfig(
            target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
            inference_mode=True, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        with open(lora_checkpoint, 'rb') as f:
            ckpt = torch.load(f, map_location = torch.device("cpu"))
        try:
            model.load_state_dict(ckpt)
        except:
            ckpt = modify_keys(ckpt)
            model.load_state_dict(ckpt)
        print("LOADED")
    return model.to(device=device, dtype=torch.bfloat16), processor, tokenizer


def score_rows(df, device):
    model, processor, tokenizer = (None, None, None) if df.empty else load_model(device)
    if args.task == 'joint':
        sa = inference(args, model, df, processor, tokenizer, 'sa')
        pc = inference(args, model, df, processor, tokenizer, 'pc')
        out = df.assign(sa_score=sa, pc_score=pc)
        out['joint_score'] = ((out['sa_score'] >= 4) & (out['pc_score'] >= 4)).astype(int)
        return out
    return df.assign(score=inference(args, model, df, processor, tokenizer, args.task))


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)

    df = pd.read_csv(args.input_csv)
    required_columns = {'videopath'}
    if args.task in {'sa', 'joint'}:
        required_columns.add('caption')
    if args.task == 'rule':
        required_columns.add('rule')
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        parser.error(f"{args.input_csv} is missing columns: {', '.join(sorted(missing_columns))}")

    if world_size > 1:
        dist.init_process_group('gloo')
    try:
        out = score_rows(df.iloc[rank::world_size], device)
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(out, gathered, dst=0)
            if rank == 0:
                out = pd.concat([part for part in gathered if not part.empty] or gathered[:1]).sort_index()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        if args.task == 'joint' and args.task_output_dir is not None:
            args.task_output_dir.mkdir(parents=True, exist_ok=True)
            df.assign(score=out['sa_score']).to_csv(args.task_output_dir / 'output_sa.csv')
            df.assign(score=out['pc_score']).to_csv(args.task_output_dir / 'output_pc.csv')
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv)

if __name__  == "__main__":
    main()

'''
    CUDA_VISIBLE_DEVICES=0 python inference.py --input_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval.csv --checkpoint /local2/hbansal/videophy2/test_videophy_training/videophy_autoeval_three_models_rule_e3_lr5e-4_bs64_part2_vta_pc_rule/videophy_2_autoeval --output_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval_lr5e-4_bs64_part2_vta_pc_rule_502_rerun.csv
'''
