#!/usr/bin/env python3
# inference_joint_slice.py  (k/N 分片 · 单 GPU · video_dir + prompt_dir)

import os, re, math, argparse, glob
from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from template import *  # PROMPT_SA, PROMPT_PHYSICS
from pdb import set_trace as stx
import json

# ─────────── CLI ────────────
parser = argparse.ArgumentParser()
parser.add_argument("-G", type=int, required=True,  help="GPU idx")
parser.add_argument("-k", type=int, required=True,  help="slice id  (1..N)")
parser.add_argument("-N", type=int, required=True,  help="total slices")
parser.add_argument("--video_dir",  type=str, required=True, help="目录下 *.mp4")
parser.add_argument("--prompt_dir", type=str, required=True, help="目录下同名 *.txt")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--lora_checkpoint", default=None, type=str)
parser.add_argument("--batch_size",   type=int, default=1)
parser.add_argument("--num_frames",   type=int, default=32)
args = parser.parse_args()

# ───────── GPU 绑定 ─────────
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[slice {args.k}/{args.N}]  device = {device}")

# ───────── 常量 ─────────
generate_kwargs = dict(do_sample=False, top_k=1, temperature=1e-3, max_length=64)
NUM_MAP = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
           "0":0,"1":1,"2":2,"3":3,"4":4,"5":5}

def parse_score(txt: str) -> int:
    txt = txt.lower().strip()
    for k, v in NUM_MAP.items():
        if k in txt:
            return v
    digits = "".join(c for c in txt if c.isdigit())
    return int(digits) if digits and int(digits) in NUM_MAP.values() else 0

def modify_keys(sd):
    new = defaultdict()
    pat = re.compile(r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight')
    for k, v in sd.items():
        if pat.match(k):
            parts = k.split(".")
            parts.insert(-1, "base_layer")
            k = ".".join(parts)
        new[k] = v
    return new

# ───────── 读取文件 & 构造 DataFrame ─────────
video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
# text_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
if not video_files:
    raise FileNotFoundError(f"No .mp4 found in {args.video_dir}")

records = []
for vp in video_files:
    basename = os.path.splitext(os.path.basename(vp))[0]      # 直接由名字保证对齐
    txt_path = os.path.join(args.prompt_dir, f"{basename}.txt")
    if not os.path.exists(txt_path):
        print(f"[Warning] prompt missing for {basename}, skip.")
        continue
    with open(txt_path, "r", encoding="utf-8") as f:
        caption = f.read().strip().replace("\n", " ")
    records.append({"videopath": vp, "caption": caption, "orig_idx": basename})

if not records:
    raise RuntimeError("No valid video/prompt pairs found.")

df_all = pd.DataFrame(records).sort_values("orig_idx").reset_index(drop=True)

# ───────── 数据切片 ─────────
total = len(df_all)
chunk = math.ceil(total / args.N)
start, end = (args.k - 1) * chunk, min(args.k * chunk, total)
df = df_all.iloc[start:end].copy().reset_index(drop=True)
print(f"处理行 {start}–{end-1}  (共 {len(df)})")

# ───────── Processor ────────
tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
img_proc = MplugOwlImageProcessor.from_pretrained(args.checkpoint)
processor = MplugOwlProcessor(img_proc, tokenizer)

# ───────── Model ────────────
model = MplugOwlForConditionalGeneration.from_pretrained(
    args.checkpoint, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
).eval()

if args.lora_checkpoint:
    cfg = LoraConfig(
        target_modules=r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)",
        inference_mode=True,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, cfg)
    sd = torch.load(args.lora_checkpoint, map_location="cpu")
    try:
        model.load_state_dict(sd)
    except:
        model.load_state_dict(modify_keys(sd))
    print(">> LoRA loaded")

model = model.to(device).to(torch.bfloat16)

def run_task(task: str):
    col = f"{task}_score"
    prompts, vids = [], []
    for _, r in df.iterrows():
        vids.append(r["videopath"])
        prompts.append(PROMPT_SA.format(caption=r["caption"]) if task == "sa" else PROMPT_PHYSICS)
    scores = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(df), args.batch_size),
            desc=f"{task.upper()} | slice{args.k}",
            position=0,
        ):
            ps = prompts[i : i + args.batch_size]
            vs = vids[i : i + args.batch_size]
            inp = processor(text=ps, videos=vs, num_frames=args.num_frames, return_tensors="pt")
            inp = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inp.items()}
            inp = {k: v.to(device) for k, v in inp.items()}
            outs = model.generate(**inp, **generate_kwargs)
            scores.extend(
                [parse_score(tokenizer.decode(o.tolist(), skip_special_tokens=True)) for o in outs]
            )
    df[col] = scores

# ---------- 推理 ----------
run_task("sa")
run_task("pc")
df["joint_score"] = ((df["sa_score"] >= 4) & (df["pc_score"] >= 4)).astype(int)

# ---------- 保存 ----------
os.makedirs(args.output_dir, exist_ok=True)
out_csv = os.path.join(args.output_dir, f"slice{args.k}.csv")
df.to_csv(out_csv, index=False)
print(f"[slice {args.k}] 完成，保存至 {out_csv}")