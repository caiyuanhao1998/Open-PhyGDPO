# -*- coding: utf-8 -*-
"""
Video + CoT + JSON verifier for Qwen2.5-VL
------------------------------------------
Reads a Markdown file containing CoT rules and a JSON list of dicts,
each with a `vid` field and corresponding LLM-extended prompt. For each video,
loads the sampled frames and structured description, sends both to the VLM,
and writes the updated JSON (corrected by VLM) to `--out`.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import tempfile
import shutil
import os
import textwrap
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pdb import set_trace as stx

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True,
                help="directory containing .mp4 files")
ap.add_argument("--cot_md", required=True, help="Markdown file with CoT rules")
ap.add_argument("--prompts_json", required=True,
                help="JSON list: each item is a dict with 'vid', 'original', etc.")
ap.add_argument("--model_dir", default="Qwen/Qwen2.5-VL-72B-Instruct")
ap.add_argument("--out", default="Qwen2.5_VL_verification.jsonl")
ap.add_argument("--num_frames", type=int, default=4)
ap.add_argument("--height", type=int, default=360)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--tp", type=int, default=8)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--retry", type=int, default=50)
ap.add_argument("--physics_score_threshold", type=float, default=0.7)
ap.add_argument("--bias", type=int, default=0)
ap.add_argument("--output_folder", default="processed_json")
args = ap.parse_args()
args.frames = min(args.num_frames, 4)

print(f"Processing {args.prompts_json}")

os.makedirs(args.output_folder,exist_ok=True)


# ---------- Load resources ----------
cot_rules = pathlib.Path(args.cot_md).read_text(encoding="utf-8").strip()
with open(args.prompts_json, "r", encoding="utf-8") as f:
    prompts_list = json.load(f)

# ---------- physics score threshold ----------
physics_score_threshold = args.physics_score_threshold

# stx()

prompts_list = [
    item for item in prompts_list if item.get("physics_related_score", 0) >= physics_score_threshold]

bias = args.bias

prompts_list = prompts_list[bias:]

# stx()

# ---------- Frame sampling ----------


def sample_frames(mp4: str, n: int, h: int, w: int) -> List[np.ndarray]:
    tmp = tempfile.mkdtemp()
    try:
        out = f"{tmp}/f_%03d.jpg"
        cmd = [
            "ffmpeg", "-i", mp4,
            "-vf", f"select='not(mod(n\\,{max(1,n)}))',scale={w}:{h}",
            "-vframes", str(n),
            out, "-hide_banner", "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        imgs = [np.array(Image.open(p).convert("RGB"))
                for p in sorted(pathlib.Path(tmp).glob("*.jpg"))]
    finally:
        shutil.rmtree(tmp)
    return imgs


# ---------- Tokenizer & LLM ----------
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

llm = LLM(
    model=args.model_dir,
    tensor_parallel_size=args.tp,
    dtype=args.dtype,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 1},
    max_model_len=32768,
    max_num_batched_tokens=32768
)

sampling = SamplingParams(temperature=0.2, top_p=1.0, max_tokens=1024)

# ---------- Prompt builder ----------


def build_prompt(system_msg: str, json_dict: dict, rules: str) -> str:
    """
    - Still use only a single <|video_pad|> placeholder
    - Place the CoT rules inside a markdown block to avoid incorrect tokenization
    """
    json_dict_slim = {
        "original":  json_dict.get("original", ""),
        "parse":     json_dict.get("parse", {}),
        "reason":    json_dict.get("reason", ""),
        "extended":  json_dict.get("extended", ""),
    }

    return textwrap.dedent(f"""\
        <|im_start|>system
        {system_msg}
        <|im_end|>
        <|im_start|>user
        <|vision_start|><|video_pad|><|vision_end|>

        Here are the reasoning rules:

        ```markdown
        {rules}
        ```

        Here is the existing structured description for the upcoming video (one JSON object):

        ```json
        {json.dumps(json_dict_slim, ensure_ascii=False)}
        ```
        <|im_end|>
        <|im_start|>assistant
    """)


system_prompt = f"You are a multi-modal assistant that reasons about physics in videos."

system_prompt += ("\n\nIMPORTANT: You MUST output **ONE single-line JSON object** ")

# stx()

# ---------- Run VLM on all entries ----------
out_path = pathlib.Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as fout:
    for item in tqdm(prompts_list):
        vid_key = item.get("vid", "")
        if not vid_key:
            print(f"[Skip] entry missing 'vid'")
            continue

        video_path = os.path.join(args.video_dir, f"{vid_key}.mp4")
        if not os.path.exists(video_path):
            print(f"[Missing] {video_path}")
            continue

        frames = sample_frames(video_path, args.frames,
                               args.height, args.width)

        prompt_cot = build_prompt(system_prompt, item, cot_rules)

        outputs_cot = llm.generate([{
            "prompt": prompt_cot,
            "multi_modal_data": {"video": [frames]}}],
            sampling)

        result_cot = outputs_cot[0].outputs[0].text.strip()
        retries = 0

        ok = False
        while retries <= args.retry:
            # ---------- check string ----------
            has_score = "physics_related_score" in result_cot
            has_label = "physics_label" in result_cot
            bracket_ok = result_cot.count('[') == result_cot.count(']')

            # ---------- parse into json ----------
            try:
                result_dict = json.loads(
                    result_cot.strip()) if has_score and has_label and bracket_ok else None
            except json.JSONDecodeError:
                result_dict = None

            # ---------- stop ----------
            if result_dict is not None:
                ok = True
                break

            # ---------- printing warning ----------
            if not has_score:
                print(f"\n[Warn] Missing physics_related_score, retry attempt {retries}")

            if not has_label:
                print(f"\n[Warn] Missing physics_label, retry attempt {retries}")

            if not bracket_ok:
                print(f"\n[Warn] Mismatched square brackets, retry attempt {retries}")

            if result_dict is None and has_score and has_label and bracket_ok:
                print(f"\n[Warn] json.loads failed, retry attempt {retries}")

            # ---------- re-gen ----------
            retries += 1
            if retries > args.retry:
                break
            retry_prompt = build_prompt(system_prompt, item, cot_rules)
            retry_out = llm.generate([{
                "prompt": retry_prompt,
                "multi_modal_data": {"video": [frames]}
            }], sampling)
            result_cot = retry_out[0].outputs[0].text.strip()

        # ---------- write or skip ----------
        if ok:
            try:
                result_cot_dict = json.loads(result_cot)
                result_cot_dict["vid"] = vid_key
                fout.write(json.dumps(result_cot_dict,
                           ensure_ascii=False) + "\n")
                # print(f"{vid_key}.mp4\t{json.dumps(result_cot_dict, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"[Error] Failed to parse {vid_key} after {args.retry} retries. Skipping.")
                continue
        else:
            print(f"[Error] Parsing {vid_key} failed after {args.retry} attempts. Skipped.")
        # stx()
    fout.seek(0, os.SEEK_END)
    fout.truncate()
