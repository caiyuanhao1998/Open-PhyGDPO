# -*- coding: utf-8 -*-
"""
Video + Gemini-1.5 Filtering / Extension + JSON Validation
----------------------------------------------------------
Inputs:
  • --prompts_json      One dict per line, containing vid / original / parse / reason / extended
  • --instruction_md    Markdown rules / CoT instructions
  • --video_dir         Directory containing .mp4 files, named as <vid>.mp4

For each record:
  1. Load the video as binary data
  2. Build Gemini content and send it to Gemini-1.5-pro together with the JSON description
  3. Parse the single-line JSON response; if parsing fails, retry with key-field checks and json.loads validation
  4. Write the updated JSONL output (automatically inserting the "vid" field)
"""
from __future__ import annotations
import argparse
import json
import pathlib
import os
import textwrap
import tempfile
import shutil
import subprocess
import sys
import time
import re
from typing import List, Dict

from google import genai
from google.genai import types
from tqdm import tqdm
from pdb import set_trace as stx

# ---------- CLI ----------
ap = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--video_dir",      required=True,
                help="directory containing .mp4 files")
ap.add_argument("--prompts_json",   required=True,
                help="JSON list of dicts, each with 'vid' etc.")
ap.add_argument("--instruction_md", required=True,
                help="Markdown file with Gemini rules / instructions")
ap.add_argument("--out",            default="gemini_verification.jsonl")
ap.add_argument("--retry",          type=int, default=10,
                help="max retry times per sample")
ap.add_argument(
    "--api_key",        default=os.getenv("GOOGLE_API_KEY"), help="Gemini API key")
ap.add_argument("--model",          default="models/gemini-2.5-pro")
ap.add_argument("--filter_prompt",  default='Is the style of this video unrealistic, such as animation, cartoon, anime, pixel art, video game, 3-D game engine, Lego stop-motion, and so on?')
ap.add_argument("--max_num",        type=int, default=10)
ap.add_argument("--physics_score_threshold", type=float, default=0.7)
ap.add_argument("--bias", type=int, default=0)
args = ap.parse_args()

if not args.api_key:
    sys.exit("Missing GOOGLE_API_KEY (env or --api_key)")

# ---------- Load resources ----------
client = genai.Client(api_key=args.api_key)


print("\n-------------------------------------------------------------------------------")
print("You can use the following Google AI models:")
models = client.models.list()
for m in models:
    name = getattr(m, "name", None)
    if name:
        print(name)
print("-------------------------------------------------------------------------------\n")

# stx()

inst_text = pathlib.Path(args.instruction_md).read_text(
    encoding="utf-8").strip()
with open(args.prompts_json, "r", encoding="utf-8") as f:
    prompts_list: List[Dict] = json.load(f)

# ---------- physics score threshold ----------
physics_score_threshold = args.physics_score_threshold
prompts_list = [
    item for item in prompts_list if item.get("physics_related_score", 0) >= physics_score_threshold]

# stx()
bias = args.bias
prompts_list = prompts_list[bias:bias+args.max_num]


# ---------- Prompt builder ----------
def build_parts(video_bytes: bytes, json_dict: Dict, realistic_filtering: str) -> List[types.Part]:
    json_slim = {
        "original": json_dict.get("original", ""),
        "parse":    json_dict.get("parse", {}),
        "reason":   json_dict.get("reason", ""),
        "extended": json_dict.get("extended", "")
    }
    # stx()
    json_text = json.dumps(json_slim, ensure_ascii=False, indent=2)

    return [
        types.Part(inline_data=types.Blob(
            data=video_bytes, mime_type="video/mp4")),
        types.Part(text=inst_text),
        types.Part(
            text=textwrap.dedent(f"""\
            Here is the structured information in JSON format:\n```json\n{json_text}\n```, {realistic_filtering},
            Please respond with ONE single-line JSON object that updates / verifies the above fields
            and MUST contain "physics_related_score" and "physics_label".
            """)
        )
    ]


# ---------- Main loop ----------
out_path = pathlib.Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
ok_cnt, fail_cnt = 0, 0

with out_path.open("w", encoding="utf-8") as fout:
    for item in tqdm(prompts_list):
        vid = item.get("vid", "")
        # stx()
        if not vid:
            print("[Skip] Missing 'vid' in item")
            continue
        mp4_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(mp4_path):
            print(f"[Missing] {mp4_path}")
            continue

        video_bytes = pathlib.Path(mp4_path).read_bytes()

        retries = 0
        while retries <= args.retry:
            parts = build_parts(video_bytes, item, args.filter_prompt)
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=types.Content(parts=parts)
                )
            except Exception as e:
                print(f"[GeminiError] {vid}, retry attempt {retries}: {e}")
                retries += 1
                continue
            # stx()
            try:
                text_out = resp.text.strip()
            except Exception as e:
                print(f"Prompt for {vid} is None, retry attempt {retries}: {e}")
                retries += 1
                continue
            if text_out.startswith("```"):
                text_out = re.sub(
                    r"^```[a-zA-Z]*\n|\n```$", "", text_out).strip()
            # stx()
            has_score = "physics_related_score" in text_out
            has_label = "physics_label" in text_out
            bracket_ok = text_out.count('[') == text_out.count(']')

            if not (has_score and has_label and bracket_ok):
                if not has_score:
                    print(f"[Warn] Missing physics_related_score (retry {retries})")
                if not has_label:
                    print(f"[Warn] Missing physics_label (retry {retries})")
                if not bracket_ok:
                    print(f"[Warn] Mismatched square brackets (retry {retries})")
                retries += 1
                continue


            try:
                result_dict = json.loads(text_out)
            except json.JSONDecodeError as e:
                print(f"[Warn] json.loads failed (retry {retries}): {e}")
                retries += 1
                continue

            # ---- 成功 ----
            result_dict["vid"] = vid
            fout.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            ok_cnt += 1
            break

        else:
            print(f"[Error] Failed to parse {vid} after {args.retry} retries.")
            fail_cnt += 1

print(f"\nFinished.  OK: {ok_cnt}   Fail: {fail_cnt}   → {args.out}")
