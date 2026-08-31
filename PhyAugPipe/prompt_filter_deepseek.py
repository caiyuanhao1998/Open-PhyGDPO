from __future__ import annotations
import argparse
import json
import pathlib
import sys
import os
import re
import time
from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pdb import set_trace as stx

# ------------------------- CLI ------------------------- #
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--prompts_json", required=True, help="raw prompt")
parser.add_argument("--md_file",   default="prompt_filter_deepseek.md")
parser.add_argument("--out",       default="results.jsonl")
parser.add_argument(
    "--model_dir", default="models/DeepSeek-R1-Distill-Llama-70B/")
parser.add_argument("--tp",        type=int, default=8)
parser.add_argument(
    "--dtype",     choices=["bfloat16", "float16"], default="bfloat16")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_tokens", type=int, default=256)     # ▲ 提到 256
parser.add_argument("--retry",     type=int,
                    default=10, help="max retry times")
parser.add_argument("--max_len",   type=int, default=1000)
parser.add_argument("--output_folder", default="processed_json")
args = parser.parse_args()


print(f"Processing {args.prompts_json}")

os.makedirs(args.output_folder, exist_ok=True)

# ------------------------- 指令 & prompt ------------------------- #
sys_inst = pathlib.Path(args.md_file).expanduser(
).read_text(encoding="utf-8").rstrip()
sys_inst += (
    "\n\nIMPORTANT: You MUST output **ONE single-line JSON object** "
    "with keys `original`, `parse`, `reason`, `extended` – nothing else."
    "**All brackets must be balanced**: every `[` must have a matching `]`, every `{` must have a matching `}`."
)

json_path = pathlib.Path(args.prompts_json).expanduser()
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

max_len = args.max_len

# 提取 caption 字段组成列表
prompts_raw: List[str] = [item["caption"].strip()
                          for item in data if "caption" in item and item["caption"].strip()]

vid_raw = [item["vid"].strip()
           for item in data if "vid" in item and item["vid"].strip()]

prompts_raw = prompts_raw[:max_len]
vid_raw = vid_raw[:max_len]

if not prompts_raw:
    sys.exit(f"[Error] {args.prompts_json} is empty.")

if not vid_raw:
    sys.exit(f"[Error] {args.prompts_json} is empty.")

assert len(prompts_raw) == len(
    vid_raw
), f"Length mismatch between prompts_raw and vid_raw: {len(prompts_raw)} != {len(vid_raw)}"

print(f"❯ Loaded {len(prompts_raw)} prompts")
print(f"❯ Loaded {len(vid_raw)} videos")

# stx()

# ------------------------- Tokenizer ------------------------- #
tok = AutoTokenizer.from_pretrained(
    os.path.expanduser(args.model_dir), trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def make_chat(user: str, strict=False) -> str:
    """
    When strict=True, append "ONLY JSON" to the end of the user prompt to force the model
    to retry and output JSON only.
    """
    extra = "\n\nONLY JSON." if strict else ""
    msgs = [
        {"role": "system", "content": sys_inst},
        {"role": "user",   "content": user + extra}
    ]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ------------------------- vLLM ------------------------- #
llm = LLM(
    model=os.path.expanduser(args.model_dir),
    tensor_parallel_size=args.tp,
    dtype=args.dtype,
    max_model_len=32768,
    max_num_batched_tokens=32768
)

sampling = SamplingParams(
    temperature=0.2,
    max_tokens=1024,
    top_p=1.0, top_k=-1
)

# ------------------------- JSON 提取函数 ------------------------- #
_json_pat = re.compile(r"\{.*\}", re.S)


def extract_json(txt: str) -> str | None:
    m = _json_pat.search(txt)
    return m.group(0).strip() if m else None

# ------------------------- 推理 ------------------------- #


def batched(seq_1, seq_2, n):
    for i in range(0, len(seq_1), n):
        yield i, seq_1[i:i+n], seq_2[i:i+n]


print("❯ Starting inference...")
with open(args.out, "w", encoding="utf-8") as fw, tqdm(total=len(prompts_raw), unit="prompt") as pbar:
    for start_idx, batch_raw, batch_vid in batched(prompts_raw, vid_raw, args.batch_size):
        # ---- 构造 chat prompt ----
        batch_chat = [make_chat(p, strict=True) for p in batch_raw]
        outputs = llm.generate(batch_chat, sampling)

        for offset, o in enumerate(outputs):
            raw_prompt = batch_raw[offset]
            vid = batch_vid[offset]
            txt = o.outputs[0].text.strip()
            json_txt = extract_json(txt)
            # stx()
            '''
                中间结果:
                    raw_prompt:
                        A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area
                    txt:
                        'Alright, I need to process the prompt "A cup of water is slowly poured out in the space station, releasing 
                        the liquid into the surrounding area" through the three steps: Parse, Reason, and Decorate.\n\nFirst, for the 
                        Parse step, I\'ll identify the entities, actions, forces, and outcomes. The entities are the cup of water and the space station. 
                        The action is the cup being poured. The forces involved are microgravity and surface tension. The outcome is the liquid floating.
                        \n\nNext, in the Reason step, I\'ll explain how the entities interact. In microgravity, the water doesn\'t pour like on Earth. 
                        Surface tension causes it to form droplets and float around the space station.\n\nFor the Decorate step, I\'ll add three visual 
                        details. I\'ll choose soft white lighting, transparent droplets, and a wide-angle shot to show the scene in the space station.\n\n
                        Putting it all together, I\'ll structure the JSON with the original prompt, parse details, reason, and extended description.\n</think>\n\n
                            ```json\n{"original":"A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area",
                                    "parse":{"entities":["cup of water (liquid)","space station (microgravity environment)"],
                                            "actions":["cup of water is poured","liquid is released"],
                                            "forces":["microgravity","surface tension"],
                                            "outcomes":["liquid floats","dispersion of droplets"]},
                                    "reason":"In the microgravity environment of the space station, the cup of water is poured, causing the liquid to float freely. Surface tension keeps the water together as it disperses into droplets in the surrounding area.",
                                    "physics_related_score": 0.65,
                                    "physics_label": 1,
                                    "extended":"In the soft white lighting of the space station, the cup of water is poured, releasing transparent droplets that float in mid-air. The water forms perfect spheres due to surface tension, captured in a wide-angle shot as they drift slowly around the station."}\n```'
            '''

            # ----------- 若失败则重试一次 -----------
            retries = 0
            while retries < args.retry:
                # ---------- 尝试把 json_txt 解析成 dict ----------
                try:
                    obj = json.loads(json_txt) if json_txt else None
                except json.JSONDecodeError as e:
                    print(f"\n[Warn] JSON parsing failed for item {start_idx + offset}, retry attempt {retries}")
                    obj = None

                # ---------- success condition ----------
                ok = (
                    obj is not None and
                    'physics_related_score' in txt and
                    'physics_label' in txt and
                    json_txt is not None and
                    json_txt.count('[') == json_txt.count(']')
                )
                if ok:
                    break

                # ---------- print warning ----------
                if 'physics_related_score' not in txt:
                    print(f"\n[Warn] Missing physics_related_score, retry attempt {retries}")

                if 'physics_label' not in txt:
                    print(f"\n[Warn] Missing physics_label, retry attempt {retries}")

                if json_txt is None:
                    print(f"\n[Warn] Failed to extract JSON, retry attempt {retries}")

                elif json_txt.count('[') != json_txt.count(']'):
                    print(f"\n[Warn] Mismatched square brackets in JSON, retry attempt {retries}")

                # ---------- re-gen ----------
                retries += 1
                retry_prompt = make_chat(raw_prompt, strict=True)
                retry_out = llm.generate([retry_prompt], sampling)[0]
                txt = retry_out.outputs[0].text.strip()
                json_txt = extract_json(txt)

            if json_txt is None:
                print(f"\n[Warn] Failed to extract JSON for item {start_idx + offset}, skipped")
                continue

            if json_txt.count('[') != json_txt.count(']'):
                print(f"\n[Warn] Invalid JSON format for item {start_idx + offset}, skipped")
                continue

            # stx()

            try:
                # obj = json.loads(json_txt.encode('utf-8').decode('unicode_escape'))
                obj = json.loads(json_txt)
            except json.JSONDecodeError:
                print(f"\n[Warn] JSON parsing failed for item {start_idx + offset}, skipped")
                continue

            obj["vid"] = vid
            obj["original"] = raw_prompt
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

        pbar.update(len(batch_raw))

print(f"✅ All items processed. Results have been written to {args.out}")
