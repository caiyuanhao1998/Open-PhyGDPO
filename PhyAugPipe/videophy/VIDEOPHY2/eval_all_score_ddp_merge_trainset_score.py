#!/usr/bin/env python3
# merge_slices.py
# 合并各 slice*.csv，并把整体 / is_hard / OI / SPA 的 joint 比例写入 joint_score.txt

import glob
import pandas as pd
import argparse
import os
import sys
from pdb import set_trace as stx

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Merge slice CSVs and compute joint score ratios")
ap.add_argument("--slice_dir",  required=True,  help="包含 slice*.csv 的目录")
ap.add_argument("--outfile",    required=True,  help="合并后输出 CSV 路径")
ap.add_argument("--ratio_file", default=None,   help="joint_score.txt 保存路径（默认与 outfile 同目录）")
args = ap.parse_args()

# ---------- 1. 收集切片 ----------
files = sorted(glob.glob(os.path.join(args.slice_dir, "slice*.csv")))
if not files:
    sys.exit(f"[Error] No slice*.csv found in {args.slice_dir}")
print(f"Found {len(files)} slice files, merging...")

# ---------- 2. 合并 ----------
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
# stx()  # 需要调试时取消注释

# ---------- 3. 计算 joint_ratio ----------
ratio_file = args.ratio_file or os.path.join(
    os.path.dirname(os.path.abspath(args.outfile)), "joint_score.txt"
)

# 若还未有 joint_score 列则先计算
if {"sa_score", "pc_score"} <= set(df.columns):
    meet_cond          = (df["sa_score"] >= 4) & (df["pc_score"] >= 4)
    df["joint_score"]  = meet_cond.astype(int)
elif "joint_score" not in df.columns:
    sys.exit("[Error] joint_score 列不存在，且缺少 sa_score/pc_score 无法计算")

# 只计算 overall_ratio
overall_ratio = round(df["joint_score"].mean(), 4)

# ---------- 4. 打印到终端 ----------
print(f"Overall={overall_ratio:.4f}")

# ---------- 5. 写 joint_score.txt ----------
with open(ratio_file, "w") as f:
    f.write(f"overall\t{overall_ratio:.4f}\n")
print("All ratios saved →", ratio_file)


# ---------- 6. 保存合并后的 CSV ----------
os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
df.to_csv(args.outfile, index=False)
print("Merged CSV saved →", args.outfile)