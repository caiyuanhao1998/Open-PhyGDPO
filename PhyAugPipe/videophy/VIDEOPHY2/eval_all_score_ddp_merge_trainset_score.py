#!/usr/bin/env python3
"""Merge one complete VideoPhy2 scoring run and compute its joint pass rate."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def publish(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice_dir", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--ratio_file", type=Path)
    parser.add_argument("--expected-slices", type=int, default=8)
    parser.add_argument("--video-dir", type=Path, help="Require a score for each MP4 in this directory")
    args = parser.parse_args()
    if args.expected_slices < 1:
        parser.error("--expected-slices must be positive")
    expected = [args.slice_dir / f"slice{i}.csv" for i in range(1, args.expected_slices + 1)]
    actual = set(args.slice_dir.glob("slice*.csv"))
    if actual != set(expected):
        missing = sorted(p.name for p in set(expected) - actual)
        extra = sorted(p.name for p in actual - set(expected))
        parser.error(f"Incomplete or mixed scoring run: missing={missing}, unexpected={extra}")
    frames = [pd.read_csv(p, dtype={"orig_idx": str}, keep_default_na=False) for p in expected]
    for path, frame in zip(expected, frames):
        if "orig_idx" not in frame or not ({"sa_score", "pc_score"} <= set(frame) or "joint_score" in frame):
            parser.error(f"Missing video IDs or score columns in {path}")
        if list(frame.columns) != list(frames[0].columns):
            parser.error(f"Inconsistent score columns in {path}")
    scores = pd.concat(frames, ignore_index=True)
    ids = scores["orig_idx"]
    if scores.empty or ids.eq("").any() or ids.duplicated().any():
        parser.error("Expected one score per video ID; found missing or duplicate IDs")
    if args.video_dir is not None:
        wanted = {path.stem for path in args.video_dir.glob("*.mp4") if path.is_file()}
        scored = set(ids)
        if scored != wanted:
            parser.error(f"Video ID mismatch: missing={sorted(wanted - scored)[:10]}, unexpected={sorted(scored - wanted)[:10]}")
    if {"sa_score", "pc_score"} <= set(scores):
        sa = pd.to_numeric(scores["sa_score"], errors="raise")
        pc = pd.to_numeric(scores["pc_score"], errors="raise")
        if not (sa.between(0, 5).all() and pc.between(0, 5).all()):
            parser.error("Missing or invalid SA/PC values in a score slice")
        scores["joint_score"] = ((sa >= 4) & (pc >= 4)).astype(int)
    elif not scores["joint_score"].isin([0, 1]).all():
        parser.error("Missing or invalid joint scores in a score slice")
    overall = scores["joint_score"].mean()
    ratio_path = args.ratio_file or args.outfile.with_name("joint_score.txt")
    publish(ratio_path, lambda stream: stream.write(f"overall\t{overall:.4f}\n"))
    publish(args.outfile, lambda stream: scores.to_csv(stream, index=False))
    print(f"Merged {len(expected)} slices and {len(scores)} videos into {args.outfile}. Overall={overall:.4f}")


if __name__ == "__main__":
    main()
