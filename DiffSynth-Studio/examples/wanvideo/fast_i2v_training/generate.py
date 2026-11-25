"""Generate a Fast-14B video or build DPO pairs from the positive training videos."""
import argparse
import csv
import fcntl
import filecmp
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory

from prepare_model import ADAPTER, ADAPTER_HASH, FOLDER, demo_inputs, verify_base
from presets import PLAN, SPACE, TEMPLATES, matching_template
from run import launch
from settings import DATA, FPS, GPUS, MODEL, ROOT, digest, positive_manifest, read_csv


def space_case(args):
    from PIL import Image, ImageOps

    if (args.width, args.height, args.frames, args.seed) != (832, 480, 81, 0):
        return None
    with Image.open(args.image) as image:
        scene = matching_template(args.prompt.strip(), ImageOps.exif_transpose(image).convert("RGB"))
    return TEMPLATES[scene]["case_id"] if scene is not None else None


def inputs(args):
    if args.mode == "video":
        image = args.image.resolve(strict=True)
        if not args.prompt.strip() or args.output.suffix != ".mp4":
            raise ValueError("Supply a nonempty prompt and an .mp4 output path")
        return [], [dict(case_id=args.output.stem, scene=args.output.stem, prompt=args.prompt.strip(),
                         seed=args.seed, image=str(image), image_sha256=digest(image))]
    if args.sft_csv is None:
        args.sft_csv = positive_manifest(args.data_dir)
    rows = read_csv(args.sft_csv)
    if rows and (not {"vid", "prompt", "sport", "width", "height", "pixels", "video"} <= rows[0].keys()
                 or {"negative_video", "negative_seed"}.intersection(rows[0])):
        raise ValueError("Expected a positive-video CSV from the Wan-2.1-Fast-14B downloader")
    if not rows or len({r["vid"] for r in rows}) != len(rows):
        raise ValueError("The positive-video manifest must contain distinct, nonempty video IDs")
    jobs = []
    for row in rows:
        vid = row["vid"]
        source = (args.data_dir / row["video"]).resolve(strict=True)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", vid) or source.stem != vid or not row["prompt"].strip():
            raise ValueError(f"Invalid positive video or prompt for {vid!r}")
        jobs.append(dict(case_id=vid, scene=vid, prompt=row["prompt"], seed=args.seed,
                         positive_video=str(source), positive_video_sha256=digest(source)))
    return rows, jobs


def publish_pairs(path, rows, media, seed, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as fp:
        tmp = Path(fp.name)
        try:
            columns = [*rows[0], "negative_seed", "negative_video"]
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                video = media / (row["vid"] + ".mp4")
                record = json.loads(video.with_suffix(".json").read_text())
                if record["seed"] != seed or record["video_sha256"] != digest(video):
                    raise ValueError(f"Incomplete generated negative: {video}")
                writer.writerow(dict(row, negative_seed=seed, negative_video=video.relative_to(data).as_posix()))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.link(tmp, path)
    except FileExistsError:
        if not filecmp.cmp(tmp, path, shallow=False):
            raise ValueError(f"Different DPO pairs already exist at {path}. Choose another --pairs-out.")
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--width", type=int, default=832)
    common.add_argument("--height", type=int, default=480)
    common.add_argument("--frames", type=int, default=81)
    common.add_argument("--steps", type=int, default=4)
    common.add_argument("--plan", action="store_true", help="Inspect the inputs without using a GPU")
    modes = parser.add_subparsers(dest="mode", required=True)
    video = modes.add_parser("video", parents=[common], help="Generate from an image and prompt")
    weights = video.add_mutually_exclusive_group()
    weights.add_argument("--base", action="store_true", help="Use the fused base model by itself")
    weights.add_argument("--adapter", type=Path, help="Use a custom PhyGDPO adapter")
    video.add_argument("--image", type=Path, required=True)
    video.add_argument("--prompt", required=True)
    video.add_argument("--output", type=Path, required=True)
    negative = modes.add_parser("negatives", parents=[common], help="Generate one negative per positive video")
    negative.add_argument("--adapter", type=Path, help="Use a custom PhyGDPO adapter")
    negative.add_argument("--positive-csv", "--sft-csv", dest="sft_csv", type=Path,
                          default=os.getenv("PHYGDPO_POSITIVE_CSV", os.getenv("PHYGDPO_SFT_CSV")))
    negative.add_argument("--pairs-out", type=Path, help="Explicit destination for the completed pairs.csv")
    negative.add_argument("--seed-manifest", action="store_true",
                          help="Write manifests/fast-generated/seed_<seed>/pairs.csv under the data directory")
    negative.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
        parser.error("width and height must be positive multiples of 16")
    if not 1 <= args.steps <= 30 or not 1 <= args.frames <= 81 or args.frames % 4 != 1:
        parser.error("use 1–30 steps and a frame count of 4n + 1 up to 81")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    rows, scenes = inputs(args)
    defaults = dict(PLAN["settings"])
    preset = space_case(args) if args.mode == "video" else None
    if args.mode == "negatives":
        defaults["fps"] = FPS
    plan = dict(settings=dict(defaults, width=args.width, height=args.height, frames=args.frames, steps=args.steps),
                scenes=scenes)
    adapter = args.adapter
    if adapter is None and not (args.mode == "video" and args.base):
        local = MODEL / ADAPTER
        adapter = local if local.is_file() else ROOT / "models/hub" / FOLDER / ADAPTER
    if args.mode == "negatives":
        data = args.data_dir.resolve()
        if args.seed_manifest and args.pairs_out is None:
            args.pairs_out = data / "manifests/fast-generated" / f"seed_{args.seed}" / "pairs.csv"
        pairs = args.pairs_out or data / "manifests/fast-generated" / "<configuration>" / f"seed_{args.seed}" / "pairs.csv"
    if args.plan:
        print(json.dumps(dict(inputs=len(scenes), settings=plan["settings"], seed=args.seed, adapter=str(adapter) if adapter is not None else None, preset=preset,
                              output=str(args.output if args.mode == "video" else pairs)), indent=2))
        return
    verify_base(MODEL)
    adapter_hash = None
    if adapter is not None:
        if not adapter.is_file():
            parser.error(f"Missing adapter {adapter}. Run prepare_model.py --published-sft.")
        adapter_hash = digest(adapter)
        if args.adapter is None and adapter_hash != ADAPTER_HASH:
            raise ValueError("Published Wan-2.1-Fast-14B adapter SHA-256 mismatch")
    if args.mode == "negatives":
        import imageio_ffmpeg
        import torch
        from cache_identity import component_hash, fingerprint, recipe_hash

        # Hardware and frame decoding can change negative pixels for the same seed.
        plan["runtime"] = dict(workers=GPUS, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                               ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),
                               reader_sha256=digest(Path(__file__).with_name("encode.py")))
        run_id = fingerprint([plan, adapter_hash, component_hash("transformer"), component_hash("scheduler"),
                              recipe_hash("evaluation")])[:20]
        pairs = args.pairs_out or data / "manifests/fast-generated" / run_id / f"seed_{args.seed}" / "pairs.csv"
        media = data / "DPO_data/Wan2.1-Fast-14B" / run_id / f"seed_{args.seed}"
    else:
        media = args.output.resolve().parent
    if preset:
        assets = demo_inputs([preset])
        scenes[0].update(conditioning=str(assets[preset]), noise=str(assets["seed-0"]),
                         conditioning_sha256=SPACE["scenes"][preset]["conditioning_sha256"],
                         noise_sha256=SPACE["noise_sha256"])
    media.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fast-i2v-", dir=media) as work:
        jobs = Path(work) / "jobs.json"
        jobs.write_text(json.dumps(plan, ensure_ascii=False))
        options = ["--jobs", jobs, "--output-dir", media]
        if adapter is not None:
            options.extend(["--adapter", adapter.resolve()])
        if args.mode == "video":
            options.extend(["--feature-dir", Path(work) / "features"])
        # A retry can reuse complete videos. Only publish a pairs CSV after every worker succeeds.
        with (media / ".fast-generation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not preset:
                launch("evaluate.py", "features", *options)
            launch("evaluate.py", "generate", *options)
            if args.mode == "negatives":
                publish_pairs(pairs, rows, media, args.seed, data)
                print(f"PHYGDPO_PAIRS_CSV={pairs}", flush=True)


if __name__ == "__main__":
    main()
