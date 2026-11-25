"""Fetch only the pinned ZIP members used by a PhyGDPO training profile."""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import NamedTuple
import zlib

import requests


HERE = Path(__file__).resolve().parent
VIDGEN = "Fudan-FUXI/VIDGEN-1M"
PAIRS = "CaiYuanhao/PhyGDPO"
VIDGEN_REV = "9158f218b6b81694a7809b6ee74e30003d58a70e"
PAIRS_REV = "f68effe3613d3a6cb2e5a69db53fa7f187adc4a3"
CODE_REV = "a27f7c541cbc62c471618b5d218ef7447f54c0c2"
PROFILES = ("original-1.3b", "fast-14b", "original-14b", "fast-sft", "all-positives")
# Preserve the manifest directory for scripts that explicitly use the older profile.
FAST_PAIR_PROFILES = ("fast-14b", "fast-dpo")
LOCAL = threading.local()


class Member(NamedTuple):
    source: str
    seed: str
    vid: str
    archive: str
    offset: int
    compressed: int
    size: int
    method: int
    crc: int

    def relative_path(self):
        if self.source in ("VidGen", "Prompt"):
            suffix = ".txt" if self.source == "Prompt" else ".mp4"
            return Path("T2V_data/VIDGEN-1M_unzip") / (self.vid + suffix)
        return Path("DPO_data") / self.source / f"seed_{self.seed}" / f"{self.vid}.mp4"


def load_index(filename="members.csv.gz"):
    index = {}
    with gzip.open(
        HERE / "manifests" / filename, "rt", newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            src, seed, vid = row["source"], row["seed"], row["vid"]
            member = Member(
                src,
                seed,
                vid,
                row["archive"],
                int(row["offset"]),
                int(row["compressed"]),
                int(row["size"]),
                int(row["method"]),
                int(row["crc32"], 16),
            )
            key = src, seed, vid
            if key in index or not re.fullmatch(r"[A-Za-z0-9_-]+", vid):
                raise ValueError(f"Invalid or duplicate member {key}")
            index[key] = member
    return index


def select_rows(args, rows):
    if args.ids:
        wanted = {v.strip() for v in args.ids.read_text().splitlines() if v.strip()}
        unknown = wanted - {r["vid"] for r in rows}
        if unknown:
            raise ValueError(f"IDs outside {args.profile}: {sorted(unknown)[:10]}")
        rows = [r for r in rows if r["vid"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Empty selection")
    return rows


def select(args, index):
    if args.profile.startswith("fast-"):
        with (HERE / "manifests/sports-1403.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1403 or len({r["vid"] for r in rows}) != len(rows):
            raise ValueError("The audited 1,403-ID manifest is incomplete")
    else:
        model = "Wan14B" if args.profile == "original-14b" else "Wan1.3B"
        ids = {v.vid for v in index.values() if v.source == model}
        rows = [{"vid": v} for v in sorted(ids)]
    rows = select_rows(args, rows)
    positives_only = getattr(args, "positives_only", False)
    negatives_only = getattr(args, "negatives_only", False)
    members = []
    for r in rows:
        vid = r["vid"]
        if not negatives_only:
            members.append(index["VidGen", "", vid])
        if args.profile in FAST_PAIR_PROFILES and not positives_only:
            members.append(index["Wan14B", r["negative_seed"], vid])
        elif args.profile.startswith("original-") and not positives_only:
            members.extend(
                index[model, seed, vid]
                for seed in ("0", "1", "3")
                if (model, seed, vid) in index
            )
    if args.unscored:
        prompts = load_index("prompts.csv.gz")
        members.extend(prompts["Prompt", "", r["vid"]] for r in rows)
    return rows, members


def all_positives(args, index):
    plans = []
    for profile in ("original-1.3b", "fast-14b", "original-14b"):
        # Preserve each trainer's row order and manifest format, including its captions.
        options = argparse.Namespace(**vars(args))
        options.profile = profile
        options.positives_only = True
        options.unscored = profile.startswith("original-")
        options.ids = options.limit = None
        rows, members = select(options, index)
        plans.append((options, rows, members))

    if not args.ids and args.limit is None:
        return plans
    wanted = None
    if args.ids:
        available = {r["vid"]: r for _, rows, _ in plans for r in rows}
        request = argparse.Namespace(profile=args.profile, ids=args.ids, limit=None)
        wanted = {r["vid"] for r in select_rows(request, available.values())}
    result = []
    for options, rows, members in plans:
        if wanted is not None:
            rows = [r for r in rows if r["vid"] in wanted]
        # Match the same per-model subset when Option-2 is run with --limit.
        rows = rows[:args.limit]
        if rows:
            ids = {r["vid"] for r in rows}
            result.append((options, rows, [m for m in members if m.vid in ids]))
    return result


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def matches(path, member):
    if not path.is_file() or path.stat().st_size != member.size:
        return False
    crc = 0
    with path.open("rb") as local:
        for chunk in iter(lambda: local.read(1024**2), b""):
            crc = zlib.crc32(chunk, crc)
    return crc == member.crc


def download(member, output, verify_only):
    dest = output / member.relative_path()
    if matches(dest, member):
        return "reused"
    if verify_only:
        raise ValueError(f"Missing or damaged file: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo, rev = (
        (VIDGEN, VIDGEN_REV) if member.source == "VidGen" else (PAIRS, PAIRS_REV)
    )
    start, end = member.offset, member.offset + member.compressed - 1
    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{member.archive}"
    for attempt in range(4):
        temp = None
        try:
            with session().get(
                url,
                params={"download": "true", "range": f"{start}-{end}"},
                stream=True,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                timeout=(15, 90),
            ) as response:
                response.raise_for_status()
                span = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", ""),
                )
                if (
                    response.status_code != 206
                    or not span
                    or tuple(map(int, span.groups()[:2])) != (start, end)
                ):
                    raise ValueError(
                        f"Server did not return the requested range for {member.archive} ({response.status_code})"
                    )
                decoder = (
                    zlib.decompressobj(-zlib.MAX_WBITS) if member.method == 8 else None
                )
                if member.method not in (0, 8):
                    raise ValueError(f"Unsupported ZIP compression {member.method}")
                received = written = crc = 0
                with tempfile.NamedTemporaryFile(
                    dir=dest.parent,
                    prefix=f".{dest.name}.",
                    suffix=".partial",
                    delete=False,
                ) as f:
                    temp = Path(f.name)
                    for chunk in response.iter_content(256 * 1024):
                        received += len(chunk)
                        block = decoder.decompress(chunk) if decoder else chunk
                        written += len(block)
                        if written > member.size or received > member.compressed:
                            raise ValueError("Oversized ZIP member")
                        f.write(block)
                        crc = zlib.crc32(block, crc)
                    if decoder and (not decoder.eof or decoder.unused_data):
                        raise ValueError("Incomplete or overlong DEFLATE stream")
                if (received, written, crc) != (
                    member.compressed,
                    member.size,
                    member.crc,
                ):
                    raise ValueError(
                        f"Byte count or CRC mismatch for {member.relative_path()}"
                    )
            os.replace(temp, dest)
            return "downloaded"
        except (requests.RequestException, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"{member.relative_path()}: {error}") from error
            time.sleep(2**attempt)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as f:
        temp = Path(f.name)
        try:
            writer = csv.DictWriter(f, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def write_prompts(meta, rows, captions):
    prompts = [{"vid": r["vid"], "original": captions[r["vid"]]} for r in rows]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=meta, delete=False) as output:
        tmp = Path(output.name)
        try:
            json.dump(prompts, output, ensure_ascii=False, indent=2)
            output.write("\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, meta / "prompts.json")


def write_metadata(args, rows, members, meta):
    positives_only = getattr(args, "positives_only", False)
    if args.profile.startswith("fast-"):
        positives = []
        pairs = []
        for row in rows:
            r = {
                k: row[k]
                for k in ("vid", "prompt", "sport", "width", "height", "pixels")
            }
            r["video"] = f"T2V_data/VIDGEN-1M_unzip/{r['vid']}.mp4"
            positives.append(r)
            pairs.append(
                dict(
                    r,
                    negative_seed=row["negative_seed"],
                    negative_video=f"DPO_data/Wan14B/seed_{row['negative_seed']}/{r['vid']}.mp4",
                )
            )
        positive_csv = "positives.csv" if positives_only else "sft.csv"
        write_csv(meta / positive_csv, list(positives[0]), positives)
        if args.profile in FAST_PAIR_PROFILES and not positives_only:
            write_csv(meta / "pairs.csv", list(pairs[0]), pairs)
        return
    if positives_only or args.unscored:
        if positives_only:
            write_csv(
                meta / "positives.csv",
                ["vid", "video"],
                (
                    {"vid": m.vid, "video": m.relative_path().as_posix()}
                    for m in members
                    if m.source == "VidGen"
                ),
            )
        else:
            records = [
                {
                    "vid": m.vid,
                    "seed": m.seed,
                    "positive": str(Path("T2V_data/VIDGEN-1M_unzip") / (m.vid + ".mp4")),
                    "negative": str(m.relative_path()),
                    "prompt_file": str(Path("T2V_data/VIDGEN-1M_unzip") / (m.vid + ".txt")),
                }
                for m in members
                if m.source.startswith("Wan")
            ]
            write_csv(meta / "pairs.csv", list(records[0]), records)
        if args.unscored:
            captions = {
                m.vid: (args.out / m.relative_path()).read_text(encoding="utf-8").strip()
                for m in members
                if m.source == "Prompt"
            }
            write_prompts(meta, rows, captions)
        return
    model = "14B" if args.profile == "original-14b" else "1.3B"
    captions = {}
    for seed in ("0", "1", "3"):
        directory = f"train_part_0_{model}_seed_{seed}"
        relative = Path(directory) / "final.csv"
        source = args.scores / relative
        if source.is_file():
            original = source.read_text(encoding="utf-8")
        else:
            url = f"https://raw.githubusercontent.com/caiyuanhao1998/Open-PhyGDPO/{CODE_REV}/PhyAugPipe/videophy/evaluated_score/{relative.as_posix()}"
            response = session().get(url, timeout=(15, 90))
            response.raise_for_status()
            original = response.content.decode("utf-8")
        desired = {m.vid for m in members if m.seed == seed}
        reader = csv.DictReader(io.StringIO(original))
        selected = []
        for r in reader:
            if r["orig_idx"] in desired:
                r["videopath"] = f"DPO_data/Wan{model}/seed_{seed}/{r['orig_idx']}.mp4"
                selected.append(r)
                captions.setdefault(r["orig_idx"], r["caption"])
        if (
            len(selected) != len(desired)
            or {r["orig_idx"] for r in selected} != desired
        ):
            raise ValueError(f"Score rows differ from indexed pairs for {relative}")
        write_csv(meta / relative, reader.fieldnames, selected)
    write_prompts(meta, rows, captions)


def metadata_directory(args, rows, subset):
    name = args.profile
    if args.positives_only:
        name += "-positives-only"
    if args.unscored:
        name += "-unscored"
    directory = args.out / "manifests" / name
    if subset:
        key = hashlib.sha256("\n".join(r["vid"] for r in rows).encode()).hexdigest()[:12]
        directory = directory / "subsets" / key
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=(*PROFILES, "fast-dpo"), metavar="{" + ",".join(PROFILES) + "}",
        help="Choose a training profile, or all-positives for all three models",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=os.getenv("PHYGDPO_DATA_DIR") or HERE.parent / "data",
        help="Shared data directory (default: PHYGDPO_DATA_DIR or DiffSynth-Studio/data)",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=HERE.parents[1] / "PhyAugPipe/videophy/evaluated_score",
        help="Optional original score CSV directory",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        help="Optional newline-separated video IDs within this profile",
    )
    parser.add_argument("--limit", type=int, help="Use the first N selected IDs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--unscored",
        action="store_true",
        help="Original T2V only: fetch the original caption texts by range and omit the score CSVs",
    )
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--positives-only",
        action="store_true",
        help="Download only the real positive videos and their manifest",
    )
    media.add_argument(
        "--negatives-only",
        action="store_true",
        help="Fetch released negatives and prepare the training manifests for previously downloaded positives",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="Print exact counts and byte totals without network access",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local video size and CRC without network access",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.negatives_only and args.profile in ("fast-sft", "all-positives"):
        parser.error(f"{args.profile} contains only positives")
    if args.unscored and args.profile.startswith("fast-"):
        parser.error(
            "--unscored is only for the original profiles; fast-14b is already score-free"
        )
    if args.profile == "fast-sft":
        args.positives_only = False  # This profile already contains only positives.
    index = load_index()
    if args.profile == "all-positives":
        plans = all_positives(args, index)
    else:
        rows, members = select(args, index)
        plans = [(args, rows, members)]
    subset = args.ids is not None or args.limit is not None
    manifests = {
        options.profile: metadata_directory(options, rows, subset)
        for options, rows, _ in plans
    }
    members = list(dict.fromkeys(member for _, _, files in plans for member in files))
    summary = {
        "profile": args.profile,
        "positive_videos": sum(m.source == "VidGen" for m in members),
        "negative_videos": sum(m.source.startswith("Wan") for m in members),
        "prompt_files": sum(m.source == "Prompt" for m in members),
        "prompt_transfer_bytes": sum(
            m.compressed for m in members if m.source == "Prompt"
        ),
        "positive_transfer_bytes": sum(
            m.compressed for m in members if m.source == "VidGen"
        ),
        "negative_transfer_bytes": sum(
            m.compressed for m in members if m.source.startswith("Wan")
        ),
        "extracted_bytes": sum(m.size for m in members),
        "metadata": (
            {name: str(path) for name, path in manifests.items()}
            if args.profile == "all-positives" else str(manifests[args.profile])
        ),
    }
    print(json.dumps(summary), flush=True)
    if args.plan:
        return
    counts = {"reused": 0, "downloaded": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        batch_size = max(1, 4 * args.workers)
        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]
            results = pool.map(lambda m: download(m, args.out, args.verify_only), batch)
            for i, status in enumerate(results, start + 1):
                counts[status] += 1
                if i % 1000 == 0:
                    print(f"{i}/{len(members)} files verified", flush=True)
    if not args.verify_only:
        for options, rows, files in plans:
            write_metadata(options, rows, files, manifests[options.profile])
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
