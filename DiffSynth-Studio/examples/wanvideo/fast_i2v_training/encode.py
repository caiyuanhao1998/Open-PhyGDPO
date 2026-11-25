"""Precompute the original 15-fps Wan text, video and first-frame features."""
import argparse
import os
import subprocess
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from transformers import T5TokenizerFast, UMT5EncoderModel
from diffusers.pipelines.wan.pipeline_wan_i2v import prompt_clean
from diffusers import AutoencoderKLWan
from cache_identity import fingerprint, recipe_hash
from settings import (
    digest,
    DATA,
    FPS,
    FRAMES,
    GPUS,
    HEIGHT,
    MODEL,
    WIDTH,
    negative_features,
    text_features,
    video_features,
    validate_manifest,
)

SHAPE = (16, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
IMAGE_SHAPE = (257, 1280)


def atomic_save(path, tensors, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.partial")
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
            str(partial),
            metadata=metadata,
        )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def raw_video(path, width, height, frames):
    vf = f"fps={FPS},scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,crop={width}:{height},setsar=1"
    raw = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            path,
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    if not raw or len(raw) % (height * width * 3):
        raise ValueError(f"Bad decoded frames in {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


class ImageToVideoEncoder:
    def __init__(self, device):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.device = device
        self.vae = (
            AutoencoderKLWan.from_pretrained(
                MODEL / "vae", torch_dtype=torch.float32, local_files_only=True
            )
            .eval()
            .to(device)
        )
        self.clip = (
            CLIPVisionModel.from_pretrained(
                MODEL / "image_encoder",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .eval()
            .to(device)
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            MODEL / "image_processor", local_files_only=True
        )
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )

    def latent(self, pixels):
        result = self.vae.encode(pixels).latent_dist.mode()
        return ((result - self.mean) / self.std)[0].bfloat16().cpu()

    @torch.inference_mode()
    def encode(self, frames):
        indexes = np.rint(np.linspace(0, len(frames) - 1, FRAMES)).astype(int)
        first_frame = Image.fromarray(frames[0])
        image_inputs = self.processor(images=first_frame, return_tensors="pt").to(
            self.device
        )
        image = (
            self.clip(**image_inputs, output_hidden_states=True)
            .hidden_states[-2][0]
            .bfloat16()
            .cpu()
        )
        pixels = (
            torch.from_numpy(frames[indexes])
            .to(self.device, dtype=torch.float32)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .contiguous()
            .div_(127.5)
            .sub_(1)
        )
        video = self.latent(pixels)
        condition_pixels = torch.zeros_like(pixels)
        condition_pixels[:, :, 0] = pixels[:, :, 0]
        return {
            "video": video,
            "condition": self.latent(condition_pixels),
            "image": image,
        }


def reusable(path, verify, *args):
    if not path.is_file():
        return False
    try:
        verify(path, *args)
    except (KeyError, TypeError, ValueError, OSError, SafetensorError) as error:
        print(f"Rebuilding {path.name}: {error}", flush=True)
        return False
    return True


def check_identity(handle, expected):
    metadata = handle.metadata() or {}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("source or encoder fingerprint changed")


def verify(path, vid, expected):
    with safe_open(str(path), framework="pt") as handle:
        check_identity(handle, expected)
        meta = handle.metadata()
        if meta.get("vid") != vid or meta.get("fps") != str(FPS):
            raise ValueError(f"Cache metadata mismatch: {path}")
        if set(handle.keys()) != {"video", "condition", "image"}:
            raise ValueError(f"Unexpected cache tensors: {path}")
        for name, shape in (
            ("video", SHAPE),
            ("condition", SHAPE),
            ("image", IMAGE_SHAPE),
        ):
            item = handle.get_slice(name)
            if tuple(item.get_shape()) != shape or item.get_dtype() != "BF16":
                raise ValueError(f"Malformed {name} feature: {path}")
        return meta["source"]


class NegativeEncoder:
    def __init__(self, device):
        self.device = device
        self.vae = (
            AutoencoderKLWan.from_pretrained(
                MODEL / "vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .eval()
            .to(device)
        )
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )

    @torch.inference_mode()
    def encode(self, row, positive_length):
        negative = raw_video(DATA / row["negative_video"], WIDTH, HEIGHT, FRAMES)
        first = raw_video(DATA / row["video"], WIDTH, HEIGHT, 1)
        common = min(positive_length, len(negative))
        if common < 2 or len(first) != 1:
            raise ValueError(f"Too few frames in pair {row['vid']}")
        indices = np.rint(np.linspace(0, common - 1, FRAMES)).astype(int)
        clip = negative[indices]
        clip[0] = first[0]
        pixels = (
            torch.from_numpy(clip)
            .to(self.device, dtype=torch.float32)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .contiguous()
            .div_(127.5)
            .sub_(1)
        )
        result = self.vae.encode(pixels).latent_dist.mode()
        return ((result - self.mean) / self.std)[0].bfloat16().cpu()


def verify_negative(path, row, key, first, expected):
    with safe_open(path, framework="pt") as source:
        check_identity(source, expected)
        metadata = source.metadata()
        if (
            metadata.get("vid") != row["vid"]
            or metadata.get("negative_seed") != row["negative_seed"]
            or metadata.get("clip_fps") != str(FPS)
        ):
            raise ValueError(f"Negative cache metadata mismatch: {path}")
        item = source.get_slice(key)
        if tuple(item.get_shape()) != SHAPE or item.get_dtype() != "BF16":
            raise ValueError(f"Malformed negative latent: {path}")
        if not torch.equal(item[:, 0], first):
            raise ValueError(f"Pair has divergent first-frame latents: {row['vid']}")


def text_identity(row):
    return {"vid": row["vid"], "encoder_sha256": recipe_hash("text"), "prompt_sha256": fingerprint(row["prompt"])}


def verify_text(path, expected):
    with safe_open(str(path), framework="pt") as f:
        check_identity(f, expected)
        item = f.get_slice("text")
        shape = item.get_shape()
        if len(shape) != 2 or shape[1] != 4096 or not 0 < shape[0] <= 512 or item.get_dtype() != "BF16":
            raise ValueError("Unexpected text tensor shape or dtype")


def encode_text(rows, device):
    todo = [r for r in rows if not reusable(text_features(r), verify_text, text_identity(r))]
    if not todo:
        return
    tokenizer = T5TokenizerFast.from_pretrained(
        MODEL / "tokenizer", local_files_only=True
    )
    encoder = UMT5EncoderModel.from_pretrained(
        MODEL / "text_encoder",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    for start in range(0, len(todo), 4):
        batch = todo[start : start + 4]
        tokens = tokenizer(
            [prompt_clean(r["prompt"]) for r in batch],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        embeds = encoder(tokens.input_ids, tokens.attention_mask).last_hidden_state
        for i, r in enumerate(batch):
            length = int(tokens.attention_mask[i].sum())
            atomic_save(
                text_features(r),
                {"text": embeds[i, :length].bfloat16()},
                text_identity(r),
            )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("text", "positive", "negative"))
    args = parser.parse_args()
    rows = validate_manifest(paired=args.stage == "negative")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != GPUS:
        raise ValueError("World size differs from the run configuration")
    rows = rows[rank::GPUS]
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if args.stage == "text":
        encode_text(rows, device)
        return
    encoder = None
    for row in rows:
        vid = row["vid"]
        pos = video_features(row)
        positive_identity = {"media_sha256": digest(DATA / row["video"]), "encoder_sha256": recipe_hash("positive")}
        if args.stage == "positive":
            if not reusable(pos, verify, vid, positive_identity):
                frames = raw_video(DATA / row["video"], WIDTH, HEIGHT, FRAMES)
                if len(frames) < 2:
                    raise ValueError(f"Too few frames for {vid}")
                if encoder is None:
                    encoder = ImageToVideoEncoder(device)
                atomic_save(
                    pos,
                    encoder.encode(frames),
                    {
                        "vid": vid,
                        "fps": str(FPS),
                        "source_frames": str(len(frames)),
                        "source": "positive_video",
                        **positive_identity,
                    },
                )
            verify(pos, vid, positive_identity)
        else:
            verify(pos, vid, positive_identity)
            path, key = negative_features(row)
            with safe_open(str(pos), framework="pt") as source:
                first = source.get_slice("video")[:, 0]
                length = int(source.metadata()["source_frames"])
            expected = {
                "negative_sha256": digest(DATA / row["negative_video"]),
                "positive_sha256": positive_identity["media_sha256"],
                "positive_encoder_sha256": positive_identity["encoder_sha256"],
                "encoder_sha256": recipe_hash("negative"),
            }
            if not reusable(path, verify_negative, row, key, first, expected):
                if encoder is None:
                    encoder = NegativeEncoder(device)
                tensor = encoder.encode(row, length)
                if not torch.equal(tensor[:, 0], first):
                    raise ValueError(
                        f"Encoded pair has divergent first-frame latents: {vid}"
                    )
                atomic_save(
                    path,
                    {key: tensor},
                    {
                        "vid": vid,
                        "clip_fps": str(FPS),
                        "negative_seed": row["negative_seed"],
                        **expected,
                    },
                )
            verify_negative(path, row, key, first, expected)


if __name__ == "__main__":
    main()
