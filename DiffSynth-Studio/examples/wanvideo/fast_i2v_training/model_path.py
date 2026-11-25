"""Resolve the local Wan-2.1-Fast-14B weights for training, inference and Gradio."""
import os
from pathlib import Path

FOLDER = "Wan2.1-Fast-TI2V-14B_PhyGDPO"


def model_directory(run_dir):
    pretrained = Path(__file__).resolve().parents[3] / "models/pretrained" / FOLDER
    default = pretrained if pretrained.is_dir() else run_dir / "models/fused"
    return Path(os.getenv("PHYGDPO_MODEL_DIR", default)).expanduser().resolve()
