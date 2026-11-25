"""Match the six public I2V examples to the evaluation prompts and starting frames."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION = HERE / "evaluation"
PLAN = json.loads((EVALUATION / "scenes.json").read_text())
SPACE = json.loads((HERE / "space-presets.json").read_text())
cases = {row["case_id"]: row for row in PLAN["scenes"]}
TEMPLATES = {preset["label"]: {**cases[key], **preset} for key, preset in SPACE["scenes"].items()}


def starting_frame(scene):
    return EVALUATION / TEMPLATES[scene]["image"]


def matching_template(prompt, image):
    from PIL import Image, ImageChops

    for name, row in TEMPLATES.items():
        if prompt == row["prompt"]:
            with Image.open(starting_frame(name)) as reference:
                if image.size == reference.size and ImageChops.difference(image, reference.convert("RGB")).getbbox() is None:
                    return name
    return None
