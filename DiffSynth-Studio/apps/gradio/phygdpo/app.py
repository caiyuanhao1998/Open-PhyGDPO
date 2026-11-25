"""Single-GPU local edition of the CaiYuanhao/PhyGDPO Space."""
from collections import deque
from functools import partial
import os
from pathlib import Path
import re
from secrets import randbits
import subprocess
import sys
import time
from uuid import uuid4

import gradio as gr
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from usage_stats import UsageStats

TITLE = "PhyGDPO · ECCV 2026"
MODEL = "PhyGDPO · Wan2.1-1.3B"
T2V_14B_MODEL = "PhyGDPO · Wan2.1-T2V-14B"
FAST_MODEL = "PhyGDPO · Wan2.1-Fast-14B"
MODEL_PANELS = {FAST_MODEL: "show-fast", MODEL: "show-small-t2v", T2V_14B_MODEL: "show-large-t2v"}
STUDIO = Path(__file__).resolve().parents[3]
RECIPE = STUDIO / "examples/wanvideo/fast_i2v_training"
sys.path.insert(0, str(RECIPE.parent))
sys.path.insert(0, str(RECIPE))
from inference import t2v_worker
from model_path import FOLDER, model_directory
from presets import EVALUATION, TEMPLATES, matching_template, starting_frame

RUN_DIR = Path(os.getenv("PHYGDPO_RUN_DIR", STUDIO / "runs/wan-fast")).expanduser().resolve()
FAST_DIR = model_directory(RUN_DIR)
OUTPUTS = RUN_DIR / "gradio/videos"
OUTPUTS.mkdir(parents=True, exist_ok=True)
stats_dir = Path(os.getenv("PHYGDPO_STATS_DIR", RUN_DIR / "gradio/stats"))
stats_dir.mkdir(parents=True, exist_ok=True)
USAGE = UsageStats(stats_dir)
GPU_AVAILABLE = torch.cuda.is_available()
SCENES = (
    ("🏍️ Motorcycling", "A motorcyclist rides along a winding paved road.", 2898027816),
    (
        "🏋🏻 Weightlifting",
        "A weightlifter successfully completes a snatch with a 25kg barbell, holding it momentarily overhead.",
        0,
    ),
    ("🏇 Horse riding", "A person rides a trotting horse along a dirt trail.", 2427626026),
    ("⛸️ Ice skating", "An ice skater glides around a curve on an outdoor ice rink.", 2292707491),
)
GYMNASTICS = ("🤸 Gymnastics", "A gymnast drops from the parallel bars and lands safely on the mat below.", 0)
T2V_SCENES = {MODEL: SCENES, T2V_14B_MODEL: (GYMNASTICS, *SCENES[1:])}
RESOLUTIONS = {
    "832 × 480": (832, 480),
    "512 × 288": (512, 288),
    "480 × 832": (480, 832),
    "320 × 192": (320, 192),
}
DEFAULT_RESOLUTION = "832 × 480"
FAST_PREVIEW_HEIGHT = "clamp(180px, 30vw, 320px)"
TEXT_PREVIEW_HEIGHT = "clamp(190px, 36vw, 340px)"
FAST_FRAMES, FAST_STEPS, FAST_MAX_STEPS = 81, 4, 30


def run_video(model, prompt, width, height, frames, steps, seed, progress, frame=None):
    if not GPU_AVAILABLE:
        raise gr.Error("Video generation requires an NVIDIA CUDA GPU.")
    started = time.monotonic()
    target = OUTPUTS / f"{uuid4().hex}.mp4"
    common = ["--prompt", prompt, "--seed", str(seed), "--width", str(width), "--height", str(height), "--frames", str(frames)]
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PHYGDPO_WORLD_SIZE="1",
               PHYGDPO_RUN_DIR=str(RUN_DIR), PYTHONUNBUFFERED="1")
    for name in ("LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
        env.pop(name, None)
    if frame is not None:
        env["PHYGDPO_MODEL_DIR"] = str(FAST_DIR)
        args = [str(RECIPE / "generate.py"), "video", "--image", str(frame), "--output", str(target), "--steps", str(steps)]
    else:
        key = "t2v-14b" if model == T2V_14B_MODEL else "t2v-1.3b"
        args = t2v_worker(key, "phygdpo", 1, steps, target)
        if model == MODEL:
            args.append("--cpu-offload")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(STUDIO), env.get("PYTHONPATH"))))
    progress(0.02, desc="Preparing model and inputs")
    recent = deque(maxlen=16)
    with subprocess.Popen([sys.executable, *args, *common], cwd=STUDIO, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
        for line in process.stdout:
            message = line.strip()
            if not message:
                continue
            print(message, flush=True)
            recent.append(message)
            step = re.search(r"Denoising · step (\d+) of (\d+)", message)
            if step:
                progress(0.06 + 0.83 * (int(step[1]) - 1) / int(step[2]), desc=step.group())
            elif "Decoding video" in message:
                progress(0.93, desc="Decoding video")
        if process.wait() or not target.is_file():
            raise gr.Error("Generation failed. See the terminal for the full log.\n" + "\n".join(recent)[-1200:])
    elapsed = time.monotonic() - started
    USAGE.record(target.stem)
    return str(target), f"**{model}** · {width}×{height} · {frames} frames · {steps} steps · {elapsed:.1f} s"


def output_settings(resolution, frames, steps, min_steps, max_steps):
    if resolution not in RESOLUTIONS:
        raise gr.Error("Select a supported resolution.")
    if frames not in range(9, 82, 4):
        raise gr.Error("Frames must be between 9 and 81, in increments of 4.")
    if steps not in range(min_steps, max_steps + 1):
        raise gr.Error(f"Denoising steps must be a whole number between {min_steps} and {max_steps}.")
    width, height = RESOLUTIONS[resolution]
    return width, height, int(frames), int(steps)


def generate_text(model, prompt, resolution, frames, steps, progress):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")
    width, height, frames, steps = output_settings(resolution, frames, steps, 2, 50)
    seed = next((seed for _, example, seed in T2V_SCENES[model] if example == prompt), None)
    if seed is None:
        seed = randbits(32)
    return run_video(model, prompt, width, height, frames, steps, seed, progress)


def generate_video(prompt, resolution, frames, steps, progress=gr.Progress()):
    return generate_text(MODEL, prompt, resolution, frames, steps, progress)


def generate_video_14b(prompt, resolution, frames, steps, progress=gr.Progress()):
    return generate_text(T2V_14B_MODEL, prompt, resolution, frames, steps, progress)


def read_starting_frame(path):
    if not path:
        raise gr.Error("Upload a starting frame or choose an example.")
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise gr.Error("Choose an image that can be opened, such as a PNG, JPEG or WebP.") from error


def generate_from_frame(prompt, starting_frame, resolution=DEFAULT_RESOLUTION,
                        frames=FAST_FRAMES, steps=FAST_STEPS, progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")
    width, height, frames, steps = output_settings(resolution, frames, steps, 1, FAST_MAX_STEPS)
    image = read_starting_frame(starting_frame)
    scene = matching_template(prompt, image)
    seed = TEMPLATES[scene]["seed"] if scene is not None else randbits(32)
    return run_video(FAST_MODEL, prompt, width, height, frames, steps, seed, progress, frame=starting_frame)


def generate_fast_video(scene, prompt, progress=gr.Progress()):
    # Existing API clients select a preset by name and pass no image.
    if scene not in TEMPLATES:
        raise gr.Error("Select a scene.")
    return generate_from_frame(prompt, starting_frame(scene), progress=progress)


def fast_scene_button_updates(prompt, frame_scene):
    preset = TEMPLATES.get(frame_scene)
    active = frame_scene if preset and (prompt or "").strip() == preset["prompt"] else None
    return [gr.update(variant="primary" if name == active else "secondary") for name in TEMPLATES]


def select_fast_scene(scene):
    prompt = TEMPLATES[scene]["prompt"]
    return prompt, str(starting_frame(scene)), scene, *fast_scene_button_updates(prompt, scene)


def upload_fast_frame(path):
    return read_starting_frame(path), *clear_fast_selection()


def clear_fast_selection():
    return None, *fast_scene_button_updates("", None)


def fast_submit_update(ready, starting_frame, generating):
    return gr.update(interactive=bool(ready and starting_frame and not generating))


def select_model(model, current):
    if model == current:
        return gr.skip()
    fast = model == FAST_MODEL
    return (
        model, gr.update(elem_classes=[MODEL_PANELS[model]]),
        gr.update(value=None, height=FAST_PREVIEW_HEIGHT if fast else TEXT_PREVIEW_HEIGHT), "",
        *(gr.update(variant="primary" if name == model else "secondary") for name in MODEL_PANELS),
    )


def prepare_fast_model(model):
    if model != FAST_MODEL or not GPU_AVAILABLE:
        return False
    filename = "epoch-0.safetensors"
    downloaded = RUN_DIR / "models/hub" / FOLDER
    ready = (FAST_DIR / "fusion.json").is_file() and any((p / filename).is_file() for p in (FAST_DIR, downloaded))
    if not ready:
        gr.Warning(f"Place the Wan-2.1-Fast-14B weights in models/pretrained/{FOLDER}, then select the model again.")
    return ready


def usage_summary():
    counts = USAGE.counts()
    last_30_days, today = (f"{value:,}" for value in counts) if counts is not None else ("—", "—")
    note = "Successful video generations · Pacific Time" if counts is not None else "Success counts are temporarily unavailable"
    return (
        f'<dl aria-label="Successful video generations" title="{note}">'
        f'<div><dt>Last 30 days</dt><dd>{last_30_days}</dd></div>'
        f'<div><dt>Today</dt><dd>{today}</dd></div>'
        '</dl>'
    )


def scene_button_updates(prompt, *, scenes):
    prompt = (prompt or "").strip()
    return [
        gr.update(variant="primary" if prompt == scene_prompt else "secondary")
        for _, scene_prompt, _ in scenes
    ]


def text_controls(model):
    scenes = T2V_SCENES[model]
    panel_id = "text-controls" if model == MODEL else "text-controls-14b"
    with gr.Column(min_width=0, elem_id=panel_id, elem_classes=["model-controls", "text-controls"]):
        _, initial_prompt, _ = scenes[0]
        with gr.Row(elem_classes=["scene-presets", "text-presets"]):
            scene_buttons = [
                gr.Button(label, size="sm", min_width=0,
                          variant="primary" if index == 0 else "secondary")
                for index, (label, _, _) in enumerate(scenes)
            ]
        prompt = gr.Textbox(value=initial_prompt, label="Prompt", lines=3, max_lines=3)
        submit = gr.Button("Generate video", variant="primary", elem_classes="generate-video")
        with gr.Row(elem_classes="quick-settings"):
            gr.HTML(f"<dl><dt>Model</dt><dd>{model}</dd></dl>", elem_classes="fixed-model", padding=False)
            resolution = gr.Dropdown(list(RESOLUTIONS), value=DEFAULT_RESOLUTION, label="Resolution", min_width=0)
        with gr.Accordion("Advanced settings", open=False, elem_classes="advanced-settings"):
            with gr.Row(elem_classes="advanced-fields"):
                frames = gr.Slider(9, 81, value=81, step=4, label="Frames", min_width=0)
                steps = gr.Slider(2, 50, value=50, step=1, label="Denoising steps", min_width=0)
        if model == T2V_14B_MODEL:
            gr.Markdown("To reproduce the published gymnastics example, run the command in README Section 2.3 on eight NVIDIA H100 GPUs. "
                        "BF16 rounding can vary across GPU architectures.", elem_classes="model-note")
        for button, (_, example, _) in zip(scene_buttons, scenes):
            button.click(lambda value=example: value, inputs=None, outputs=prompt,
                         queue=False, show_progress="hidden", api_visibility="private")
        prompt.change(partial(scene_button_updates, scenes=scenes), inputs=prompt, outputs=scene_buttons,
                      queue=False, show_progress="hidden", api_visibility="private")
    return submit, [prompt, resolution, frames, steps]


def settings_controls(default_steps, min_steps, max_steps):
    with gr.Row(elem_classes="output-fields"):
        resolution = gr.Dropdown(list(RESOLUTIONS), value=DEFAULT_RESOLUTION,
                                 label="Resolution", elem_classes="resolution-setting", min_width=0)
        frames = gr.Dropdown(list(range(9, 82, 4)), value=81, label="Frames", elem_classes="frames-setting", min_width=0)
        steps = gr.Number(value=default_steps, precision=0, step=1, minimum=min_steps, maximum=max_steps,
                          label="Denoising steps", elem_classes="steps-setting", min_width=0)
    return resolution, frames, steps


with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
    with gr.Row(elem_id="space-header"):
        gr.Markdown(f"# {TITLE}", elem_id="space-title", min_width=0)
        usage = gr.HTML(usage_summary(), elem_id="usage-panel", padding=False)
    current_model = gr.State(FAST_MODEL)
    fast_ready = gr.State(False)
    generating = gr.State(False)
    with gr.Row(elem_id="model-switch"):
        gr.Markdown("Select a model:", elem_id="model-switch-label", min_width=0)
        fast_model = gr.Button(FAST_MODEL, size="sm", min_width=0, variant="primary")
        text_model = gr.Button(MODEL, size="sm", min_width=0, variant="secondary")
        large_model = gr.Button(T2V_14B_MODEL, size="sm", min_width=0, variant="secondary")
        model_buttons = [fast_model, text_model, large_model]
    if not GPU_AVAILABLE:
        gr.Markdown("Video generation requires an NVIDIA CUDA GPU.")
    with gr.Row(elem_id="space-workspace", elem_classes=[MODEL_PANELS[FAST_MODEL]]) as workspace:
        with gr.Column(min_width=0, elem_classes="space-pane", elem_id="input-pane"):
            small_submit, small_inputs = text_controls(MODEL)
            large_submit, large_inputs = text_controls(T2V_14B_MODEL)
            with gr.Column(min_width=0, elem_classes="model-controls", elem_id="fast-controls"):
                first_scene = next(iter(TEMPLATES))
                initial_fast_prompt = TEMPLATES[first_scene]["prompt"]
                initial_frame = str(starting_frame(first_scene))
                frame_scene = gr.State(first_scene)
                with gr.Row(elem_id="fast-templates", elem_classes="scene-presets"):
                    fast_buttons = [
                        gr.Button(f"{preset['icon']} {preset['button_label']}", size="sm", min_width=0,
                                  variant="primary" if name == first_scene else "secondary")
                        for name, preset in TEMPLATES.items()
                    ]
                fast_prompt = gr.Textbox(value=initial_fast_prompt, label="Prompt", lines=3, max_lines=3)
                with gr.Column(min_width=0, elem_id="frame-input"):
                    with gr.Row(elem_id="frame-toolbar"):
                        gr.Markdown("Starting frame", elem_id="frame-heading")
                        upload_frame = gr.UploadButton("Upload image", file_types=["image"], size="sm",
                                                       min_width=0, elem_id="upload-frame")
                    starting_image = gr.Image(
                        value=initial_frame, label="Starting frame", show_label=False, container=False,
                        interactive=True, sources=["upload"], type="filepath", buttons=[], format="png",
                        placeholder="Drop an image here or click to upload a starting frame.",
                        height=164, min_width=0, elem_id="fast-frame",
                    )
        with gr.Column(min_width=0, elem_classes="space-pane", elem_id="output-pane"):
            video = gr.Video(label="Video preview", height=FAST_PREVIEW_HEIGHT,
                             format="mp4", interactive=False, elem_id="video-preview")
            summary = gr.Markdown(elem_id="generation-summary")
            with gr.Column(scale=0, min_width=0, elem_classes="output-controls", elem_id="fast-output"):
                fast_settings = settings_controls(FAST_STEPS, 1, FAST_MAX_STEPS)
                fast_submit = gr.Button("Generate video", variant="primary", interactive=False,
                                        elem_classes="generate-video")

    locked_controls = [small_submit, large_submit, *model_buttons]
    generation_controls = [generating, fast_submit, *locked_controls]

    def start_generation():
        return None, "", True, *(gr.update(interactive=False) for _ in [fast_submit, *locked_controls])

    def finish_generation(ready, frame):
        return False, fast_submit_update(ready, frame, False), *(gr.update(interactive=True) for _ in locked_controls)

    def bind_generation(button, predict, inputs, api_name):
        clear_preview = button.click(start_generation, outputs=[video, summary, *generation_controls],
                                     queue=False, show_progress="hidden", api_visibility="private")
        result = clear_preview.success(
            predict, inputs, [video, summary], api_name=api_name, show_progress_on=video,
            concurrency_limit=1, concurrency_id="video-generation",
        )
        for terminal in (result.success, result.failure):
            terminal(finish_generation, [fast_ready, starting_image], generation_controls, queue=False,
                     show_progress="hidden", api_visibility="private")
        result.success(usage_summary, outputs=usage, queue=False, api_visibility="private")

    bind_generation(small_submit, generate_video, small_inputs, "generate_video")
    bind_generation(large_submit, generate_video_14b, large_inputs, "generate_video_14b")
    bind_generation(fast_submit, generate_from_frame, [fast_prompt, starting_image, *fast_settings], "generate_from_frame")
    # Keep the two-argument preset endpoint available to existing Gradio clients.
    with gr.Row(visible=False):
        api_model = gr.Dropdown(list(MODEL_PANELS), value=FAST_MODEL, label="Model")
        api_scene = gr.Dropdown(list(TEMPLATES), value=first_scene, label="Scene")
    gr.on([], generate_fast_video, [api_scene, fast_prompt], [video, summary], api_name="generate_fast_video",
          concurrency_limit=1, concurrency_id="video-generation")

    for name, model_button in zip(MODEL_PANELS, model_buttons):
        model_button.click(
            partial(select_model, name), current_model,
            [current_model, workspace, video, summary, *model_buttons],
            queue=False, show_progress="hidden", api_visibility="private",
        )
    prepared = gr.on([demo.load, fast_model.click], prepare_fast_model, api_model, fast_ready,
                     api_name="prepare_fast_model", show_progress_on=fast_submit,
                     concurrency_limit=1, concurrency_id="fast-preparation")
    for event in (prepared.success, starting_image.change):
        event(fast_submit_update, [fast_ready, starting_image, generating], fast_submit,
              queue=False, show_progress="hidden", api_visibility="private")
    gr.on([starting_image.clear, starting_image.upload], clear_fast_selection, outputs=[frame_scene, *fast_buttons],
          queue=False, show_progress="hidden", api_visibility="private")
    for name, button in zip(TEMPLATES, fast_buttons):
        button.click(partial(select_fast_scene, name), outputs=[fast_prompt, starting_image, frame_scene, *fast_buttons],
                     queue=False, show_progress="hidden", api_visibility="private")
    upload_frame.upload(upload_fast_frame, upload_frame, [starting_image, frame_scene, *fast_buttons],
                        queue=False, show_progress="hidden", api_visibility="private")
    fast_prompt.input(fast_scene_button_updates, [fast_prompt, frame_scene], fast_buttons,
                      queue=False, trigger_mode="always_last", show_progress="hidden", api_visibility="private")
    demo.load(usage_summary, outputs=usage, queue=False, api_visibility="private")
    gr.Timer(30).tick(usage_summary, outputs=usage, queue=False, api_visibility="private")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate", font=("Inter", "system-ui", "sans-serif")).set(
            body_background_fill="#f5f7fc",
            body_background_fill_dark="*neutral_950",
            body_text_color_subdued="*neutral_700",
            body_text_color_subdued_dark="*neutral_300",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*primary_50",
            background_fill_secondary_dark="*neutral_800",
            border_color_primary="#d8e0ef",
            border_color_primary_dark="*neutral_700",
            link_text_color="#3555a1",
            link_text_color_dark="#a5b7ff",
            button_primary_background_fill="#3555a1",
            button_primary_background_fill_hover="#294584",
            button_primary_background_fill_dark="#4761b7",
            button_primary_background_fill_hover_dark="#38529f",
            button_primary_border_color="*button_primary_background_fill",
            button_primary_border_color_dark="*button_primary_background_fill",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            block_background_fill="transparent",
            block_background_fill_dark="transparent",
            block_border_width="0px",
            block_padding="0px",
            block_label_background_fill="transparent",
            block_label_background_fill_dark="transparent",
            block_label_border_width="0px",
            block_title_background_fill="transparent",
            block_title_background_fill_dark="transparent",
            input_background_fill="*background_fill_primary",
            input_background_fill_dark="*background_fill_primary",
        ),
        css_paths=Path(__file__).with_name("style.css"),
        allowed_paths=[str(OUTPUTS), str(EVALUATION / "inputs")],
        footer_links=[],
        share=False,
    )
