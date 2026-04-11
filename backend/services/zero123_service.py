"""Zero123++ service — single image → 6 multi-view images.

Pipeline:
  1. Load Zero123++ in float16 (~6 GB VRAM)
  2. Generate 6 novel-view images (2x3 grid)
  3. Free GPU memory completely before returning
  4. Return 6 PIL Images for downstream reconstruction

This sequential unload pattern lets TripoSR (~4 GB) run afterwards
without OOM on 8 GB cards.
"""
from __future__ import annotations

import torch
from PIL import Image


def generate_multiview(
    image: Image.Image,
    progress_cb=None,
    num_steps: int = 28,
) -> list:
    """
    Generate 6 novel-view images with Zero123++.
    Downloads ~5.5 GB of weights on first use (cached afterwards).
    Returns a list of 6 PIL Images split from the 2x3 output grid.
    """
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        raise RuntimeError(
            "diffusers no instalado. Ejecuta:\n"
            "  pip install diffusers>=0.24.0 accelerate"
        )

    if progress_cb:
        progress_cb(5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    pipeline.to(device)
    pipeline.enable_attention_slicing(1)   # reduce VRAM peak

    if progress_cb:
        progress_cb(25)

    # Zero123++ expects 320x320 RGBA input
    img_input = image.convert("RGBA").resize((320, 320), Image.LANCZOS)

    result_grid = pipeline(
        img_input,
        num_inference_steps=num_steps,
    ).images[0]

    if progress_cb:
        progress_cb(58)

    # Free VRAM immediately so TripoSR can load next
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if progress_cb:
        progress_cb(62)

    # Split 2x3 grid → 6 individual views
    w, h = result_grid.size
    tw, th = w // 3, h // 2
    views = []
    for row in range(2):
        for col in range(3):
            crop = result_grid.crop(
                (col * tw, row * th, (col + 1) * tw, (row + 1) * th)
            )
            views.append(crop.convert("RGB"))

    # Views layout (Zero123++ v1.2):
    # [0] azimuth=30°   [1] azimuth=90°  [2] azimuth=150°
    # [3] azimuth=210°  [4] azimuth=270° [5] azimuth=330°
    # All at alternating elevations ±20°
    return views
