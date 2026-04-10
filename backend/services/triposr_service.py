"""TripoSR service — image → 3D mesh (STL / OBJ / GLB)."""
from __future__ import annotations

import os
import sys
import torch
from PIL import Image

# TripoSR has no setup.py — clone it into backend/TripoSR and add to path
_tsr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TripoSR")
if os.path.isdir(_tsr_dir) and _tsr_dir not in sys.path:
    sys.path.insert(0, _tsr_dir)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from tsr.system import TSR
        except ImportError:
            raise RuntimeError(
                "TripoSR no está instalado. Ejecuta en la carpeta backend/:\n"
                "  git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR\n"
                "  pip install -r TripoSR/requirements.txt"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _model.renderer.set_chunk_size(131072)
        _model.to(device)
        _model.eval()
    return _model


def generate_from_image(img_path: str, output_base: str, progress_cb=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model()

    if progress_cb:
        progress_cb(15)

    # Remove background and composite on white → RGB (3 channels)
    from rembg import remove as rembg_remove
    raw = Image.open(img_path).convert("RGBA")
    no_bg = rembg_remove(raw)
    bg = Image.new("RGB", no_bg.size, (255, 255, 255))
    bg.paste(no_bg, mask=no_bg.split()[3])
    image = bg

    if progress_cb:
        progress_cb(30)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    if progress_cb:
        progress_cb(65)

    meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=128)
    mesh = meshes[0]

    if progress_cb:
        progress_cb(85)

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)
