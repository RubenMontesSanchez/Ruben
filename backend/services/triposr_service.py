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

    # Use TripoSR's built-in preprocessing: remove bg + center/resize object
    from tsr.utils import remove_background, resize_foreground
    from rembg import new_session as rembg_new_session

    rembg_session = rembg_new_session()
    raw = Image.open(img_path)
    image = remove_background(raw, rembg_session)  # → RGBA, bg transparent
    image = resize_foreground(image, 0.85)          # center object, 85% of frame

    # Composite on white → RGB for CLIP encoder
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    image = bg.convert("RGB")

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
