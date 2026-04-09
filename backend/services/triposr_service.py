"""TripoSR service — image → 3D mesh (STL / OBJ / GLB)."""
from __future__ import annotations

import torch
from PIL import Image

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from tsr.system import TSR
        except ImportError:
            raise RuntimeError(
                "TripoSR is not installed. Run:\n"
                "  pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
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

    image = Image.open(img_path).convert("RGBA")

    if progress_cb:
        progress_cb(30)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    if progress_cb:
        progress_cb(65)

    meshes = model.extract_mesh(scene_codes, resolution=256)
    mesh = meshes[0]

    if progress_cb:
        progress_cb(85)

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)
