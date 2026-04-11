"""TripoSR service — image → 3D mesh (STL / OBJ / GLB).

Two pipelines available:
  - standard:  rembg + TripoSR
  - advanced:  rembg + Zero123++ (6 views) → free VRAM → TripoSR
"""
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
                "TripoSR no esta instalado. Ejecuta en la carpeta backend/:\n"
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


def _preprocess(img_path: str, remove_bg: bool, enhance: bool) -> Image.Image:
    """Load, optionally enhance, optionally remove background."""
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    raw = Image.open(img_path)
    raw.load()  # force full read before file handle closes

    if enhance:
        from PIL import ImageEnhance
        raw = raw.convert("RGB")
        raw = ImageEnhance.Contrast(raw).enhance(1.3)
        raw = ImageEnhance.Sharpness(raw).enhance(1.5)

    if remove_bg:
        from tsr.utils import remove_background, resize_foreground
        from rembg import new_session as rembg_new_session
        sess = rembg_new_session()
        image = remove_background(raw, sess)
        image = resize_foreground(image, 0.85)
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg.convert("RGB")

    return raw.convert("RGB")


def _reconstruct(image: Image.Image, output_base: str, resolution: int, progress_cb, p_start: int, p_end: int):
    """Run TripoSR on a single PIL image and export meshes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model()

    if progress_cb:
        progress_cb(p_start)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    if progress_cb:
        progress_cb(p_start + int((p_end - p_start) * 0.6))

    meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=resolution)
    mesh = meshes[0]

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(p_end)


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_from_image(
    img_path: str,
    output_base: str,
    progress_cb=None,
    resolution: int = 256,
    remove_bg: bool = True,
    enhance: bool = False,
):
    """Standard pipeline: preprocess → TripoSR."""
    if progress_cb:
        progress_cb(10)

    image = _preprocess(img_path, remove_bg, enhance)

    if progress_cb:
        progress_cb(30)

    _reconstruct(image, output_base, resolution, progress_cb, p_start=30, p_end=100)


def generate_from_image_advanced(
    img_path: str,
    output_base: str,
    progress_cb=None,
    resolution: int = 256,
    remove_bg: bool = True,
    enhance: bool = False,
):
    """
    Advanced pipeline: preprocess → Zero123++ (6 views) → free VRAM → TripoSR.

    VRAM usage (sequential, never simultaneous):
      Step 1  rembg preprocessing        ~0.5 GB
      Step 2  Zero123++ inference         ~6-7 GB  → freed after
      Step 3  TripoSR reconstruction      ~4 GB
    """
    if progress_cb:
        progress_cb(5)

    # Step 1 — preprocess (CPU / small VRAM)
    image = _preprocess(img_path, remove_bg, enhance)

    if progress_cb:
        progress_cb(10)

    # Step 2 — Zero123++ multi-view (loads ~6 GB, then freed)
    from services.zero123_service import generate_multiview

    def _z_progress(p):
        if progress_cb:
            # map 0-100 from zero123 → 10-65 overall
            progress_cb(10 + int(p * 0.55))

    views = generate_multiview(image, progress_cb=_z_progress)

    # Save generated views alongside the output
    for i, v in enumerate(views):
        v.save(f"{output_base}_view{i}.png")

    if progress_cb:
        progress_cb(65)

    # Step 3 — TripoSR from the best generated view
    # View 0  (azimuth ~30°, slight elevation) is the best canonical
    # angle in the Zero123++ v1.2 grid for downstream reconstruction.
    best_view = views[0]

    _reconstruct(best_view, output_base, resolution, progress_cb, p_start=65, p_end=100)
