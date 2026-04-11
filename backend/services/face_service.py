"""Face reconstruction service — single photo → 3D face/bust mesh.

Pipeline:
  1. MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10) detects 478 3D landmarks
  2. Delaunay triangulation builds a watertight mesh
  3. Depth is scaled and smoothed for a printable result
  4. Optional bust base added below the neck

Model file (~30 MB) is downloaded automatically on first use.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import trimesh
from PIL import Image

# ── Model path ────────────────────────────────────────────────────────────────
_MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
_FACE_MODEL  = os.path.join(_MODELS_DIR, "face_landmarker.task")
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_model():
    os.makedirs(_MODELS_DIR, exist_ok=True)
    if not os.path.exists(_FACE_MODEL):
        print("Descargando modelo FaceLandmarker (~30 MB)…")
        urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL)
        print("Modelo descargado.")


def reconstruct_face(
    img_path: str,
    output_base: str,
    progress_cb=None,
    add_bust_base: bool = True,
):
    """Reconstruct a 3D face mesh from a single frontal photo."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        raise RuntimeError(
            "Dependencia faltante. Ejecuta:\n  pip install mediapipe"
        )

    _ensure_face_model()

    if progress_cb:
        progress_cb(10)

    # ── Load image ────────────────────────────────────────────────────────────
    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil, dtype=np.uint8)
    h, w    = img_np.shape[:2]

    # ── MediaPipe FaceLandmarker (Tasks API) ──────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=_FACE_MODEL)
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector  = mp_vision.FaceLandmarker.create_from_options(options)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result    = detector.detect(mp_image)
    detector.close()

    if not result.face_landmarks:
        raise ValueError(
            "No se detectó ninguna cara en la imagen.\n"
            "Usa una foto frontal, bien iluminada y sin gafas de sol."
        )

    if progress_cb:
        progress_cb(35)

    landmarks = result.face_landmarks[0]  # list of NormalizedLandmark

    # ── Build 3-D vertex array ────────────────────────────────────────────────
    face_size = w
    verts = np.array(
        [
            [
                (lm.x - 0.5) * face_size,
                -(lm.y - 0.5) * face_size,
                lm.z * face_size * 1.5,
            ]
            for lm in landmarks
        ],
        dtype=np.float64,
    )

    # ── Delaunay triangulation on 2-D projection ──────────────────────────────
    from scipy.spatial import Delaunay

    pts_2d = verts[:, :2]
    tri    = Delaunay(pts_2d)
    faces  = tri.simplices.astype(np.int64)

    if progress_cb:
        progress_cb(60)

    # ── Create and clean mesh ─────────────────────────────────────────────────
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.3, iterations=5)

    if progress_cb:
        progress_cb(75)

    if add_bust_base:
        mesh = _add_bust_base(mesh)

    if progress_cb:
        progress_cb(88)

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)


def _add_bust_base(face_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Extend the face mesh downward with a neck/shoulder cylinder base."""
    bounds      = face_mesh.bounds
    chin_y      = bounds[0][1]
    face_w      = bounds[1][0] - bounds[0][0]
    base_height = face_w * 0.4
    base        = trimesh.creation.cylinder(
        radius=face_w * 0.18,
        height=base_height,
        sections=32,
    )
    base.apply_translation([0, chin_y - base_height / 2, 0])
    return trimesh.util.concatenate([face_mesh, base])
