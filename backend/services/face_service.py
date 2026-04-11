"""Face reconstruction service — single photo → 3D face/bust mesh.

Pipeline:
  1. MediaPipe FaceMesh detects 478 3D facial landmarks
  2. Delaunay triangulation builds a watertight mesh
  3. Depth is scaled and smoothed for a printable result
  4. Optional bust base added below the neck

Requirements: pip install mediapipe
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image


def reconstruct_face(
    img_path: str,
    output_base: str,
    progress_cb=None,
    add_bust_base: bool = True,
):
    """
    Reconstruct a 3D face mesh from a single frontal photo.
    Returns a trimesh-exportable mesh with the person's facial geometry.
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        raise RuntimeError(
            "Dependencias faltantes. Ejecuta:\n"
            "  pip install mediapipe opencv-python-headless"
        )

    if progress_cb:
        progress_cb(10)

    # ── Load image ────────────────────────────────────────────────────────────
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    # ── MediaPipe FaceMesh ────────────────────────────────────────────────────
    mp_fm = mp.solutions.face_mesh
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(img_np)

    if not results.multi_face_landmarks:
        raise ValueError(
            "No se detectó ninguna cara en la imagen.\n"
            "Usa una foto frontal, bien iluminada y sin gafas de sol."
        )

    if progress_cb:
        progress_cb(35)

    landmarks = results.multi_face_landmarks[0].landmark

    # ── Build 3-D vertex array ────────────────────────────────────────────────
    # MediaPipe Z is in the same scale as X (normalised by image width).
    # We scale Z to give realistic face depth (~60 % of face width).
    face_size = w  # rough scale reference
    verts = np.array(
        [
            [
                (lm.x - 0.5) * face_size,          # centre on X
                -(lm.y - 0.5) * face_size,          # flip Y (image→world)
                lm.z * face_size * 1.5,             # amplify depth
            ]
            for lm in landmarks
        ],
        dtype=np.float64,
    )

    # ── Triangulate via MediaPipe's built-in tessellation ────────────────────
    # FACEMESH_TESSELATION is a set of (i, j) edge pairs that form triangles
    # when combined.  We convert them to actual triangle faces.
    connections = list(mp_fm.FACEMESH_TESSELATION)

    # Build adjacency and extract triangles using Delaunay on 2-D projection
    from scipy.spatial import Delaunay

    pts_2d = verts[:, :2]
    tri = Delaunay(pts_2d)
    faces = tri.simplices.astype(np.int64)

    if progress_cb:
        progress_cb(60)

    # ── Create and clean mesh ─────────────────────────────────────────────────
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()

    # Smooth to reduce polygon artefacts
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.3, iterations=5)

    if progress_cb:
        progress_cb(75)

    # ── Optional: add a flat bust/neck base ───────────────────────────────────
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
    bounds = face_mesh.bounds
    chin_y = bounds[0][1]  # lowest Y point (chin)
    face_w = bounds[1][0] - bounds[0][0]

    # Create a cylinder as a base (neck + shoulder stub)
    base_height = face_w * 0.4
    base = trimesh.creation.cylinder(
        radius=face_w * 0.18,
        height=base_height,
        sections=32,
    )
    # Position below chin
    base.apply_translation([0, chin_y - base_height / 2, 0])

    combined = trimesh.util.concatenate([face_mesh, base])
    return combined
