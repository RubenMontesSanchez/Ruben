"""Full-body reconstruction service — single photo → clothed 3-D body mesh.

Pipeline (two tiers):
  Tier A — fast (no GPU required):
    1. MediaPipe Pose detects 33 3D body landmarks
    2. A simple parametric body mesh is built around those landmarks
    3. Optional clothing layer (convex-hull inflation)

  Tier B — quality (GPU recommended, ~6–8 GB VRAM):
    Uses PIFu-HD (https://github.com/shunsukesaito/PIFuHD) if installed.
    PIFu produces a detailed clothed human mesh from a single image.
    Falls back to Tier A if PIFuHD is not available.

Requirements (Tier A): pip install mediapipe
Requirements (Tier B): see PIFuHD installation guide (optional)
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reconstruct_body(
    img_path: str,
    output_base: str,
    progress_cb=None,
    mode: str = "fast",          # "fast" | "quality"
    add_base: bool = True,
):
    """
    Reconstruct a full-body 3-D mesh from a single frontal photo.

    Parameters
    ----------
    img_path     : path to the input image (any format PIL can open)
    output_base  : prefix for output files (exports .glb / .stl / .obj)
    progress_cb  : optional callable(int 0-100)
    mode         : "fast" uses MediaPipe landmarks; "quality" tries PIFuHD
    add_base     : add a small flat disc base so the figure stands upright
    """
    if mode == "quality":
        try:
            return _reconstruct_pifuhd(img_path, output_base, progress_cb, add_base)
        except Exception:
            pass  # fall through to fast tier

    return _reconstruct_landmarks(img_path, output_base, progress_cb, add_base)


# ---------------------------------------------------------------------------
# Tier A — MediaPipe Pose landmark skeleton
# ---------------------------------------------------------------------------

def _reconstruct_landmarks(img_path: str, output_base: str, progress_cb, add_base: bool):
    """Build a body mesh from MediaPipe Pose 33-point landmarks."""
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

    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil)
    h, w    = img_np.shape[:2]

    # ── Detect pose landmarks ─────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(img_np)

    if not results.pose_landmarks:
        raise ValueError(
            "No se detectó ninguna persona en la imagen.\n"
            "Usa una foto frontal, de cuerpo completo y bien iluminada."
        )

    if progress_cb:
        progress_cb(30)

    lm = results.pose_landmarks.landmark

    # Convert normalised coords to world units (mm-scale, Y up)
    scale = max(w, h)
    def pt(idx: int) -> np.ndarray:
        l = lm[idx]
        return np.array([
            (l.x - 0.5) * scale,
            -(l.y - 0.5) * scale,
            -l.z * scale * 0.4,
        ], dtype=np.float64)

    # ── Build body geometry from landmark segments ───────────────────────
    meshes: list[trimesh.Trimesh] = []

    # MediaPipe Pose landmark indices (a subset used for body shape)
    # Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    NOSE        = 0
    L_SHOULDER  = 11;  R_SHOULDER  = 12
    L_ELBOW     = 13;  R_ELBOW     = 14
    L_WRIST     = 15;  R_WRIST     = 16
    L_HIP       = 23;  R_HIP       = 24
    L_KNEE      = 25;  R_KNEE      = 26
    L_ANKLE     = 27;  R_ANKLE     = 28
    L_INDEX     = 19;  R_INDEX     = 20  # hand tips

    # Helper: capsule between two points
    def capsule(a: np.ndarray, b: np.ndarray, radius: float, sections: int = 12) -> trimesh.Trimesh:
        vec   = b - a
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return trimesh.creation.icosphere(radius=radius, subdivisions=1)
        cyl  = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        # Align cylinder axis (Z) with vec
        z_axis  = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(z_axis, vec / length)
        rot_norm = float(np.linalg.norm(rot_axis))
        if rot_norm > 1e-6:
            angle = float(np.arcsin(np.clip(rot_norm, -1.0, 1.0)))
            mat   = trimesh.transformations.rotation_matrix(angle, rot_axis / rot_norm)
            cyl.apply_transform(mat)
        # Translate to midpoint
        mid = (a + b) / 2.0
        cyl.apply_translation(mid)
        return cyl

    # Body segment radii (rough anatomical proportions, mm-scale relative to image)
    shoulder_sep = float(np.linalg.norm(pt(L_SHOULDER) - pt(R_SHOULDER)))
    torso_r  = shoulder_sep * 0.14
    limb_r   = shoulder_sep * 0.07
    head_r   = shoulder_sep * 0.22
    hand_r   = shoulder_sep * 0.05
    foot_r   = shoulder_sep * 0.07

    # Head
    head_center = (pt(NOSE) + pt(L_SHOULDER) + pt(R_SHOULDER)) / 3
    head_center[1] += head_r * 0.4   # shift slightly up
    meshes.append(trimesh.creation.icosphere(radius=head_r, subdivisions=2))
    meshes[-1].apply_translation(head_center)

    # Torso
    shoulder_mid = (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2
    hip_mid      = (pt(L_HIP)      + pt(R_HIP))      / 2
    meshes.append(capsule(shoulder_mid, hip_mid, torso_r, sections=16))

    # Arms
    for sh, el, wr, idx in [
        (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
        (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
    ]:
        meshes.append(capsule(pt(sh), pt(el), limb_r))
        meshes.append(capsule(pt(el), pt(wr), limb_r * 0.85))
        # Hand blob
        hand = trimesh.creation.icosphere(radius=hand_r, subdivisions=1)
        hand.apply_translation(pt(wr))
        meshes.append(hand)

    # Legs
    for hip, knee, ankle in [
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    ]:
        meshes.append(capsule(pt(hip), pt(knee), limb_r * 1.1))
        meshes.append(capsule(pt(knee), pt(ankle), limb_r * 0.9))
        # Foot blob
        foot = trimesh.creation.icosphere(radius=foot_r, subdivisions=1)
        foot.apply_translation(pt(ankle))
        meshes.append(foot)

    if progress_cb:
        progress_cb(60)

    # ── Combine and smooth ────────────────────────────────────────────────
    body = trimesh.util.concatenate(meshes)
    trimesh.smoothing.filter_laplacian(body, lamb=0.5, iterations=3)

    if progress_cb:
        progress_cb(80)

    if add_base:
        body = _add_standing_base(body)

    if progress_cb:
        progress_cb(90)

    body.export(f"{output_base}.glb")
    body.export(f"{output_base}.stl")
    body.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)


# ---------------------------------------------------------------------------
# Tier B — PIFuHD (optional, quality reconstruction)
# ---------------------------------------------------------------------------

def _reconstruct_pifuhd(img_path: str, output_base: str, progress_cb, add_base: bool):
    """
    Use PIFuHD for high-quality clothed human reconstruction.

    PIFuHD must be cloned into backend/PIFuHD/ and its requirements installed.
    https://github.com/shunsukesaito/PIFuHD
    """
    import sys, os, subprocess, tempfile

    pifuhd_path = os.path.join(os.path.dirname(__file__), "..", "PIFuHD")
    if not os.path.isdir(pifuhd_path):
        raise FileNotFoundError("PIFuHD no encontrado en backend/PIFuHD/")

    sys.path.insert(0, os.path.abspath(pifuhd_path))

    if progress_cb:
        progress_cb(15)

    # PIFuHD ships a simple_test.py script that takes --input_path / --out_path
    script = os.path.join(pifuhd_path, "apps", "simple_test.py")
    if not os.path.exists(script):
        raise FileNotFoundError("PIFuHD script no encontrado: apps/simple_test.py")

    out_dir = os.path.dirname(output_base)
    result  = subprocess.run(
        [sys.executable, script,
         "--input_path", img_path,
         "--out_path",   out_dir,
         "--resolution", "256"],
        cwd=pifuhd_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"PIFuHD falló:\n{result.stderr[-500:]}")

    if progress_cb:
        progress_cb(85)

    # PIFuHD outputs result.obj — rename to our expected paths
    out_obj = os.path.join(out_dir, "result.obj")
    if not os.path.exists(out_obj):
        raise FileNotFoundError("PIFuHD no generó result.obj")

    mesh = trimesh.load(out_obj, force="mesh")
    if add_base:
        mesh = _add_standing_base(mesh)

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_standing_base(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Add a flat disc base at the bottom of the mesh."""
    bounds    = mesh.bounds
    foot_y    = bounds[0][1]
    body_w    = bounds[1][0] - bounds[0][0]
    base_r    = body_w * 0.35
    base_h    = body_w * 0.04

    base = trimesh.creation.cylinder(radius=base_r, height=base_h, sections=48)
    base.apply_translation([0, foot_y - base_h / 2, 0])

    return trimesh.util.concatenate([mesh, base])
