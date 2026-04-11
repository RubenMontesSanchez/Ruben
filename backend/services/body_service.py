"""Full-body reconstruction service — single photo → clothed 3-D body mesh.

Pipeline (Tier A — fast):
  1. MediaPipe PoseLandmarker (Tasks API, mediapipe >= 0.10) detects 33 3D landmarks
  2. Capsule-based body mesh built around those landmarks
  3. Laplacian smoothing + optional standing disc base

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
_POSE_MODEL  = os.path.join(_MODELS_DIR, "pose_landmarker_full.task")
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _ensure_pose_model():
    os.makedirs(_MODELS_DIR, exist_ok=True)
    if not os.path.exists(_POSE_MODEL):
        print("Descargando modelo PoseLandmarker (~30 MB)…")
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL)
        print("Modelo descargado.")


def reconstruct_body(
    img_path: str,
    output_base: str,
    progress_cb=None,
    mode: str = "fast",
    add_base: bool = True,
):
    """Reconstruct a full-body 3-D mesh from a single frontal photo."""
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
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        raise RuntimeError(
            "Dependencia faltante. Ejecuta:\n  pip install mediapipe"
        )

    _ensure_pose_model()

    if progress_cb:
        progress_cb(10)

    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil, dtype=np.uint8)
    h, w    = img_np.shape[:2]

    # ── Detect pose landmarks ─────────────────────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=_POSE_MODEL)
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector  = mp_vision.PoseLandmarker.create_from_options(options)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result    = detector.detect(mp_image)
    detector.close()

    if not result.pose_landmarks:
        raise ValueError(
            "No se detectó ninguna persona en la imagen.\n"
            "Usa una foto frontal, de cuerpo completo y bien iluminada."
        )

    if progress_cb:
        progress_cb(30)

    lm    = result.pose_landmarks[0]
    scale = max(w, h)

    def pt(idx: int) -> np.ndarray:
        l = lm[idx]
        return np.array(
            [(l.x - 0.5) * scale, -(l.y - 0.5) * scale, -l.z * scale * 0.4],
            dtype=np.float64,
        )

    # ── Build body geometry ───────────────────────────────────────────────────
    meshes: list[trimesh.Trimesh] = []

    NOSE = 0
    L_SHOULDER = 11; R_SHOULDER = 12
    L_ELBOW    = 13; R_ELBOW    = 14
    L_WRIST    = 15; R_WRIST    = 16
    L_HIP      = 23; R_HIP      = 24
    L_KNEE     = 25; R_KNEE     = 26
    L_ANKLE    = 27; R_ANKLE    = 28

    def capsule(a: np.ndarray, b: np.ndarray, radius: float, sections: int = 12) -> trimesh.Trimesh:
        vec    = b - a
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return trimesh.creation.icosphere(radius=radius, subdivisions=1)
        cyl      = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        z_axis   = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(z_axis, vec / length)
        rot_norm = float(np.linalg.norm(rot_axis))
        if rot_norm > 1e-6:
            angle = float(np.arcsin(np.clip(rot_norm, -1.0, 1.0)))
            mat   = trimesh.transformations.rotation_matrix(angle, rot_axis / rot_norm)
            cyl.apply_transform(mat)
        cyl.apply_translation((a + b) / 2.0)
        return cyl

    shoulder_sep = float(np.linalg.norm(pt(L_SHOULDER) - pt(R_SHOULDER)))
    torso_r = shoulder_sep * 0.14
    limb_r  = shoulder_sep * 0.07
    head_r  = shoulder_sep * 0.22
    hand_r  = shoulder_sep * 0.05
    foot_r  = shoulder_sep * 0.07

    # Head
    head_center    = (pt(NOSE) + pt(L_SHOULDER) + pt(R_SHOULDER)) / 3
    head_center[1] += head_r * 0.4
    head = trimesh.creation.icosphere(radius=head_r, subdivisions=2)
    head.apply_translation(head_center)
    meshes.append(head)

    # Torso
    meshes.append(capsule(
        (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2,
        (pt(L_HIP)      + pt(R_HIP))      / 2,
        torso_r, sections=16,
    ))

    # Arms
    for sh, el, wr in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
        meshes.append(capsule(pt(sh), pt(el), limb_r))
        meshes.append(capsule(pt(el), pt(wr), limb_r * 0.85))
        hand = trimesh.creation.icosphere(radius=hand_r, subdivisions=1)
        hand.apply_translation(pt(wr))
        meshes.append(hand)

    # Legs
    for hip, knee, ankle in [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)]:
        meshes.append(capsule(pt(hip), pt(knee), limb_r * 1.1))
        meshes.append(capsule(pt(knee), pt(ankle), limb_r * 0.9))
        foot = trimesh.creation.icosphere(radius=foot_r, subdivisions=1)
        foot.apply_translation(pt(ankle))
        meshes.append(foot)

    if progress_cb:
        progress_cb(60)

    body = trimesh.util.concatenate(meshes)
    trimesh.smoothing.filter_laplacian(body, lamb=0.5, iterations=3)

    if progress_cb:
        progress_cb(80)

    if add_base:
        body = _add_standing_base(body)

    body.export(f"{output_base}.glb")
    body.export(f"{output_base}.stl")
    body.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)


# ---------------------------------------------------------------------------
# Tier B — PIFuHD stub
# ---------------------------------------------------------------------------

def _reconstruct_pifuhd(img_path: str, output_base: str, progress_cb, add_base: bool):
    import sys, subprocess

    pifuhd_path = os.path.join(os.path.dirname(__file__), "..", "PIFuHD")
    if not os.path.isdir(pifuhd_path):
        raise FileNotFoundError("PIFuHD no encontrado en backend/PIFuHD/")

    script = os.path.join(pifuhd_path, "apps", "simple_test.py")
    out_dir = os.path.dirname(output_base)
    result  = subprocess.run(
        [sys.executable, script, "--input_path", img_path,
         "--out_path", out_dir, "--resolution", "256"],
        cwd=pifuhd_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PIFuHD falló:\n{result.stderr[-500:]}")

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
    bounds = mesh.bounds
    foot_y = bounds[0][1]
    body_w = bounds[1][0] - bounds[0][0]
    base_r = body_w * 0.35
    base_h = body_w * 0.04
    base   = trimesh.creation.cylinder(radius=base_r, height=base_h, sections=48)
    base.apply_translation([0, foot_y - base_h / 2, 0])
    return trimesh.util.concatenate([mesh, base])
