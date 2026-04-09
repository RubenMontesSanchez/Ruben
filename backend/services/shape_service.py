"""Shap-E service — text → 3D mesh (STL / OBJ / GLB)."""
from __future__ import annotations

import numpy as np
import torch
import trimesh

_models = None


def _get_models():
    global _models
    if _models is None:
        try:
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_config, load_model
        except ImportError:
            raise RuntimeError(
                "Shap-E is not installed. Run:\n"
                "  pip install git+https://github.com/openai/shap-e.git"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xm = load_model("transmitter", device=device)
        model = load_model("text300M", device=device)
        diffusion = diffusion_from_config(load_config("diffusion"))
        _models = (xm, model, diffusion, device)
    return _models


def generate_from_text(prompt: str, output_base: str, progress_cb=None):
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh

    xm, model, diffusion, device = _get_models()

    if progress_cb:
        progress_cb(15)

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    if progress_cb:
        progress_cb(80)

    t_mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.array(t_mesh.verts),
        faces=np.array(t_mesh.faces),
    )

    mesh.export(f"{output_base}.glb")
    mesh.export(f"{output_base}.stl")
    mesh.export(f"{output_base}.obj")

    if progress_cb:
        progress_cb(100)
