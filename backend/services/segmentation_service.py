"""Segmentation and recognition service.

- recognize_object(): CLIP zero-shot classification
- segment_foreground(): SAM-based foreground extraction (better than rembg for
  complex shapes like humans/animals — keeps the object as one connected region)
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# ── Object categories for CLIP recognition ───────────────────────────────────
_CATEGORIES = [
    "person", "human figure", "animal", "dog", "cat", "bird", "horse",
    "chair", "table", "sofa", "lamp", "vase", "bottle", "cup", "mug",
    "shoe", "bag", "helmet", "toy", "figurine", "car", "motorcycle",
    "sculpture", "bust", "face", "hand", "tree", "rock", "building",
]

_LABELS_ES = {
    "person": "persona", "human figure": "figura humana", "animal": "animal",
    "dog": "perro", "cat": "gato", "bird": "pájaro", "horse": "caballo",
    "chair": "silla", "table": "mesa", "sofa": "sofá", "lamp": "lámpara",
    "vase": "jarrón", "bottle": "botella", "cup": "taza", "mug": "taza",
    "shoe": "zapato", "bag": "bolso", "helmet": "casco", "toy": "juguete",
    "figurine": "figurita", "car": "coche", "motorcycle": "moto",
    "sculpture": "escultura", "bust": "busto", "face": "cara",
    "hand": "mano", "tree": "árbol", "rock": "roca", "building": "edificio",
}

# Cached model to avoid reloading on every request
_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor


def recognize_object(image: Image.Image) -> dict:
    """
    Run CLIP zero-shot classification and return the top label + confidence.
    Model is cached after first load (~600 MB download on first use).
    Runs on CPU to leave VRAM free for TripoSR.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        return {"label": "objeto", "label_es": "objeto", "confidence": 0.0, "error": "CLIP no instalado"}

    try:
        model, processor = _get_clip()

        img_rgb = image.convert("RGB").resize((224, 224))
        inputs = processor(
            text=_CATEGORIES,
            images=img_rgb,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        top_idx = int(probs.argmax())
        label = _CATEGORIES[top_idx]

        return {
            "label": label,
            "label_es": _LABELS_ES.get(label, label),
            "confidence": float(probs[top_idx]),
        }
    except Exception as e:
        return {"label": "objeto", "label_es": "objeto", "confidence": 0.0, "error": str(e)}


def segment_foreground(image: Image.Image) -> Image.Image:
    """
    Use SAM (ViT-B, ~375 MB) to extract the main foreground object.
    Returns an RGBA image with the background transparent.
    Falls back to rembg if SAM is not available.
    """
    try:
        from transformers import SamModel, SamProcessor
    except ImportError:
        return _rembg_fallback(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model.eval()
    except Exception:
        return _rembg_fallback(image)

    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # Use center point as the foreground prompt
    cx, cy = w // 2, h // 2
    input_points = [[[cx, cy]]]

    inputs = sam_processor(
        images=img_rgb,
        input_points=input_points,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    # Pick the mask with the highest IoU score
    scores = outputs.iou_scores[0, 0]
    best = int(scores.argmax())
    mask = masks[0][0][best].numpy().astype(np.uint8) * 255

    del sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply mask as alpha channel
    rgba = img_rgb.convert("RGBA")
    mask_img = Image.fromarray(mask, mode="L")
    rgba.putalpha(mask_img)
    return rgba


def _rembg_fallback(image: Image.Image) -> Image.Image:
    from rembg import remove, new_session
    return remove(image.convert("RGBA"), session=new_session())
