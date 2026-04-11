import os
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter()

# In-memory job store
jobs: dict = {}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Quick CLIP recognition — call after image upload, before generating."""
    import io
    from PIL import Image
    from services.segmentation_service import recognize_object
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
        return recognize_object(img)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"label": "objeto", "label_es": "objeto", "confidence": 0.0, "error": str(e)}


@router.post("/generate/image")
async def generate_image_to_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    resolution: int = Form(256),
    remove_bg: bool = Form(True),
    enhance: bool = Form(False),
    pipeline: str = Form("standard"),   # "standard" | "advanced"
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "output": None, "error": None}

    img_path = f"outputs/{job_id}_input.png"
    content = await file.read()
    with open(img_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_run_image_generation, job_id, img_path, resolution, remove_bg, enhance, pipeline)
    return {"job_id": job_id}


@router.post("/generate/text")
async def generate_text_to_3d(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    quality: str = Form("normal"),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "output": None, "error": None}

    background_tasks.add_task(_run_text_generation, job_id, prompt, quality)
    return {"job_id": job_id}


@router.post("/generate/face")
async def generate_face_to_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    add_bust_base: bool = Form(True),
):
    """Reconstruct a 3-D face/bust from a single frontal photo (MediaPipe FaceMesh)."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "output": None, "error": None}

    img_path = f"outputs/{job_id}_input.png"
    content = await file.read()
    with open(img_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_run_face_generation, job_id, img_path, add_bust_base)
    return {"job_id": job_id}


@router.post("/generate/body")
async def generate_body_to_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("fast"),       # "fast" | "quality"
    add_base: bool = Form(True),
):
    """Reconstruct a full-body 3-D mesh from a single frontal photo."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "output": None, "error": None}

    img_path = f"outputs/{job_id}_input.png"
    content = await file.read()
    with open(img_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_run_body_generation, job_id, img_path, mode, add_base)
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/download/{job_id}/{fmt}")
def download_model(job_id: str, fmt: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    if fmt not in ("stl", "obj", "glb"):
        raise HTTPException(status_code=400, detail="Format must be stl, obj, or glb")

    file_path = f"outputs/{job_id}.{fmt}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_types = {"stl": "model/stl", "obj": "text/plain", "glb": "model/gltf-binary"}
    return FileResponse(
        file_path,
        media_type=media_types[fmt],
        filename=f"model_{job_id[:8]}.{fmt}",
    )


# --- Background workers ---

def _update(job_id: str, progress: int):
    jobs[job_id]["progress"] = progress


def _run_image_generation(job_id: str, img_path: str, resolution: int, remove_bg: bool, enhance: bool, pipeline: str):
    try:
        jobs[job_id]["status"] = "processing"
        if pipeline == "advanced":
            from services.triposr_service import generate_from_image_advanced as _gen
        else:
            from services.triposr_service import generate_from_image as _gen
        _gen(
            img_path, f"outputs/{job_id}",
            lambda p: _update(job_id, p),
            resolution=resolution,
            remove_bg=remove_bg,
            enhance=enhance,
        )
        jobs[job_id].update({"status": "completed", "progress": 100, "output": f"/outputs/{job_id}.glb"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "error", "error": str(e)})


def _run_text_generation(job_id: str, prompt: str, quality: str):
    try:
        jobs[job_id]["status"] = "processing"
        from services.shape_service import generate_from_text
        generate_from_text(prompt, f"outputs/{job_id}", lambda p: _update(job_id, p), quality=quality)
        jobs[job_id].update({"status": "completed", "progress": 100, "output": f"/outputs/{job_id}.glb"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "error", "error": str(e)})


def _run_face_generation(job_id: str, img_path: str, add_bust_base: bool):
    try:
        jobs[job_id]["status"] = "processing"
        from services.face_service import reconstruct_face
        reconstruct_face(
            img_path, f"outputs/{job_id}",
            progress_cb=lambda p: _update(job_id, p),
            add_bust_base=add_bust_base,
        )
        jobs[job_id].update({"status": "completed", "progress": 100, "output": f"/outputs/{job_id}.glb"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "error", "error": str(e)})


def _run_body_generation(job_id: str, img_path: str, mode: str, add_base: bool):
    try:
        jobs[job_id]["status"] = "processing"
        from services.body_service import reconstruct_body
        reconstruct_body(
            img_path, f"outputs/{job_id}",
            progress_cb=lambda p: _update(job_id, p),
            mode=mode,
            add_base=add_base,
        )
        jobs[job_id].update({"status": "completed", "progress": 100, "output": f"/outputs/{job_id}.glb"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "error", "error": str(e)})
