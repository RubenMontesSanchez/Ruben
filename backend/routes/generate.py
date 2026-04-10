import os
import uuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter()

# In-memory job store
jobs: dict = {}


@router.post("/generate/image")
async def generate_image_to_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    resolution: int = Form(256),
    remove_bg: bool = Form(True),
    enhance: bool = Form(False),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "output": None, "error": None}

    img_path = f"outputs/{job_id}_input.png"
    content = await file.read()
    with open(img_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_run_image_generation, job_id, img_path, resolution, remove_bg, enhance)
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


def _run_image_generation(job_id: str, img_path: str, resolution: int, remove_bg: bool, enhance: bool):
    try:
        jobs[job_id]["status"] = "processing"
        from services.triposr_service import generate_from_image
        generate_from_image(
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
