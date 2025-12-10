
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Job, Prediction, JobStatus
from app.schemas import JobResponse, PredictionResponse, JobCreate
from app.services.model import classifier
from app.services.cache import cache
from app.utils.image_utils import validate_image
from app.utils.hash_utils import calculate_image_hash
from app.workers.tasks import process_batch_images
import uuid
import time
import json
import shutil
import os
import zipfile

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.post("/batch", status_code=202)
async def batch_classify_images(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Upload a ZIP file containing multiple images for batch classification.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive.")

    # 1. Create Job ID and Temp Directory
    job_id = str(uuid.uuid4())
    upload_dir = os.path.join("data", "uploads", job_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    zip_path = os.path.join(upload_dir, "upload.zip")

    try:
        # 2. Save Uploaded ZIP
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Extract ZIP
        extracted_images = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(upload_dir)
            
            # 4. Filter for valid images
            for root, dirs, files in os.walk(upload_dir):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('.'):
                        full_path = os.path.join(root, filename)
                        extracted_images.append(os.path.abspath(full_path))
        
        if not extracted_images:
            shutil.rmtree(upload_dir)
            raise HTTPException(status_code=400, detail="No valid images found in ZIP.")

        # 5. Create Job Record
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            total_images=len(extracted_images)
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # 6. Dispatch Celery Task
        process_batch_images.delay(extracted_images, job_id)

        return {
            "job_id": job_id,
            "message": "Batch processing started.",
            "images_queued": len(extracted_images),
            "status_url": f"/api/v1/jobs/{job_id}"
        }

    except zipfile.BadZipFile:
        shutil.rmtree(upload_dir)
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")
    except Exception as e:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify", response_model=PredictionResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a single image (Synchronous for Phase 1/2).
    Includes Redis Caching (Phase 2).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    contents = await file.read()
    
    if not validate_image(contents):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 1. Calculate Image Hash
    image_hash = calculate_image_hash(contents)

    # 2. Check Cache
    cached_result = cache.get_prediction(image_hash)
    if cached_result:
        return {
            "filename": file.filename,
            "predicted_class": cached_result["class"],
            "confidence": cached_result["confidence"],
            "inference_time": 0.0, # Instant
            "top_3_classes": cached_result.get("top_3_classes", []),
            "from_cache": True
        }

    try:
        start_time = time.time()
        result = classifier.predict(contents)
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # 3. Cache the result
        cache_data = {
            "class": result["class"],
            "confidence": result["confidence"],
            "top_3_classes": [], # Todo: Populate this from model
            "cached_at": time.time()
        }
        cache.set_prediction(image_hash, cache_data)
        
        return {
            "filename": file.filename,
            "predicted_class": result["class"],
            "confidence": result["confidence"],
            "inference_time": inference_time,
            "top_3_classes": [], 
            "from_cache": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=list[JobResponse])
def read_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List all jobs.
    """
    jobs = db.query(Job).offset(skip).limit(limit).all()
    return jobs

@router.get("/{job_id}", response_model=JobResponse)
def read_job(job_id: str, db: Session = Depends(get_db)):
    """
    Get job status.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
