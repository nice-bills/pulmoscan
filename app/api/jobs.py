
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Job, Prediction
from app.schemas import JobResponse, PredictionResponse
from app.services.model import classifier
from app.services.cache import cache
from app.utils.image_utils import validate_image
from app.utils.hash_utils import calculate_image_hash
import uuid
import time
import json

router = APIRouter(prefix="/jobs", tags=["Jobs"])

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
