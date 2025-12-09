import time
import os
from celery import shared_task
from app.services.model import classifier
from app.services.cache import cache
from app.database import SessionLocal
from app.models import Job, Prediction, JobStatus
from app.utils.hash_utils import calculate_image_hash
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_single_image(self, image_path: str, job_id: str):
    """
    Process a single image asynchronously.
    """
    db = SessionLocal()
    try:
        # 1. Update Job Status to PROCESSING
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found.")
            return
        
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.PROCESSING
            job.started_at = func.now()
            db.commit()

        # 2. Load Image
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            # Handle error
            return

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # 3. Calculate Hash
        image_hash = calculate_image_hash(image_bytes)

        # 4. Check Cache
        cached_result = cache.get_prediction(image_hash)
        
        prediction_data = {}
        from_cache = False
        processing_time = 0.0

        if cached_result:
            from_cache = True
            prediction_data = cached_result
            job.cached_images += 1
        else:
            # 5. Inference
            start_time = time.time()
            result = classifier.predict(image_bytes)
            processing_time = (time.time() - start_time) * 1000
            
            prediction_data = {
                "class": result["class"],
                "confidence": result["confidence"],
                "top_3_classes": result.get("top_3_classes", []) # Assuming model returns this
            }
            
            # Cache the result
            cache_data = prediction_data.copy()
            cache_data["cached_at"] = time.time()
            cache.set_prediction(image_hash, cache_data)

        # 6. Save Prediction
        prediction = Prediction(
            job_id=job_id,
            image_filename=os.path.basename(image_path),
            image_hash=image_hash,
            predicted_class=prediction_data["class"],
            confidence=prediction_data["confidence"],
            top_3_classes=prediction_data.get("top_3_classes"),
            processing_time_ms=processing_time,
            from_cache=from_cache
        )
        db.add(prediction)
        
        # 7. Update Job Progress
        job.processed_images += 1
        if job.total_images > 0:
            job.cache_hit_rate = (job.cached_images / job.processed_images) * 100

        # Mark as completed if this was the only image (simplification for single image task)
        # For batch processing, we'd check if processed == total
        job.status = JobStatus.COMPLETED
        job.completed_at = func.now()
        
        db.commit()
        
        return {
            "status": "success",
            "prediction": prediction_data,
            "from_cache": from_cache
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        db.rollback()
        # Update job with failure
        # job.failed_images += 1
        # db.commit()
        raise e
    finally:
        db.close()

from sqlalchemy.sql import func
