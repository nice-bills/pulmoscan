import time
import os
from celery import shared_task
from app.services.model import classifier
from app.services.cache import cache
from app.database import SessionLocal
from app.models import Job, Prediction, JobStatus
from app.utils.hash_utils import calculate_image_hash
from sqlalchemy.orm import Session
from typing import List
import logging
from sqlalchemy.sql import func
from tqdm import tqdm

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_single_image(self, image_path: str, job_id: str):
    """
    Process a single image asynchronously.
    """
    print(f"[Task {job_id}] Starting process_single_image for {image_path}")
    try:
        db = SessionLocal()
        print(f"[Task {job_id}] DB Session created")
    except Exception as e:
        print(f"[Task {job_id}] FAILED to create DB session: {e}")
        raise e

    try:
        # 1. Update Job Status to PROCESSING
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found.")
            return JobStatus.FAILED
        
        # Only transition from PENDING to PROCESSING once
        if job.status == JobStatus.PENDING:
            print(f"[Task {job_id}] Setting status to PROCESSING")
            job.status = JobStatus.PROCESSING
            job.started_at = func.now()
            db.commit()
            db.refresh(job)

        # 2. Load Image
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            job.failed_images += 1
            job.processed_images += 1
            db.commit()
            return JobStatus.FAILED

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        print(f"[Task {job_id}] Image loaded")

        # 3. Calculate Hash
        image_hash = calculate_image_hash(image_bytes)

        # 4. Check Cache
        cached_result = cache.get_prediction(image_hash)
        
        prediction_data = {}
        from_cache = False
        processing_time = 0.0

        if cached_result:
            print(f"[Task {job_id}] Cache HIT")
            from_cache = True
            prediction_data = cached_result
            job.cached_images += 1
        else:
            # 5. Inference
            print(f"[Task {job_id}] Starting Inference...")
            start_time = time.time()
            result = classifier.predict(image_bytes)
            processing_time = (time.time() - start_time) * 1000
            print(f"[Task {job_id}] Inference DONE in {processing_time:.2f}ms")
            
            prediction_data = {
                "class": result["class"],
                "confidence": result["confidence"],
                "top_3_classes": result.get("top_3_classes", [])
            }
            
            # Cache the result
            cache_data = prediction_data.copy()
            cache_data["cached_at"] = time.time()
            cache.set_prediction(image_hash, cache_data)

        # 6. Save Prediction
        print(f"[Task {job_id}] Saving prediction to DB")
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
        
        # For a single image task, mark job completed here if it's the only one
        if job.processed_images + job.failed_images == job.total_images:
            job.status = JobStatus.COMPLETED if job.failed_images == 0 else JobStatus.FAILED
            job.completed_at = func.now()
        
        db.commit()
        print(f"[Task {job_id}] DONE & Committed")
        
        return {
            "status": "success",
            "prediction": prediction_data,
            "from_cache": from_cache
        }

    except Exception as e:
        logger.error(f"Error processing single image for job {job_id}: {e}")
        print(f"[Task {job_id}] EXCEPTION: {e}")
        db.rollback()
        # Ensure job stats are updated even on unhandled error
        job = db.query(Job).filter(Job.id == job_id).first() 
        if job:
            job.failed_images += 1
            job.processed_images += 1
            if job.total_images > 0:
                job.cache_hit_rate = (job.cached_images / job.processed_images) * 100
            if job.processed_images + job.failed_images == job.total_images:
                job.status = JobStatus.FAILED
                job.completed_at = func.now()
            db.commit()
        raise e
    finally:
        db.close()
        print(f"[Task {job_id}] DB Session Closed")


@shared_task(bind=True)
def process_batch_images(self, image_paths: List[str], job_id: str):
    """
    Process a batch of images asynchronously.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Batch Job {job_id} not found.")
            return JobStatus.FAILED

        # Only transition from PENDING to PROCESSING once
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.PROCESSING
            job.started_at = func.now()
            db.commit()
            db.refresh(job)

        images_to_infer_bytes = []
        images_to_infer_metadata = [] # Stores (original_path, hash) for non-cached images
        predictions_to_add = [] # Collect all predictions (cached + inferred) for bulk add
        
        total_images_in_batch = len(image_paths)
        
        current_processed = 0
        current_failed = 0
        current_cached = 0

        for original_path in tqdm(image_paths, desc=f"Job {job_id}"):
            image_filename = os.path.basename(original_path)
            image_bytes = None
            image_hash = None
                    
            try:
                if not os.path.exists(original_path):
                    raise FileNotFoundError(f"Image not found: {original_path}")

                with open(original_path, "rb") as f:
                    image_bytes = f.read()
                
                image_hash = calculate_image_hash(image_bytes)
                cached_result = cache.get_prediction(image_hash)

                if cached_result:
                    prediction_data = {
                        "class": cached_result["class"],
                        "confidence": cached_result["confidence"],
                        "top_3_classes": cached_result.get("top_3_classes", []),
                        "processing_time_ms": 0.0, # Cached, instant
                        "from_cache": True
                    }
                    predictions_to_add.append(Prediction(
                        job_id=job_id,
                        image_filename=image_filename,
                        image_hash=image_hash,
                        predicted_class=prediction_data["class"],
                        confidence=prediction_data["confidence"],
                        top_3_classes=prediction_data["top_3_classes"],
                        processing_time_ms=prediction_data["processing_time_ms"],
                        from_cache=True
                    ))
                    current_cached += 1
                else:
                    images_to_infer_bytes.append(image_bytes)
                    images_to_infer_metadata.append({
                        "original_path": original_path,
                        "image_filename": image_filename,
                        "image_hash": image_hash
                    })

            except Exception as e:
                logger.error(f"Error processing image {original_path} for batch job {job_id}: {e}")
                current_failed += 1
                # Save a 'failed' prediction record
                predictions_to_add.append(Prediction(
                    job_id=job_id,
                    image_filename=image_filename if image_filename else "unknown",
                    image_hash=image_hash if image_hash else "error",
                    predicted_class="FAILED",
                    confidence=0.0,
                    top_3_classes=[],
                    processing_time_ms=0.0,
                    from_cache=False 
                ))
            
            # Update job progress after each image is processed (or failed)
            current_processed_total = current_cached + (len(images_to_infer_metadata)) + current_failed # Total images considered so far
            
            job.processed_images = current_processed_total - current_failed
            job.failed_images = current_failed
            job.cached_images = current_cached
            if job.total_images > 0 and job.processed_images > 0:
                job.cache_hit_rate = (job.cached_images / job.processed_images) * 100
            elif job.processed_images == 0:
                job.cache_hit_rate = 0.0
            
            # Commit intermediate progress to allow external polling
            db.commit()
            db.refresh(job) # Refresh job object to prevent stale data in session
        
        # Perform Batch Inference for non-cached images
        if images_to_infer_bytes:
            batch_inference_results = classifier.predict_batch(images_to_infer_bytes)
            
            for i, result in enumerate(batch_inference_results):
                metadata = images_to_infer_metadata[i]
                
                prediction_data = {
                    "class": result["class"],
                    "confidence": result["confidence"],
                    "top_3_classes": result.get("top_3_classes", []),
                    "processing_time_ms": result["inference_time"],
                    "from_cache": False
                }
                
                # Cache the new result
                cache_data = prediction_data.copy()
                cache_data["cached_at"] = time.time()
                cache.set_prediction(metadata["image_hash"], cache_data)

                predictions_to_add.append(Prediction(
                    job_id=job_id,
                    image_filename=metadata["image_filename"],
                    image_hash=metadata["image_hash"],
                    predicted_class=prediction_data["class"],
                    confidence=prediction_data["confidence"],
                    top_3_classes=prediction_data["top_3_classes"],
                    processing_time_ms=prediction_data["processing_time_ms"],
                    from_cache=False
                ))
        
        # Add all collected predictions to the session
        if predictions_to_add:
            db.add_all(predictions_to_add)

        # Final Job Status Update
        job.processed_images = total_images_in_batch - current_failed # Total successfully processed
        job.failed_images = current_failed
        job.cached_images = current_cached
        
        if job.total_images > 0 and job.processed_images > 0:
            job.cache_hit_rate = (job.cached_images / job.processed_images) * 100
        elif job.processed_images == 0:
            job.cache_hit_rate = 0.0

        if current_failed == total_images_in_batch:
            job.status = JobStatus.FAILED
        elif current_failed > 0:
            job.status = JobStatus.COMPLETED # Partially completed implies completed with some failures
        else:
            job.status = JobStatus.COMPLETED
            
        job.completed_at = func.now()
        db.commit() # Final commit
        
        return job.status.value

    except Exception as e:
        logger.error(f"Unhandled error in batch job {job_id}: {e}")
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first() # Re-fetch in case of rollback
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = func.now()
            db.commit()
        raise e
    finally:
        db.close()


