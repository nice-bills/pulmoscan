import sys
import os
import time
import uuid
import shutil
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from app.database import SessionLocal
from app.models import Job, JobStatus, Prediction
from app.workers.tasks import process_batch_images
from app.workers.celery_app import celery_app # Import to ensure config load

def test_batch_worker():
    print("--- Starting Celery BATCH Worker Integration Test ---")
    
    # Setup directories
    test_dir = "temp_batch_test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    image_paths = []
    num_images = 5
    
    # 1. Create Dummy Images
    print(f"1. Creating {num_images} dummy images...")
    try:
        for i in range(num_images):
            img_path = os.path.join(test_dir, f"test_img_{i}.png")
            # Vary color slightly to ensure unique hashes if needed, though simpler is fine
            color = (i * 50 % 255, 100, 100) 
            img = Image.new('RGB', (224, 224), color=color)
            img.save(img_path)
            image_paths.append(os.path.abspath(img_path))
        print(f"   Created {len(image_paths)} images in {test_dir}")
    except Exception as e:
        print(f"   Error creating images: {e}")
        return

    # 2. Setup DB Data
    print("2. Creating test BATCH job entry in Database...")
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        job = Job(id=job_id, status=JobStatus.PENDING, total_images=num_images)
        db.add(job)
        db.commit()
        print(f"   Created Job ID: {job_id}")
    except Exception as e:
        print(f"   Error creating job: {e}")
        return
    finally:
        db.close()

    # 3. Dispatch Task
    print("3. Dispatching BATCH task to Celery...")
    task_dispatched = False
    for attempt in range(5):
        try:
            task = process_batch_images.delay(image_paths, job_id)
            print(f"   Task dispatched! Task ID: {task.id}")
            task_dispatched = True
            break
        except Exception as e:
            print(f"   [Attempt {attempt+1}/5] Error dispatching task: {e}")
            time.sleep(2)
            
    if not task_dispatched:
        print("   ❌ Failed to dispatch task.")
        return

    # 4. Monitor
    print("4. Polling Database for Job completion...")
    max_retries = 15 # Longer timeout for batch
    success = False
    
    for i in range(max_retries):
        db = SessionLocal()
        refreshed_job = db.query(Job).filter(Job.id == job_id).first()
        status = refreshed_job.status
        processed = refreshed_job.processed_images
        
        print(f"   [{i+1}/{max_retries}] Job Status: {status} | Processed: {processed}/{num_images}")
        
        if status == JobStatus.COMPLETED:
            print("\n✅ SUCCESS: Job marked as COMPLETED!")
            
            # Verify predictions count
            pred_count = db.query(Prediction).filter(Prediction.job_id == job_id).count()
            print(f"   Predictions found in DB: {pred_count}")
            
            if pred_count == num_images:
                 print("   ✅ Prediction count matches total images.")
                 success = True
            else:
                 print(f"   ❌ Prediction count mismatch! Expected {num_images}, got {pred_count}")
            
            db.close()
            break
        elif status == JobStatus.FAILED:
            print("\n❌ FAILURE: Job marked as FAILED.")
            db.close()
            break
            
        db.close()
        time.sleep(1)

    if not success:
        print("\n⚠️ TIMEOUT: Job did not complete in time.")

    # 5. Cleanup
    print("5. Cleaning up...")
    try:
        shutil.rmtree(test_dir)
        print("   Removed test images directory.")
    except Exception as e:
        print(f"   Error cleanup: {e}")

if __name__ == "__main__":
    test_batch_worker()
