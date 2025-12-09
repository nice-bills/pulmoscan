import sys
import os
import time
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path so we can import app modules
sys.path.append(os.getcwd())

from app.database import SessionLocal, Base, engine
from app.models import Job, JobStatus
from app.workers.tasks import process_single_image
from app.workers.celery_app import celery_app  # Ensure app is loaded/configured
from PIL import Image

def test_worker():
    print("--- Starting Celery Worker Integration Test ---")
    
    # 1. Setup DB Data
    print("1. Creating test job entry in Database...")
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        job = Job(id=job_id, status=JobStatus.PENDING, total_images=1)
        db.add(job)
        db.commit()
        print(f"   Created Job ID: {job_id}")
    except Exception as e:
        print(f"   Error creating job: {e}")
        return
    finally:
        db.close()
    
    # 2. Create Dummy Image
    print("2. Creating dummy image file...")
    img_path = "temp_celery_test.png"
    try:
        img = Image.new('RGB', (224, 224), color='blue')
        img.save(img_path)
        print(f"   Created {img_path}")
    except Exception as e:
        print(f"   Error creating image: {e}")
        return

    # 3. Trigger Task
    # 3. Dispatching task to Celery...
    print("3. Dispatching task to Celery...")
    task_dispatched = False
    for attempt in range(5):
        try:
            # We assume Redis is running and Worker is listening
            task = process_single_image.delay(os.path.abspath(img_path), job_id)
            print(f"   Task dispatched! Task ID: {task.id}")
            task_dispatched = True
            break
        except Exception as e:
            print(f"   [Attempt {attempt+1}/5] Error dispatching task: {e}")
            print("   Retrying in 2 seconds...")
            time.sleep(2)
    
    if not task_dispatched:
        print("   ❌ Failed to dispatch task after 5 attempts.")
        return

    # 4. Monitor
    print("4. Polling Database for Job completion...")
    max_retries = 10
    success = False
    
    for i in range(max_retries):
        db = SessionLocal()
        refreshed_job = db.query(Job).filter(Job.id == job_id).first()
        status = refreshed_job.status
        db.close()
        
        print(f"   [{i+1}/{max_retries}] Job Status: {status}")
        
        if status == JobStatus.COMPLETED:
            print("\n✅ SUCCESS: Job marked as COMPLETED by worker!")
            success = True
            break
        elif status == JobStatus.FAILED:
            print("\n❌ FAILURE: Job marked as FAILED.")
            break
            
        time.sleep(1)

    if not success:
        print("\n⚠️ TIMEOUT: Job did not complete in time. Is the worker running?")

    # 5. Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
        print("   Cleaned up temporary image.")

if __name__ == "__main__":
    test_worker()
