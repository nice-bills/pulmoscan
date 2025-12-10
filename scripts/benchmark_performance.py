import sys
import os
import time
import uuid
import shutil
from PIL import Image
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from app.database import SessionLocal
from app.models import Job, JobStatus
from app.workers.tasks import process_single_image, process_batch_images
from app.workers.celery_app import celery_app # Import to ensure config load

def create_dummy_images(num_images: int, target_dir: str) -> List[str]:
    """Creates dummy image files for testing."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    image_paths = []
    for i in range(num_images):
        img_path = os.path.join(target_dir, f"benchmark_img_{i}.png")
        color = (i * 40 % 255, (i * 60 + 50) % 255, (i * 80 + 100) % 255) 
        img = Image.new('RGB', (224, 224), color=color)
        img.save(img_path)
        image_paths.append(os.path.abspath(img_path))
    return image_paths

def wait_for_job_completion(job_id: str, total_images: int, timeout: int = 300) -> JobStatus:
    """Polls the database for job completion."""
    db = SessionLocal()
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            # Expire all objects in the session to force a fresh reload from the DB
            db.expire_all() 
            refreshed_job = db.query(Job).filter(Job.id == job_id).first()
            if refreshed_job:
                if refreshed_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    return refreshed_job.status
                print(f"   Job {job_id} Status: {refreshed_job.status} | Processed: {refreshed_job.processed_images}/{total_images}")
            else:
                print(f"   Job {job_id} not found yet.")
            time.sleep(1)
    finally:
        db.close()
    return JobStatus.FAILED # Timeout

def run_single_image_benchmark(image_paths: List[str]):
    print("\n--- Running Single Image Processing Benchmark ---")
    start_overall_time = time.time()
    
    job_ids = []
    
    for path in image_paths:
        db = SessionLocal()
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, status=JobStatus.PENDING, total_images=1)
        db.add(job)
        db.commit()
        db.close()
        job_ids.append(job_id)

        try:
            print(f"   Dispatching single task for {os.path.basename(path)} (Job ID: {job_id})...")
            # Retry mechanism for dispatching task
            task_dispatched = False
            for attempt in range(3):
                try:
                    process_single_image.delay(path, job_id)
                    task_dispatched = True
                    break
                except Exception as e:
                    # print(f"   [Dispatch Attempt {attempt+1}/3] Error: {e}. Retrying...")
                    time.sleep(1)
            if not task_dispatched:
                print(f"   Failed to dispatch task for {os.path.basename(path)}")
        except Exception as e:
            print(f"Error dispatching task for {os.path.basename(path)}: {e}")
            
    # Wait for all individual jobs to complete
    for job_id in job_ids:
        status = wait_for_job_completion(job_id, 1)
        if status != JobStatus.COMPLETED:
            print(f"   Warning: Single job {job_id} did not complete successfully (Status: {status})")

    end_overall_time = time.time()
    total_duration = end_overall_time - start_overall_time
    print(f"\nSingle Image Benchmark Results for {len(image_paths)} images:")
    print(f"  Total time: {total_duration:.2f} seconds")
    print(f"  Avg time per image: {(total_duration / len(image_paths)):.2f} seconds")
    return total_duration

def run_batch_image_benchmark(image_paths: List[str]):
    print("\n--- Running Batch Image Processing Benchmark ---")
    start_overall_time = time.time()

    db = SessionLocal()
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, status=JobStatus.PENDING, total_images=len(image_paths))
    db.add(job)
    db.commit()
    db.close()

    try:
        print(f"   Dispatching batch task for {len(image_paths)} images (Job ID: {job_id})...")
        # Retry mechanism for dispatching task
        task_dispatched = False
        for attempt in range(3):
            try:
                process_batch_images.delay(image_paths, job_id)
                task_dispatched = True
                break
            except Exception as e:
                print(f"   [Dispatch Attempt {attempt+1}/3] Error: {e}. Retrying...")
                time.sleep(1)
        if not task_dispatched:
            print(f"   Failed to dispatch batch task for {job_id}")
            return None
    except Exception as e:
        print(f"Error dispatching batch task for {job_id}: {e}")
        return None

    status = wait_for_job_completion(job_id, len(image_paths))
    if status != JobStatus.COMPLETED:
        print(f"   Warning: Batch job {job_id} did not complete successfully (Status: {status})")

    end_overall_time = time.time()
    total_duration = end_overall_time - start_overall_time
    print(f"\nBatch Image Benchmark Results for {len(image_paths)} images:")
    print(f"  Total time: {total_duration:.2f} seconds")
    print(f"  Avg time per image: {(total_duration / len(image_paths)):.2f} seconds")
    return total_duration

if __name__ == "__main__":
    NUM_IMAGES_TO_BENCHMARK = 10
    BENCHMARK_DIR = "temp_benchmark_images"

    try:
        # Create images
        image_paths = create_dummy_images(NUM_IMAGES_TO_BENCHMARK, BENCHMARK_DIR)
        
        # Run benchmarks
        single_time = run_single_image_benchmark(image_paths)
        batch_time = run_batch_image_benchmark(image_paths)

        print("\n--- Overall Benchmark Summary ---")
        print(f"Number of images: {NUM_IMAGES_TO_BENCHMARK}")
        if single_time:
            print(f"Single processing total time: {single_time:.2f} seconds")
            print(f"Single processing avg time per image: {(single_time / NUM_IMAGES_TO_BENCHMARK):.2f} seconds")
        if batch_time:
            print(f"Batch processing total time: {batch_time:.2f} seconds")
            print(f"Batch processing avg time per image: {(batch_time / NUM_IMAGES_TO_BENCHMARK):.2f} seconds")
        
        if single_time and batch_time:
            if batch_time < single_time:
                speedup = single_time / batch_time
                print(f"\nBatch processing was {speedup:.2f}x faster than single processing!")
            else:
                print("\nSingle processing was faster or equal to batch processing.")

    finally:
        # Cleanup
        if os.path.exists(BENCHMARK_DIR):
            shutil.rmtree(BENCHMARK_DIR)
            print(f"\nCleaned up {BENCHMARK_DIR}.")
