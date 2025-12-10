import time
import os
import sys
import uuid

# Add project root to path
sys.path.append(os.getcwd())

from tqdm import tqdm
import onnxruntime as ort
import torch
import numpy as np
from app.services.cache import cache
from app.workers.tasks import process_batch_images
from app.models import Job, JobStatus
from app.database import SessionLocal

ONNX_PATH = "models/covid/mobilenetv3.onnx"

def test_model_load_time():
    print("\n[1/4] Testing Model Load Time...")
    start = time.time()
    sess = ort.InferenceSession(ONNX_PATH)
    duration = time.time() - start
    print(f"   Load Time: {duration:.4f}s")
    if duration < 3.0:
        print("    PASS (< 3s)")
    else:
        print("    FAIL (> 3s)")
    return sess

def test_single_inference_latency(sess):
    print("\n[2/4] Testing Single Image Inference Latency (ONNX)...")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    
    # Warmup
    for _ in range(50):
        sess.run(None, {input_name: dummy_input})
        
    # Measure
    latencies = []
    for _ in range(100):
        start = time.time()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"   Avg Latency: {avg_latency:.2f}ms")
    if avg_latency < 100:
        print("    PASS (< 100ms)")
    else:
        print("    FAIL (> 100ms)")

def test_cache_latency():
    print("\n[3/4] Testing Cache Latency...")
    key = f"bench_{uuid.uuid4().hex}"
    data = {"test": "data"}
    
    # Write
    cache.set_prediction(key, data)
    
    # Measure Read
    start = time.time()
    _ = cache.get_prediction(key)
    duration = (time.time() - start) * 1000
    
    print(f"   Cache Retrieval: {duration:.3f}ms")
    if duration < 5.0:
        print("    PASS (< 5ms)")
    else:
        print("    FAIL (> 5ms)")

def create_dummy_images(num_images: int, target_dir: str):
    from PIL import Image
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    paths = []
    for i in range(num_images):
        p = os.path.join(target_dir, f"perf_{i}.png")
        Image.new('RGB', (224, 224)).save(p)
        paths.append(os.path.abspath(p))
    return paths

def test_batch_throughput():
    print("\n[4/4] Testing Batch Throughput (Projected)...")
    # We will run a batch of 50 images directly via the worker logic 
    # to avoid API overhead for this pure throughput test.
    
    num_images = 50
    temp_dir = f"temp_perf_{uuid.uuid4().hex[:8]}"
    paths = create_dummy_images(num_images, temp_dir)
    
    # Create Job in DB
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, status=JobStatus.PENDING, total_images=num_images)
    db.add(job)
    db.commit()
    db.close()
    
    print(f"   Processing {num_images} images via Celery task (Synchronous call for timing)...")
    start = time.time()
    
    # Calling the task function directly (not .delay) to measure pure execution time 
    # of the worker logic, excluding Redis queue latency (which varies by environment).
    # This gives the "Processing Capacity" metric.
    process_batch_images(paths, job_id)
    
    duration = time.time() - start
    avg_per_image = duration / num_images
    projected_1000 = avg_per_image * 1000
    
    print(f"   Time for {num_images}: {duration:.2f}s")
    print(f"   Avg per image: {avg_per_image:.4f}s")
    print(f"   Projected 1000 images: {projected_1000:.2f}s ({projected_1000/60:.1f} min)")
    
    if projected_1000 < 120:
        print("    PASS (< 2 minutes)")
    else:
        print("    FAIL (> 2 minutes)")
        
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("STARTING PERFORMANCE VALIDATION")
    sess = test_model_load_time()
    test_single_inference_latency(sess)
    test_cache_latency()
    test_batch_throughput()
    print("\nDONE")
