import requests
import os
import zipfile
import shutil
import time
import uuid
from PIL import Image

BASE_URL = "http://localhost:8000"

def create_dummy_images_and_zip(num_images: int, temp_dir: str) -> str:
    """Creates dummy images and zips them into a file."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    image_paths = []
    for i in range(num_images):
        img_path = os.path.join(temp_dir, f"dummy_image_{i}.png")
        color = (i * 20 % 255, (i * 30 + 50) % 255, (i * 40 + 100) % 255)
        img = Image.new('RGB', (224, 224), color=color)
        img.save(img_path)
        image_paths.append(img_path)

    zip_filename = os.path.join(temp_dir, "test_images.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        for img_path in image_paths:
            zf.write(img_path, os.path.basename(img_path))
    print(f"Created ZIP file: {zip_filename} with {num_images} images.")
    return zip_filename

def test_zip_upload():
    print("--- Starting ZIP Upload Test ---")
    num_images = 10 # Increased for better progress tracking
    temp_test_dir = "temp_zip_test"

    zip_file_path = create_dummy_images_and_zip(num_images, temp_test_dir)

    # 1. Upload ZIP file
    print(f"1. Uploading {zip_file_path} to {BASE_URL}/api/v1/jobs/batch")
    try:
        with open(zip_file_path, "rb") as f:
            files = {"file": (os.path.basename(zip_file_path), f, "application/zip")}
            response = requests.post(f"{BASE_URL}/api/v1/jobs/batch", files=files)
            response.raise_for_status() # Raise an exception for bad status codes
        
        response_data = response.json()
        job_id = response_data["job_id"]
        images_queued = response_data["images_queued"]
        print(f"   Upload successful. Job ID: {job_id}, Images Queued: {images_queued}")
        assert images_queued == num_images
    except requests.exceptions.ConnectionError:
        print(f"   Error: Could not connect to API at {BASE_URL}. Is the FastAPI server running?")
        shutil.rmtree(temp_test_dir)
        return
    except Exception as e:
        print(f"   Error uploading ZIP: {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'No response'}")
        shutil.rmtree(temp_test_dir)
        return

    # 2. Poll Job Status
    print(f"2. Polling job status for Job ID: {job_id}")
    status_url = f"{BASE_URL}/api/v1/jobs/{job_id}"
    job_completed = False
    for i in range(30): # Poll for up to 30 seconds
        try:
            response = requests.get(status_url)
            response.raise_for_status()
            job_status = response.json()
            
            print(f"   [{i+1}/30] Job Status: {job_status['status']} | Processed: {job_status['processed_images']}/{job_status['total_images']}")
            
            if job_status["status"] == "completed":
                job_completed = True
                print("   Job completed!")
                assert job_status["processed_images"] == num_images
                break
            elif job_status["status"] == "failed":
                print("   Job failed!")
                break
        except Exception as e:
            print(f"   Error polling job status: {e}")
            break
        time.sleep(1)

    if not job_completed:
        print("3. Job did not complete within the timeout.")
        
    # 3. Cleanup
    print("4. Cleaning up temporary files...")
    if os.path.exists(temp_test_dir):
        shutil.rmtree(temp_test_dir)
    print("   Cleanup complete.")

    if job_completed:
        print("\n✅ ZIP Upload Test PASSED!")
    else:
        print("\n❌ ZIP Upload Test FAILED!")

if __name__ == "__main__":
    test_zip_upload()
