import requests
import os
import zipfile
import shutil
import time
from PIL import Image
import pandas as pd
import io
from tqdm import tqdm
import uuid

BASE_URL = "http://localhost:8000"

def create_dummy_images_and_zip(num_images: int, temp_dir: str) -> str:
    if os.path.exists(temp_dir):
        # robust cleanup
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass # Ignore if locked, we'll just overwrite or fail later if crucial
            
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []
    for i in range(num_images):
        img_path = os.path.join(temp_dir, f"csv_test_img_{i}.png")
        img = Image.new('RGB', (224, 224), color='green')
        img.save(img_path)
        image_paths.append(img_path)

    zip_filename = os.path.join(temp_dir, "csv_test.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        for img_path in image_paths:
            zf.write(img_path, os.path.basename(img_path))
    return zip_filename

def test_csv_export():
    print("--- Testing CSV Export ---")
    # Use unique dir to avoid lock issues
    temp_dir = f"temp_csv_test_{uuid.uuid4().hex[:8]}"
    
    try:
        zip_path = create_dummy_images_and_zip(3, temp_dir)
        
        # 1. Start Job
        print("1. Starting batch job...")
        with open(zip_path, "rb") as f:
            files = {"file": ("csv_test.zip", f, "application/zip")}
            resp = requests.post(f"{BASE_URL}/api/v1/jobs/batch", files=files)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
        
        print(f"   Job ID: {job_id}")
        
        # 2. Wait for Completion
        print("2. Waiting for job completion...")
        for _ in tqdm(range(30), desc="Polling Status"):
            status_resp = requests.get(f"{BASE_URL}/api/v1/jobs/{job_id}")
            if status_resp.json()["status"] == "completed":
                break
            time.sleep(1)
        
        # 3. Download CSV
        print("3. Downloading CSV...")
        export_url = f"{BASE_URL}/api/v1/jobs/{job_id}/results/download"
        csv_resp = requests.get(export_url)
        
        if csv_resp.status_code == 200:
            print("    CSV Downloaded successfully.")
            # Verify content
            csv_content = csv_resp.text
            print(f"   Content Length: {len(csv_content)} bytes")
            
            # Parse with pandas to verify structure
            try:
                df = pd.read_csv(io.StringIO(csv_content))
                print("    CSV Parsed successfully with Pandas.")
                print("   Columns:", df.columns.tolist())
                print(f"   Rows: {len(df)}")
                
                if len(df) == 3 and "predicted_class" in df.columns:
                    print("    Data validation passed.")
                else:
                    print("    Data validation failed.")
            except Exception as e:
                print(f"    CSV Parsing failed: {e}")
                print(csv_content)
        else:
            print(f"    Download failed: {csv_resp.status_code}")
            print(csv_resp.text)

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                print(f"   Warning: Cleanup failed ({e})")

if __name__ == "__main__":
    test_csv_export()
