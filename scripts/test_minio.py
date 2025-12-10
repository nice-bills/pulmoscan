import sys
import os
import io

# Add project root to path
sys.path.append(os.getcwd())

from app.services.storage import storage

def test_minio_connection():
    print("--- Testing MinIO Connection ---")
    
    test_content = b"Hello MinIO! This is a test file."
    object_name = "test_file.txt"
    download_path = "downloaded_test_file.txt"

    # 1. Upload
    print(f"1. Uploading '{object_name}'...")
    file_obj = io.BytesIO(test_content)
    result = storage.upload_file(file_obj, object_name)
    
    if result:
        print("    Upload successful.")
    else:
        print("    Upload failed.")
        return

    # 2. Generate Presigned URL
    print("2. Generating presigned URL...")
    url = storage.generate_presigned_url(object_name)
    if url:
        print(f"    URL generated: {url}")
    else:
        print("     URL generation failed.")

    # 3. Download
    print(f"3. Downloading back to '{download_path}'...")
    success = storage.download_file(object_name, download_path)
    
    if success:
        with open(download_path, "rb") as f:
            content = f.read()
        if content == test_content:
            print("    Download successful and content matches.")
        else:
            print("   Content mismatch!")
    else:
        print("       Download failed.")

    # Cleanup
    if os.path.exists(download_path):
        os.remove(download_path)

if __name__ == "__main__":
    test_minio_connection()
