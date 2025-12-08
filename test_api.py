
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_single_classification():
    print("\nTesting Single Image Classification...")
    # Create a dummy image for testing
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v1/jobs/classify", files=files)
        response.raise_for_status()
        data = response.json()
        duration = (time.time() - start_time) * 1000
        
        print(f"✅ Prediction: {data['predicted_class']} (Conf: {data['confidence']:.2f})")
        print(f"✅ Response time: {duration:.2f}ms")
        return True
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        print(response.text if 'response' in locals() else "")
        return False

if __name__ == "__main__":
    print(f"Running tests against {BASE_URL}")
    
    health_ok = test_health()
    class_ok = test_single_classification()
    
    if health_ok and class_ok:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
