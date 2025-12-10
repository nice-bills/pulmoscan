import sys
import os
import time

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from app.services.cache import cache

def test_cache():
    print("Testing Redis Cache...")
    try:
        # 1. Test Set
        data = {"class": "Test", "confidence": 0.99, "top_3_classes": []}
        cache.set_prediction("test_hash_123", data)
        print("Set Key: prediction:test_hash_123")
        
        # 2. Test Get
        time.sleep(0.1) # Ensure write prop
        result = cache.get_prediction("test_hash_123")
        if result and result["class"] == "Test":
             print(f"Get Key (Hit): {result}")
        else:
             print(f"Get Key Failed. Result: {result}")
             
        # 3. Test Miss
        miss = cache.get_prediction("non_existent_hash")
        if miss is None:
            print("Cache Miss handled correctly (None)")
        else:
            print(f"Cache Miss failed. Got: {miss}")
            
        stats = cache.get_stats()
        print(f"Cache Stats: {stats}")
            
    except Exception as e:
        print(f"Redis Connection Error: {e}")
        print("Ensure Docker is running: docker-compose up -d")

if __name__ == "__main__":
    test_cache()
