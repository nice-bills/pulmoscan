import redis
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.config import settings

def test_redis_connection():
    print(f"Attempting to connect to Redis at: {settings.REDIS_URL}")
    try:
        # Use from_url to parse the REDIS_URL from settings
        client = redis.from_url(settings.REDIS_URL)
        
        # Attempt to ping the server
        client.ping()
        print("Successfully connected to Redis and received PONG!")
        return True
    except redis.exceptions.ConnectionError as e:
        print(f"Redis Connection Error: {e}")
        print("Please ensure your Redis Docker container is running and accessible on localhost:6379.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection()
