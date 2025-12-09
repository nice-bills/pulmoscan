import json
import redis
from typing import Optional, Any
from app.config import settings

class RedisCache:
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.default_ttl = 604800  # 7 days in seconds

    def get_prediction(self, image_hash: str) -> Optional[dict]:
        """
        Get prediction from cache by image hash.
        Returns deserialized JSON or None if not found.
        """
        if not settings.CACHE_ENABLED:
            return None
            
        key = f"prediction:{image_hash}"
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except redis.RedisError:
            # Fail silently on cache errors to avoid breaking the app
            pass
        return None

    def set_prediction(self, image_hash: str, prediction: dict, ttl: int = None) -> None:
        """
        Store prediction in cache.
        """
        if not settings.CACHE_ENABLED:
            return

        key = f"prediction:{image_hash}"
        try:
            self.client.set(
                key, 
                json.dumps(prediction), 
                ex=ttl or self.default_ttl
            )
        except redis.RedisError:
            pass

    def get_stats(self) -> dict:
        """Get basic cache statistics."""
        try:
            info = self.client.info()
            return {
                "total_keys": self.client.dbsize(),
                "used_memory": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0)
            }
        except redis.RedisError:
            return {"status": "error"}

# Singleton instance
cache = RedisCache()
