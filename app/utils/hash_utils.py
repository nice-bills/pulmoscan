
import hashlib

def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA-256 hash of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()
