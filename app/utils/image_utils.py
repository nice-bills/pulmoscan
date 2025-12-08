
from PIL import Image
import io

def validate_image(image_bytes: bytes) -> bool:
    """Check if bytes are a valid image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False
