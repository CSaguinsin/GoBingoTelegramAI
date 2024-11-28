from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.size[0] < 100 or img.size[1] < 100:
                return False, "Image is too small"
            if os.path.getsize(image_path) < 1024:
                return False, "Image file is too small"
            return True, "Image is valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}" 