"""
Image preprocessing for ML model.

The MobileNetV3Small model used by the training pipeline keeps its built-in
preprocessing enabled and therefore expects image pixels in the normal
[0, 255] range. This module only loads/converts/resizes images and does NOT
perform /255 normalization.
"""

import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE = (224, 224)


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load an image and convert it to RGB."""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        image = Image.open(image_path)

        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def validate_image(image_path: str, max_size_mb: int = 10) -> Tuple[bool, str]:
    """Validate image before processing."""
    try:
        if not os.path.exists(image_path):
            return False, "Image file not found"

        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"Image too large ({file_size_mb:.1f} MB, max {max_size_mb} MB)"

        image = Image.open(image_path)
        width, height = image.size
        if width < 64 or height < 64:
            return False, "Image too small (minimum 64x64 pixels)"

        if image.format not in ["JPEG", "PNG", "GIF", "BMP", "WEBP"]:
            return False, f"Unsupported image format: {image.format}"

        return True, "valid"

    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Invalid image: {str(e)}"


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Prepare an image for MobileNetV3 inference.

    Resize to 224x224 and add the batch dimension. Pixel values intentionally
    remain in the [0, 255] range because MobileNetV3 performs its own
    preprocessing inside the model.
    """
    try:
        image = load_image(image_path)
        if image is None:
            return None

        image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None


def preprocess_image_imagenet(image_path: str) -> Optional[np.ndarray]:
    """
    Optional ImageNet preprocessing helper for models that explicitly require
    ImageNet mean/std normalization. Not used by the MobileNetV3 pipeline.
    """
    try:
        image = load_image(image_path)
        if image is None:
            return None

        image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - imagenet_mean) / imagenet_std
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        logger.error(f"ImageNet preprocessing error: {e}")
        return None


def resize_image(image_path: str, size: Tuple[int, int] = MODEL_INPUT_SIZE) -> Optional[Image.Image]:
    """Resize image to specified size."""
    try:
        image = load_image(image_path)
        if image is None:
            return None

        return image.resize(size, Image.Resampling.LANCZOS)

    except Exception as e:
        logger.error(f"Image resize error: {e}")
        return None
