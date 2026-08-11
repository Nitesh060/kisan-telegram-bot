"""
Image preprocessing for ML model
"""

import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Standard image size for model
MODEL_INPUT_SIZE = (224, 224)


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load image from file path."""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
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
        # Check file exists
        if not os.path.exists(image_path):
            return False, "Image file not found"
        
        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"Image too large ({file_size_mb:.1f} MB, max {max_size_mb} MB)"
        
        # Try to open image
        image = Image.open(image_path)
        
        # Check image dimensions (not too small)
        width, height = image.size
        if width < 64 or height < 64:
            return False, "Image too small (minimum 64x64 pixels)"
        
        # Check image format
        if image.format not in ["JPEG", "PNG", "GIF", "BMP", "WEBP"]:
            return False, f"Unsupported image format: {image.format}"
        
        return True, "valid"
    
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Invalid image: {str(e)}"


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Preprocess image for model inference.
    
    Steps:
    1. Load image
    2. Resize to model input size
    3. Normalize pixel values
    4. Add batch dimension
    
    Returns:
        Preprocessed image array or None if error
    """
    try:
        # Load image
        image = load_image(image_path)
        if image is None:
            return None
        
        # Resize
        image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize (MobileNetV3 expects values in [0, 1])
        image_array = image_array / 255.0
        
        # Add batch dimension (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None


def preprocess_image_imagenet(image_path: str) -> Optional[np.ndarray]:
    """
    Preprocess image using ImageNet normalization.
    
    This is useful if model was trained with ImageNet normalization.
    Uses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    try:
        image = load_image(image_path)
        if image is None:
            return None
        
        # Resize
        image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # ImageNet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        image_array = (image_array - imagenet_mean) / imagenet_std
        
        # Add batch dimension
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
        
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image
    
    except Exception as e:
        logger.error(f"Image resize error: {e}")
        return None
