"""
ML model inference pipeline
"""

import logging
import json
import numpy as np
from typing import Optional, Dict, Tuple
import os

logger = logging.getLogger(__name__)

# Lazy imports for ML (only when needed)
_tensorflow = None
_keras = None


def _ensure_tf():
    """Lazy load TensorFlow."""
    global _tensorflow, _keras
    if _tensorflow is None:
        try:
            import tensorflow as tf
            _tensorflow = tf
            _keras = tf.keras
        except ImportError:
            logger.warning("TensorFlow not installed - ML inference will be unavailable")
    return _tensorflow, _keras


class ModelInference:
    """ML model inference handler."""
    
    def __init__(self, model_path: str = "models/crop_disease_model.keras"):
        """Initialize model inference engine."""
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self._load_model()
    
    def _load_model(self):
        """Load model from disk."""
        try:
            tf, keras = _ensure_tf()
            if tf is None:
                logger.warning("TensorFlow not available - cannot load model")
                return
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return
            
            self.model = keras.models.load_model(self.model_path)
            self.num_classes = self.model.output_shape[-1]
            
            logger.info(f"✅ Model loaded: {self.model_path}")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output classes: {self.num_classes}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def load_class_names(self, class_names_path: str = "models/class_names.json") -> bool:
        """Load class names mapping from JSON file."""
        try:
            if not os.path.exists(class_names_path):
                logger.warning(f"Class names file not found: {class_names_path}")
                return False
            
            with open(class_names_path, "r") as f:
                data = json.load(f)
            
            # Expect format: {"0": "Tomato_Early_Blight", "1": "Potato_Late_Blight", ...}
            self.class_names = [data[str(i)] for i in range(len(data))]
            
            logger.info(f"✅ Loaded {len(self.class_names)} class names")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load class names: {e}")
            return False
    
    def predict(self, image_array: np.ndarray) -> Optional[Dict]:
        """
        Run inference on preprocessed image.
        
        Args:
            image_array: Preprocessed image (batch_size, 224, 224, 3)
        
        Returns:
            Dict with prediction results or None if error
        """
        if self.model is None:
            logger.error("Model not loaded - cannot run inference")
            return None
        
        try:
            # Run prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            # predictions shape: (batch_size, num_classes)
            # Get first batch
            prediction = predictions[0]
            
            # Get top prediction
            top_class_idx = np.argmax(prediction)
            top_confidence = float(prediction[top_class_idx])
            
            # Get class name
            if self.class_names and top_class_idx < len(self.class_names):
                class_name = self.class_names[top_class_idx]
            else:
                class_name = f"Class_{top_class_idx}"
            
            # Parse class name to extract crop and disease
            # Expect format like "Tomato_Early_Blight" or "Tomato Early Blight"
            crop, disease = self._parse_class_name(class_name)
            
            # Get top 3 predictions for debugging
            top_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = [
                {
                    "class": self.class_names[idx] if self.class_names and idx < len(self.class_names) else f"Class_{idx}",
                    "confidence": float(prediction[idx]),
                }
                for idx in top_indices
            ]
            
            result = {
                "crop": crop,
                "disease": disease,
                "class_name": class_name,
                "confidence": top_confidence,
                "confidence_percent": round(top_confidence * 100, 2),
                "top_predictions": top_predictions,
                "success": True,
            }
            
            logger.info(f"✅ Prediction: {crop} - {disease} ({top_confidence:.1%})")
            return result
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """
        Parse class name to extract crop and disease.
        
        Expects format like:
        - "Tomato_Early_Blight"
        - "Potato Early Blight"
        - "Rice Brown Spot"
        
        Returns:
            Tuple of (crop, disease)
        """
        try:
            # Replace underscores with spaces
            name = class_name.replace("_", " ")
            
            # Try common patterns
            parts = name.split()
            
            if len(parts) == 0:
                return "Unknown", "Unknown"
            elif len(parts) == 1:
                # Single word - assume it's the disease
                return "Crop", parts[0]
            elif len(parts) == 2:
                # Two words - assume crop + disease
                return parts[0], parts[1]
            else:
                # Multiple words - first is crop, rest is disease
                crop = parts[0]
                disease = " ".join(parts[1:])
                return crop, disease
        
        except Exception as e:
            logger.error(f"Error parsing class name: {e}")
            return "Unknown", "Unknown"
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            return {"ready": False, "error": "Model not loaded"}
        
        return {
            "ready": True,
            "model_path": self.model_path,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "num_classes": self.num_classes,
            "num_class_names": len(self.class_names),
            "class_names_loaded": len(self.class_names) > 0,
        }
