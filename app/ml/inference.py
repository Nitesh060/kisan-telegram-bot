"""
ML model inference pipeline
"""

import logging
import json
import numpy as np
from typing import Optional, Dict, Tuple
import os

logger = logging.getLogger(__name__)

_tensorflow = None
_keras = None


def _ensure_tf():
    """Lazy load TensorFlow/Keras."""
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
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self._load_model()

    def _load_model(self):
        """Load Keras model with compatibility for Keras 3 .keras files."""
        try:
            tf, keras = _ensure_tf()
            if tf is None:
                return

            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return

            # compile=False avoids restoring optimizer/loss state that is not
            # needed for inference and improves compatibility across versions.
            self.model = keras.models.load_model(self.model_path, compile=False)
            self.num_classes = int(self.model.output_shape[-1])

            logger.info(f"✅ Model loaded: {self.model_path}")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output classes: {self.num_classes}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            self.model = None

    def load_class_names(self, class_names_path: str = "models/class_names.json") -> bool:
        """Load class names from either a JSON list or dictionary."""
        try:
            if not os.path.exists(class_names_path):
                logger.warning(f"Class names file not found: {class_names_path}")
                return False

            with open(class_names_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                self.class_names = [str(x) for x in data]
            elif isinstance(data, dict):
                # Supports {"0": "Class A", "1": "Class B"}
                try:
                    self.class_names = [str(data[str(i)]) for i in range(len(data))]
                except (KeyError, TypeError):
                    # Also support {"0": ..., "2": ...} / numeric keys safely.
                    ordered = sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
                    self.class_names = [str(v) for _, v in ordered]
            else:
                raise ValueError("class_names.json must contain a JSON list or object")

            logger.info(f"✅ Loaded {len(self.class_names)} class names")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load class names: {e}")
            self.class_names = []
            return False

    def predict(self, image_array: np.ndarray) -> Optional[Dict]:
        """Run inference on a preprocessed image."""
        if self.model is None:
            logger.error("Model not loaded - cannot run inference")
            return None

        try:
            predictions = self.model.predict(image_array, verbose=0)
            prediction = predictions[0]

            top_class_idx = int(np.argmax(prediction))
            top_confidence = float(prediction[top_class_idx])

            if self.class_names and top_class_idx < len(self.class_names):
                class_name = self.class_names[top_class_idx]
            else:
                class_name = f"Class_{top_class_idx}"

            crop, disease = self._parse_class_name(class_name)

            top_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = [
                {
                    "class": self.class_names[int(idx)] if self.class_names and int(idx) < len(self.class_names) else f"Class_{int(idx)}",
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
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            return None

    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Parse class name into crop and disease."""
        try:
            name = class_name.replace("_", " ").strip()
            parts = name.split()

            if not parts:
                return "Unknown", "Unknown"
            if len(parts) == 1:
                return "Crop", parts[0]
            if len(parts) == 2:
                return parts[0], parts[1]
            return parts[0], " ".join(parts[1:])

        except Exception as e:
            logger.error(f"Error parsing class name: {e}")
            return "Unknown", "Unknown"

    def is_model_ready(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Dict:
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
