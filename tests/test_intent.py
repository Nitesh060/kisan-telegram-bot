"""
Tests for intent detection
"""

import pytest
from app.nlp.intent import IntentDetector


class TestIntentDetector:
    """Test intent detection."""
    
    def test_diagnosis_intent_english(self):
        """Test English diagnosis intent."""
        text = "What disease is this?"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.DIAGNOSIS
    
    def test_diagnosis_intent_hindi(self):
        """Test Hindi diagnosis intent."""
        text = "यह कौन सा रोग है?"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.DIAGNOSIS
    
    def test_management_intent_english(self):
        """Test English management intent."""
        text = "What should I do to treat this?"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.MANAGEMENT
    
    def test_management_intent_hindi(self):
        """Test Hindi management intent."""
        text = "इसका क्या करें?"
        intent = IntentDetector.detect(text)
        assert intent in [IntentDetector.MANAGEMENT, IntentDetector.DIAGNOSIS]
    
    def test_symptoms_intent(self):
        """Test symptoms intent."""
        text = "What are the symptoms?"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.SYMPTOMS
    
    def test_prevention_intent(self):
        """Test prevention intent."""
        text = "How can I prevent this?"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.PREVENTION
    
    def test_unknown_intent(self):
        """Test unknown intent."""
        text = "Tell me a joke"
        intent = IntentDetector.detect(text)
        assert intent == IntentDetector.UNKNOWN
    
    def test_empty_text(self):
        """Test empty text."""
        intent = IntentDetector.detect("")
        assert intent == IntentDetector.UNKNOWN
    
    def test_crop_detection_english(self):
        """Test crop detection from English text."""
        crop = IntentDetector.detect_crop_from_keywords("my tomato plants")
        assert crop == "Tomato"
    
    def test_crop_detection_hindi(self):
        """Test crop detection from Hindi text."""
        crop = IntentDetector.detect_crop_from_keywords("गेहूं में समस्या")
        assert crop == "Wheat"
    
    def test_disease_keywords_extraction(self):
        """Test disease keyword extraction."""
        keywords = IntentDetector.extract_disease_keywords("brown spots on tomato leaf")
        assert "spot" in keywords or "spots" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
