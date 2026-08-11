"""
Rule-based intent detection for farmer questions
No LLM or complex NLP - just keyword matching
"""

import logging
from typing import Optional, List
from app.nlp.keywords import INTENT_KEYWORDS

logger = logging.getLogger(__name__)


class IntentDetector:
    """Rule-based intent detection."""
    
    # Intent types
    DIAGNOSIS = "DIAGNOSIS"
    SYMPTOMS = "SYMPTOMS"
    CAUSE = "CAUSE"
    MANAGEMENT = "MANAGEMENT"
    TREATMENT = "TREATMENT"
    PREVENTION = "PREVENTION"
    SEVERITY = "SEVERITY"
    GENERAL_INFO = "GENERAL_INFO"
    UNKNOWN = "UNKNOWN"
    
    ALL_INTENTS = [
        DIAGNOSIS,
        SYMPTOMS,
        CAUSE,
        MANAGEMENT,
        TREATMENT,
        PREVENTION,
        SEVERITY,
        GENERAL_INFO,
        UNKNOWN,
    ]
    
    @staticmethod
    def detect(text: str) -> str:
        """
        Detect intent from text.
        
        Args:
            text: User question/input
        
        Returns:
            Intent type (string)
        """
        if not text:
            return IntentDetector.UNKNOWN
        
        # Normalize text
        text_lower = text.lower()
        
        # Check each intent's keywords
        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    logger.info(f"Detected intent: {intent} (keyword: {keyword})")
                    return intent
        
        logger.warning(f"Could not detect intent from: {text}")
        return IntentDetector.UNKNOWN
    
    @staticmethod
    def detect_crop_from_keywords(text: str) -> Optional[str]:
        """
        Try to detect crop name from text.
        
        Common crops:
        - Rice, Wheat, Maize, Cotton, Sugarcane
        - Tomato, Potato, Onion, Chilli
        - Mango, Apple, Citrus
        """
        CROPS = {
            "rice": ["rice", "chawal", "چاول"],
            "wheat": ["wheat", "gehu", "گیہو"],
            "maize": ["maize", "corn", "makka", "مکّہ"],
            "cotton": ["cotton", "kapas", "کپاس"],
            "sugarcane": ["sugarcane", "ganna", "گنا"],
            "tomato": ["tomato", "tamatar", "ٹماٹر"],
            "potato": ["potato", "alu", "آلو"],
            "onion": ["onion", "pyaz", "پیاز"],
            "chilli": ["chilli", "mirch", "مرچ"],
            "mango": ["mango", "aam", "آم"],
            "apple": ["apple", "seb", "سیب"],
            "citrus": ["citrus", "orange", "lemon", "santra", "نارنگی"],
        }
        
        text_lower = text.lower()
        
        for crop, keywords in CROPS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.info(f"Detected crop: {crop}")
                    return crop.title()
        
        return None
    
    @staticmethod
    def extract_disease_keywords(text: str) -> List[str]:
        """Extract potential disease keywords from text."""
        keywords = []
        
        # Common disease symptoms/names in English and Hindi
        DISEASE_KEYWORDS = [
            # Blight
            "blight", "blast",
            # Spot/spots
            "spot", "spots", "daag",
            # Rot/rotting
            "rot", "gali",
            # Mildew/powder
            "mildew", "powder", "chatri", "चत्री",
            # Rust
            "rust", "khaad",
            # Wilt/wilting
            "wilt", "sukhna", "सूखना",
            # Yellowing
            "yellow", "pila", "पीला",
            # Leaf issues
            "leaf", "patta", "पत्ता",
            # Stem issues
            "stem", "dand", "दंड",
            # Root issues
            "root", "jad", "जड",
            # Fruit issues
            "fruit", "phal", "फल",
        ]
        
        text_lower = text.lower()
        
        for keyword in DISEASE_KEYWORDS:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    @staticmethod
    def is_disease_question(text: str) -> bool:
        """Check if question is about disease."""
        intent = IntentDetector.detect(text)
        return intent in [
            IntentDetector.DIAGNOSIS,
            IntentDetector.SYMPTOMS,
            IntentDetector.CAUSE,
            IntentDetector.MANAGEMENT,
            IntentDetector.TREATMENT,
            IntentDetector.PREVENTION,
            IntentDetector.SEVERITY,
        ]
