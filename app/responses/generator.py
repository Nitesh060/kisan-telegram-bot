"""
Response generation engine - deterministic, no LLM
"""

import logging
from typing import Optional, Dict
from app.responses.templates import (
    DISEASE_PREDICTION_TEMPLATE,
    CONFIDENCE_TEMPLATES,
    ERROR_MESSAGES,
)
from app.database.repository import DiseaseRepository
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses from templates and database data."""

    @staticmethod
    def _get_disease_info(session: Session, crop: str, disease: str):
        """Find disease information using tolerant name matching.

        The ML model may return labels such as ``late blight`` while the
        database may store ``Late Blight``, ``late-blight`` or include the
        crop name. Try safe normalized variants without changing the DB.
        """
        crop_clean = (crop or "").strip()
        disease_clean = (disease or "").strip()

        candidates = []

        def add(value: str):
            value = (value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        # Original value first.
        add(disease_clean)

        # Handle labels like "Tomato - late blight" if they reach this layer.
        if " - " in disease_clean:
            left, right = disease_clean.split(" - ", 1)
            if not crop_clean:
                crop_clean = left.strip()
            add(right)
        elif crop_clean and disease_clean.lower().startswith(crop_clean.lower()):
            remainder = disease_clean[len(crop_clean):].strip(" -:_")
            add(remainder)

        # Common formatting variants.
        for value in list(candidates):
            add(value.lower())
            add(value.title())
            add(value.replace("_", " "))
            add(value.replace("-", " "))
            add(value.replace("_", "-").replace(" ", "-"))

        # Also try combined crop+disease forms for databases that store the
        # full label in one disease column.
        disease_variants = list(candidates)
        if crop_clean and disease_clean:
            for value in disease_variants:
                add(f"{crop_clean} - {value}")
                add(f"{crop_clean} {value}")

        for candidate in candidates:
            try:
                result = DiseaseRepository.get_disease(session, crop_clean, candidate)
                if result:
                    logger.info(
                        "Disease DB match: %s - %s (requested: %s - %s)",
                        crop_clean, candidate, crop, disease
                    )
                    return result
            except Exception as exc:
                logger.debug("Disease lookup failed for '%s': %s", candidate, exc)

        return None

    
    @staticmethod
    def generate_disease_response(
        session: Session,
        crop: str,
        disease: str,
        confidence: float,
        language: str = "en",
    ) -> Optional[str]:
        """
        Generate disease identification response.
        
        Args:
            session: Database session
            crop: Crop name
            disease: Disease name
            confidence: Confidence score (0-1)
            language: Response language
        
        Returns:
            Formatted response string
        """
        try:
            # Get disease info from database
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            
            if not disease_info:
                logger.warning(f"Disease not found in DB: {crop} - {disease}")
                return ResponseGenerator.get_error_message(
                    "no_disease_found",
                    language,
                )
            
            # Get confidence level
            confidence_percent = round(confidence * 100, 1)
            confidence_level = ResponseGenerator._get_confidence_level(confidence)
            confidence_text = CONFIDENCE_TEMPLATES.get(language, {}).get(
                confidence_level,
                f"Confidence: {confidence_percent}%"
            )
            
            # Build prevention section
            prevention_section = ""
            if disease_info.prevention:
                if language == "hi":
                    prevention_section = f"🛡️ *रोकथाम:*\n{disease_info.prevention}\n\n"
                else:
                    prevention_section = f"🛡️ *Prevention:*\n{disease_info.prevention}\n\n"
            
            # Build treatment section
            treatment_section = ""
            if disease_info.treatment:
                if language == "hi":
                    treatment_section = f"💊 *उपचार:*\n{disease_info.treatment}\n\n"
                else:
                    treatment_section = f"💊 *Treatment:*\n{disease_info.treatment}\n\n"
            
            # Get template
            template = DISEASE_PREDICTION_TEMPLATE.get(language, DISEASE_PREDICTION_TEMPLATE["en"])
            
            # Format response
            response = template.format(
                crop=crop,
                disease=disease,
                confidence_percent=confidence_percent,
                confidence_level_text=confidence_text,
                symptoms=disease_info.symptoms or "Detailed symptoms not available",
                management=disease_info.management or "Management info not available",
                prevention_section=prevention_section,
                treatment_section=treatment_section,
                severity=disease_info.severity or "Unknown",
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating disease response: {e}")
            return None
    
    @staticmethod
    def generate_info_response(
        session: Session,
        crop: str,
        disease: str,
        language: str = "en",
    ) -> Optional[str]:
        """Generate disease information response."""
        return ResponseGenerator.generate_disease_response(
            session,
            crop,
            disease,
            confidence=1.0,  # High confidence for database queries
            language=language,
        )
    
    @staticmethod
    def generate_symptom_response(
        session: Session,
        crop: str,
        disease: str,
        language: str = "en",
    ) -> Optional[str]:
        """Generate symptom information response."""
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            
            if language == "hi":
                response = f"🌾 *{crop} - {disease}*\n\n"
                response += f"🔍 *लक्षण:*\n{disease_info.symptoms}\n"
            else:
                response = f"🌾 *{crop} - {disease}*\n\n"
                response += f"🔍 *Symptoms:*\n{disease_info.symptoms}\n"
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating symptom response: {e}")
            return None
    
    @staticmethod
    def generate_management_response(
        session: Session,
        crop: str,
        disease: str,
        language: str = "en",
    ) -> Optional[str]:
        """Generate management guidance response."""
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            
            if language == "hi":
                response = f"🌾 *{crop} - {disease}*\n\n"
                response += f"🛠️ *प्रबंधन:*\n{disease_info.management}\n"
            else:
                response = f"🌾 *{crop} - {disease}*\n\n"
                response += f"🛠️ *Management:*\n{disease_info.management}\n"
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating management response: {e}")
            return None
    
    @staticmethod
    def generate_prevention_response(
        session: Session,
        crop: str,
        disease: str,
        language: str = "en",
    ) -> Optional[str]:
        """Generate prevention guidance response."""
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            
            if language == "hi":
                response = f"🌾 *{crop} - {disease}*\n\n"
                if disease_info.prevention:
                    response += f"🛡️ *रोकथाम:*\n{disease_info.prevention}\n"
                else:
                    response += "कोई विशिष्ट रोकथाम जानकारी उपलब्ध नहीं है।\n"
            else:
                response = f"🌾 *{crop} - {disease}*\n\n"
                if disease_info.prevention:
                    response += f"🛡️ *Prevention:*\n{disease_info.prevention}\n"
                else:
                    response += "No specific prevention information available.\n"
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating prevention response: {e}")
            return None
    
    @staticmethod
    def get_error_message(error_key: str, language: str = "en") -> str:
        """Get error message."""
        messages = ERROR_MESSAGES.get(language, ERROR_MESSAGES["en"])
        return messages.get(error_key, "An error occurred. Please try again.")
    
    @staticmethod
    def _get_confidence_level(confidence: float) -> str:
        """Determine confidence level."""
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.60:
            return "medium"
        else:
            return "low"
