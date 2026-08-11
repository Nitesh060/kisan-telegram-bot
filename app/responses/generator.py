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
            disease_info = DiseaseRepository.get_disease(session, crop, disease)
            
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
            disease_info = DiseaseRepository.get_disease(session, crop, disease)
            
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
            disease_info = DiseaseRepository.get_disease(session, crop, disease)
            
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
            disease_info = DiseaseRepository.get_disease(session, crop, disease)
            
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
