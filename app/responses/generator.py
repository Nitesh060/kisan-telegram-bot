"""
Response generation engine - deterministic, no LLM
"""

import logging
import re
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
    def _normalize(value: str) -> str:
        """Normalize labels so ML output and DB naming styles can match."""
        value = (value or "").strip().lower()
        value = value.replace("_", " ").replace("-", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _get_disease_info(session: Session, crop: str, disease: str):
        """Find disease information using exact and normalized matching."""
        crop_clean = (crop or "").strip()
        disease_clean = (disease or "").strip()

        # Handle accidental combined labels such as "Apple - black rot".
        combined_crop = crop_clean
        combined_disease = disease_clean
        if " - " in disease_clean:
            left, right = disease_clean.split(" - ", 1)
            if not combined_crop:
                combined_crop = left.strip()
            if ResponseGenerator._normalize(left) == ResponseGenerator._normalize(crop_clean):
                combined_disease = right.strip()

        # First: exact/tolerant repository lookup using common variants.
        candidates = []
        for value in [combined_disease, disease_clean]:
            if not value:
                continue
            variants = [
                value,
                value.lower(),
                value.title(),
                value.replace("_", " "),
                value.replace("-", " "),
                value.replace("_", "-").replace(" ", "-"),
            ]
            for candidate in variants:
                if candidate and candidate not in candidates:
                    candidates.append(candidate)

        for candidate in candidates:
            try:
                result = DiseaseRepository.get_disease(session, combined_crop, candidate)
                if result:
                    logger.info(
                        "Disease DB match: %s - %s (requested: %s - %s)",
                        combined_crop, candidate, crop, disease,
                    )
                    return result
            except Exception as exc:
                logger.debug("Disease lookup failed for '%s': %s", candidate, exc)

        # Second: normalized scan within the crop. This handles cases like:
        # DB: "Black Rot" vs model: "black rot" / "black_rot".
        try:
            crop_diseases = DiseaseRepository.get_diseases_by_crop(session, combined_crop)
            wanted = ResponseGenerator._normalize(combined_disease)
            wanted_combined = ResponseGenerator._normalize(
                f"{combined_crop} {combined_disease}"
            )

            for record in crop_diseases:
                db_name = ResponseGenerator._normalize(record.disease_name)
                db_combined = ResponseGenerator._normalize(
                    f"{record.crop} {record.disease_name}"
                )

                if db_name == wanted or db_combined == wanted_combined:
                    logger.info(
                        "Normalized disease DB match: %s - %s (requested: %s - %s)",
                        record.crop, record.disease_name, crop, disease,
                    )
                    return record
        except Exception as exc:
            logger.debug("Normalized disease scan failed: %s", exc)

        return None

    @staticmethod
    def generate_disease_response(
        session: Session,
        crop: str,
        disease: str,
        confidence: float,
        language: str = "en",
    ) -> Optional[str]:
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)

            if not disease_info:
                logger.warning(f"Disease not found in DB: {crop} - {disease}")
                return ResponseGenerator.get_error_message("no_disease_found", language)

            confidence_percent = round(confidence * 100, 1)
            confidence_level = ResponseGenerator._get_confidence_level(confidence)
            confidence_text = CONFIDENCE_TEMPLATES.get(language, {}).get(
                confidence_level,
                f"Confidence: {confidence_percent}%"
            )

            prevention_section = ""
            if disease_info.prevention:
                if language == "hi":
                    prevention_section = f"🛡️ *रोकथाम:*\n{disease_info.prevention}\n\n"
                else:
                    prevention_section = f"🛡️ *Prevention:*\n{disease_info.prevention}\n\n"

            treatment_section = ""
            if disease_info.treatment:
                if language == "hi":
                    treatment_section = f"💊 *उपचार:*\n{disease_info.treatment}\n\n"
                else:
                    treatment_section = f"💊 *Treatment:*\n{disease_info.treatment}\n\n"

            template = DISEASE_PREDICTION_TEMPLATE.get(language, DISEASE_PREDICTION_TEMPLATE["en"])

            response = template.format(
                crop=crop,
                disease=disease_info.disease_name,
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
        return ResponseGenerator.generate_disease_response(
            session, crop, disease, confidence=1.0, language=language
        )

    @staticmethod
    def generate_symptom_response(
        session: Session,
        crop: str,
        disease: str,
        language: str = "en",
    ) -> Optional[str]:
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)

            if language == "hi":
                return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🔍 *लक्षण:*\n{disease_info.symptoms}\n"
            return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🔍 *Symptoms:*\n{disease_info.symptoms}\n"

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
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)

            if language == "hi":
                return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🛠️ *प्रबंधन:*\n{disease_info.management}\n"
            return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🛠️ *Management:*\n{disease_info.management}\n"

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
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)

            if language == "hi":
                if disease_info.prevention:
                    return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🛡️ *रोकथाम:*\n{disease_info.prevention}\n"
                return "कोई विशिष्ट रोकथाम जानकारी उपलब्ध नहीं है।\n"

            if disease_info.prevention:
                return f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n🛡️ *Prevention:*\n{disease_info.prevention}\n"
            return "No specific prevention information available.\n"

        except Exception as e:
            logger.error(f"Error generating prevention response: {e}")
            return None

    @staticmethod
    def get_error_message(error_key: str, language: str = "en") -> str:
        messages = ERROR_MESSAGES.get(language, ERROR_MESSAGES["en"])
        return messages.get(error_key, "An error occurred. Please try again.")

    @staticmethod
    def _get_confidence_level(confidence: float) -> str:
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.60:
            return "medium"
        else:
            return "low"
