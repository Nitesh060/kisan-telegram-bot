"""
Response generation engine - deterministic, no LLM.
Disease content comes from the database; visible section labels are multilingual.
"""

import logging
import re
from typing import Optional

from app.responses.templates import (
    DISEASE_PREDICTION_TEMPLATE,
    CONFIDENCE_TEMPLATES,
    ERROR_MESSAGES,
)
from app.database.repository import DiseaseRepository
from app.services.language_service import get_text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses from templates and database data."""

    @staticmethod
    def _normalize(value: str) -> str:
        value = (value or "").strip().lower()
        value = value.replace("_", " ").replace("-", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _canonical_crop(crop: str) -> str:
        normalized = ResponseGenerator._normalize(crop)
        aliases = {
            "chili": "Chilli",
            "chilli": "Chilli",
            "pepper bell": "Pepper_bell",
            "gauva": "Gauva",
            "guava": "Gauva",
        }
        return aliases.get(normalized, (crop or "").strip())

    @staticmethod
    def _canonical_disease(crop: str, disease: str) -> str:
        normalized = ResponseGenerator._normalize(disease)
        crop_norm = ResponseGenerator._normalize(crop)
        aliases = {
            ("rice", "leaf blast"): "Blast",
            ("rice", "neck blast"): "Blast",
            ("wheat", "brown rust"): "Rust",
            ("wheat", "yellow rust"): "Rust",
            ("apple", "black rot"): "Black Rot",
            ("apple", "rust"): "Rust",
            ("apple", "scab"): "Scab",
        }
        return aliases.get((crop_norm, normalized), (disease or "").strip())

    @staticmethod
    def _get_disease_info(session: Session, crop: str, disease: str):
        crop_clean = ResponseGenerator._canonical_crop(crop)
        disease_clean = (disease or "").strip()
        combined_crop = crop_clean
        combined_disease = disease_clean

        if " - " in disease_clean:
            left, right = disease_clean.split(" - ", 1)
            if not combined_crop:
                combined_crop = ResponseGenerator._canonical_crop(left.strip())
            if ResponseGenerator._normalize(left) == ResponseGenerator._normalize(crop_clean):
                combined_disease = right.strip()

        combined_crop = ResponseGenerator._canonical_crop(combined_crop)
        combined_disease = ResponseGenerator._canonical_disease(combined_crop, combined_disease)

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
    def _label(key: str, language: str) -> str:
        return get_text(key, language)

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
                logger.warning("Disease not found in DB: %s - %s", crop, disease)
                return ResponseGenerator.get_error_message("no_disease_found", language)

            confidence_percent = round(confidence * 100, 1)
            confidence_level = ResponseGenerator._get_confidence_level(confidence)
            confidence_text = CONFIDENCE_TEMPLATES.get(language, {}).get(
                confidence_level,
                f"{ResponseGenerator._label('confidence', language)}: {confidence_percent}%",
            )

            # Keep disease content exactly as stored in the DB. Only the labels
            # are translated here, so we never silently invent a translation.
            symptoms = disease_info.symptoms or get_text("symptoms_unavailable", language)
            management = disease_info.management or get_text("management_unavailable", language)

            prevention_section = ""
            if disease_info.prevention:
                prevention_section = (
                    f"🛡️ *{ResponseGenerator._label('prevention', language)}:*\n"
                    f"{disease_info.prevention}\n\n"
                )

            treatment_section = ""
            if disease_info.treatment:
                treatment_section = (
                    f"💊 *{ResponseGenerator._label('treatment', language)}:*\n"
                    f"{disease_info.treatment}\n\n"
                )

            # Existing templates remain compatible; for the new languages we
            # build the disease card directly so every visible label is localized.
            if language not in ("en", "hi"):
                return (
                    f"🌱 *{ResponseGenerator._label('disease_result', language)}*\n\n"
                    f"*{ResponseGenerator._label('crop', language)}:* {crop}\n"
                    f"*{ResponseGenerator._label('disease', language)}:* {disease_info.disease_name}\n"
                    f"*{ResponseGenerator._label('confidence', language)}:* {confidence_percent}%\n\n"
                    f"🔍 *{ResponseGenerator._label('symptoms', language)}:*\n{symptoms}\n\n"
                    f"🛠️ *{ResponseGenerator._label('management', language)}:*\n{management}\n\n"
                    f"{prevention_section}{treatment_section}"
                    f"⚠️ *{ResponseGenerator._label('important_note', language)}:*\n"
                    f"{get_text('preliminary_note', language)}\n"
                )

            template = DISEASE_PREDICTION_TEMPLATE.get(
                language,
                DISEASE_PREDICTION_TEMPLATE["en"],
            )
            response = template.format(
                crop=crop,
                disease=disease_info.disease_name,
                confidence_percent=confidence_percent,
                confidence_level_text=confidence_text,
                symptoms=symptoms,
                management=management,
                prevention_section=prevention_section,
                treatment_section=treatment_section,
                severity=disease_info.severity or "Unknown",
            )
            return response

        except Exception as e:
            logger.error("Error generating disease response: %s", e, exc_info=True)
            return None

    @staticmethod
    def generate_info_response(session: Session, crop: str, disease: str, language: str = "en") -> Optional[str]:
        return ResponseGenerator.generate_disease_response(session, crop, disease, 1.0, language)

    @staticmethod
    def generate_symptom_response(session: Session, crop: str, disease: str, language: str = "en") -> Optional[str]:
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            return (
                f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n"
                f"🔍 *{ResponseGenerator._label('symptoms', language)}:*\n"
                f"{disease_info.symptoms}\n"
            )
        except Exception as e:
            logger.error("Error generating symptom response: %s", e, exc_info=True)
            return None

    @staticmethod
    def generate_management_response(session: Session, crop: str, disease: str, language: str = "en") -> Optional[str]:
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            return (
                f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n"
                f"🛠️ *{ResponseGenerator._label('management', language)}:*\n"
                f"{disease_info.management}\n"
            )
        except Exception as e:
            logger.error("Error generating management response: %s", e, exc_info=True)
            return None

    @staticmethod
    def generate_prevention_response(session: Session, crop: str, disease: str, language: str = "en") -> Optional[str]:
        try:
            disease_info = ResponseGenerator._get_disease_info(session, crop, disease)
            if not disease_info:
                return ResponseGenerator.get_error_message("no_disease_found", language)
            if disease_info.prevention:
                return (
                    f"🌾 *{disease_info.crop} - {disease_info.disease_name}*\n\n"
                    f"🛡️ *{ResponseGenerator._label('prevention', language)}:*\n"
                    f"{disease_info.prevention}\n"
                )
            return get_text("prevention_unavailable", language)
        except Exception as e:
            logger.error("Error generating prevention response: %s", e, exc_info=True)
            return None

    @staticmethod
    def get_error_message(error_key: str, language: str = "en") -> str:
        messages = ERROR_MESSAGES.get(language, ERROR_MESSAGES["en"])
        return messages.get(error_key, ERROR_MESSAGES["en"].get(error_key, "An error occurred. Please try again."))

    @staticmethod
    def _get_confidence_level(confidence: float) -> str:
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.60:
            return "medium"
        return "low"
