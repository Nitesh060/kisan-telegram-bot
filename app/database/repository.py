"""
Repository classes for database operations
"""

import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.models import User, Disease, Interaction, Session as SessionModel, FeedbackSample

logger = logging.getLogger(__name__)


class UserRepository:
    @staticmethod
    def get_or_create_user(session: Session, telegram_user_id: int, username: Optional[str] = None, first_name: Optional[str] = None) -> User:
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        if not user:
            user = User(telegram_user_id=telegram_user_id, telegram_username=username, first_name=first_name, language="en")
            session.add(user)
            session.commit()
            logger.info(f"✅ New user created: {telegram_user_id}")
        return user

    @staticmethod
    def get_user_language(session: Session, telegram_user_id: int) -> str:
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        return user.language if user else "en"

    @staticmethod
    def set_user_language(session: Session, telegram_user_id: int, language: str) -> bool:
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        if user:
            user.language = language
            session.commit()
            return True
        return False


class DiseaseRepository:
    @staticmethod
    def get_disease(session: Session, crop: str, disease_name: str) -> Optional[Disease]:
        return session.query(Disease).filter(Disease.crop == crop, Disease.disease_name == disease_name).first()

    @staticmethod
    def get_diseases_by_crop(session: Session, crop: str) -> List[Disease]:
        return session.query(Disease).filter(Disease.crop == crop).all()

    @staticmethod
    def search_diseases(session: Session, keyword: str) -> List[Disease]:
        keyword_lower = f"%{keyword.lower()}%"
        return session.query(Disease).filter(
            (Disease.disease_name.ilike(keyword_lower)) |
            (Disease.crop.ilike(keyword_lower)) |
            (Disease.keywords.ilike(keyword_lower))
        ).all()

    @staticmethod
    def get_all_crops(session: Session) -> List[str]:
        crops = session.query(Disease.crop).distinct().all()
        return [crop[0] for crop in crops if crop[0]]

    @staticmethod
    def disease_exists(session: Session, crop: str, disease_name: str) -> bool:
        return session.query(Disease).filter(Disease.crop == crop, Disease.disease_name == disease_name).first() is not None

    @staticmethod
    def create_or_update_disease(session: Session, crop: str, disease_name: str, disease_type: str, symptoms: str, management: str, **kwargs) -> Disease:
        disease = session.query(Disease).filter(Disease.crop == crop, Disease.disease_name == disease_name).first()
        if disease:
            disease.disease_type = disease_type
            disease.symptoms = symptoms
            disease.management = management
            for key, value in kwargs.items():
                if hasattr(disease, key):
                    setattr(disease, key, value)
        else:
            disease = Disease(crop=crop, disease_name=disease_name, disease_type=disease_type, symptoms=symptoms, management=management, **kwargs)
            session.add(disease)
        session.commit()
        return disease


class InteractionRepository:
    @staticmethod
    def create_interaction(session: Session, user_id: int, crop: Optional[str] = None, predicted_disease: Optional[str] = None, confidence: Optional[float] = None, question: Optional[str] = None, intent: Optional[str] = None, response_text: Optional[str] = None, response_language: str = "en", image_filename: Optional[str] = None) -> Interaction:
        interaction = Interaction(
            user_id=user_id,
            crop=crop,
            predicted_disease=predicted_disease,
            confidence=confidence,
            question=question,
            intent=intent,
            response_text=response_text,
            response_language=response_language,
            image_filename=image_filename,
        )
        session.add(interaction)
        session.commit()
        return interaction

    @staticmethod
    def get_user_interactions(session: Session, user_id: int, limit: int = 10) -> List[Interaction]:
        return session.query(Interaction).filter(Interaction.user_id == user_id).order_by(Interaction.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_interaction_by_id(session: Session, interaction_id: int) -> Optional[Interaction]:
        return session.query(Interaction).filter(Interaction.id == interaction_id).first()


class FeedbackRepository:
    """Human feedback operations for active learning."""

    @staticmethod
    def create_pending_sample(session: Session, user_id: int, interaction_id: Optional[int], telegram_file_id: str, crop: str, model_prediction: str, model_confidence: Optional[float]) -> FeedbackSample:
        sample = FeedbackSample(
            user_id=user_id,
            interaction_id=interaction_id,
            telegram_file_id=telegram_file_id,
            crop=crop,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            status="pending",
            source="user_feedback",
        )
        session.add(sample)
        session.commit()
        return sample

    @staticmethod
    def verify_sample(session: Session, sample_id: int, correct_disease: str, status: str = "verified") -> Optional[FeedbackSample]:
        sample = session.query(FeedbackSample).filter(FeedbackSample.id == sample_id).first()
        if not sample:
            return None
        sample.correct_disease = correct_disease
        sample.status = status
        sample.verified_at = datetime.utcnow()
        session.commit()
        return sample

    @staticmethod
    def get_sample(session: Session, sample_id: int) -> Optional[FeedbackSample]:
        return session.query(FeedbackSample).filter(FeedbackSample.id == sample_id).first()


class SessionRepository:
    @staticmethod
    def get_or_create_session(session: Session, user_id: int) -> SessionModel:
        user_session = session.query(SessionModel).filter(SessionModel.user_id == user_id).first()
        if not user_session:
            user_session = SessionModel(user_id=user_id)
            session.add(user_session)
            session.commit()
        return user_session

    @staticmethod
    def update_session_context(session: Session, user_id: int, crop: Optional[str] = None, disease: Optional[str] = None, last_interaction_id: Optional[int] = None) -> SessionModel:
        user_session = SessionRepository.get_or_create_session(session, user_id)
        if crop:
            user_session.last_crop = crop
        if disease:
            user_session.last_disease = disease
        if last_interaction_id:
            user_session.last_interaction_id = last_interaction_id
        session.commit()
        return user_session

    @staticmethod
    def get_last_context(session: Session, user_id: int) -> tuple:
        user_session = session.query(SessionModel).filter(SessionModel.user_id == user_id).first()
        if user_session:
            return user_session.last_crop, user_session.last_disease
        return None, None
