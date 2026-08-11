"""
Repository classes for database operations
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.models import User, Disease, Interaction, Session as SessionModel

logger = logging.getLogger(__name__)


class UserRepository:
    """User data operations."""
    
    @staticmethod
    def get_or_create_user(
        session: Session,
        telegram_user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
    ) -> User:
        """Get or create user by Telegram ID."""
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        
        if not user:
            user = User(
                telegram_user_id=telegram_user_id,
                telegram_username=username,
                first_name=first_name,
                language="en",
            )
            session.add(user)
            session.commit()
            logger.info(f"✅ New user created: {telegram_user_id}")
        
        return user
    
    @staticmethod
    def get_user_language(session: Session, telegram_user_id: int) -> str:
        """Get user's preferred language."""
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        return user.language if user else "en"
    
    @staticmethod
    def set_user_language(session: Session, telegram_user_id: int, language: str) -> bool:
        """Set user's preferred language."""
        user = session.query(User).filter(User.telegram_user_id == telegram_user_id).first()
        if user:
            user.language = language
            session.commit()
            return True
        return False


class DiseaseRepository:
    """Disease information operations."""
    
    @staticmethod
    def get_disease(
        session: Session,
        crop: str,
        disease_name: str,
    ) -> Optional[Disease]:
        """Get disease by crop and disease name."""
        return session.query(Disease).filter(
            Disease.crop == crop,
            Disease.disease_name == disease_name,
        ).first()
    
    @staticmethod
    def get_diseases_by_crop(session: Session, crop: str) -> List[Disease]:
        """Get all diseases for a crop."""
        return session.query(Disease).filter(Disease.crop == crop).all()
    
    @staticmethod
    def search_diseases(session: Session, keyword: str) -> List[Disease]:
        """Search diseases by keyword (in keywords field or disease name)."""
        keyword_lower = f"%{keyword.lower()}%"
        return session.query(Disease).filter(
            (Disease.disease_name.ilike(keyword_lower)) |
            (Disease.crop.ilike(keyword_lower)) |
            (Disease.keywords.ilike(keyword_lower))
        ).all()
    
    @staticmethod
    def get_all_crops(session: Session) -> List[str]:
        """Get list of all crops in database."""
        crops = session.query(Disease.crop).distinct().all()
        return [crop[0] for crop in crops if crop[0]]
    
    @staticmethod
    def disease_exists(session: Session, crop: str, disease_name: str) -> bool:
        """Check if disease exists in database."""
        return session.query(Disease).filter(
            Disease.crop == crop,
            Disease.disease_name == disease_name,
        ).first() is not None
    
    @staticmethod
    def create_or_update_disease(
        session: Session,
        crop: str,
        disease_name: str,
        disease_type: str,
        symptoms: str,
        management: str,
        **kwargs
    ) -> Disease:
        """Create or update disease record."""
        disease = session.query(Disease).filter(
            Disease.crop == crop,
            Disease.disease_name == disease_name,
        ).first()
        
        if disease:
            # Update existing
            disease.disease_type = disease_type
            disease.symptoms = symptoms
            disease.management = management
            for key, value in kwargs.items():
                if hasattr(disease, key):
                    setattr(disease, key, value)
        else:
            # Create new
            disease = Disease(
                crop=crop,
                disease_name=disease_name,
                disease_type=disease_type,
                symptoms=symptoms,
                management=management,
                **kwargs
            )
            session.add(disease)
        
        session.commit()
        return disease


class InteractionRepository:
    """User interaction/history operations."""
    
    @staticmethod
    def create_interaction(
        session: Session,
        user_id: int,
        crop: Optional[str] = None,
        predicted_disease: Optional[str] = None,
        confidence: Optional[float] = None,
        question: Optional[str] = None,
        intent: Optional[str] = None,
        response_text: Optional[str] = None,
        response_language: str = "en",
        image_filename: Optional[str] = None,
    ) -> Interaction:
        """Create new interaction record."""
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
    def get_user_interactions(
        session: Session,
        user_id: int,
        limit: int = 10,
    ) -> List[Interaction]:
        """Get user's recent interactions."""
        return session.query(Interaction).filter(
            Interaction.user_id == user_id
        ).order_by(Interaction.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_interaction_by_id(session: Session, interaction_id: int) -> Optional[Interaction]:
        """Get interaction by ID."""
        return session.query(Interaction).filter(Interaction.id == interaction_id).first()


class SessionRepository:
    """Session/context operations."""
    
    @staticmethod
    def get_or_create_session(session: Session, user_id: int) -> SessionModel:
        """Get or create session for user."""
        user_session = session.query(SessionModel).filter(SessionModel.user_id == user_id).first()
        
        if not user_session:
            user_session = SessionModel(user_id=user_id)
            session.add(user_session)
            session.commit()
        
        return user_session
    
    @staticmethod
    def update_session_context(
        session: Session,
        user_id: int,
        crop: Optional[str] = None,
        disease: Optional[str] = None,
        last_interaction_id: Optional[int] = None,
    ) -> SessionModel:
        """Update session context."""
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
        """Get user's last disease context."""
        user_session = session.query(SessionModel).filter(SessionModel.user_id == user_id).first()
        
        if user_session:
            return user_session.last_crop, user_session.last_disease
        
        return None, None
