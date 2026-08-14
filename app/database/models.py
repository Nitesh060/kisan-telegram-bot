"""
Database models for Kisan Telegram Bot
"""

from datetime import datetime
from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """Farmer/User model."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    telegram_user_id = Column(BigInteger, unique=True, index=True)
    telegram_username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    feedback_samples = relationship("FeedbackSample", back_populates="user", cascade="all, delete-orphan")
    __table_args__ = (
        Index("idx_user_telegram_id", "telegram_user_id"),
        Index("idx_user_created_at", "created_at"),
    )


class Disease(Base):
    """Disease information model."""
    __tablename__ = "diseases"
    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String(255), index=True)
    disease_name = Column(String(255), index=True)
    disease_type = Column(String(100))
    scientific_name = Column(String(255), nullable=True)
    symptoms = Column(Text)
    causes = Column(Text, nullable=True)
    favorable_conditions = Column(Text, nullable=True)
    management = Column(Text)
    prevention = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True)
    severity = Column(String(50))
    keywords = Column(String(500), nullable=True)
    confidence_threshold = Column(Float, default=0.60)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        Index("idx_disease_crop_name", "crop", "disease_name"),
        Index("idx_disease_keywords", "keywords"),
    )


class Interaction(Base):
    """User interaction/request history."""
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    image_path = Column(String(500), nullable=True)
    image_filename = Column(String(255), nullable=True)
    crop = Column(String(255), nullable=True)
    predicted_disease = Column(String(255), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    question = Column(Text, nullable=True)
    intent = Column(String(50), nullable=True)
    response_text = Column(Text, nullable=True)
    response_language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    user = relationship("User", back_populates="interactions")
    __table_args__ = (
        Index("idx_interaction_user_id", "user_id"),
        Index("idx_interaction_created_at", "created_at"),
        Index("idx_interaction_disease", "predicted_disease"),
    )


class FeedbackSample(Base):
    """Human-verified samples collected for active learning."""
    __tablename__ = "feedback_samples"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    interaction_id = Column(Integer, ForeignKey("interactions.id"), nullable=True, index=True)
    telegram_file_id = Column(String(500), nullable=False)
    crop = Column(String(255), nullable=False, index=True)
    model_prediction = Column(String(255), nullable=False)
    model_confidence = Column(Float, nullable=True)
    correct_disease = Column(String(255), nullable=True, index=True)
    status = Column(String(30), default="pending", index=True)  # pending/verified/rejected
    source = Column(String(30), default="user_feedback")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    verified_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="feedback_samples")
    __table_args__ = (
        Index("idx_feedback_status_crop", "status", "crop"),
        Index("idx_feedback_created_at", "created_at"),
    )


class ModelInfo(Base):
    """ML model information and metadata."""
    __tablename__ = "model_info"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), unique=True)
    model_path = Column(String(500))
    model_version = Column(String(50))
    num_classes = Column(Integer)
    accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (Index("idx_model_name", "model_name"),)


class Session(Base):
    """Lightweight session tracking for conversation context."""
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    last_crop = Column(String(255), nullable=True)
    last_disease = Column(String(255), nullable=True)
    last_interaction_id = Column(Integer, ForeignKey("interactions.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (Index("idx_session_user_id", "user_id"),)
