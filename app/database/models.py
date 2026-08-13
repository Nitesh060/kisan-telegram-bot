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
    # Telegram user IDs can exceed PostgreSQL's 32-bit INTEGER range.
    # BIGINT safely supports current and future Telegram user IDs.
    telegram_user_id = Column(BigInteger, unique=True, index=True)
    telegram_username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    
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
    disease_type = Column(String(100))  # Viral, Fungal, Bacterial, etc.
    scientific_name = Column(String(255), nullable=True)
    
    # Disease details (text)
    symptoms = Column(Text)
    causes = Column(Text, nullable=True)
    favorable_conditions = Column(Text, nullable=True)
    management = Column(Text)
    prevention = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True)
    
    # Metadata
    severity = Column(String(50))  # Low, Medium, High
    keywords = Column(String(500), nullable=True)  # Comma-separated
    confidence_threshold = Column(Float, default=0.60)
    
    # Timestamps
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
    
    # Image data
    image_path = Column(String(500), nullable=True)
    image_filename = Column(String(255), nullable=True)
    
    # Prediction
    crop = Column(String(255), nullable=True)
    predicted_disease = Column(String(255), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    
    # User input
    question = Column(Text, nullable=True)
    
    # Intent detection
    intent = Column(String(50), nullable=True)  # DIAGNOSIS, MANAGEMENT, etc.
    
    # Response
    response_text = Column(Text, nullable=True)
    response_language = Column(String(10), default="en")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    user = relationship("User", back_populates="interactions")
    
    __table_args__ = (
        Index("idx_interaction_user_id", "user_id"),
        Index("idx_interaction_created_at", "created_at"),
        Index("idx_interaction_disease", "predicted_disease"),
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
    
    __table_args__ = (
        Index("idx_model_name", "model_name"),
    )


class Session(Base):
    """Lightweight session tracking for conversation context."""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Last identified disease for context
    last_crop = Column(String(255), nullable=True)
    last_disease = Column(String(255), nullable=True)
    last_interaction_id = Column(Integer, ForeignKey("interactions.id"), nullable=True)
    
    # Session state
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_session_user_id", "user_id"),
    )
