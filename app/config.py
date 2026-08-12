"""
Configuration management for Kisan Telegram Bot
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    USE_WEBHOOK: bool = os.getenv("USE_WEBHOOK", "false").lower() == "true"
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "https://your-app.onrender.com/telegram/webhook")
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./kisan_bot.db"
    )
    
    # ML Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/crop_disease_model.keras")
    CLASS_NAMES_PATH: str = os.getenv("CLASS_NAMES_PATH", "models/class_names.json")
    
    # Confidence thresholds
    CONFIDENCE_HIGH: float = float(os.getenv("CONFIDENCE_HIGH", "0.85"))
    CONFIDENCE_LOW: float = float(os.getenv("CONFIDENCE_LOW", "0.40"))
    
    # Image processing
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    IMAGE_RETENTION_HOURS: int = int(os.getenv("IMAGE_RETENTION_HOURS", "1"))
    TEMP_IMAGE_DIR: str = os.getenv("TEMP_IMAGE_DIR", "/tmp/kisan_images")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Language
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    SUPPORTED_LANGUAGES: list = ["en", "hi", "hinglish"]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def validate_settings() -> bool:
    """Validate critical settings."""
    errors = []
    
    if not settings.TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN not set")
    
    if settings.USE_WEBHOOK and not settings.WEBHOOK_URL:
        errors.append("WEBHOOK_URL required when USE_WEBHOOK is True")
    
    if settings.CONFIDENCE_HIGH <= settings.CONFIDENCE_LOW:
        errors.append("CONFIDENCE_HIGH must be greater than CONFIDENCE_LOW")
    
    if errors:
        print("⚠️ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True
