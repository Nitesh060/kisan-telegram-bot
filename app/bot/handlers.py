"""
Telegram bot handlers for commands and messages
"""

import logging
import os
import tempfile
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from app.config import settings
from app.database.database import DatabaseManager
from app.database.repository import (
    UserRepository,
    DiseaseRepository,
    InteractionRepository,
    SessionRepository,
)
from app.ml.inference import ModelInference
from app.ml.preprocessing import preprocess_image, validate_image
from app.nlp.intent import IntentDetector
from app.responses.generator import ResponseGenerator
from app.responses.templates import (
    GREETING_MESSAGES,
    HELP_MESSAGES,
    ABOUT_MESSAGES,
    LANGUAGE_MESSAGE,
)

logger = logging.getLogger(__name__)

# Global model instance
_model_inference: Optional[ModelInference] = None


def get_model() -> ModelInference:
    """Get or initialize model."""
    global _model_inference
    if _model_inference is None:
        _model_inference = ModelInference(settings.MODEL_PATH)
        _model_inference.load_class_names(settings.CLASS_NAMES_PATH)
    return _model_inference


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        logger.info(f"👤 User started bot: {user.id} (@{user.username})")
        
        # Get or create user in database
        with DatabaseManager() as session:
            UserRepository.get_or_create_user(
                session,
                user.id,
                user.username,
                user.first_name,
            )
        
        # Send greeting
        language = "en"
        greeting = GREETING_MESSAGES.get(language, GREETING_MESSAGES["en"])
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=greeting,
            parse_mode="Markdown",
        )
    
    except Exception as e:
        logger.error(f"❌ Error in start handler: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ An error occurred. Please try again.",
        )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    try:
        chat_id = update.effective_chat.id
        
        with DatabaseManager() as session:
            language = UserRepository.get_user_language(session, update.effective_user.id)
        
        help_text = HELP_MESSAGES.get(language, HELP_MESSAGES["en"])
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=help_text,
            parse_mode="Markdown",
        )
    
    except Exception as e:
        logger.error(f"❌ Error in help handler: {e}")


async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about command."""
    try:
        chat_id = update.effective_chat.id
        
        with DatabaseManager() as session:
            language = UserRepository.get_user_language(session, update.effective_user.id)
        
        about_text = ABOUT_MESSAGES.get(language, ABOUT_MESSAGES["en"])
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=about_text,
            parse_mode="Markdown",
        )
    
    except Exception as e:
        logger.error(f"❌ Error in about handler: {e}")


async def language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /language command."""
    try:
        chat_id = update.effective_chat.id
        
        keyboard = [
            [
                InlineKeyboardButton("English", callback_data="lang_en"),
                InlineKeyboardButton("हिंदी", callback_data="lang_hi"),
            ],
            [
                InlineKeyboardButton("Hinglish", callback_data="lang_hinglish"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="Please select your preferred language:",
            reply_markup=reply_markup,
        )
    
    except Exception as e:
        logger.error(f"❌ Error in language handler: {e}")


# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image messages."""
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        logger.info(f"📷 Image received from {user.id}")
        
        # Get or create user
        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(
                session,
                user.id,
                user.username,
                user.first_name,
            )
            language = user_obj.language
        
        # Download image
        photo = update.message.photo[-1]  # Get highest resolution
        
        # Save to temp file
        temp_dir = settings.TEMP_IMAGE_DIR
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir,
            suffix=".jpg",
            delete=False,
        )
        image_path = temp_file.name
        temp_file.close()
        
        # Download from Telegram
        file = await context.bot.get_file(photo.file_id)
        await file.download_to_drive(image_path)
        
        logger.info(f"✅ Image saved: {image_path}")
        
        # Validate image
        is_valid, validation_msg = validate_image(
            image_path,
            settings.MAX_IMAGE_SIZE_MB,
        )
        
        if not is_valid:
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message("invalid_image", language),
            )
            os.unlink(image_path)
            return
        
        # Check if ML model is ready
        model = get_model()
        if not model.is_model_ready():
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message("model_not_ready", language),
            )
            os.unlink(image_path)
            return
        
        # Send processing message
        processing_msg = (
            "🔄 Processing your image..." if language == "en"
            else "🔄 आपकी छवि को संसाधित किया जा रहा है..."
        )
        await context.bot.send_message(chat_id=chat_id, text=processing_msg)
        
        # Preprocess and predict
        image_array = preprocess_image(image_path)
        if image_array is None:
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message("processing_error", language),
            )
            os.unlink(image_path)
            return
        
        # Run inference
        prediction = model.predict(image_array)
        if not prediction or not prediction.get("success"):
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message("processing_error", language),
            )
            os.unlink(image_path)
            return
        
        crop = prediction["crop"]
        disease = prediction["disease"]
        confidence = prediction["confidence"]
        confidence_percent = prediction["confidence_percent"]
        
        logger.info(
            f"✅ Prediction: {crop} - {disease} ({confidence_percent:.1f}%)"
        )
        
        # Check confidence threshold
        if confidence < settings.CONFIDENCE_LOW:
            response = ResponseGenerator.get_error_message("low_confidence", language)
            await context.bot.send_message(
                chat_id=chat_id,
                text=response,
            )
        else:
            # Generate response from database
            with DatabaseManager() as session:
                response = ResponseGenerator.generate_disease_response(
                    session,
                    crop,
                    disease,
                    confidence,
                    language,
                )
                
                # Store interaction
                user_obj = UserRepository.get_or_create_user(session, user.id, user.username, user.first_name)
                interaction = InteractionRepository.create_interaction(
                    session,
                    user_obj.id,
                    crop=crop,
                    predicted_disease=disease,
                    confidence=confidence,
                    response_text=response,
                    response_language=language,
                    image_filename=os.path.basename(image_path),
                )
                
                # Update session context
                SessionRepository.update_session_context(
                    session,
                    user_obj.id,
                    crop=crop,
                    disease=disease,
                    last_interaction_id=interaction.id,
                )
            
            if response:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=response,
                    parse_mode="Markdown",
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=ResponseGenerator.get_error_message("database_error", language),
                )
        
        # Clean up temp file
        try:
            os.unlink(image_path)
        except:
            pass
    
    except Exception as e:
        logger.error(f"❌ Error in image handler: {e}", exc_info=True)
        try:
            language = "en"
            with DatabaseManager() as session:
                language = UserRepository.get_user_language(session, update.effective_user.id)
        except:
            pass
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=ResponseGenerator.get_error_message("processing_error", language),
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages."""
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        text = update.message.text
        
        logger.info(f"💬 Text received from {user.id}: {text[:50]}")
        
        # Get user language
        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(
                session,
                user.id,
                user.username,
                user.first_name,
            )
            language = user_obj.language
        
        # Detect intent
        intent = IntentDetector.detect(text)
        
        # Get last context
        with DatabaseManager() as session:
            last_crop, last_disease = SessionRepository.get_last_context(
                session,
                user_obj.id,
            )
        
        if not last_crop or not last_disease:
            # No previous context - ask for image
            if language == "hi":
                msg = "🤔 कृपया पहले एक फसल की तस्वीर भेजें।"
            else:
                msg = "🤔 Please send a crop photo first."
            
            await context.bot.send_message(
                chat_id=chat_id,
                text=msg,
            )
            return
        
        # Generate response based on intent
        with DatabaseManager() as session:
            response = None
            
            if intent == IntentDetector.SYMPTOMS:
                response = ResponseGenerator.generate_symptom_response(
                    session, last_crop, last_disease, language
                )
            elif intent == IntentDetector.MANAGEMENT:
                response = ResponseGenerator.generate_management_response(
                    session, last_crop, last_disease, language
                )
            elif intent == IntentDetector.PREVENTION:
                response = ResponseGenerator.generate_prevention_response(
                    session, last_crop, last_disease, language
                )
            else:
                # Default to disease info
                response = ResponseGenerator.generate_info_response(
                    session, last_crop, last_disease, language
                )
        
        if response:
            await context.bot.send_message(
                chat_id=chat_id,
                text=response,
                parse_mode="Markdown",
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message("database_error", language),
            )
    
    except Exception as e:
        logger.error(f"❌ Error in text handler: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ An error occurred. Please try again.",
        )
