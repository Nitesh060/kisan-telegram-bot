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
_model_inference: Optional[ModelInference] = None


def get_model() -> ModelInference:
    global _model_inference
    if _model_inference is None:
        _model_inference = ModelInference(settings.MODEL_PATH)
        _model_inference.load_class_names(settings.CLASS_NAMES_PATH)
    return _model_inference


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        logger.info(f"👤 User started bot: {user.id} (@{user.username})")
        with DatabaseManager() as session:
            UserRepository.get_or_create_user(session, user.id, user.username, user.first_name)
        greeting = GREETING_MESSAGES.get("en", GREETING_MESSAGES["en"])
        await context.bot.send_message(chat_id=chat_id, text=greeting, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"❌ Error in start handler: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ An error occurred. Please try again.")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        with DatabaseManager() as session:
            language = UserRepository.get_user_language(session, update.effective_user.id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_MESSAGES.get(language, HELP_MESSAGES["en"]), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"❌ Error in help handler: {e}")


async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        with DatabaseManager() as session:
            language = UserRepository.get_user_language(session, update.effective_user.id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ABOUT_MESSAGES.get(language, ABOUT_MESSAGES["en"]), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"❌ Error in about handler: {e}")


async def language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        keyboard = [
            [InlineKeyboardButton("English", callback_data="lang_en"), InlineKeyboardButton("हिंदी", callback_data="lang_hi")],
            [InlineKeyboardButton("Hinglish", callback_data="lang_hinglish")],
        ]
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please select your preferred language:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"❌ Error in language handler: {e}")


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receive an image, then ask the user which crop it belongs to."""
    image_path = None
    keep_for_crop_selection = False

    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        logger.info(f"📷 Image received from {user.id}")

        # Keep only primitive values after the DB session closes.
        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(
                session,
                user.id,
                user.username,
                user.first_name,
            )
            language = user_obj.language or "en"
            user_id = user_obj.id

        photo = update.message.photo[-1]
        os.makedirs(settings.TEMP_IMAGE_DIR, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=settings.TEMP_IMAGE_DIR,
            suffix=".jpg",
            delete=False,
        )
        image_path = temp_file.name
        temp_file.close()

        file = await context.bot.get_file(photo.file_id)
        await file.download_to_drive(image_path)
        logger.info(f"✅ Image saved: {image_path}")

        is_valid, _ = validate_image(
            image_path,
            settings.MAX_IMAGE_SIZE_MB,
        )
        if not is_valid:
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message(
                    "invalid_image",
                    language,
                ),
            )
            return

        model = get_model()
        if not model.is_model_ready():
            await context.bot.send_message(
                chat_id=chat_id,
                text=ResponseGenerator.get_error_message(
                    "model_not_ready",
                    language,
                ),
            )
            return

        crops = model.get_crops()

        if not crops:
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Crop classes could not be loaded. Please try again.",
            )
            return

        # Save the pending image until the user selects a crop.
        context.user_data["pending_image_path"] = image_path
        context.user_data["pending_language"] = language
        context.user_data["pending_user_id"] = user_id
        context.user_data["pending_chat_id"] = chat_id
        keep_for_crop_selection = True

        # Dynamic buttons from class_names.json, rather than hard-coding crops.
        keyboard = []
        row = []

        for crop in crops:
            # Callback payload stays short and the actual crop is kept in user_data.
            key = str(len(context.user_data.get("pending_crop_options", {})))
            options = context.user_data.setdefault("pending_crop_options", {})
            options[key] = crop

            row.append(
                InlineKeyboardButton(
                    f"🌱 {crop}",
                    callback_data=f"crop:{key}",
                )
            )

            if len(row) == 2:
                keyboard.append(row)
                row = []

        if row:
            keyboard.append(row)

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "🌱 *Which crop/plant is this?*\n\n"
                "Please select the crop before I analyse the disease."
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.error(
            f"❌ Error in image handler: {e}",
            exc_info=True,
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=ResponseGenerator.get_error_message(
                "processing_error",
                "en",
            ),
        )

    finally:
        # The image must remain available while the crop-selection button
        # is waiting for a callback.
        if image_path and not keep_for_crop_selection:
            try:
                os.unlink(image_path)
            except OSError:
                pass


async def crop_selection_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Run crop-restricted ML prediction after the user selects a crop."""
    query = update.callback_query

    try:
        await query.answer()

        if not query.data or not query.data.startswith("crop:"):
            return

        option_key = query.data.split(":", 1)[1]
        options = context.user_data.get("pending_crop_options", {})
        selected_crop = options.get(option_key)

        image_path = context.user_data.get("pending_image_path")
        language = context.user_data.get("pending_language", "en")
        user_id = context.user_data.get("pending_user_id")

        if not selected_crop or not image_path or not os.path.exists(image_path):
            await query.message.reply_text(
                "⚠️ Image session expired. Please send the photo again."
            )
            return

        await query.edit_message_text(
            f"🌱 *Crop selected:* {selected_crop}\n\n"
            "🔄 Analysing the image...",
            parse_mode="Markdown",
        )

        model = get_model()

        if not model.is_model_ready():
            await query.message.reply_text(
                ResponseGenerator.get_error_message(
                    "model_not_ready",
                    language,
                )
            )
            return

        image_array = preprocess_image(image_path)

        if image_array is None:
            await query.message.reply_text(
                ResponseGenerator.get_error_message(
                    "processing_error",
                    language,
                )
            )
            return

        # IMPORTANT: only classes belonging to the selected crop compete.
        prediction = model.predict(
            image_array,
            selected_crop=selected_crop,
        )

        if not prediction or not prediction.get("success"):
            await query.message.reply_text(
                "⚠️ I couldn't analyse this crop with the current model.\n"
                "Please upload a clearer image or select another crop."
            )
            return

        crop = prediction["crop"]
        disease = prediction["disease"]
        confidence = prediction["confidence"]
        confidence_percent = prediction["confidence_percent"]

        logger.info(
            f"✅ Crop-selected prediction: {selected_crop} -> "
            f"{crop} - {disease} ({confidence_percent:.2f}%)"
        )

        if confidence < settings.CONFIDENCE_LOW:
            top_predictions = prediction.get("top_predictions", [])

            lines = [
                "🔎 *Low-Confidence Prediction*",
                "",
                f"🌱 *Selected Crop:* {selected_crop}",
                f"🦠 *Possible Disease:* {disease}",
                f"📊 *Confidence:* {confidence_percent:.2f}%",
                "",
                "*Top alternatives:*",
            ]

            for rank, item in enumerate(top_predictions[:3], start=1):
                class_name = (
                    str(item.get("class", "Unknown"))
                    .replace("___", " – ")
                    .replace("__", " – ")
                    .replace("_", " ")
                )
                item_confidence = float(item.get("confidence", 0.0)) * 100
                lines.append(
                    f"{rank}. {class_name} — {item_confidence:.2f}%"
                )

            lines.extend(
                [
                    "",
                    "⚠️ *This result is uncertain.*",
                    "Please send a clear close-up photo of the affected "
                    "leaf, stem, or fruit in good lighting.",
                ]
            )

            await query.message.reply_text(
                "\n".join(lines),
                parse_mode="Markdown",
            )
            return

        with DatabaseManager() as session:
            response = ResponseGenerator.generate_disease_response(
                session,
                crop,
                disease,
                confidence,
                language,
            )

            interaction = InteractionRepository.create_interaction(
                session,
                user_id,
                crop=crop,
                predicted_disease=disease,
                confidence=confidence,
                response_text=response,
                response_language=language,
                image_filename=os.path.basename(image_path),
            )

            SessionRepository.update_session_context(
                session,
                user_id,
                crop=crop,
                disease=disease,
                last_interaction_id=interaction.id,
            )

        if response:
            await query.message.reply_text(
                response,
                parse_mode="Markdown",
            )
        else:
            await query.message.reply_text(
                ResponseGenerator.get_error_message(
                    "database_error",
                    language,
                )
            )

    except Exception as e:
        logger.error(
            f"❌ Error in crop selection handler: {e}",
            exc_info=True,
        )
        try:
            await query.message.reply_text(
                "⚠️ Something went wrong while analysing the image."
            )
        except Exception:
            pass

    finally:
        image_path = context.user_data.pop("pending_image_path", None)
        context.user_data.pop("pending_language", None)
        context.user_data.pop("pending_user_id", None)
        context.user_data.pop("pending_chat_id", None)
        context.user_data.pop("pending_crop_options", None)

        if image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        text = update.message.text
        logger.info(f"💬 Text received from {user.id}: {text[:50]}")

        # Do not keep a SQLAlchemy ORM object after its session closes.
        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(session, user.id, user.username, user.first_name)
            language = user_obj.language or "en"
            user_id = user_obj.id

        intent = IntentDetector.detect(text)

        with DatabaseManager() as session:
            last_crop, last_disease = SessionRepository.get_last_context(session, user_id)

        if not last_crop or not last_disease:
            msg = "🤔 कृपया पहले एक फसल की तस्वीर भेजें।" if language == "hi" else "🤔 Please send a crop photo first."
            await context.bot.send_message(chat_id=chat_id, text=msg)
            return

        with DatabaseManager() as session:
            if intent == IntentDetector.SYMPTOMS:
                response = ResponseGenerator.generate_symptom_response(session, last_crop, last_disease, language)
            elif intent == IntentDetector.MANAGEMENT:
                response = ResponseGenerator.generate_management_response(session, last_crop, last_disease, language)
            elif intent == IntentDetector.PREVENTION:
                response = ResponseGenerator.generate_prevention_response(session, last_crop, last_disease, language)
            else:
                response = ResponseGenerator.generate_info_response(session, last_crop, last_disease, language)

        await context.bot.send_message(
            chat_id=chat_id,
            text=response if response else ResponseGenerator.get_error_message("database_error", language),
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.error(f"❌ Error in text handler: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ An error occurred. Please try again.")
