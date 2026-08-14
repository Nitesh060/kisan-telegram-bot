"""Human-in-the-loop active learning workflow for crop disease predictions."""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from app.database.database import DatabaseManager
from app.database.repository import UserRepository, InteractionRepository, FeedbackRepository
from app.ml.inference import ModelInference
from app.config import settings

logger = logging.getLogger(__name__)


async def image_handler_with_active_learning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Remember Telegram's permanent file_id, then run the existing image flow."""
    from app.bot.handlers import image_handler

    try:
        if update.message and update.message.photo:
            context.user_data["active_learning_file_id"] = update.message.photo[-1].file_id
    except Exception:
        logger.exception("Could not store Telegram file_id for active learning")

    return await image_handler(update, context)


async def crop_selection_with_active_learning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run the existing crop prediction, then attach a feedback request."""
    from app.bot.handlers import crop_selection_handler

    await crop_selection_handler(update, context)

    try:
        telegram_user_id = update.effective_user.id
        file_id = context.user_data.get("active_learning_file_id")
        if not file_id:
            return

        with DatabaseManager() as session:
            user = UserRepository.get_or_create_user(
                session,
                telegram_user_id,
                update.effective_user.username,
                update.effective_user.first_name,
            )
            interactions = InteractionRepository.get_user_interactions(session, user.id, limit=1)
            if not interactions:
                return

            interaction = interactions[0]
            if not interaction.predicted_disease or not interaction.crop:
                return

            sample = FeedbackRepository.create_pending_sample(
                session,
                user_id=user.id,
                interaction_id=interaction.id,
                telegram_file_id=file_id,
                crop=interaction.crop,
                model_prediction=interaction.predicted_disease,
                model_confidence=interaction.confidence,
            )
            sample_id = sample.id

        keyboard = [[
            InlineKeyboardButton("✅ Correct", callback_data=f"feedback_yes:{sample_id}"),
            InlineKeyboardButton("❌ Wrong", callback_data=f"feedback_no:{sample_id}"),
        ]]

        await update.callback_query.message.reply_text(
            "🤖 *Was my disease prediction correct?*\n\n"
            "Your feedback helps improve the model.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("Active-learning feedback creation failed")


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle correct/wrong feedback and collect verified labels."""
    query = update.callback_query
    await query.answer()

    try:
        data = query.data or ""
        parts = data.split(":")
        action = parts[0]

        if action == "feedback_yes":
            sample_id = int(parts[1])
            with DatabaseManager() as session:
                sample = FeedbackRepository.get_sample(session, sample_id)
                if not sample:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return
                predicted = sample.model_prediction
                FeedbackRepository.verify_sample(session, sample_id, predicted, status="verified")

            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "✅ Thank you! This image has been added as a *verified training sample*.",
                parse_mode="Markdown",
            )
            return

        if action == "feedback_no":
            sample_id = int(parts[1])
            with DatabaseManager() as session:
                sample = FeedbackRepository.get_sample(session, sample_id)
                if not sample:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return
                crop = sample.crop

            model = ModelInference(settings.MODEL_PATH)
            model.load_class_names(settings.CLASS_NAMES_PATH)
            class_indices = model.get_allowed_class_indices(crop)

            options = {}
            keyboard = []
            row = []
            for key, idx in enumerate(class_indices):
                if idx >= len(model.class_names):
                    continue
                class_name = model.class_names[idx]
                _, disease = model._parse_class_name(class_name)
                options[str(key)] = disease
                row.append(
                    InlineKeyboardButton(
                        disease[:30],
                        callback_data=f"feedback_label:{sample_id}:{key}",
                    )
                )
                if len(row) == 2:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)

            if not keyboard:
                await query.message.reply_text("⚠️ No disease labels are available for this crop.")
                return

            context.user_data[f"feedback_options:{sample_id}"] = options

            await query.edit_message_text(
                f"❌ Thanks for correcting it.\n\n🌱 Crop: *{crop}*\n"
                "Please select the correct disease:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
            return

        if action == "feedback_label":
            sample_id = int(parts[1])
            key = parts[2]
            options = context.user_data.get(f"feedback_options:{sample_id}", {})
            correct_disease = options.get(key)

            if not correct_disease:
                await query.message.reply_text(
                    "⚠️ Feedback options expired. Please send the image again."
                )
                return

            with DatabaseManager() as session:
                sample = FeedbackRepository.get_sample(session, sample_id)
                if not sample:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return
                crop = sample.crop
                predicted = sample.model_prediction
                verified = FeedbackRepository.verify_sample(
                    session,
                    sample_id,
                    correct_disease,
                    status="verified",
                )
                if not verified:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return

            context.user_data.pop(f"feedback_options:{sample_id}", None)
            await query.edit_message_text(
                f"✅ Saved!\n\n🌱 Crop: *{crop}*\n"
                f"🤖 Model predicted: *{predicted}*\n"
                f"👨‍🌾 Verified label: *{correct_disease}*\n\n"
                "This verified image can be used for the next model-training cycle.",
                parse_mode="Markdown",
            )
            return

    except Exception:
        logger.exception("Active-learning feedback handler failed")
        await query.message.reply_text("⚠️ Could not save the feedback. Please try again.")
