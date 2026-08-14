"""Human-in-the-loop active learning workflow with Cloudinary image storage."""

import logging
import os
from urllib.parse import urlparse, unquote
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from app.database.database import DatabaseManager
from app.database.repository import UserRepository, InteractionRepository, FeedbackRepository
from app.ml.inference import ModelInference
from app.config import settings

logger = logging.getLogger(__name__)


def _cloudinary_credentials():
    """Read Cloudinary credentials, preferring the official CLOUDINARY_URL."""
    cloudinary_url = os.getenv("CLOUDINARY_URL")

    if cloudinary_url:
        try:
            parsed = urlparse(cloudinary_url.strip())
            if parsed.scheme != "cloudinary" or not parsed.hostname:
                raise ValueError("CLOUDINARY_URL must start with cloudinary://")

            cloud_name = parsed.hostname.strip()
            api_key = unquote(parsed.username or "").strip()
            api_secret = unquote(parsed.password or "").strip()

            if not all([cloud_name, api_key, api_secret]):
                raise ValueError("CLOUDINARY_URL is missing cloud name, API key, or API secret")

            return cloud_name, api_key, api_secret
        except Exception as exc:
            logger.error("Invalid CLOUDINARY_URL configuration: %s", exc)
            return None, None, None

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    return (
        cloud_name.strip() if cloud_name else None,
        api_key.strip() if api_key else None,
        api_secret.strip() if api_secret else None,
    )


def _cloudinary_upload(file_path: str, crop: str, disease: str, file_id: str):
    """Upload a verified training image into crop/disease folders in Cloudinary."""
    try:
        import cloudinary
        import cloudinary.uploader

        cloud_name, api_key, api_secret = _cloudinary_credentials()

        if not all([cloud_name, api_key, api_secret]):
            logger.error(
                "Cloudinary credentials missing. Configure either "
                "CLOUDINARY_CLOUD_NAME/CLOUDINARY_API_KEY/CLOUDINARY_API_SECRET "
                "or CLOUDINARY_URL."
            )
            return None

        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )

        logger.info(
            "☁️ Cloudinary configured: cloud=%s api_key_suffix=%s",
            cloud_name,
            api_key[-4:] if len(api_key) >= 4 else "****",
        )

        safe_crop = "_".join(str(crop).strip().lower().split())
        safe_disease = "_".join(str(disease).strip().lower().split())
        safe_file_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(file_id)
        )

        # Cloudinary product environments can use dynamic folders. In that mode,
        # slashes inside public_id do NOT necessarily create Media Library folders.
        # asset_folder explicitly controls the visible folder structure.
        asset_folder = f"kisan-bot/verified/{safe_crop}/{safe_disease}"

        result = cloudinary.uploader.upload(
            file_path,
            asset_folder=asset_folder,
            public_id=safe_file_id,
            resource_type="image",
            tags=["verified", "training", safe_crop, safe_disease],
            context={
                "crop": str(crop),
                "disease": str(disease),
                "source": "telegram_verified_feedback",
            },
            overwrite=False,
        )

        logger.info(
            "✅ Cloudinary upload successful: public_id=%s asset_folder=%s",
            result.get("public_id"),
            result.get("asset_folder", asset_folder),
        )
        return result.get("secure_url")

    except ImportError:
        logger.error("Cloudinary package is not installed")
        return None
    except Exception as exc:
        logger.exception(
            "❌ Cloudinary upload failed (%s): %s",
            type(exc).__name__,
            str(exc),
        )
        return None


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
    """Run existing crop prediction and create a feedback sample."""
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
    """Handle feedback; verified images are uploaded to Cloudinary."""
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
                crop = sample.crop
                file_id = sample.telegram_file_id
                FeedbackRepository.verify_sample(
                    session, sample_id, predicted, status="verified"
                )

            image_url = await _download_and_upload_verified(
                context, file_id, crop, predicted
            )

            if image_url:
                message = (
                    "✅ Thank you! The image has been verified and securely saved "
                    "to the training dataset."
                )
            else:
                message = (
                    "✅ Thank you! The label is verified and saved. "
                    "⚠️ The image could not be uploaded to Cloudinary; "
                    "check the Cloudinary configuration."
                )

            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(message)
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
                row.append(InlineKeyboardButton(
                    disease[:30],
                    callback_data=f"feedback_label:{sample_id}:{key}",
                ))
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
                await query.message.reply_text("⚠️ Feedback options expired. Please send the image again.")
                return

            with DatabaseManager() as session:
                sample = FeedbackRepository.get_sample(session, sample_id)
                if not sample:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return

                crop = sample.crop
                predicted = sample.model_prediction
                file_id = sample.telegram_file_id
                verified = FeedbackRepository.verify_sample(
                    session, sample_id, correct_disease, status="verified"
                )
                if not verified:
                    await query.message.reply_text("⚠️ Feedback session expired.")
                    return

            image_url = await _download_and_upload_verified(
                context, file_id, crop, correct_disease
            )

            context.user_data.pop(f"feedback_options:{sample_id}", None)

            if image_url:
                await query.edit_message_text(
                    f"✅ Saved!\n\n🌱 Crop: *{crop}*\n"
                    f"🤖 Model predicted: *{predicted}*\n"
                    f"👨‍🌾 Verified label: *{correct_disease}*\n\n"
                    "📸 Image uploaded to the verified training dataset.",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text(
                    f"✅ Label verified!\n\n🌱 Crop: *{crop}*\n"
                    f"🤖 Model predicted: *{predicted}*\n"
                    f"👨‍🌾 Verified label: *{correct_disease}*\n\n"
                    "⚠️ Image upload failed. Please check Cloudinary settings.",
                    parse_mode="Markdown",
                )
            return

    except Exception:
        logger.exception("Active-learning feedback handler failed")
        await query.message.reply_text("⚠️ Could not save the feedback. Please try again.")


async def _download_and_upload_verified(context, file_id: str, crop: str, disease: str):
    """Download a Telegram file temporarily and upload it to Cloudinary."""
    import tempfile

    temp_path = None
    try:
        telegram_file = await context.bot.get_file(file_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name

        await telegram_file.download_to_drive(temp_path)
        return _cloudinary_upload(temp_path, crop, disease, file_id)

    except Exception as exc:
        logger.exception(
            "Could not download Telegram image for Cloudinary upload (%s): %s",
            type(exc).__name__,
            str(exc),
        )
        return None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
