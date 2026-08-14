"""Rule-based conversational chat for the Kisan Telegram Bot.

No LLM or external AI API is used. The handler uses the existing intent,
disease database and session context to make text interactions conversational.
"""

import logging
import re
from telegram import Update
from telegram.ext import ContextTypes

from app.database.database import DatabaseManager
from app.database.repository import UserRepository, SessionRepository
from app.nlp.intent import IntentDetector
from app.responses.generator import ResponseGenerator

logger = logging.getLogger(__name__)


# Common crop names used to detect when a user has started talking about a
# different crop. This prevents stale session context (for example Rice Blast)
# from being reused for a new Tomato question.
CROP_ALIASES = {
    "tomato": "tomato",
    "टमाटर": "tomato",
    "rice": "rice",
    "paddy": "rice",
    "धान": "rice",
    "wheat": "wheat",
    "गेहूं": "wheat",
    "potato": "potato",
    "आलू": "potato",
    "apple": "apple",
    "सेब": "apple",
    "grape": "grape",
    "अंगूर": "grape",
    "pepper": "pepper",
    "chilli": "pepper",
    "chili": "pepper",
    "मिर्च": "pepper",
    "corn": "corn",
    "maize": "corn",
    "मक्का": "corn",
    "soybean": "soybean",
    "soya": "soybean",
    "cotton": "cotton",
    "कपास": "cotton",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _mentioned_crop(text: str) -> str | None:
    """Return the normalized crop explicitly mentioned in the message."""
    t = _normalize(text)
    # Match aliases as words/phrases, including Hindi text.
    for alias, normalized in sorted(CROP_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in t:
            return normalized
    return None


def _simple_chat(text: str, language: str) -> str | None:
    """Return a conversational reply when no disease context is required."""
    t = _normalize(text)

    greetings = {
        "hi", "hello", "hey", "hii", "namaste", "namaskar",
        "नमस्ते", "नमस्कार", "bhai", "good morning", "good afternoon",
        "good evening",
    }
    thanks = {
        "thanks", "thank you", "thankyou", "thx", "dhanyavad",
        "धन्यवाद", "shukriya", "shukriya bhai",
    }
    goodbye = {"bye", "goodbye", "see you", "ok bye", "ठीक है bye"}

    if t in greetings:
        if language == "hi":
            return "🙏 नमस्ते! मैं आपकी फसल और पौधों की बीमारी पहचानने में मदद कर सकता हूँ। फोटो भेजिए या अपना सवाल पूछिए।"
        if language == "hinglish":
            return "🙏 Namaste bhai! Main crop/plant disease identify karne mein help kar sakta hoon. Photo bhejo ya apna sawaal pucho."
        return "🙏 Hello! I can help with crop/plant disease detection. Send a photo or ask me a question."

    if t in thanks or any(x in t for x in ["thank you bhai", "thanks bhai", "shukriya bhai"]):
        return "😊 You're welcome! Agar aur koi crop/disease question ho to pooch sakte ho."

    if t in goodbye:
        return "👍 Theek hai bhai. Jab bhi zarurat ho, message kar dena."

    if t in {"what can you do", "what do you do", "tum kya kar sakte ho", "aap kya kar sakte ho", "kya kar sakte ho"}:
        return (
            "🌱 Main aapki help kar sakta hoon:\n\n"
            "📷 Plant/crop photo analyse karna\n"
            "🦠 Possible disease batana\n"
            "🔍 Symptoms samjhana\n"
            "🛠️ Management/treatment information dena\n"
            "🛡️ Prevention information dena\n\n"
            "Photo bhejo ya seedha sawaal pucho."
        )

    return None


async def chat_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle normal text as a conversational, context-aware message."""
    try:
        user = update.effective_user
        chat_id = update.effective_chat.id
        text = update.message.text or ""

        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(
                session, user.id, user.username, user.first_name
            )
            language = user_obj.language or "en"
            user_id = user_obj.id
            last_crop, last_disease = SessionRepository.get_last_context(
                session, user_id
            )

        quick_reply = _simple_chat(text, language)
        if quick_reply:
            await context.bot.send_message(chat_id=chat_id, text=quick_reply)
            return

        intent = IntentDetector.detect(text)
        mentioned_crop = _mentioned_crop(text)

        # CRITICAL CONTEXT GUARD:
        # If the user explicitly mentions another crop, never answer using
        # the previous crop/disease context. For example, after Rice Blast,
        # "Tomato ke patte pe daag hain" must NOT return Rice Blast information.
        if mentioned_crop and last_crop:
            normalized_last_crop = _normalize(last_crop).replace("_", " ")
            if mentioned_crop not in normalized_last_crop:
                last_crop = None
                last_disease = None

        # A new crop symptom/question without an actual diagnosis should not
        # receive a fabricated disease answer. Ask for a photo instead.
        if mentioned_crop and not last_disease:
            if language == "hi":
                response = (
                    f"🍅 *{mentioned_crop.title()}* ki baat samajh gaya.\n\n"
                    "Patte par daag ka exact reason photo dekhe bina confirm nahi kar sakta. "
                    "Kripya affected leaf ki clear photo bhejiye; main pehle crop select karwaunga "
                    "aur phir available ML model se analyse karunga."
                )
            else:
                response = (
                    f"🌱 I understand you're asking about *{mentioned_crop.title()}*.\n\n"
                    "I can't confirm the disease from text alone. Please send a clear photo "
                    "of the affected leaf; I'll analyse it after you select the crop."
                )
            await context.bot.send_message(
                chat_id=chat_id,
                text=response,
                parse_mode="Markdown",
            )
            return

        # If there is a valid previous diagnosis, answer follow-up questions
        # using that diagnosis as context.
        if last_crop and last_disease:
            with DatabaseManager() as session:
                if intent == IntentDetector.SYMPTOMS:
                    response = ResponseGenerator.generate_symptom_response(
                        session, last_crop, last_disease, language
                    )
                elif intent in (IntentDetector.MANAGEMENT, IntentDetector.TREATMENT):
                    response = ResponseGenerator.generate_management_response(
                        session, last_crop, last_disease, language
                    )
                elif intent == IntentDetector.PREVENTION:
                    response = ResponseGenerator.generate_prevention_response(
                        session, last_crop, last_disease, language
                    )
                elif intent == IntentDetector.CAUSE:
                    disease = ResponseGenerator._get_disease_info(
                        session, last_crop, last_disease
                    )
                    if disease:
                        if language == "hi":
                            response = f"🌱 *{disease.crop} - {disease.disease_name}*\n\n🔬 *कारण:*\n{disease.causes or 'कारण की जानकारी उपलब्ध नहीं है।'}"
                        else:
                            response = f"🌱 *{disease.crop} - {disease.disease_name}*\n\n🔬 *Cause:*\n{disease.causes or 'Cause information is not available.'}"
                    else:
                        response = ResponseGenerator.get_error_message("no_disease_found", language)
                elif intent == IntentDetector.SEVERITY:
                    disease = ResponseGenerator._get_disease_info(
                        session, last_crop, last_disease
                    )
                    if disease:
                        if language == "hi":
                            response = f"⚠️ *गंभीरता:* {disease.severity or 'जानकारी उपलब्ध नहीं है।'}"
                        else:
                            response = f"⚠️ *Severity:* {disease.severity or 'Information is not available.'}"
                    else:
                        response = ResponseGenerator.get_error_message("no_disease_found", language)
                elif intent in (IntentDetector.GENERAL_INFO, IntentDetector.UNKNOWN):
                    response = ResponseGenerator.generate_info_response(
                        session, last_crop, last_disease, language
                    )
                else:
                    response = ResponseGenerator.generate_info_response(
                        session, last_crop, last_disease, language
                    )

            await context.bot.send_message(
                chat_id=chat_id,
                text=response or ResponseGenerator.get_error_message("database_error", language),
                parse_mode="Markdown",
            )
            return

        if intent in (
            IntentDetector.SYMPTOMS,
            IntentDetector.CAUSE,
            IntentDetector.MANAGEMENT,
            IntentDetector.TREATMENT,
            IntentDetector.PREVENTION,
            IntentDetector.DIAGNOSIS,
            IntentDetector.SEVERITY,
        ):
            response = (
                "🌱 I can help with that. Please send the crop/plant photo first, "
                "then I can keep the diagnosis as context for your follow-up questions."
            )
        else:
            response = (
                "😊 Haan bhai, chat kar sakte ho. Crop/disease ke baare mein sawaal pucho "
                "ya photo bhejo, phir main usi context mein aage ke questions ka jawab dunga."
            )

        await context.bot.send_message(chat_id=chat_id, text=response)

    except Exception as e:
        logger.error("❌ Error in conversational chat handler: %s", e, exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ Sorry, I couldn't process that message. Please try again.",
        )
