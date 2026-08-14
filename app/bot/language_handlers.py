"""Telegram handlers for multilingual UI and language selection."""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from app.database.database import DatabaseManager
from app.database.repository import UserRepository
from app.services.language_service import SUPPORTED_LANGUAGES, get_text

logger = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user = update.effective_user
        with DatabaseManager() as session:
            user_obj = UserRepository.get_or_create_user(
                session, user.id, user.username, user.first_name
            )
            language = user_obj.language or "en"

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=get_text("greeting", language),
        )
        await language_handler(update, context, silent=True)
    except Exception as exc:
        logger.exception("Error in multilingual start handler: %s", exc)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ Please try again.",
        )


async def language_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    silent: bool = False,
) -> None:
    try:
        buttons = []
        row = []
        for code, name in SUPPORTED_LANGUAGES.items():
            row.append(InlineKeyboardButton(name, callback_data=f"lang:{code}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        language = "en"
        if update.effective_user:
            with DatabaseManager() as session:
                language = UserRepository.get_user_language(
                    session, update.effective_user.id
                )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=get_text("language_prompt", language),
            reply_markup=InlineKeyboardMarkup(buttons),
        )
    except Exception as exc:
        logger.exception("Error in language handler: %s", exc)


async def language_selection_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    query = update.callback_query
    await query.answer()

    try:
        language = query.data.split(":", 1)[1]
        if language not in SUPPORTED_LANGUAGES:
            return

        with DatabaseManager() as session:
            UserRepository.set_user_language(
                session,
                update.effective_user.id,
                language,
            )

        await query.edit_message_text(get_text("language_saved", language))
        await query.message.reply_text(get_text("greeting", language))
    except Exception as exc:
        logger.exception("Error saving language selection: %s", exc)
        await query.message.reply_text("⚠️ Could not change language. Please try again.")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with DatabaseManager() as session:
        language = UserRepository.get_user_language(session, update.effective_user.id)
    text = get_text("help", language)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with DatabaseManager() as session:
        language = UserRepository.get_user_language(session, update.effective_user.id)
    text = get_text("about", language)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)
