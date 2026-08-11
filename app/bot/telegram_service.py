"""
Telegram service for bot operations
"""

import logging
from typing import Optional
from telegram import Update, Chat
from telegram.ext import Application

logger = logging.getLogger(__name__)


class TelegramService:
    """Service layer for Telegram operations."""
    
    def __init__(self, app: Application):
        """Initialize Telegram service."""
        self.app = app
        self.bot = app.bot
    
    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "Markdown",
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """
        Send text message to user.
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            parse_mode: Markdown or HTML
            reply_to_message_id: Reply to message ID
        
        Returns:
            True if successful
        """
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_to_message_id=reply_to_message_id,
            )
            logger.info(f"✅ Message sent to {chat_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send message to {chat_id}: {e}")
            return False
    
    async def download_file(self, file_id: str, file_path: str) -> bool:
        """
        Download file from Telegram.
        
        Args:
            file_id: Telegram file ID
            file_path: Local file path to save
        
        Returns:
            True if successful
        """
        try:
            file = await self.bot.get_file(file_id)
            await file.download_to_drive(file_path)
            logger.info(f"✅ File downloaded: {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download file {file_id}: {e}")
            return False
    
    async def get_user_info(self, update: Update) -> dict:
        """
        Extract user information from update.
        
        Returns:
            Dict with user information
        """
        user = update.effective_user
        return {
            "user_id": user.id if user else None,
            "username": user.username if user else None,
            "first_name": user.first_name if user else None,
            "last_name": user.last_name if user else None,
        }
    
    async def get_chat_id(self, update: Update) -> Optional[int]:
        """Get chat ID from update."""
        return update.effective_chat.id if update.effective_chat else None
    
    async def send_photo(
        self,
        chat_id: int,
        photo_path: str,
        caption: str = "",
        parse_mode: str = "Markdown",
    ) -> bool:
        """Send photo message."""
        try:
            with open(photo_path, "rb") as photo:
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=caption,
                    parse_mode=parse_mode,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to send photo to {chat_id}: {e}")
            return False
    
    def get_user_display_name(self, update: Update) -> str:
        """Get user display name."""
        user = update.effective_user
        if user.first_name:
            return user.first_name
        elif user.username:
            return user.username
        else:
            return f"User {user.id}"
