"""
Kisan Telegram Bot - Main Application
Production-ready Crop Disease Detection Bot
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from app.config import settings
from app.database.database import init_db, get_session
from app.services.disease_service import DiseaseService
from app.bot.handlers import (
    start_handler,
    help_handler,
    about_handler,
    language_handler,
    text_handler,
)
from app.bot.active_learning import (
    image_handler_with_active_learning,
    crop_selection_with_active_learning,
    feedback_handler,
)
from app.bot.telegram_service import TelegramService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

telegram_app: Application = None
telegram_service: TelegramService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Kisan Telegram Bot...")

    try:
        init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    try:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "diseases.csv",
        )
        if os.path.exists(csv_path):
            session = get_session()
            try:
                results = DiseaseService.import_from_csv(
                    session,
                    csv_path,
                    skip_duplicates=False,
                )
                logger.info(
                    "✅ Disease database ready: total=%s, created=%s, updated=%s, errors=%s",
                    results["total"], results["created"], results["updated"], results["errors"],
                )
            finally:
                session.close()
        else:
            logger.warning("⚠️ Disease CSV not found: %s", csv_path)
    except Exception as e:
        logger.error(f"❌ Disease database import failed: {e}")
        raise

    global telegram_app, telegram_service
    try:
        telegram_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        await telegram_app.initialize()

        telegram_app.add_handler(CommandHandler("start", start_handler))
        telegram_app.add_handler(CommandHandler("help", help_handler))
        telegram_app.add_handler(CommandHandler("about", about_handler))
        telegram_app.add_handler(CommandHandler("language", language_handler))

        # Crop selection is handled by the active-learning wrapper, which calls
        # the existing crop prediction and then asks for human verification.
        telegram_app.add_handler(
            CallbackQueryHandler(
                crop_selection_with_active_learning,
                pattern=r"^crop:",
            )
        )

        # Human feedback: correct/wrong/corrected disease.
        telegram_app.add_handler(
            CallbackQueryHandler(
                feedback_handler,
                pattern=r"^feedback_(yes|no|label):",
            )
        )

        telegram_app.add_handler(
            MessageHandler(filters.PHOTO, image_handler_with_active_learning)
        )
        telegram_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
        )

        telegram_service = TelegramService(telegram_app)
        await telegram_app.start()

        if settings.USE_WEBHOOK:
            await telegram_app.bot.set_webhook(
                url=settings.WEBHOOK_URL,
                drop_pending_updates=True,
            )
            logger.info(f"✅ Telegram webhook set to {settings.WEBHOOK_URL}")
        else:
            logger.warning("⚠️ Using polling mode - not recommended for production")

        logger.info("✅ Telegram bot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Telegram bot initialization failed: {e}")
        raise

    yield

    logger.info("🛑 Shutting down Kisan Telegram Bot...")
    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
    logger.info("✅ Bot shutdown complete")


app = FastAPI(
    title="Kisan Crop Disease Detection Bot",
    description="AI-powered crop disease detection without LLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "service": "kisan-telegram-bot",
        "version": "1.0.0",
    }


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> JSONResponse:
    try:
        if not telegram_app:
            logger.error("Telegram app not initialized")
            raise HTTPException(status_code=503, detail="Bot not ready")

        update_data = await request.json()
        update = Update.de_json(update_data, telegram_app.bot)
        await telegram_app.process_update(update)
        return JSONResponse({"ok": True})

    except Exception as e:
        logger.error(f"❌ Webhook error: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/telegram/polling")
async def telegram_polling() -> Dict:
    if not telegram_app:
        raise HTTPException(status_code=503, detail="Bot not ready")

    if settings.USE_WEBHOOK:
        raise HTTPException(
            status_code=400,
            detail="Polling disabled when webhook mode is enabled",
        )

    try:
        await telegram_app.start_polling(allowed_updates=Update.ALL_TYPES)
        return {"status": "polling started"}
    except Exception as e:
        logger.error(f"❌ Polling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/database/health")
async def database_health() -> Dict:
    try:
        session = get_session()
        session.execute("SELECT 1")
        session.close()
        return {"status": "connected", "database": settings.DATABASE_URL}
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return {"status": "disconnected", "error": str(e)}


@app.get("/ml/health")
async def ml_health() -> Dict:
    try:
        from app.ml.inference import ModelInference
        model = ModelInference()
        if model.model is None:
            return {
                "status": "not_loaded",
                "message": "ML model file not found",
                "expected_path": settings.MODEL_PATH,
            }
        return {
            "status": "ready",
            "model_path": settings.MODEL_PATH,
        }
    except Exception as e:
        logger.error(f"ML model check failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/")
async def root() -> Dict:
    return {
        "name": "Kisan Crop Disease Detection Bot",
        "description": "AI-powered crop disease detection for farmers",
        "endpoints": {
            "health": "/health",
            "database": "/database/health",
            "ml_model": "/ml/health",
            "telegram_webhook": "/telegram/webhook",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    if settings.ENVIRONMENT == "development" and not settings.USE_WEBHOOK:
        logger.info("🚀 Starting in development mode with polling...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=settings.PORT,
            reload=True,
        )
    else:
        logger.info("🚀 Starting in production mode with webhook...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=settings.PORT,
        )
