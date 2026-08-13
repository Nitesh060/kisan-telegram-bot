"""
Database connection and session management
"""

import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# Create engine based on database URL
if "sqlite" in settings.DATABASE_URL:
    # SQLite configuration
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=settings.DEBUG,
    )
    
    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL configuration
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,  # Test connections before using
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _migrate_postgres_schema():
    """Apply small backward-compatible schema fixes required by the current models."""
    if "postgresql" not in settings.DATABASE_URL:
        return

    try:
        with engine.begin() as connection:
            # Telegram user IDs can be larger than PostgreSQL INTEGER (32-bit).
            # Convert an existing INTEGER column to BIGINT so existing deployments
            # are fixed automatically instead of requiring a manual DB migration.
            connection.execute(text("""
                ALTER TABLE users
                ALTER COLUMN telegram_user_id TYPE BIGINT
                USING telegram_user_id::BIGINT
            """))
        logger.info("✅ PostgreSQL schema migration completed: telegram_user_id -> BIGINT")
    except Exception as e:
        # If the column is already BIGINT (or the table is not yet present), do not
        # prevent startup. Base.metadata.create_all() below remains the source of
        # truth for creating missing tables.
        logger.warning(f"⚠️ PostgreSQL schema migration skipped: {e}")


def init_db():
    """Initialize database - create all tables and apply safe schema fixes."""
    try:
        Base.metadata.create_all(bind=engine)
        _migrate_postgres_schema()
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        raise


def get_session() -> Session:
    """Get database session."""
    return SessionLocal()


def close_session(session: Session):
    """Close database session."""
    if session:
        session.close()


class DatabaseManager:
    """Context manager for database sessions."""
    
    def __init__(self):
        self.session: Session = None
    
    def __enter__(self) -> Session:
        self.session = get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()
