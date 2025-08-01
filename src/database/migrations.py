# src/database/migrations.py
"""
Database migration management using Alembic.

This module handles both the initialization of the migration system
and the execution of migrations.
"""

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)


def init_alembic():
    """
    Initialize Alembic migration repository.

    This is like setting up the blueprint filing system for your building project.
    Once initialized, Alembic can track and apply database structure changes
    systematically across all environments.

    This function is idempotent - safe to call multiple times.
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        alembic_dir = project_root / "alembic"
        alembic_ini = project_root / "alembic.ini"

        # Check if already initialized
        if alembic_dir.exists() and alembic_ini.exists():
            logger.info("✅ Alembic already initialized, skipping setup")
            return True

        logger.info("🔧 Initializing Alembic migration system...")

        # Ensure we have a database URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        # Create Alembic configuration
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        # Initialize Alembic repository
        command.init(alembic_cfg, str(alembic_dir))

        # Create a custom env.py that imports our models
        env_py_content = f'''"""
Alembic environment configuration for trading bot.

This file is automatically generated and customized for our SQLAlchemy models.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add the project root to Python path so we can import our models
project_root = r"{project_root}"
sys.path.insert(0, project_root)

# Import our models so Alembic can detect schema changes
from src.models.base import Base
from src.models.price_data import HistoricalPrice
from src.models.news_data import HistoricalNews

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the target metadata for autogenerate support
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={{"paramstyle": "named"}},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

        # Write the custom env.py
        env_py_path = alembic_dir / "env.py"
        with open(env_py_path, "w") as f:
            f.write(env_py_content)

        logger.info("✅ Alembic migration system initialized successfully")
        logger.info(f"📁 Migration repository: {alembic_dir}")
        logger.info(f"⚙️  Configuration file: {alembic_ini}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Alembic: {e}")
        raise


async def run_migrations():
    """
    Run database migrations to latest version.

    This is like following a cookbook to build your database structure.
    Each migration is a recipe step that transforms the database
    from one version to the next.
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        alembic_cfg_path = project_root / "alembic.ini"

        if not alembic_cfg_path.exists():
            logger.warning("⚠️  No alembic.ini found - creating tables directly")
            from src.database.connection import db_manager

            await db_manager.create_tables()
            return

        # Configure Alembic
        alembic_cfg = Config(str(alembic_cfg_path))

        # Run migrations
        logger.info("🔄 Running database migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("✅ Database migrations completed")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


def create_migration(message: str):
    """
    Create a new migration file.

    Use this when you modify the data models and need to generate
    a migration to apply those changes to the database.
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        alembic_cfg_path = project_root / "alembic.ini"

        if not alembic_cfg_path.exists():
            raise FileNotFoundError("Alembic not initialized. Run init_alembic() first.")

        alembic_cfg = Config(str(alembic_cfg_path))
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"✅ Created migration: {message}")

    except Exception as e:
        logger.error(f"❌ Failed to create migration: {e}")
        raise
