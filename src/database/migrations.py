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

from src.config.settings import settings

logger = logging.getLogger(__name__)

def init_alembic():
    """
    Initialize Alembic migration repository without attempting autogenerate.
    
    This is like setting up your trading platform infrastructure - we create
    the framework but don't try to execute trades until everything is ready.
    
    Key insight: We separate initialization from migration generation because
    they have different operational requirements (sync vs async).
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

        # Use SYNC URL for migrations (this is the critical fix!)
        database_url = settings.database.sync_url
        if not database_url:
            raise ValueError("Sync database URL not available")

        # Create Alembic configuration with sync URL
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        # Initialize Alembic repository
        command.init(alembic_cfg, str(alembic_dir))

        # Create enhanced env.py that handles both sync and async scenarios
        env_py_content = create_enhanced_env_py(project_root)

        # Write the enhanced env.py
        env_py_path = alembic_dir / "env.py"
        with open(env_py_path, "w") as f:
            f.write(env_py_content)

        logger.info("✅ Alembic migration system initialized successfully")
        logger.info(f"📁 Migration repository: {alembic_dir}")
        logger.info(f"⚙️  Configuration file: {alembic_ini}")
        
        # Note: We do NOT create an initial migration here!
        # That will be done later with the proper sync connection.

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Alembic: {e}")
        raise


def create_enhanced_env_py(project_root: Path) -> str:
    """
    Create an enhanced env.py that intelligently handles sync/async URLs.
    
    This is like having a smart trading system that adapts its connection
    strategy based on the type of operation it needs to perform.
    """
    return f'''"""
Enhanced Alembic environment for trading systems.

This configuration intelligently handles both sync and async database URLs,
ensuring migrations work correctly regardless of your application's 
connection strategy.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the project root to Python path
project_root = r"{project_root}"
sys.path.insert(0, project_root)

# Import our models so Alembic can detect schema changes
try:
    from src.data.base import Base
    from src.data.price_data import HistoricalPrice
    from src.data.news_data import HistoricalNews
    target_metadata = Base.metadata
except ImportError as e:
    print(f"Warning: Could not import models: {{e}}")
    target_metadata = None

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_sync_database_url():
    """
    Intelligently determine the correct sync database URL.
    
    This function converts async URLs to sync URLs automatically,
    ensuring migrations always work regardless of your app configuration.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        database_url = config.get_main_option("sqlalchemy.url")
    
    # Convert async URLs to sync URLs for migrations
    if database_url and "postgresql+asyncpg" in database_url:
        database_url = database_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        print(f"🔄 Converted async URL to sync for migrations")
    
    return database_url


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_sync_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={{"paramstyle": "named"}},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode with enhanced error handling."""
    # Use our intelligent URL detection
    database_url = get_sync_database_url()
    if database_url:
        config.set_main_option("sqlalchemy.url", database_url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No connection pooling for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''


def create_initial_migration():
    """
    Create the initial migration after Alembic is properly initialized.
    
    This is like placing your first trade after your trading platform is
    fully set up and connected. We separate this from initialization to
    ensure everything is working before we attempt database introspection.
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        alembic_cfg_path = project_root / "alembic.ini"

        if not alembic_cfg_path.exists():
            raise FileNotFoundError("Alembic not initialized. Run init_alembic() first.")

        # Create configuration with sync URL
        alembic_cfg = Config(str(alembic_cfg_path))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.sync_url)

        # Now we can safely create the initial migration
        command.revision(
            alembic_cfg,
            message="Initial trading database schema with historical data tables",
            autogenerate=True
        )
        
        logger.info("✅ Initial migration created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to create initial migration: {e}")
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