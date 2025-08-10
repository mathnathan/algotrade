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
    Initialize Alembic migration repository with proper model integration.
    
    This function ensures that Alembic can see and track your trading data models
    by creating a customized env.py that imports your specific schema.
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        alembic_dir = project_root / "alembic"
        alembic_ini = project_root / "alembic.ini"

        # Check if already initialized (both directory AND working env.py must exist)
        if alembic_dir.exists() and alembic_ini.exists():
            env_py_path = alembic_dir / "env.py"
            if env_py_path.exists():
                # Quick check that env.py has our model imports
                env_content = env_py_path.read_text()
                if "from src.data.base import Base" in env_content:
                    logger.info("✅ Alembic already initialized with proper model integration")
                    return True
        
        logger.info("🔧 Initializing Alembic migration system...")

        # Get database URL from your settings
        from src.config.settings import settings
        database_url = settings.database.async_url
        if not database_url:
            raise ValueError("Database URL could not be constructed from settings")

        # Create Alembic configuration with the ini file path
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        # Let Alembic create the basic structure first
        logger.info("🔧 Creating Alembic directory structure...")
        command.init(alembic_cfg, str(alembic_dir))
        
        # Now customize the env.py file to work with your trading models
        logger.info("🔧 Customizing env.py for trading data models...")
        env_py_path = alembic_dir / "env.py"
        
        # Create our custom env.py that knows about your models
        custom_env_content = f'''"""
Alembic environment configuration for trading bot.

This file configures how database migrations are executed, ensuring
your trading system's database schema evolves safely and predictably.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the project root to Python path so we can import our models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our models so Alembic can detect schema changes
# This is the critical step that populates Base.metadata with your table definitions
from src.data.base import Base
from src.data.price_data import HistoricalPrice
from src.data.news_data import HistoricalNews
# Add imports for any other model files you create in src.data

# This is the Alembic Config object, which provides access to configuration values
config = context.config

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the target metadata for 'autogenerate' support
# This tells Alembic what your ideal database structure should look like
target_metadata = Base.metadata


def run_migrations_offline():
    """
    Run migrations in 'offline' mode.

    This generates SQL scripts without connecting to a database.
    Useful for generating migration files that can be reviewed before execution.
    """
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
    """
    Run migrations in 'online' mode.

    Creates a database connection and executes migrations directly.
    This is what happens when your trading system starts up.
    """
    # Override the database URL with environment variable if present
    # This ensures migrations use the same connection as your trading application
    database_url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if database_url:
        config.set_main_option("sqlalchemy.url", database_url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No connection pooling for migrations
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


# Determine which mode to run based on context
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

        # Write our custom env.py file
        with open(env_py_path, 'w') as f:
            f.write(custom_env_content)

        logger.info("✅ Created customized env.py with trading model integration")

        # Now create the initial migration that captures your current models
        logger.info("🔧 Creating initial migration to capture trading schema...")
        
        # The imports in env.py will be executed when this command runs,
        # which will populate Base.metadata with your model definitions
        command.revision(
            alembic_cfg,
            message="Initial trading database schema with historical data tables",
            autogenerate=True
        )
        
        logger.info("✅ Created initial migration capturing all trading tables")
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
