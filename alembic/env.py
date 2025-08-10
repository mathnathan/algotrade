"""
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
project_root = r"/workspace"
sys.path.insert(0, project_root)

# Import our models so Alembic can detect schema changes
try:
    from src.data.base import Base
    from src.data.price_data import HistoricalPrice
    from src.data.news_data import HistoricalNews
    target_metadata = Base.metadata
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
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
        dialect_opts={"paramstyle": "named"},
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
