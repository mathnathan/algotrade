# alembic/env.py
"""
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
from src.data.base import Base

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
        dialect_opts={"paramstyle": "named"},
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
    database_url = os.getenv("DATABASE_URL")
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

# Determine which mode to run based on context
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
