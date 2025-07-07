# src/database/__init__.py
"""
Database connection and session management.

This module provides the foundation for all database operations.
Think of it as the 'phone system' that lets your Python code 
talk to the PostgreSQL database.
"""

from src.database.connection import DatabaseManager, get_db_session
from src.database.migrations import run_migrations

__all__ = ["DatabaseManager", "get_db_session", "run_migrations"]