# src/data/__init__.py
"""
Database models for Alpaca trading data.

These models are Python classes that represent database tables.
Each class attribute becomes a database column with proper types and constraints.
"""

from src.data.base import Base
from src.data.news_data import HistoricalNews
from src.data.price_data import HistoricalPrice

__all__ = ["Base", "HistoricalNews", "HistoricalPrice"]
