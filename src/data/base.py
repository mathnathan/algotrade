# src/data/base.py
"""
Base model configuration for SQLAlchemy.

This is our foundation. It provides common functionality and ensures
consistency across all our data models.
"""

from typing import Any

from pandas import DataFrame
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Define naming convention for indexes and constraints
# This ensures consistent, predictable naming across all database objects
convention = {
    "ix": "ix_%(column_0_label)s",                    # Index naming
    "uq": "uq_%(table_name)s_%(column_0_name)s",      # Unique constraint naming
    "ck": "ck_%(table_name)s_%(constraint_name)s",    # Check constraint naming
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",  # Foreign key naming
    "pk": "pk_%(table_name)s"                         # Primary key naming
}

metadata = MetaData(naming_convention=convention)

class Base(DeclarativeBase):
    """
    Base class for all database models.

    This parent class gives all our data models their database superpowers
    like connection handling, query building, and automatic SQL generation.
    """
    metadata = metadata

    # This allows us to convert model instances to dictionaries easily
    def to_dataframe(self) -> DataFrame:
        """Convert ORM instance to pandas dataframe for convenient access"""
        return DataFrame(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary for easy JSON serialization."""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
