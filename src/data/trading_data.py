# src/data/trading_data.py

import uuid

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from src.data.base import Base


class TradingSession(Base):
    """
    Model for tracking trading sessions and system performance.

    This table helps monitor the algorithmic trading system's performance
    and provides audit trails for trading decisions.
    """

    __tablename__ = "trading_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Session identification
    session_name = Column(String(100), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    session_start = Column(DateTime(timezone=True), nullable=False, default=func.now())
    session_end = Column(DateTime(timezone=True), nullable=True)

    # Performance metrics
    total_trades = Column(BigInteger, nullable=False, default=0)
    successful_trades = Column(BigInteger, nullable=False, default=0)
    total_pnl = Column(Numeric(precision=15, scale=2), nullable=False, default=0)
    max_drawdown = Column(Numeric(precision=15, scale=2), nullable=True)

    # System performance metrics
    avg_query_time = Column(Float, nullable=True)
    max_query_time = Column(Float, nullable=True)
    total_database_errors = Column(BigInteger, nullable=False, default=0)

    # Session metadata
    configuration = Column(JSONB, nullable=True)  # Strategy parameters
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_trading_sessions_strategy_time", "strategy_name", "session_start"),
        Index("ix_trading_sessions_performance", "total_pnl", "session_start"),
    )
