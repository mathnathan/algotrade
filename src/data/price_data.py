# src/data/price_data.py
"""
Historical price data model for Alpaca stock bars.

This model represents the exact structure returned by Alpaca's StockHistoricalDataClient.
Each field maps directly to what we receive from the API, ensuring no data loss.

Think of this like a digital ledger entry - every price tick gets recorded
with all the essential market data that traders need for analysis.
"""

from sqlalchemy import Column, String, DateTime, Numeric, BigInteger, Index
from sqlalchemy.sql import func
from .base import Base

class HistoricalPrice(Base):
    """
    Historical stock price data from Alpaca API.
    
    This table stores OHLCV (Open, High, Low, Close, Volume) data plus additional
    market metrics like VWAP (Volume Weighted Average Price) and trade count.
    
    Why these fields matter in algorithmic trading:
    - OHLC: Core price action - the story of each time period
    - Volume: Market interest - high volume = high conviction moves  
    - Trade Count: Market participation - more trades = more liquidity
    - VWAP: Institutional benchmark - where big money thinks fair value is
    """
    __tablename__ = "historical_prices"
    
    # Primary key - auto-incrementing ID for database efficiency
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Stock symbol - this is our main grouping field
    # Index this heavily since we'll filter by symbol constantly
    symbol = Column(String(10), nullable=False, index=True)
    
    # Timestamp with timezone - CRITICAL for time series analysis
    # Alpaca provides UTC timestamps which is perfect for global markets
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLC prices - using Numeric for precise decimal handling
    # Never use Float for financial data! Precision matters when money is involved.
    open = Column(Numeric(precision=12, scale=4), nullable=False)   # $9,999,999.9999 max
    high = Column(Numeric(precision=12, scale=4), nullable=False)   
    low = Column(Numeric(precision=12, scale=4), nullable=False)    
    close = Column(Numeric(precision=12, scale=4), nullable=False)  
    
    # Volume and trade metrics - BigInteger handles large numbers
    volume = Column(BigInteger, nullable=False)           # Total shares traded
    trade_count = Column(BigInteger, nullable=False)      # Number of individual trades
    
    # VWAP - Volume Weighted Average Price (institutional benchmark)
    vwap = Column(Numeric(precision=12, scale=4), nullable=True)
    
    # Audit fields - when we inserted this record  
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Composite indexes for efficient querying
    # These are like database "shortcuts" for common query patterns
    __table_args__ = (
        # Most common query: symbol + time range
        Index('ix_historical_prices_symbol_timestamp', 'symbol', 'timestamp'),
        
        # Time-based queries across all symbols  
        Index('ix_historical_prices_timestamp_symbol', 'timestamp', 'symbol'),
        
        # Unique constraint - prevent duplicate data
        # One price bar per symbol per timestamp
        Index('uq_historical_prices_symbol_timestamp', 'symbol', 'timestamp', unique=True),
    )
    
    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return f"<HistoricalPrice(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"