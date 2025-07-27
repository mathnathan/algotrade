from sqlalchemy import Column, String, DateTime, Numeric, BigInteger, Index, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.schema import CheckConstraint
import uuid

from src.data.base import Base

class HistoricalPrice(Base):
    """
    Historical price data model with comprehensive indexing and constraints.
    
    This model incorporates optimizations learned from production trading systems
    where query performance directly impacts trading profitability.
    """
    __tablename__ = "historical_prices"
    
    # Primary key using UUID for better distribution in partitioned tables
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Core market data fields with comprehensive constraints
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLC prices with precision optimized for financial calculations
    open = Column(Numeric(precision=15, scale=6), nullable=False)
    high = Column(Numeric(precision=15, scale=6), nullable=False)
    low = Column(Numeric(precision=15, scale=6), nullable=False)
    close = Column(Numeric(precision=15, scale=6), nullable=False)
    
    # Volume metrics with proper sizing for high-volume stocks
    volume = Column(BigInteger, nullable=False)
    trade_count = Column(BigInteger, nullable=False)
    vwap = Column(Numeric(precision=15, scale=6), nullable=True)
    
    # Additional trading metrics for analysis
    dollar_volume = Column(Numeric(precision=20, scale=2), nullable=True)  # price * volume
    average_trade_size = Column(Numeric(precision=15, scale=6), nullable=True)  # volume / trade_count
    
    # Data quality and lineage tracking
    data_source = Column(String(50), nullable=False, default='alpaca')
    data_quality_score = Column(Float, nullable=True)  # 0.0 to 1.0 quality rating
    is_validated = Column(Boolean, nullable=False, default=False)
    
    # Audit fields with automatic timestamping
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Check constraints to ensure data integrity
    __table_args__ = (
        # Basic OHLC constraints
        CheckConstraint('open > 0', name='ck_historical_prices_open_positive'),
        CheckConstraint('high > 0', name='ck_historical_prices_high_positive'),
        CheckConstraint('low > 0', name='ck_historical_prices_low_positive'),
        CheckConstraint('close > 0', name='ck_historical_prices_close_positive'),
        
        # OHLC logical constraints
        CheckConstraint('high >= open', name='ck_historical_prices_high_ge_open'),
        CheckConstraint('high >= close', name='ck_historical_prices_high_ge_close'),
        CheckConstraint('low <= open', name='ck_historical_prices_low_le_open'),
        CheckConstraint('low <= close', name='ck_historical_prices_low_le_close'),
        CheckConstraint('high >= low', name='ck_historical_prices_high_ge_low'),
        
        # Volume constraints
        CheckConstraint('volume >= 0', name='ck_historical_prices_volume_non_negative'),
        CheckConstraint('trade_count >= 0', name='ck_historical_prices_trade_count_non_negative'),
        
        # VWAP constraint
        CheckConstraint('vwap IS NULL OR (vwap >= low AND vwap <= high)', 
                       name='ck_historical_prices_vwap_range'),
        
        # Data quality constraint
        CheckConstraint('data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)',
                       name='ck_historical_prices_quality_score_range'),
        
        # Performance-optimized indexes
        
        # Primary query pattern: symbol + time range
        Index('ix_prices_symbol_timestamp', 'symbol', 'timestamp'),
        
        # Time-series analysis across symbols
        Index('ix_prices_timestamp_symbol', 'timestamp', 'symbol'),
        
        # Unique constraint preventing duplicates
        Index('uq_prices_symbol_timestamp', 'symbol', 'timestamp', unique=True),
        
        # Volume analysis queries
        Index('ix_prices_volume_analysis', 'symbol', 'timestamp', 'volume'),
        
        # Price movement analysis
        Index('ix_prices_price_analysis', 'symbol', 'timestamp', 'close', 'volume'),
        
        # Data quality monitoring
        Index('ix_prices_quality', 'data_quality_score', 'is_validated'),
        
        # Partial index for high-volume trading days (performance optimization)
        Index('ix_prices_high_volume', 'symbol', 'timestamp', 
              postgresql_where='volume > 1000000'),
        
        # Partial index for validated data only
        Index('ix_prices_validated', 'symbol', 'timestamp',
              postgresql_where='is_validated = true'),
    )
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics for analysis."""
        if self.volume and self.vwap:
            self.dollar_volume = self.volume * self.vwap
        
        if self.volume and self.trade_count and self.trade_count > 0:
            self.average_trade_size = self.volume / self.trade_count
    
    def validate_ohlc_integrity(self) -> bool:
        """Validate OHLC data integrity."""
        if not all([self.open, self.high, self.low, self.close]):
            return False
        
        return (self.low <= self.open <= self.high and 
                self.low <= self.close <= self.high)
    
    def __repr__(self) -> str:
        return (f"<HistoricalPrice(symbol='{self.symbol}', "
                f"timestamp='{self.timestamp}', close={self.close}, "
                f"volume={self.volume})>")
