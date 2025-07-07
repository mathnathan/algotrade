# src/data/news_data.py
"""
Historical news data model for Alpaca news API.

This model handles the complex structure of financial news data, including
arrays of symbols and images. In algorithmic trading, news sentiment often
drives market movements, so capturing this data properly is crucial.

Think of this as a digital newspaper archive, but structured for machine analysis.
"""

from sqlalchemy import Column, String, Text, DateTime, BigInteger, Index, Float
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func  
from typing import Any
from .base import Base

class HistoricalNews(Base):
    """
    Historical news data from Alpaca API.
    
    Financial news is a critical signal in algorithmic trading because:
    - Breaking news can cause immediate price movements
    - Sentiment analysis helps predict market direction  
    - Multiple symbols per article show market correlations
    - Source credibility affects market ipact
    
    This table captures the full news context
    """
    __tablename__ = "historical_news"
    
    # Use the news ID from Alpaca as primary key - ensures no duplicates
    id = Column(BigInteger, primary_key=True, autoincrement=False)
    
    # Core news content fields
    headline = Column(Text, nullable=False)              # The hook that gets attention
    summary = Column(Text, nullable=True)                # AI-generated summary
    content = Column(Text, nullable=True)                # Full article text (if available)
    
    # Source information - credibility matters in trading
    source = Column(String(100), nullable=True, index=True)  # Reuters, Bloomberg, etc.
    author = Column(String(200), nullable=True)              # Writer attribution
    url = Column(Text, nullable=True)                        # Link to full article
    
    # Timing information - critical for correlation with price movements
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)   # When published
    updated_at = Column(DateTime(timezone=True), nullable=False)               # When last modified
    
    # Symbol associations - PostgreSQL ARRAY type for multiple symbols per article
    # Example: ['AAPL', 'MSFT', 'GOOGL'] for a big tech industry article
    symbols = Column(ARRAY(String(10)), nullable=True, index=True)
    
    # Images - store as JSONB for flexible structure
    # Alpaca provides different image sizes (large, small, thumb) with URLs
    images = Column(JSONB, nullable=True)
    
    # Sentiment analysis field - populated later by our ML pipeline
    # Scale: -1.0 (very negative) to +1.0 (very positive), 0.0 (neutral)
    sentiment = Column(Float, nullable=True, index=True)
    
    # Audit fields
    data_inserted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    data_updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Indexes for efficient querying patterns
    __table_args__ = (
        # Time-based news queries (most common)
        Index('ix_historical_news_created_at', 'created_at'),
        
        # Symbol-specific news lookup  
        # PostgreSQL GIN index for array columns - super efficient for "contains" queries
        Index('ix_historical_news_symbols_gin', 'symbols', postgresql_using='gin'),
        
        # Source credibility filtering
        Index('ix_historical_news_source_created', 'source', 'created_at'),
        
        # Sentiment analysis queries
        Index('ix_historical_news_sentiment_created', 'sentiment', 'created_at'),
        
        # Prevent duplicate news entries
        Index('uq_historical_news_id', 'id', unique=True),
    )
    
    def get_symbol_list(self) -> list[str]:
        """Extract symbols as a clean Python list."""
        return self.symbols if self.symbols is not None else []
    
    def get_images_dict(self) -> dict[str, Any]:
        """Extract images as a clean Python dictionary."""
        return self.images if self.images is not None else {}
    
    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        symbol_str = ', '.join(self.get_symbol_list()[:3])  # Show first 3 symbols
        if len(self.get_symbol_list()) > 3:
            symbol_str += '...'
        return f"<HistoricalNews(id={self.id}, symbols=[{symbol_str}], headline='{self.headline[:50]}...')>"