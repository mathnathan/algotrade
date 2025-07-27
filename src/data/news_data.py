# src/data/news_data.py

from sqlalchemy import Column, String, DateTime, BigInteger, Index, Text, Float, Boolean
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func
from sqlalchemy.schema import CheckConstraint
from typing import Any

from src.data.base import Base

class HistoricalNews(Base):
    """
    Historical news data model with improved indexing and validation.
    
    This model optimizes news data storage and retrieval for sentiment analysis
    and correlation with price movements.
    """
    __tablename__ = "historical_news"
    
    # Use Alpaca's news ID as primary key
    id = Column(BigInteger, primary_key=True, autoincrement=False)
    
    # Core news content with size constraints
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    
    # Source credibility and metadata
    source = Column(String(100), nullable=True, index=True)
    author = Column(String(200), nullable=True)
    url = Column(Text, nullable=True)
    
    # Timestamp fields with timezone awareness
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    
    # Symbol associations with indexing
    symbols = Column(ARRAY(String(10)), nullable=True)
    primary_symbol = Column(String(10), nullable=True, index=True)  # Most relevant symbol
    
    # Structured image metadata
    images = Column(JSONB, nullable=True)
    
    # Sentiment analysis results
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    sentiment_magnitude = Column(Float, nullable=True)  # 0.0 to 1.0 (confidence)
    sentiment_model_version = Column(String(50), nullable=True)
    
    # News classification and filtering
    news_category = Column(String(50), nullable=True, index=True)  # 'earnings', 'merger', 'general', etc.
    market_relevance_score = Column(Float, nullable=True)  # 0.0 to 1.0
    is_market_moving = Column(Boolean, nullable=False, default=False)
    
    # Data quality and processing status
    is_processed = Column(Boolean, nullable=False, default=False)
    processing_errors = Column(Text, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    
    # Audit fields
    data_inserted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    data_updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        # Data quality constraints
        CheckConstraint('sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)',
                       name='ck_news_sentiment_range'),
        CheckConstraint('sentiment_magnitude IS NULL OR (sentiment_magnitude >= 0 AND sentiment_magnitude <= 1)',
                       name='ck_news_magnitude_range'),
        CheckConstraint('market_relevance_score IS NULL OR (market_relevance_score >= 0 AND market_relevance_score <= 1)',
                       name='ck_news_relevance_range'),
        CheckConstraint('data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)',
                       name='ck_news_quality_range'),
        
        # Content validation
        CheckConstraint("headline IS NOT NULL AND trim(headline) != ''",
                       name='ck_news_headline_not_empty'),
        
        # Performance-optimized indexes
        
        # Time-based news queries (most common pattern)
        Index('ix_news_created_at', 'created_at'),
        
        # Symbol-specific news lookup with GIN index for array efficiency
        Index('ix_news_symbols_gin', 'symbols', postgresql_using='gin'),

        # Primary symbol queries (for focused analysis)
        Index('ix_news_primary_symbol_time', 'primary_symbol', 'created_at'),

        # Source credibility analysis
        Index('ix_news_source_time', 'source', 'created_at'),
        
        # Sentiment analysis queries
        Index('ix_news_sentiment_analysis', 'sentiment_score', 'sentiment_magnitude', 'created_at'),

        # Market-moving news identification
        Index('ix_news_market_moving', 'is_market_moving', 'created_at'),
        
        # News category analysis
        Index('ix_news_category_time', 'news_category', 'created_at'),
        
        # Processing status tracking
        Index('ix_news_processing_status', 'is_processed', 'data_inserted_at'),

        # Data quality monitoring
        Index('ix_news_quality_monitoring', 'data_quality_score', 'is_processed'),

        # Composite index for sentiment correlation analysis
        Index('ix_news_sentiment_correlation', 'primary_symbol', 'created_at',
              'sentiment_score', 'market_relevance_score'),
        
        # Partial index for high-quality, market-relevant news
        Index('ix_news_high_quality', 'primary_symbol', 'created_at',
              postgresql_where='is_market_moving = true AND data_quality_score > 0.8'),
    )
    
    def get_symbol_list(self) -> list[str]:
        """Get symbols as a clean Python list."""
        return self.symbols if self.symbols is not None else []
    
    def get_images_dict(self) -> dict[str, Any]:
        """Get images as a clean Python dictionary."""
        return self.images if self.images is not None else {}
    
    def calculate_content_metrics(self) -> dict[str, Any]:
        """Calculate content-based metrics for analysis."""
        metrics = {
            'headline_length': len(self.headline) if self.headline else 0,
            'summary_length': len(self.summary) if self.summary else 0,
            'content_length': len(self.content) if self.content else 0,
            'symbol_count': len(self.get_symbol_list()),
            'has_images': bool(self.get_images_dict())
        }
        
        # Calculate content richness score
        content_score = 0
        if metrics['headline_length'] > 20:
            content_score += 0.2
        if metrics['summary_length'] > 100:
            content_score += 0.3
        if metrics['content_length'] > 500:
            content_score += 0.3
        if metrics['has_images']:
            content_score += 0.1
        if metrics['symbol_count'] > 0:
            content_score += 0.1
        
        metrics['content_richness_score'] = content_score
        return metrics
    
    def __repr__(self) -> str:
        symbol_str = ', '.join(self.get_symbol_list()[:2])
        if len(self.get_symbol_list()) > 2:
            symbol_str += '...'
        return (f"<HistoricalNews(id={self.id}, "
                f"symbols=[{symbol_str}], "
                f"headline='{self.headline[:40]}...')>")