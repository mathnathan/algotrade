# src/config/settings.py

from pathlib import Path
from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



@dataclass
class ModelConfig:
    """Configuration for the deep learning model architecture and training."""
    
    # Sequence and embedding parameters
    price_sequence_length: int = 21  # Trading days (~1 month) - optimal for daily predictions
    news_sequence_length: int = 7    # Days of news to consider
    embedding_dim: int = 384         # Sentence transformer output dimension
    hidden_dim: int = 256           # Transformer hidden dimension
    num_attention_heads: int = 8    # Multi-head attention
    num_encoder_layers: int = 4     # Transformer encoder depth
    num_decoder_layers: int = 2     # Transformer decoder depth
    dropout_rate: float = 0.1       # Regularization
    
    # Multi-task learning parameters
    regression_weight: float = 1.0   # Weight for price prediction loss
    classification_weight: float = 1.0  # Weight for direction prediction loss
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

@dataclass 
class NewsConfig:
    """Configuration for news data collection and filtering."""
    
    # News source filtering - these categories are most predictive for SPY
    include_sources: list[str] = None  # None means all sources
    exclude_sources: list[str] = None
    
    # News type filtering for SPY (broad market index)
    news_keywords_macro: list[str] = None  # Macro-economic news
    news_keywords_financial: list[str] = None  # Financial sector news
    max_news_per_day: int = 50        # Prevent information overload
    min_headline_length: int = 10     # Filter out very short headlines
    
    def __post_init__(self):
        if self.news_keywords_macro is None:
            # These keywords capture macro sentiment that drives SPY
            self.news_keywords_macro = [
                "federal reserve", "fed", "inflation", "interest rate", "gdp", 
                "unemployment", "jobs report", "economic outlook", "recession",
                "monetary policy", "fiscal policy", "treasury", "dollar", "trump",
                "musk", "biden", "geopolitical", "china", "europe", "ukraine",
                "oil prices", "commodity prices", "supply chain", "global economy",
                "trade war", "tariffs", "sanctions", "central bank", "interest rates",
                "bond yields", "stock market", "equity markets", "market volatility",
                "market correction", "bull market", "bear market", "market sentiment",
                "market trends", "economic indicators", "consumer confidence", "futures market",
                "market analysis", "investment outlook", "financial stability", "credit markets",
                "capital markets", "asset prices", "wealth management", "financial crisis",
                "financial regulation", "financial markets", "market dynamics", "market forces",
                "market participants", "market trends", "market performance", "market outlook",
                "market news", "market updates", "market commentary", "market analysis",
                "market research", "market intelligence", "market insights", "market forecasts",
                "market predictions", "market expectations", "market developments", "market events",
                "market movements", "market fluctuations", "market trends", "market cycles",
                "market corrections", "market rallies", "market downturns", "market rebounds",
                "market volatility", "market risks", "market opportunities", "market challenges",
                "market strategies", "market positioning", "market sentiment analysis", "elon",
                "elon musk", "tesla", "spacex", "twitter", "x", "social media",
                "technology sector", "tech stocks", "big tech"
            ]
        
        if self.news_keywords_financial is None:
            # Financial sector news that affects overall market sentiment
            self.news_keywords_financial = [
                "earnings", "revenue", "profit", "guidance", "outlook",
                "market", "trading", "wall street", "nasdaq", "dow jones",
                "s&p 500", "stock market", "equities", "bonds", "commodities",
                "oil", "gold", "silver", "crypto", "bitcoin", "ethereum",
                "banking", "finance", "investment", "portfolio", "hedge fund",
                "private equity", "venture capital", "mergers and acquisitions",
                "ipo", "spac", "etf", "index fund", "mutual fund", "dividend",
            ]

@dataclass
class TradingConfig:
    """Configuration for trading strategy and risk management."""


    # Prediction and execution timing
    prediction_time_et: str = "12:00"    # When to make daily prediction
    entry_time_et: str = "12:30"         # When to enter position  
    exit_time_et: str = "15:45"          # When to close position (15 min before close)
    
    # Confidence and thresholds
    min_confidence_threshold: float = 0.62  # Based on research, 60-65% is optimal starting point
    min_prediction_magnitude: float = 0.005  # Minimum 0.5% predicted move to trade
    
    # Position sizing (Kelly Criterion inspired)
    base_position_size: float = 0.02     # 2% of portfolio per trade
    max_position_size: float = 0.10      # Never risk more than 10%
    confidence_multiplier: float = 1.5   # Scale position by confidence
    volatility_adjustment: bool = True   # Adjust for recent volatility
    
    # Risk management
    stop_loss_pct: float = 0.015         # 1.5% stop loss
    take_profit_pct: float = 0.025       # 2.5% take profit
    max_daily_loss_pct: float = 0.05     # 5% maximum daily portfolio loss
    max_drawdown_pct: float = 0.15       # 15% maximum drawdown before stopping
    
    # Order execution
    order_type: str = "market"           # Start with market orders for simplicity
    time_in_force: str = "day"           # Day orders only

class DatabaseConfig(BaseSettings):
    """
    Database configuration with intelligent defaults for trading environments.
    
    This configuration class handles all the complexity of database URLs,
    connection parameters, and environment-specific settings in one place.
    """
    
    # Core connection parameters
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    database: str = Field("trading_db", env="DB_DATABASE")
    username: str = Field("trading_user", env="DB_USERNAME")
    password: str = Field(..., env="DB_PASSWORD")  # Required
    
    # Connection pool settings optimized for trading workloads
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(15, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    
    # Retry and resilience settings
    connection_retries: int = Field(3, env="DB_CONNECTION_RETRIES")
    retry_delay: float = Field(1.0, env="DB_RETRY_DELAY")
    health_check_interval: int = Field(60, env="DB_HEALTH_CHECK_INTERVAL")
    
    # Performance and monitoring settings
    enable_query_logging: bool = Field(False, env="DB_ENABLE_QUERY_LOGGING")
    slow_query_threshold: float = Field(0.1, env="DB_SLOW_QUERY_THRESHOLD")
    enable_metrics: bool = Field(True, env="DB_ENABLE_METRICS")
    
    @validator('password')
    def password_must_be_secure(cls, v):
        if len(v) < 12:
            raise ValueError('Database password must be at least 12 characters for security')
        return v
    
    @validator('pool_size')
    def pool_size_reasonable(cls, v):
        if v > 50:
            raise ValueError('Pool size over 50 may consume excessive resources')
        return v
    
    @property
    def async_url(self) -> str:
        """Generate the async database URL with proper escaping and driver specification."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def masked_url(self) -> str:
        """Generate a URL safe for logging with password masked."""
        return f"postgresql+asyncpg://{self.username}:***@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    database: DatabaseConfig = DatabaseConfig()
    
    # Dynamically construct the database URL
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.trading_db_user}:"
            f"{self.trading_db_password}@"
            f"postgres:"
            f"5432/"
            f"{self.trading_db_name}"
        )
    
    # Alpaca API Configuration
    # PAPER CONFIGURATION
    apca_paper_api_key_id: str = Field(..., env="APCA_PAPER_API_KEY_ID")
    apca_paper_api_secret_key: str = Field(..., env="APCA_PAPER_API_SECRET_KEY") 
    apca_paper_base_url: str = Field("https://paper-api.alpaca.markets/v2")

    # LIVE CONFIGURATION
    apca_live_api_key_id: str = Field(..., env="APCA_LIVE_API_KEY_ID")
    apca_live_api_secret_key: str = Field(..., env="APCA_LIVE_API_SECRET_KEY") 
    apca_live_base_url: str = Field("https://api.alpaca.markets/v2")

    # Alpaca Trading Configuration
    paper: bool = Field(True, env="PAPER")  # Use paper trading by default
    
    # Model Configuration
    huggingface_cache_dir: Path = Field(env="HF_HOME")
    huggingface_hub_token: str = Field(env="HUGGINGFACE_HUB_TOKEN")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    
    # Trading Configuration
    model: ModelConfig = ModelConfig()
    news: NewsConfig = NewsConfig()
    trading: TradingConfig = TradingConfig()

    # Logging Configuration
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./logs/trading.log"))

    # Model persistence
    model_save_path: Path = Field(Path("./models"))
    checkpoint_frequency: int = Field(10)  # Save every N epochs
    
    # Save configurations
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

# Global settings instance
settings = Settings()
