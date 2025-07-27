# src/config/settings.py

from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
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

class DatabaseConfig(BaseModel):
    """Database connection settings."""
    host: str = "postgres"
    port: int = 5432
    name: str = Field("trading_db", alias="DB_TRADING_DB_NAME")
    user: str = Field(..., alias="DB_TRADING_USER_NAME")
    password: str = Field(..., alias="DB_TRADING_USER_PASSWORD")

    @property
    def async_url(self) -> str:
        """Generate the async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def masked_url(self) -> str:
        """Generate a URL safe for logging."""
        return f"postgresql+asyncpg://{self.user}:***@{self.host}:{self.port}/{self.name}"

class AlpacaConfig(BaseModel):
    """Alpaca API settings."""
    # Secrets loaded from .env
    paper_api_key_id: str = Field(..., alias="APCA_PAPER_API_KEY_ID")
    paper_api_secret_key: str = Field(..., alias="APCA_PAPER_API_SECRET_KEY")
    live_api_key_id: str = Field(..., alias="APCA_LIVE_API_KEY_ID")
    live_api_secret_key: str = Field(..., alias="APCA_LIVE_API_SECRET_KEY")

    # Non-secrets with hardcoded defaults
    paper_base_url: str = "https://paper-api.alpaca.markets/v2"
    live_base_url: str = "https://api.alpaca.markets/v2"

class Settings(BaseSettings):
    """
    Main application settings.
    Loads secrets from .env and defines all other configuration directly.
    """
    
    # Configuration for pydantic-settings to load from a .env file
    # extra='ignore' is crucial: it prevents errors if .env contains variables
    # not defined in this model (e.g., PAPER=true, comments).
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Nested Configuration Models ---
    # Pydantic will automatically populate these nested models
    # using the aliases defined within them.
    database: DatabaseConfig
    alpaca: AlpacaConfig

    # --- Other Application Settings (non-secrets with defaults) ---
    paper: bool = Field(True, alias="PAPER")
    
    # Huggingface secrets and configs
    huggingface_hub_token: str = Field(..., alias="HUGGINGFACE_HUB_TOKEN")
    huggingface_cache_dir: Path = Path("/app/.cache/huggingface")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Path = Path("./logs/trading.log")

    # Model Persistence
    model_save_path: Path = Path("./models")
    checkpoint_frequency: int = 10

    # Static Configurations (using the dataclasses from above)
    model: ModelConfig = ModelConfig()
    news: NewsConfig = NewsConfig()
    trading: TradingConfig = TradingConfig()

# Global settings instance, ready to be imported across the application
settings = Settings()