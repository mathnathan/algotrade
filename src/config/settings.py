# src/config/settings.py

from pathlib import Path
from dataclasses import dataclass
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

@dataclass
class ModelConfig:
    price_sequence_length: int = 21
    news_sequence_length: int = 7
    embedding_dim: int = 384
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 2
    dropout_rate: float = 0.1
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

@dataclass
class NewsConfig:
    include_sources: list[str] = None
    exclude_sources: list[str] = None
    news_keywords_macro: list[str] = None
    news_keywords_financial: list[str] = None
    max_news_per_day: int = 50
    min_headline_length: int = 10
    
    def __post_init__(self):
        if self.news_keywords_macro is None:
            self.news_keywords_macro = ["federal reserve", "fed", "inflation"] # Truncated for brevity
        if self.news_keywords_financial is None:
            self.news_keywords_financial = ["earnings", "revenue", "profit"] # Truncated for brevity

@dataclass
class TradingConfig:
    prediction_time_et: str = "12:00"
    entry_time_et: str = "12:30"
    exit_time_et: str = "15:45"
    min_confidence_threshold: float = 0.62
    min_prediction_magnitude: float = 0.005
    base_position_size: float = 0.02
    max_position_size: float = 0.10
    confidence_multiplier: float = 1.5
    volatility_adjustment: bool = True
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.025
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    order_type: str = "market"
    time_in_force: str = "day"

class DatabaseConfig(BaseSettings):
    """Database connection settings that loads secrets from .env."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore" # Ignore other variables in the .env file
    )
    
    # Non-secret defaults
    host: str = "postgres"
    port: int = 5432
    
    # Secrets to be loaded from .env using aliases
    name: str = Field("trading_db", alias="DB_TRADING_DB_NAME")
    user: str = Field(..., alias="DB_TRADING_USER_NAME")
    password: str = Field(..., alias="DB_TRADING_USER_PASSWORD")

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def masked_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:***@{self.host}:{self.port}/{self.name}"

class AlpacaConfig(BaseSettings):
    """Alpaca API settings that loads secrets from .env."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # Secrets loaded from .env
    paper_api_key_id: str = Field(..., alias="APCA_PAPER_API_KEY_ID")
    paper_api_secret_key: str = Field(..., alias="APCA_PAPER_API_SECRET_KEY")
    live_api_key_id: str = Field(..., alias="APCA_LIVE_API_KEY_ID")
    live_api_secret_key: str = Field(..., alias="APCA_LIVE_API_SECRET_KEY")

    # Non-secrets with hardcoded defaults
    paper_base_url: str = "https://paper-api.alpaca.markets/v2"
    live_base_url: str = "https://api.alpaca.markets/v2"


class Settings(BaseSettings):
    """Main application settings orchestrator."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # This triggers each one to load its own environment variables.
    database: DatabaseConfig = DatabaseConfig()
    alpaca: AlpacaConfig = AlpacaConfig()

    # --- Other Application Settings ---
    paper_trading: bool = Field(True, alias="PAPER")
    
    # Huggingface secrets and configs
    huggingface_hub_token: str = Field(..., alias="HUGGINGFACE_HUB_TOKEN")
    huggingface_cache_dir: Path = Path("/app/.cache/huggingface")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Logging, Persistence, and static configs
    log_level: str = "INFO"
    log_file: Path = Path("./logs/trading.log")
    model_save_path: Path = Path("./models")
    checkpoint_frequency: int = 10
    model: ModelConfig = ModelConfig()
    news: NewsConfig = NewsConfig()
    trading: TradingConfig = TradingConfig()


# Global settings instance
settings = Settings()
