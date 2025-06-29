from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    postgres_admin_url: str = Field(
        "postgresql://postgres:postgres_admin_password@postgres:5432/postgres",
        env="POSTGRES_ADMIN_URL"
    )
    
    # Alpaca API Configuration
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY") 
    alpaca_base_url: str = Field(
        "https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    
    # Model Configuration
    huggingface_cache_dir: Path = Field(
        Path("./data/models"), 
        env="HUGGINGFACE_CACHE_DIR"
    )
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    
    # Trading Configuration
    position_size_pct: float = Field(0.02, env="POSITION_SIZE_PCT")
    stop_loss_pct: float = Field(0.01, env="STOP_LOSS_PCT")
    take_profit_pct: float = Field(0.015, env="TAKE_PROFIT_PCT")
    trading_symbol: str = Field("SPY", env="TRADING_SYMBOL")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Path = Field(Path("./logs/trading.log"), env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
