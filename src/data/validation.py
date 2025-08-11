# src/data/validation.py
"""
Comprehensive data validation for financial market data.
This ensures that only valid, sensible market data reaches your trading algorithms.
"""

# Python imports
import datetime
import logging
from decimal import Decimal
from typing import Any, ClassVar

# Third-party imports
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ValidatedPrice(BaseModel):
    """
    Pydantic model for validating historical price data from Alpaca API.

    This model ensures that price data makes logical sense within financial
    market constraints before it reaches your database.
    """

    symbol: str = Field(..., pattern=r"^[A-Z]{1,10}$", description="Valid stock symbol")
    timestamp: datetime.datetime = Field(..., description="Price observation timestamp")
    open: Decimal = Field(..., ge=0, description="Opening price")
    high: Decimal = Field(..., ge=0, description="High price")
    low: Decimal = Field(..., ge=0, description="Low price")
    close: Decimal = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    trade_count: int = Field(..., ge=0, description="Number of trades")
    vwap: Decimal | None = Field(None, ge=0, description="Volume weighted average price")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_have_timezone(cls, v):
        """Ensure all timestamps include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @field_validator("high")
    @classmethod
    def high_must_be_highest(cls, v, info):
        """Validate that high price is actually the highest price of the period."""
        values = info.data
        if "open" in values and v < values["open"]:
            raise ValueError("High price cannot be less than open price")
        if "low" in values and v < values["low"]:
            raise ValueError("High price cannot be less than low price")
        if "close" in values and v < values["close"]:
            raise ValueError("High price cannot be less than close price")
        return v

    @field_validator("low")
    @classmethod
    def low_must_be_lowest(cls, v, info):
        """Validate that low price is actually the lowest price of the period."""
        values = info.data
        if "open" in values and v > values["open"]:
            raise ValueError("Low price cannot be greater than open price")
        if "high" in values and v > values["high"]:
            raise ValueError("Low price cannot be greater than high price")
        if "close" in values and v > values["close"]:
            raise ValueError("Low price cannot be greater than close price")
        return v

    @field_validator("vwap")
    @classmethod
    def vwap_must_be_reasonable(cls, v, info):
        """Validate that VWAP falls within the high-low range."""
        if v is None:
            return v
        values = info.data
        if "high" in values and "low" in values and not (values["low"] <= v <= values["high"]):
            raise ValueError("VWAP must fall between low and high prices")
        return v

    @field_validator("volume")
    @classmethod
    def volume_must_be_reasonable(cls, v):
        """Validate that volume is within reasonable bounds for equity markets."""
        if v > 10_000_000_000:  # 10 billion shares seems unreasonable
            raise ValueError("Volume appears unreasonably high")
        return v

    class Config:
        # Allow decimal types to be properly handled
        json_encoders: ClassVar[dict] = {
            Decimal: lambda v: float(v),
            datetime.datetime: lambda v: v.isoformat(),
        }


class ValidatedNews(BaseModel):
    """
    Pydantic model for validating news data from Alpaca API.

    This ensures that news data has all required fields and makes logical sense
    before being stored in your database.
    """

    id: int = Field(..., description="Unique news article ID")
    headline: str = Field(..., min_length=1, max_length=1000, description="Article headline")
    summary: str | None = Field(None, max_length=5000, description="Article summary")
    content: str | None = Field(None, max_length=50000, description="Full article content")
    source: str | None = Field(None, max_length=100, description="News source")
    author: str | None = Field(None, max_length=200, description="Article author")
    url: str | None = Field(None, description="Link to full article")
    created_at: datetime.datetime = Field(..., description="Publication timestamp")
    updated_at: datetime.datetime = Field(..., description="Last modification timestamp")
    symbols: list[str] | None = Field(None, description="Associated stock symbols")
    images: dict[str, Any] | None = Field(None, description="Associated images")
    sentiment: float | None = Field(None, ge=-1.0, le=1.0, description="Sentiment score")

    @field_validator("headline", "summary", "content")
    @classmethod
    def clean_text_fields(cls, v):
        """
        Clean text fields but don't reject empty strings at field level.

        """
        if v is None:
            return None

        # Clean the text but allow empty strings to pass through
        cleaned = str(v).strip()

        # Return None for empty strings to simplify business logic
        return cleaned if cleaned else None

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamps_must_have_timezone(cls, v):
        """Ensure all timestamps include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamps must include timezone information")
        return v

    @field_validator("symbols")
    @classmethod
    def symbols_must_be_valid(cls, v):
        """Validate that stock symbols follow expected format."""
        if v is None:
            return v

        for symbol in v:
            if not isinstance(symbol, str) or not symbol.isupper() or len(symbol) > 10:
                raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v):
        """Basic URL validation for news links."""
        if v is None:
            return v

        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        return v

    @model_validator(mode="after")
    def validate_content_requirements(self):
        """
        Business rule validation: (headline OR summary OR content) must be present.

        Model validators run after all field validators have passed, ensuring we're
        working with clean, properly formatted data.
        """
        headline = self.headline
        summary = self.summary
        content = self.content

        # Check if we have at least one meaningful piece of content
        has_headline = headline is not None and len(headline.strip()) > 0
        has_summary = summary is not None and len(summary.strip()) > 0
        has_content = content is not None and len(content.strip()) > 0

        if not (has_headline or has_summary or has_content):
            raise ValueError(
                "Article must have at least a non-empty headline OR a non-empty summary. "
                f"Got headline: '{headline}', summary: '{summary}'"
            )

        # Additional business rule: If we only have a summary, it should be substantial
        if not has_headline and has_summary and len(summary.strip()) < 10:
            raise ValueError("If headline is missing, summary must be at least 10 characters long")

        return self

    @model_validator(mode="after")
    def validate_timestamp_logic(self):
        """
        Business rule: updated_at should not be before created_at.

        """
        if self.updated_at < self.created_at:
            raise ValueError("Updated timestamp cannot be before created timestamp")

        return self


class DataValidator:
    """
    Service class for validating and cleaning market data before database insertion.

    This provides a centralized location for all validation logic and makes it
    easy to add new validation rules as your trading system evolves.
    """

    def __init__(self):
        self.validation_errors = []
        self.processed_count = 0
        self.error_count = 0

    def validate_price_data(self, raw_data: dict[str, Any]) -> ValidatedPrice | None:
        """
        Validate and clean price data from Alpaca API.

        Returns None if validation fails, otherwise returns validated price data.
        """
        try:
            validated = ValidatedPrice(**raw_data)
            self.processed_count += 1
            return validated

        except Exception as e:
            self.error_count += 1
            error_msg = f"Price validation failed for {raw_data.get('symbol', 'unknown')}: {e}"
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            return None

    def validate_news_data(self, raw_data: dict[str, Any]) -> ValidatedNews | None:
        """
        Validate and clean news data from Alpaca API.

        Returns None if validation fails, otherwise returns validated news data.
        """
        try:
            validated = ValidatedNews(**raw_data)
            self.processed_count += 1
            return validated

        except Exception as e:
            self.error_count += 1
            error_msg = f"News validation failed for article {raw_data.get('id', 'unknown')}: {e}"
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            return None

    def get_validation_stats(self) -> dict[str, Any]:
        """Get statistics about validation performance."""
        return {
            "total_processed": self.processed_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.processed_count, 1),
            "recent_errors": self.validation_errors[-10:],  # Last 10 errors
        }

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_errors.clear()
        self.processed_count = 0
        self.error_count = 0
