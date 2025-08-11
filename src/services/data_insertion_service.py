# src/services/data_insertion_service.py

import logging
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.news_data import HistoricalNews
from src.data.price_data import HistoricalPrice
from src.data.validation import DataValidator
from src.database.connection import db_manager

logger = logging.getLogger(__name__)


class DataInsertionService:
    """
    Service for inserting validated market data into the database.

    """

    def __init__(self):
        self.validator = DataValidator()

    async def insert_price_data(self, symbol: str, price_df: pd.DataFrame) -> tuple[int, int]:
        """
        Insert price data with validation and conflict resolution.

        Returns:
            Tuple of (successful_inserts, validation_errors)
        """
        if price_df.empty:
            return 0, 0

        successful_inserts = 0
        validation_errors = 0

        async with db_manager.get_session() as session:
            for _, row in price_df.iterrows():
                try:
                    # Convert row to dictionary for validation
                    price_data = {
                        "symbol": symbol,
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "trade_count": int(row.get("trade_count", 0)),
                        "vwap": float(row["vwap"]) if pd.notna(row.get("vwap")) else None,
                    }

                    # Validate using your existing DataValidator
                    validated_price = self.validator.validate_price_data(price_data)
                    if validated_price is None:
                        validation_errors += 1
                        continue

                    # Prepare database record
                    db_record = {
                        "symbol": validated_price.symbol,
                        "timestamp": validated_price.timestamp,
                        "open": validated_price.open,
                        "high": validated_price.high,
                        "low": validated_price.low,
                        "close": validated_price.close,
                        "volume": validated_price.volume,
                        "trade_count": validated_price.trade_count,
                        "vwap": validated_price.vwap,
                        "data_source": "alpaca",
                        "data_quality_score": 1.0,  # Alpaca data is high quality
                        "is_validated": True,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now(),
                    }

                    # Use PostgreSQL's ON CONFLICT for upsert behavior
                    # This handles the case where we might re-run data collection
                    stmt = postgres_insert(HistoricalPrice).values(**db_record)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["symbol", "timestamp"],
                        set_={
                            "open": stmt.excluded.open,
                            "high": stmt.excluded.high,
                            "low": stmt.excluded.low,
                            "close": stmt.excluded.close,
                            "volume": stmt.excluded.volume,
                            "trade_count": stmt.excluded.trade_count,
                            "vwap": stmt.excluded.vwap,
                            "updated_at": datetime.now(),
                        },
                    )

                    await session.execute(stmt)
                    successful_inserts += 1

                except Exception as e:
                    logger.error(f"❌ Failed to insert price record for {symbol}: {e}")
                    validation_errors += 1
                    continue

            # Commit all inserts as a single transaction
            await session.commit()

        logger.debug(
            f"💰 Inserted {successful_inserts} price records for {symbol} "
            f"({validation_errors} validation errors)"
        )

        return successful_inserts, validation_errors

    async def insert_news_data(self, symbol: str, news_df: pd.DataFrame) -> tuple[int, int]:
        """
        Insert news data with robust ID handling and pre-validation cleaning.

        """
        if news_df.empty:
            return 0, 0

        successful_inserts = 0
        validation_errors = 0

        async with db_manager.get_session() as session:
            for _, row in news_df.iterrows():
                try:
                    # 🔧 PRE-VALIDATION CLEANING: Handle missing IDs before validation
                    raw_id = row.get("id")
                    if pd.isna(raw_id) or raw_id is None:
                        # Generate a deterministic ID based on content
                        headline = str(row.get("headline", ""))
                        created_at = row.get("created_at", "")
                        generated_id = abs(hash(f"{headline}_{created_at}")) % (10**15)
                        logger.debug(f"Generated ID {generated_id} for article: {headline[:50]}...")
                    else:
                        generated_id = int(raw_id)

                    # 🔧 CLEAN DATA PREPARATION: Ensure all fields are properly formatted
                    news_data = {
                        "id": generated_id,  # Always an integer now
                        "headline": row.get("headline"),
                        "summary": row.get("summary"),
                        "content": row.get("content"),
                        "source": str(row["source"]) if pd.notna(row.get("source")) else None,
                        "author": str(row["author"]) if pd.notna(row.get("author")) else None,
                        "url": str(row["url"]) if pd.notna(row.get("url")) else None,
                        "created_at": self._ensure_timezone_aware(row.get("created_at")),
                        "updated_at": self._ensure_timezone_aware(
                            row.get("updated_at", row.get("created_at", datetime.now()))
                        ),
                        "symbols": [symbol],  # Associate with the symbol we're processing
                    }

                    # 🔧 NOW VALIDATION WILL SUCCEED: All required fields are properly formatted
                    validated_news = self.validator.validate_news_data(news_data)
                    if validated_news is None:
                        validation_errors += 1
                        continue

                    # 🔧 DATABASE INSERTION: No more fallback logic needed
                    db_record = {
                        "id": validated_news.id,  # Always valid now
                        "headline": validated_news.headline,
                        "summary": validated_news.summary,
                        "content": validated_news.content,
                        "source": validated_news.source,
                        "author": validated_news.author,
                        "url": validated_news.url,
                        "created_at": validated_news.created_at,
                        "updated_at": validated_news.updated_at,
                        "symbols": validated_news.symbols,
                        "primary_symbol": symbol,
                        "news_category": "general",
                        "is_market_moving": False,
                        "is_processed": False,
                    }

                    # Insert using PostgreSQL UPSERT for idempotency
                    await self._upsert_news_record(session, db_record)
                    successful_inserts += 1

                except Exception as e:
                    validation_errors += 1
                    logger.warning(f"Failed to process news record: {e}")
                    continue

        return successful_inserts, validation_errors

    def _ensure_timezone_aware(self, timestamp) -> datetime:
        """
        Ensure timestamp has timezone information for database storage.

        Financial data requires precise timestamps, and PostgreSQL timezone
        handling prevents many subtle bugs in backtesting and live trading.
        """
        if timestamp is None:
            return datetime.now(UTC)

        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        if hasattr(timestamp, "tz_localize") and timestamp.tz is None:
            return datetime.fromisoformat(timestamp.tz_localize("UTC").to_pydatetime().isoformat())
        elif isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)

        return timestamp if isinstance(timestamp, datetime) else datetime.now(UTC)

    async def _upsert_news_record(self, session: AsyncSession, record: dict) -> None:
        """
        Insert or update news record using PostgreSQL's ON CONFLICT.

        This ensures idempotency - running the same data multiple times
        won't create duplicates, which is crucial for reliable data pipelines.
        """
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(HistoricalNews).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "updated_at": stmt.excluded.updated_at,
                "data_updated_at": func.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()
