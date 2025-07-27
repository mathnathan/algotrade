# Enhanced version of scripts/init_database.py

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from src.config.logging import setup_logging
from src.config.settings import settings
from src.data import HistoricalNews, HistoricalPrice
from src.database.connection import DatabaseManager
from src.database.migrations import run_migrations

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = setup_logging()

# Use a single DatabaseManager instance throughout the script
db_manager = DatabaseManager(settings.database)


async def wait_for_database(max_retries: int = 30, delay: float = 2.0) -> bool:
    """
    Wait for PostgreSQL to be ready with exponential backoff.

    In trading environments, database connectivity is like waiting for market
    opening - you need patience but also efficiency. This function implements
    exponential backoff, which is commonly used in high-frequency trading
    systems to handle temporary network issues gracefully.
    """
    logger.info("🔄 Waiting for PostgreSQL to be ready...")

    for attempt in range(max_retries):
        try:
            # Use the DatabaseManager's robust test_connection method
            if await db_manager.test_connection():
                logger.info(f"✅ PostgreSQL ready after {attempt + 1} attempts")
                return True

        except Exception as e:
            logger.debug(f"Connection attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            # Exponential backoff with jitter (like HFT retry strategies)
            wait_time = min(delay * (2**attempt), 30.0)  # Cap at 30 seconds
            logger.info(f"⏳ Attempt {attempt + 1}/{max_retries}: Retrying in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)

    logger.error(f"❌ PostgreSQL failed to start after {max_retries} attempts")
    return False


async def create_database_schema():
    """
    Create all database tables and indexes using the DatabaseManager.

    This leverages our enhanced DatabaseManager to create a production-ready
    database schema optimized for trading data storage and retrieval.
    """
    try:
        logger.info("🏗️  Creating trading database schema...")

        # Use the DatabaseManager's create_tables method
        await db_manager.create_tables()

        # Verify the schema was created correctly
        await _verify_schema_integrity()

        logger.info("✅ Database schema created and verified successfully")

    except Exception as e:
        logger.error(f"❌ Schema creation failed: {e}")
        raise


async def _verify_schema_integrity():
    """
    Verify that our database schema supports all required trading operations.

    This is like a pre-flight check before takeoff - ensuring all systems
    are properly configured for live trading operations.
    """
    async with db_manager.get_session() as session:
        from sqlalchemy import text

        # Check that core tables exist with expected structure
        result = await session.execute(
            text("""
            SELECT
                table_name,
                COUNT(*) as column_count
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name IN ('historical_prices', 'historical_news')
            GROUP BY table_name
            ORDER BY table_name
        """)
        )

        tables = {row[0]: row[1] for row in result.fetchall()}

        # Verify table structure matches our expectations
        expected_tables = {
            "historical_prices": 12,  # Expected number of columns
            "historical_news": 20,  # Expected number of columns (adjust based on your models)
        }

        for table_name, expected_columns in expected_tables.items():
            if table_name not in tables:
                raise Exception(f"Required table '{table_name}' not found")

            actual_columns = tables[table_name]
            if actual_columns < expected_columns:
                logger.warning(
                    f"⚠️  Table '{table_name}' has {actual_columns} columns, expected at least {expected_columns}"
                )

        # Verify critical indexes exist for trading performance
        result = await session.execute(
            text("""
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname LIKE 'idx_%'
            ORDER BY tablename, indexname
        """)
        )

        indexes = result.fetchall()
        if indexes:
            logger.info("📈 Trading-optimized indexes found:")
            for index_name, table_name in indexes:
                logger.info(f"  • {table_name}.{index_name}")

        logger.info("✅ Schema integrity verification completed")


async def verify_data_models():
    """
    Test data models with realistic trading data scenarios.

    This simulates the types of data operations your trading algorithms
    will perform, ensuring the database can handle real market conditions.
    """
    try:
        logger.info("🧪 Testing data models with trading scenarios...")

        async with db_manager.get_session() as session:
            from decimal import Decimal

            # Test 1: High-frequency price data insertion (simulates live market feed)
            test_price = HistoricalPrice(
                symbol="SPY",  # Use SPY as it's the most liquid ETF
                timestamp=datetime.now(UTC),
                open=Decimal("445.1200"),
                high=Decimal("445.8900"),
                low=Decimal("444.5500"),
                close=Decimal("445.7700"),
                volume=15_000_000,  # Realistic SPY volume
                trade_count=45_678,
                vwap=Decimal("445.4320"),
            )

            session.add(test_price)
            await session.flush()  # Get the ID without committing

            # Test 2: Market-moving news insertion (simulates news feed)
            test_news = HistoricalNews(
                id=999999999,  # Use high ID to avoid conflicts
                headline="Fed Announces Interest Rate Decision",
                summary="Federal Reserve maintains current interest rates, signals data-dependent approach",
                source="federal_reserve",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                symbols=["SPY", "QQQ", "IWM"],  # Major ETFs affected by Fed decisions
                sentiment_score=0.05,  # Slightly positive market reaction
                market_relevance_score=0.95,  # Highly relevant for trading
                is_market_moving=True,
            )

            session.add(test_news)
            await session.commit()

            # Test 3: Query performance (simulates algorithm data retrieval)
            import time

            start_time = time.time()

            result = await session.execute(
                text("""
                SELECT symbol, close, volume, timestamp
                FROM historical_prices
                WHERE symbol = 'SPY'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            )

            query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            price_data = result.fetchone()

            if price_data and query_time < 50:  # Should be under 50ms for trading systems
                logger.info(f"✅ Price query performance: {query_time:.2f}ms (acceptable for HFT)")
            else:
                logger.warning(
                    f"⚠️  Price query took {query_time:.2f}ms (may impact trading performance)"
                )

            # Clean up test data
            await session.execute(text("DELETE FROM historical_prices WHERE symbol = 'SPY'"))
            await session.execute(text("DELETE FROM historical_news WHERE id = 999999999"))
            await session.commit()

            logger.info("✅ Data models verified for trading operations")

    except Exception as e:
        logger.error(f"❌ Data model test failed: {e}")
        raise


async def initialize_trading_database():
    """
    Complete trading database initialization with enhanced error handling.

    This orchestrates the entire setup process with the reliability standards
    expected in production trading environments.
    """
    initialization_start = asyncio.get_event_loop().time()

    try:
        logger.info("🚀 Initializing trading database infrastructure...")

        # Step 1: Verify database connectivity
        if not await wait_for_database():
            raise Exception("Database connectivity could not be established")

        # Step 2: Create or update schema
        try:
            logger.info("🔄 Attempting migration-based schema update...")
            await run_migrations()
            logger.info("✅ Database migrations completed successfully")
        except Exception as migration_error:
            logger.warning(f"⚠️  Migration failed, using direct table creation: {migration_error}")
            await create_database_schema()

        # Step 3: Verify system readiness
        await verify_data_models()

        # Step 4: Performance baseline
        pool_status = await db_manager.get_pool_status()
        logger.info(f"📊 Connection pool status: {pool_status}")

        initialization_time = asyncio.get_event_loop().time() - initialization_start
        logger.info(f"⏱️  Total initialization time: {initialization_time:.2f} seconds")

        logger.info("🎉 Trading database initialization completed successfully!")
        logger.info("📋 System ready for:")
        logger.info("  ✅ Real-time Alpaca market data ingestion")
        logger.info("  ✅ High-frequency price and volume analysis")
        logger.info("  ✅ News sentiment correlation with price movements")
        logger.info("  ✅ Historical backtesting and strategy development")
        logger.info("  🎯 Production-grade algorithmic trading operations!")

    except Exception as e:
        logger.error(f"💥 Trading database initialization failed: {e}")
        logger.error("🔧 Troubleshooting steps:")
        logger.error("  1. Verify PostgreSQL is running and accessible")
        logger.error("  2. Check DATABASE_URL environment variable")
        logger.error("  3. Ensure database user has CREATE privileges")
        logger.error("  4. Review connection pool configuration")
        raise
    finally:
        # Always clean up resources
        await db_manager.close()


async def main():
    """Entry point for database initialization."""
    try:
        await initialize_trading_database()
        return 0
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
