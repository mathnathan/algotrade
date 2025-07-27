# src/database/connection.py
"""
Database connection management with resilience and monitoring.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import DatabaseConfig
from src.database.monitoring import DatabaseMetrics

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.metrics = DatabaseMetrics()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker | None = None
        self._health_task: asyncio.Task | None = None
        self._last_health_check: float = 0
        self._connection_failures: int = 0
        self._initialized: bool = False

    async def initialize(self) -> None:
        """
        Initialize the database engine with comprehensive error handling.

        This method implements the retry logic and validation needed for
        reliable startup in production trading environments.
        """
        if self._initialized:
            return

        for attempt in range(self.config.connection_retries):
            try:
                await self._create_engine()
                await self._validate_connection()
                await self._start_health_monitoring()
                self._initialized = True
                logger.info(f"✅ Database initialized successfully on attempt {attempt + 1}")
                return

            except Exception as e:
                self._connection_failures += 1
                logger.warning(f"Database initialization attempt {attempt + 1} failed: {e}")

                if attempt < self.config.connection_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All database initialization attempts failed")
                    raise

    async def _create_engine(self) -> None:
        """Create the SQLAlchemy async engine with optimized settings."""
        self._engine = create_async_engine(
            self.config.async_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True,  # Validates connections before use
            echo=self.config.enable_query_logging,
            future=True,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info(f"Database engine created: {self.config.masked_url}")

    async def _validate_connection(self) -> None:
        """Validate that the database connection works and has expected capabilities."""
        if self._engine is None:
            raise RuntimeError("Database engine is not initialized")
        async with self._engine.begin() as conn:
            # Test basic connectivity
            result = await conn.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            if test_value != 1:
                raise RuntimeError("Database connectivity test failed")

            # Verify PostgreSQL version compatibility
            result = await conn.execute(text("SELECT version()"))
            version_info = result.scalar()
            logger.info(f"Connected to: {version_info}")

            # Verify required extensions are available
            result = await conn.execute(
                text("""
                SELECT extname FROM pg_extension
                WHERE extname IN ('pg_trgm', 'btree_gin', 'uuid-ossp')
            """)
            )
            extensions = [row[0] for row in result.fetchall()]
            logger.info(f"Available extensions: {extensions}")

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background task that monitors database health and connection pool status."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.metrics.record_health_check_failure()

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check of database and connection pool."""
        start_time = time.time()

        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))

            # Record successful health check
            check_duration = time.time() - start_time
            self.metrics.record_health_check_success(check_duration)
            self._last_health_check = time.time()

        except Exception as e:
            self.metrics.record_health_check_failure()
            logger.warning(f"Health check failed: {e}")

    async def test_connection(self) -> bool:
        """
        Test database connectivity with comprehensive validation.

        In algorithmic trading, database availability is like checking if the stock exchange
        is open before placing trades. We need to know immediately if our data pipeline
        can function properly.

        This method provides more than just a ping - it validates that our trading system
        can actually execute the types of operations it needs for market data storage.
        """
        try:
            # Initialize the connection if not already done
            if not self._initialized:
                await self.initialize()

            # Test actual database operations that our trading system will need
            async with self.get_session() as session:
                # Basic connectivity test
                result = await session.execute(text("SELECT 1 as connectivity_test"))
                test_value = result.scalar()

                if test_value != 1:
                    logger.error(
                        "Database connectivity test failed - basic query returned unexpected result"
                    )
                    return False

                # Test transaction handling (critical for trade execution integrity)
                async with session.begin():
                    await session.execute(text("SELECT NOW() as transaction_test"))

                # Test that we can handle the data types we'll use for market data
                await session.execute(
                    text("""
                    SELECT
                        'AAPL'::varchar as symbol_test,
                        150.25::numeric(12,4) as price_test,
                        NOW()::timestamp with time zone as timestamp_test,
                        1000000::bigint as volume_test
                """)
                )

                logger.info("✅ Database connection test passed - ready for market data operations")
                return True

        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            self.metrics.record_session_error()
            return False

    async def create_tables(self) -> None:
        """
        Create all database tables and indexes defined in our SQLAlchemy models.

        Think of this as building the trading floor infrastructure. Just like how
        the NYSE needs specific trading stations, communication systems, and data
        feeds before opening for business, our trading system needs its database
        schema properly established before it can store and analyze market data.

        This method uses SQLAlchemy's metadata system to create tables that match
        our Python model definitions - ensuring type safety and proper indexing
        for high-frequency data operations.
        """
        try:
            if not self._initialized:
                await self.initialize()

            if self._engine is None:
                raise RuntimeError("Database engine not initialized")

            logger.info("🏗️  Creating trading database schema...")

            # Import our models to ensure they're registered with the metadata
            from src.data import Base

            # Use async connection for table creation
            async with self._engine.begin() as conn:
                # Create all tables defined in our models
                await conn.run_sync(Base.metadata.create_all)

                logger.info("✅ Database tables created successfully")

                # Log what tables were created for verification
                result = await conn.execute(
                    text("""
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name IN ('historical_prices', 'historical_news')
                    ORDER BY table_name, ordinal_position
                """)
                )

                tables_info = result.fetchall()
                if tables_info:
                    logger.info("📊 Created tables for market data storage:")
                    current_table = None
                    for row in tables_info:
                        table_name, column_name, data_type, is_nullable = row
                        if table_name != current_table:
                            logger.info(f"  📋 {table_name}:")
                            current_table = table_name
                        nullable_indicator = "NULL" if is_nullable == "YES" else "NOT NULL"
                        logger.info(f"    • {column_name} ({data_type}) {nullable_indicator}")

                # Create indexes for optimal query performance
                await self._create_trading_indexes(conn)

        except Exception as e:
            logger.error(f"❌ Table creation failed: {e}")
            raise

    async def _create_trading_indexes(self, conn) -> None:
        """
        Create specialized indexes optimized for trading data queries.

        In trading systems, query speed directly impacts profitability. These indexes
        are like having express lanes on a highway - they ensure that common queries
        (like "get all SPY prices from the last hour") execute in milliseconds rather
        than seconds.
        """
        try:
            # Index for time-series price queries (most common in trading algorithms)
            await conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_timestamp
                ON historical_prices (symbol, timestamp DESC)
            """)
            )

            # Index for volume analysis (important for liquidity assessment)
            await conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_historical_prices_volume_analysis
                ON historical_prices (symbol, timestamp DESC, volume)
                WHERE volume > 0
            """)
            )

            # Index for news sentiment analysis queries
            await conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_historical_news_sentiment_time
                ON historical_news (created_at DESC, sentiment_score)
                WHERE sentiment_score IS NOT NULL
            """)
            )

            # Index for symbol-specific news searches
            await conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_historical_news_symbols_gin
                ON historical_news USING GIN (symbols)
            """)
            )

            logger.info("✅ Trading-optimized indexes created successfully")

        except Exception as e:
            logger.warning(f"⚠️  Some indexes may already exist or failed to create: {e}")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with comprehensive error handling and metrics.

        This context manager tracks session usage patterns and provides
        the reliability guarantees that trading systems require.
        """
        if not self._initialized:
            await self.initialize()

        if self._session_factory is None:
            raise RuntimeError("Session factory is not initialized")

        session_start = time.time()
        session = None

        try:
            session = self._session_factory()
            self.metrics.record_session_created()
            yield session

        except Exception as e:
            self.metrics.record_session_error()
            if session:
                await session.rollback()
            logger.error(f"Session error: {e}")
            raise

        finally:
            if session:
                await session.close()
                session_duration = time.time() - session_start
                self.metrics.record_session_closed(session_duration)

    async def execute_with_retry(self, operation, *args, **kwargs) -> Any:
        """
        Execute database operations with automatic retry logic.

        This method provides resilience for critical trading operations
        that must succeed even during temporary database issues.
        """
        last_exception = None

        for attempt in range(self.config.connection_retries):
            try:
                async with self.get_session() as session:
                    return await operation(session, *args, **kwargs)

            except Exception as e:
                last_exception = e
                logger.warning(f"Operation attempt {attempt + 1} failed: {e}")

                if attempt < self.config.connection_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        logger.error(f"Operation failed after {self.config.connection_retries} attempts")
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Operation failed but no exception was captured")

    async def get_pool_status(self) -> dict[str, Any]:
        """Get detailed connection pool status for monitoring."""
        if not self._engine:
            return {"status": "not_initialized"}

        pool = self._engine.pool
        # Async pools do not expose size/checkedin/checkedout/overflow methods
        return {
            "pool_class": type(pool).__name__,
            "last_health_check": self._last_health_check,
            "connection_failures": self._connection_failures,
            "status": "healthy" if time.time() - self._last_health_check < 120 else "unhealthy",
        }

    async def close(self) -> None:
        """Clean shutdown with proper resource cleanup."""
        if self._health_task:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task

        if self._engine:
            await self._engine.dispose()

        logger.info("Database manager shut down cleanly")
