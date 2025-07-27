# src/database/connection.py
"""
Enhanced database connection management with proper transaction handling.

This implementation follows SQLAlchemy best practices for async session management,
ensuring that your trading system has rock-solid database reliability - just like
how professional trading firms handle their mission-critical data infrastructure.
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

from src.config.settings import DatabaseConfig, settings
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
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize the database engine with comprehensive error handling.

        Uses a lock to prevent multiple simultaneous initializations, which is
        critical in trading environments where multiple services might try to
        access the database simultaneously during startup.
        """
        async with self._initialization_lock:
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
                    logger.warning(
                        f"🔄 Database initialization attempt {attempt + 1}/{self.config.connection_retries} failed: {type(e).__name__}: {e}"
                    )

                    if attempt < self.config.connection_retries - 1:
                        delay = self.config.retry_delay * (2**attempt)  # Exponential backoff
                        logger.info(f"⏳ Retrying in {delay:.1f} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error("❌ All database initialization attempts failed")
                        raise

    async def _create_engine(self) -> None:
        """
        Create the SQLAlchemy async engine with production-optimized settings.

        These settings are tuned for algorithmic trading where you need:
        - Fast connection recovery (pool_pre_ping=True)
        - Proper connection lifecycle management (pool_recycle)
        - Adequate capacity for concurrent operations (pool_size + max_overflow)
        """
        self._engine = create_async_engine(
            self.config.async_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True,  # Critical for detecting stale connections
            echo=self.config.enable_query_logging,
            future=True,
        )

        # Configure session factory with proper transaction handling
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects accessible after commit
            autoflush=True,  # Auto-flush before queries
            autocommit=False,  # Explicit transaction control
        )

        logger.info(f"🔗 Database engine created: {self.config.masked_url}")

    async def _validate_connection(self) -> None:
        """
        Validate database connection with detailed error reporting.

        This is like checking all your trading systems before market open -
        ensuring everything works correctly before you start processing real money.
        """
        if self._engine is None:
            raise RuntimeError("Database engine is not initialized")

        try:
            async with self._engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1 as connectivity_test"))
                test_value = result.scalar()
                if test_value != 1:
                    raise RuntimeError("Database connectivity test failed - unexpected result")

                # Verify PostgreSQL version compatibility
                result = await conn.execute(text("SELECT version()"))
                version_info = result.scalar()
                logger.info(f"📊 Connected to: {version_info}")

                # Test timestamp handling (critical for trading data)
                result = await conn.execute(text("SELECT NOW() as timestamp_test"))
                timestamp_test = result.scalar()
                logger.info(f"🕒 Database time: {timestamp_test}")

        except Exception as e:
            logger.error(f"❌ Database validation failed: {type(e).__name__}: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with proper transaction lifecycle management.

        This is the heart of your database operations - like the trading desk
        where all your market data operations happen. Each session is isolated
        and properly managed to prevent the transaction conflicts you were seeing.
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
            logger.debug(f"📝 Created new database session: {id(session)}")

            yield session

            # Commit any pending changes if the session is still active
            if session.is_active:
                await session.commit()
                logger.debug(f"✅ Session {id(session)} committed successfully")

        except Exception as e:
            self.metrics.record_session_error()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "session_id": id(session) if session else "unknown",
                "session_active": session.is_active if session else False,
                "session_in_transaction": session.in_transaction() if session else False,
            }

            logger.error(f"💥 Session error: {error_details}")

            # Only rollback if there's an active transaction
            if session and session.in_transaction():
                try:
                    await session.rollback()
                    logger.debug(f"🔄 Session {id(session)} rolled back")
                except Exception as rollback_error:
                    logger.error(f"❌ Rollback failed: {rollback_error}")

            raise

        finally:
            if session:
                await session.close()
                session_duration = time.time() - session_start
                self.metrics.record_session_closed(session_duration)
                logger.debug(f"🔒 Session {id(session)} closed (duration: {session_duration:.3f}s)")

    async def test_connection(self) -> bool:
        """
        Test database connectivity with comprehensive validation.

        Fixed version that doesn't create nested transactions. This method now
        properly tests your database without the transaction conflicts that
        were causing your startup issues.
        """
        try:
            # Initialize if not already done
            if not self._initialized:
                await self.initialize()

            # Test with a clean session (no nested transactions!)
            async with self.get_session() as session:
                # Basic connectivity test
                result = await session.execute(text("SELECT 1 as connectivity_test"))
                test_value = result.scalar()

                if test_value != 1:
                    logger.error(
                        "❌ Database connectivity test failed - basic query returned unexpected result"
                    )
                    return False

                # Test data types relevant to trading operations
                result = await session.execute(
                    text("""
                    SELECT
                        'AAPL'::varchar as symbol_test,
                        150.25::numeric(12,4) as price_test,
                        NOW()::timestamp with time zone as timestamp_test,
                        1000000::bigint as volume_test
                """)
                )

                test_row = result.fetchone()
                if not test_row:
                    logger.error("❌ Data type test failed")
                    return False

                logger.info("✅ Database connection test passed - ready for trading operations")
                return True

        except Exception as e:
            logger.error(f"❌ Database connection test failed: {type(e).__name__}: {e}")
            self.metrics.record_session_error()
            return False

    async def execute_with_retry(self, operation, *args, **kwargs) -> Any:
        """
        Execute database operations with automatic retry logic.

        This provides the resilience that trading systems need - if a database
        operation fails due to temporary issues, it automatically retries with
        exponential backoff, just like how trading algorithms handle temporary
        market data feed interruptions.
        """
        last_exception = None

        for attempt in range(self.config.connection_retries):
            try:
                async with self.get_session() as session:
                    return await operation(session, *args, **kwargs)

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"🔄 Database operation attempt {attempt + 1}/{self.config.connection_retries} failed: {type(e).__name__}: {e}"
                )

                if attempt < self.config.connection_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        logger.error(
            f"❌ Database operation failed after {self.config.connection_retries} attempts"
        )
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Operation failed but no exception was captured")

    async def create_tables(self) -> None:
        """
        Create all database tables and indexes with proper error handling.
        """
        try:
            if not self._engine:
                await self.initialize()

            if self._engine is None:
                raise RuntimeError("Database engine is not initialized")

            # Import here to avoid circular imports
            from src.data.base import Base

            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("✅ Database tables created successfully")

        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {type(e).__name__}: {e}")
            raise

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background task that monitors database health."""
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
        """Perform health check with detailed diagnostics."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                self._last_health_check = time.time()

        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            self.metrics.record_health_check_failure()

    async def get_pool_status(self) -> dict[str, Any]:
        """Get detailed connection pool status for monitoring."""
        if not self._engine:
            return {"status": "not_initialized"}

        pool = self._engine.pool
        return {
            "pool_class": type(pool).__name__,
            "last_health_check": self._last_health_check,
            "connection_failures": self._connection_failures,
            "status": "healthy" if time.time() - self._last_health_check < 120 else "unhealthy",
            "initialized": self._initialized,
        }

    async def close(self) -> None:
        """Clean shutdown with proper resource cleanup."""
        logger.info("🔄 Shutting down database manager...")

        if self._health_task:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task

        if self._engine:
            await self._engine.dispose()

        self._initialized = False
        logger.info("✅ Database manager shut down cleanly")


db_manager = DatabaseManager(settings.database)
