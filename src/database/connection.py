# src/database/connection.py
"""
Database connection management with resilience and monitoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from src.config.database_config import DatabaseConfig
from src.database.monitoring import DatabaseMetrics

logger = logging.getLogger(__name__)

class DatabaseManager:
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.metrics = DatabaseMetrics()
        self._engine = None
        self._session_factory = None
        self._health_task = None
        self._last_health_check = 0
        self._connection_failures = 0
        self._initialized = False
    
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
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
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
            future=True
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info(f"Database engine created: {self.config.masked_url}")
    
    async def _validate_connection(self) -> None:
        """Validate that the database connection works and has expected capabilities."""
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
            result = await conn.execute(text("""
                SELECT extname FROM pg_extension 
                WHERE extname IN ('pg_trgm', 'btree_gin', 'uuid-ossp')
            """))
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
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with comprehensive error handling and metrics.
        
        This context manager tracks session usage patterns and provides
        the reliability guarantees that trading systems require.
        """
        if not self._initialized:
            await self.initialize()
        
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
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        logger.error(f"Operation failed after {self.config.connection_retries} attempts")
        raise last_exception
    
    async def get_pool_status(self) -> dict[str, Any]:
        """Get detailed connection pool status for monitoring."""
        if not self._engine:
            return {"status": "not_initialized"}
        
        pool = self._engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
            "last_health_check": self._last_health_check,
            "connection_failures": self._connection_failures,
            "status": "healthy" if time.time() - self._last_health_check < 120 else "unhealthy"
        }
    
    async def close(self) -> None:
        """Clean shutdown with proper resource cleanup."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        if self._engine:
            await self._engine.dispose()
            
        logger.info("Database manager shut down cleanly")