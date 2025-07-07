# src/database/connection.py (Fixed with lazy initialization)
"""
Database connection management with lazy initialization to avoid import-time conflicts.
"""

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text, make_url
from typing import AsyncGenerator, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager with lazy initialization to avoid driver conflicts.
    
    We delay the actual engine creation until we're sure about the connection 
    parameters, avoiding an import-time driver selection issues.
    """
    
    def __init__(self):
        # Don't create the engine immediately - wait until first use
        self._engine: Optional = None
        self._async_session: Optional = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """
        Initialize the database engine on first use (lazy initialization).
        """
        if self._initialized:
            return
        
        try:
            # Import settings here to avoid circular import issues
            from src.config.settings import settings
            
            # Create the properly formatted async URL
            async_url = self._create_async_url(settings.database_url)
            logger.info(f"🔄 Initializing database engine with: {self._mask_password(async_url)}")
            
            # Create the async engine with the corrected URL
            self._engine = create_async_engine(
                async_url,
                
                # Connection pool settings optimized for trading workloads
                pool_size=20,           # Core connections always ready
                max_overflow=30,        # Extra connections during busy periods  
                pool_pre_ping=True,     # Validate connections before use
                pool_recycle=3600,      # Refresh connections every hour
                
                # Performance and debugging settings
                future=True,           # Use SQLAlchemy 2.0+ features
            )
            
            # Create session factory
            self._async_session = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False  # Keep objects usable after commit
            )
            
            self._initialized = True
            logger.info("✅ Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database engine: {e}")
            raise
    
    def _create_async_url(self, original_url: str) -> str:
        """
        Convert any PostgreSQL URL to use the asyncpg driver explicitly.
        """
        try:
            # Parse the original URL to understand its components
            parsed_url = make_url(original_url)
            logger.debug(f"🔍 Original driver: {parsed_url.drivername}")
            
            # Check if it's already using the async driver
            if parsed_url.drivername == 'postgresql+asyncpg':
                logger.info("✅ URL already configured for asyncpg")
                return original_url
            
            # Convert any PostgreSQL variant to use asyncpg
            elif parsed_url.drivername in ['postgresql', 'postgresql+psycopg2']:
                # Use SQLAlchemy's URL manipulation to ensure correctness
                async_url = parsed_url.set(drivername='postgresql+asyncpg')
                result_url = str(async_url)
                logger.info(f"🔄 Converted {parsed_url.drivername} -> postgresql+asyncpg")
                return result_url
            
            else:
                # Unknown driver - raise an error rather than guessing
                raise ValueError(f"Unsupported database driver: {parsed_url.drivername}")
                
        except Exception as e:
            logger.error(f"❌ URL conversion failed: {e}")
            logger.error(f"   Original URL: {self._mask_password(original_url)}")
            raise
    
    @staticmethod
    def _mask_password(url: str) -> str:
        """Hide password in URL for logging purposes."""
        try:
            parsed = make_url(url)
            if parsed.password:
                masked_url = parsed.set(password='***')
                return str(masked_url)
        except:
            pass
        return url
    
    @property
    def engine(self):
        """Get the database engine, initializing if necessary."""
        self._ensure_initialized()
        return self._engine
    
    @property
    def async_session(self):
        """Get the session factory, initializing if necessary."""
        self._ensure_initialized()
        return self._async_session
    
    async def verify_async_driver(self):
        """Verify that we're using the correct async driver."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version_info = await result.fetchone()
                
                driver_name = conn.engine.url.drivername
                logger.info(f"✅ Database connection verified using driver: {driver_name}")
                logger.debug(f"PostgreSQL version: {version_info[0]}")
                
                if 'asyncpg' not in driver_name:
                    raise RuntimeError(f"Expected asyncpg driver, but got: {driver_name}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Async driver verification failed: {e}")
            raise
    
    async def create_tables(self):
        """Create all tables using the async engine."""
        try:
            # Verify we have the right driver first
            await self.verify_async_driver()
            
            async with self.engine.begin() as conn:
                # Import models to ensure they're registered
                from src.data import Base, HistoricalPrice, HistoricalNews
                
                logger.info("🏗️  Creating database tables...")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("✅ Database tables created successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test database connectivity with proper error handling."""
        try:
            await self.verify_async_driver()
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup."""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Clean shutdown of database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")

# Create the manager instance (but don't initialize the engine yet)
db_manager = DatabaseManager()

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for database sessions."""
    async with db_manager.get_session() as session:
        yield session