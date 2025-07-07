# scripts/init_database.py
"""
Modern database initialization script using SQLAlchemy and Alembic.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.logging import setup_logging
from src.database.connection import db_manager
from src.database.migrations import run_migrations
from src.data import HistoricalPrice, HistoricalNews

logger = setup_logging()

async def wait_for_database(max_retries: int = 30, delay: float = 2.0) -> bool:
    """
    Wait for PostgreSQL to be ready.
    """
    logger.info("🔄 Waiting for PostgreSQL to be ready...")
    
    for attempt in range(max_retries):
        if await db_manager.test_connection():
            logger.info(f"✅ PostgreSQL ready after {attempt + 1} attempts")
            return True
            
        logger.info(f"⏳ Attempt {attempt + 1}/{max_retries}: PostgreSQL not ready, waiting {delay}s...")
        await asyncio.sleep(delay)
    
    logger.error(f"❌ PostgreSQL failed to start after {max_retries} attempts")
    return False

async def create_database_schema():
    """
    Create all database tables and indexes.
    """
    try:
        logger.info("🏗️  Creating database schema...")
        
        # Create all tables
        await db_manager.create_tables()
        
        # Verify tables were created
        async with db_manager.get_session() as session:
            # Check if our tables exist
            from sqlalchemy import text
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('historical_prices', 'historical_news')
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            if 'historical_prices' in tables and 'historical_news' in tables:
                logger.info("✅ Core tables created successfully:")
                logger.info("  📊 historical_prices - Ready for OHLCV data")
                logger.info("  📰 historical_news - Ready for news data")
            else:
                raise Exception(f"Expected tables not found. Found: {tables}")
                
    except Exception as e:
        logger.error(f"❌ Schema creation failed: {e}")
        raise

async def verify_data_models():
    """
    Test that our data models work correctly.
    
    This is like doing a test drive after building a car - we want to make
    sure everything works before we start collecting real data.
    """
    try:
        logger.info("🧪 Testing data models...")
        
        async with db_manager.get_session() as session:
            from datetime import datetime, timezone
            from decimal import Decimal
            
            # Test HistoricalPrice model
            test_price = HistoricalPrice(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("100.0000"),
                high=Decimal("101.0000"), 
                low=Decimal("99.0000"),
                close=Decimal("100.5000"),
                volume=50000,
                trade_count=250,
                vwap=Decimal("100.2500")
            )
            
            session.add(test_price)
            await session.commit()
            
            # Test HistoricalNews model
            test_news = HistoricalNews(
                id=999999999,  # Use high ID to avoid conflicts
                headline="Test Market Update",
                summary="This is a test news article for verification",
                source="test_source",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                symbols=["TEST", "SPY"],
                images=[{"size": "large", "url": "https://example.com/test.jpg"}],
                sentiment=0.1  # Slightly positive
            )
            
            session.add(test_news)
            await session.commit()
            
            # Clean up test data
            await session.delete(test_price)
            await session.delete(test_news)
            await session.commit()
            
            logger.info("✅ Data models working correctly")
            
    except Exception as e:
        logger.error(f"❌ Data model test failed: {e}")
        raise

async def main():
    """
    Complete database initialization process.
    
    This orchestrates the entire setup process in the correct order,
    like a conductor directing an orchestra to create beautiful music.
    """
    try:
        logger.info("🚀 Starting database initialization...")
        
        # Step 1: Wait for PostgreSQL to be ready
        if not await wait_for_database():
            sys.exit(1)
        
        # Step 2: Run migrations (or create tables if no migrations exist)
        try:
            await run_migrations()
        except Exception as e:
            logger.warning(f"Migration failed, falling back to direct table creation: {e}")
            await create_database_schema()
        
        # Step 3: Verify everything works
        await verify_data_models()
        
        logger.info("🎉 Database initialization completed successfully!")
        logger.info("📋 Summary:")
        logger.info("  ✅ PostgreSQL connection established")
        logger.info("  ✅ Database schema created")
        logger.info("  ✅ Data models verified")
        logger.info("  🎯 Ready for Alpaca data collection!")
        
    except Exception as e:
        logger.error(f"💥 Database initialization failed: {e}")
        sys.exit(1)
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())