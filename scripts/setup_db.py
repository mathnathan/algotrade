#!/usr/bin/env python3
"""
Database setup script for the trading bot.
Creates necessary tables and indexes after the database and user are created.
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

# Use admin connection to create database/user, then trading connection for schema
POSTGRES_ADMIN_URL = os.getenv("POSTGRES_ADMIN_URL", "postgresql://postgres:postgres_admin_password@postgres:5432/postgres")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_user:secure_trading_password@postgres:5432/trading_db")

# Application schema SQL
SCHEMA_SQL = """
-- News data table
CREATE TABLE IF NOT EXISTS news_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    url TEXT,
    sentiment_score FLOAT,
    embedding_vector FLOAT[], -- Store as float array for now
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Price data table  
CREATE TABLE IF NOT EXISTS price_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,4) NOT NULL,
    high_price DECIMAL(10,4) NOT NULL,
    low_price DECIMAL(10,4) NOT NULL, 
    close_price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, date)
);

-- Trading predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    prediction_time TIMESTAMP WITH TIME ZONE NOT NULL,
    predicted_direction VARCHAR(10) NOT NULL, -- 'higher' or 'lower'
    confidence_score FLOAT NOT NULL,
    model_version VARCHAR(50),
    features JSONB, -- store feature data for analysis
    actual_outcome VARCHAR(10), -- filled after market close
    actual_return DECIMAL(8,4), -- actual return percentage
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    alpaca_order_id VARCHAR(50) UNIQUE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quantity DECIMAL(10,4) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    limit_price DECIMAL(10,4),
    stop_price DECIMAL(10,4),
    time_in_force VARCHAR(10) NOT NULL,
    extended_hours BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) NOT NULL,
    filled_qty DECIMAL(10,4) DEFAULT 0,
    filled_price DECIMAL(10,4),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    prediction_id INTEGER REFERENCES predictions(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(10,4) NOT NULL,
    avg_entry_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    unrealized_pnl DECIMAL(10,4),
    side VARCHAR(10) NOT NULL, -- 'long' or 'short'
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE,
    strategy_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_news_symbol_published ON news_data(symbol, published_at);
CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_alpaca_id ON trading_orders(alpaca_order_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_opened ON positions(opened_at);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for positions table
DROP TRIGGER IF EXISTS update_positions_updated_at ON positions;
CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant all permissions to trading_user on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO trading_user;
"""

async def wait_for_postgres(max_retries=30, delay=2):
    """Wait for PostgreSQL to be ready."""
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(POSTGRES_ADMIN_URL)
            await conn.close()
            print("PostgreSQL is ready!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}: PostgreSQL not ready yet... ({e})")
            await asyncio.sleep(delay)
    
    raise Exception("PostgreSQL failed to start after maximum retries")

async def setup_database_complete():
    """Create database, user, and schema all as admin."""
    try:
        print("Connecting as PostgreSQL admin...")
        conn = await asyncpg.connect(POSTGRES_ADMIN_URL)
        
        # Check if database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'trading_db'"
        )
        
        if not result:
            print("Creating trading_db database...")
            await conn.execute("CREATE DATABASE trading_db")
        else:
            print("Database trading_db already exists")
            
        # Check if user exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_roles WHERE rolname = 'trading_user'"
        )
        
        if not result:
            print("Creating trading_user...")
            await conn.execute(
                "CREATE ROLE trading_user LOGIN PASSWORD 'secure_trading_password'"
            )
        else:
            print("User trading_user already exists")
            
        # Grant privileges on database
        await conn.execute("GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user")
        await conn.close()
        
        # Connect to trading_db as admin and create schema
        print("Creating schema in trading_db...")
        
        # Parse the admin URL to construct trading_db admin URL properly
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(POSTGRES_ADMIN_URL)
        # Replace the database name (path) from '/postgres' to '/trading_db'
        new_parsed = parsed._replace(path='/trading_db')
        trading_db_admin_url = urlunparse(new_parsed)
        
        print(f"Connecting to: {trading_db_admin_url.replace(parsed.password, '***')}")
        
        conn = await asyncpg.connect(trading_db_admin_url)
        
        print("Creating application schema and granting permissions...")
        await conn.execute(SCHEMA_SQL)
        
        print("Schema creation completed successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        print("Created tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
            
        await conn.close()
        return True
        
    except Exception as e:
        print(f"Error in database setup: {e}")
        return False

async def test_trading_user_connection():
    """Test that trading_user can connect and query tables."""
    try:
        print("Testing trading_user connection...")
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Test basic query
        result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        print(f"Trading user can see {result} tables")
        
        # Test insert (should work)
        await conn.execute("""
            INSERT INTO price_data (symbol, date, open_price, high_price, low_price, close_price, volume)
            VALUES ('TEST', CURRENT_DATE, 100.0, 101.0, 99.0, 100.5, 1000)
            ON CONFLICT (symbol, date) DO NOTHING
        """)
        
        # Test select (should work)  
        test_data = await conn.fetchrow("""
            SELECT * FROM price_data WHERE symbol = 'TEST' LIMIT 1
        """)
        
        if test_data:
            print("✅ Trading user can read/write data successfully")
            # Clean up test data
            await conn.execute("DELETE FROM price_data WHERE symbol = 'TEST'")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Trading user connection test failed: {e}")
        return False

async def setup_database():
    """Complete database setup process."""
    try:
        # Wait for PostgreSQL to be ready
        await wait_for_postgres()
        
        # Create everything as admin
        if not await setup_database_complete():
            raise Exception("Failed to create database and schema")
            
        # Test trading user can connect
        if not await test_trading_user_connection():
            print("⚠️  Warning: Trading user connection test failed, but setup may still work")
            
        print("\n✅ Database setup completed successfully!")
        print("📊 Ready to collect data and start trading!")
        
    except Exception as e:
        print(f"\n❌ Database setup failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())