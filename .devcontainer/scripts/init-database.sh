#!/bin/bash
# scripts/init-database.sh
# Clean, interleaved database initialization for trading systems

set -e  # Exit immediately on any error - critical for financial systems

echo "🏗️  Starting trading database initialization..."
echo "📋 Configuration check:"

# Validate environment variables - fail fast if anything is missing
if [[ -z "$DB_TRADING_DB_NAME" || -z "$DB_TRADING_USER_NAME" || -z "$DB_TRADING_USER_PASSWORD" ]]; then
    echo "❌ Missing required environment variables"
    echo "Required: DB_TRADING_DB_NAME, DB_TRADING_USER_NAME, DB_TRADING_USER_PASSWORD"
    exit 1
fi

echo "  ✅ Database: $DB_TRADING_DB_NAME"
echo "  ✅ User: $DB_TRADING_USER_NAME"  
echo "  ✅ Admin: $POSTGRES_USER"
echo ""

# Step 1: Create the trading database
echo "🔧 Creating trading database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE $DB_TRADING_DB_NAME'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_TRADING_DB_NAME')\gexec
EOSQL
echo "  ✅ Trading database ready"
echo ""

# Step 2: Create the trading user with secure credentials
echo "🔧 Setting up trading user..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$DB_TRADING_USER_NAME') THEN
            CREATE USER $DB_TRADING_USER_NAME WITH PASSWORD '$DB_TRADING_USER_PASSWORD';
        ELSE
            ALTER USER $DB_TRADING_USER_NAME WITH PASSWORD '$DB_TRADING_USER_PASSWORD';
        END IF;
    END
    \$\$;
    
    GRANT CONNECT ON DATABASE $DB_TRADING_DB_NAME TO $DB_TRADING_USER_NAME;
EOSQL
echo "  ✅ Trading user configured with secure access"
echo ""

# Step 3: Grant comprehensive schema permissions for trading operations
echo "🔧 Configuring trading permissions..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$DB_TRADING_DB_NAME" <<-EOSQL
    -- Core schema access for trading operations
    GRANT USAGE, CREATE ON SCHEMA public TO $DB_TRADING_USER_NAME;
    
    -- Table and sequence permissions for market data storage
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_TRADING_USER_NAME;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_TRADING_USER_NAME;
    
    -- Future object permissions (critical for SQLAlchemy table creation)
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_TRADING_USER_NAME;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_TRADING_USER_NAME;
EOSQL
echo "  ✅ Trading permissions configured for market data operations"
echo ""

# Step 4: Test the complete setup with a real-world simulation
echo "🧪 Testing trading user capabilities..."
psql -v ON_ERROR_STOP=1 --username "$DB_TRADING_USER_NAME" --dbname "$DB_TRADING_DB_NAME" <<-EOSQL
    -- Test table creation (simulates historical_prices table)
    CREATE TABLE test_market_data (
        id BIGSERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        price NUMERIC(12, 4) NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Test data insertion (simulates Alpaca data ingestion)
    INSERT INTO test_market_data (symbol, price) VALUES ('AAPL', 150.25);
    
    -- Test data querying (simulates trading algorithm analysis)
    SELECT symbol, price FROM test_market_data WHERE symbol = 'AAPL';
    
    -- Clean up test data
    DROP TABLE test_market_data;
EOSQL
echo "  ✅ All trading operations verified successfully"
echo ""

echo "🎉 Trading database initialization complete!"
echo "📊 Your system is now ready for:"
echo "  • Alpaca market data storage and retrieval"
echo "  • Real-time price analysis and technical indicators"
echo "  • Historical backtesting and strategy development"
echo "  • Secure, performant trading operations"