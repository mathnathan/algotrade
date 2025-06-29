-- This script runs automatically when the PostgreSQL container starts
-- Create the trading database and user

-- Create trading user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_user') THEN
        CREATE ROLE trading_user LOGIN PASSWORD 'secure_trading_password';
    END IF;
END
$$;

-- Create trading database if it doesn't exist
SELECT 'CREATE DATABASE trading_db OWNER trading_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading_db')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;