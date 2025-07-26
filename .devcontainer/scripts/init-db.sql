-- scripts/init-db.sql
-- This script runs when PostgreSQL first starts up

-- Create the trading database
CREATE DATABASE trading_db;

-- Create the trading user with limited privileges
CREATE USER trading_user WITH PASSWORD 'secure_trading_password';

-- Grant just the permissions needed for trading operations
GRANT CONNECT ON DATABASE trading_db TO trading_user;

-- Switch to the trading database to set schema permissions
\c trading_db;

-- Allow trading_user to create tables and use the public schema
GRANT CREATE, USAGE ON SCHEMA public TO trading_user;

-- Grant sequence permissions (needed for auto-incrementing IDs)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trading_user;

-- Grant table permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_user;