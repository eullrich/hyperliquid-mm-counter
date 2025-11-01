-- Create database for Hyperliquid data collection
-- Run with: psql postgres -f db_schema.sql

-- Create database
DROP DATABASE IF EXISTS hyperliquid_data;
CREATE DATABASE hyperliquid_data;

-- Connect to the new database
\c hyperliquid_data

-- Candles table (OHLCV data for different intervals)
CREATE TABLE candles (
    id BIGSERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,  -- '5m', '1h', '4h', '1d'
    timestamp BIGINT NOT NULL,      -- Unix timestamp in milliseconds
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin, interval, timestamp)
);

-- Index for fast queries
CREATE INDEX idx_candles_coin_interval_timestamp ON candles(coin, interval, timestamp DESC);
CREATE INDEX idx_candles_timestamp ON candles(timestamp DESC);

-- Open Interest table
CREATE TABLE open_interest (
    id BIGSERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    value DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin, timestamp)
);

CREATE INDEX idx_oi_coin_timestamp ON open_interest(coin, timestamp DESC);

-- Orderbook snapshots (best bid/ask)
CREATE TABLE orderbook (
    id BIGSERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    best_bid DECIMAL(20, 8) NOT NULL,
    best_ask DECIMAL(20, 8) NOT NULL,
    bid_size DECIMAL(30, 8) NOT NULL,
    ask_size DECIMAL(30, 8) NOT NULL,
    spread_bps DECIMAL(10, 2),  -- Spread in basis points
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin, timestamp)
);

CREATE INDEX idx_orderbook_coin_timestamp ON orderbook(coin, timestamp DESC);

-- Funding rates
CREATE TABLE funding_rates (
    id BIGSERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    rate DECIMAL(10, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin, timestamp)
);

CREATE INDEX idx_funding_coin_timestamp ON funding_rates(coin, timestamp DESC);

-- Collector health/heartbeat table
CREATE TABLE collector_health (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    last_heartbeat BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'running', 'error', 'stopped'
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_health_service_heartbeat ON collector_health(service_name, last_heartbeat DESC);

-- Tokens metadata table
CREATE TABLE tokens (
    id SERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100),
    added_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Insert initial tokens (will be populated from API)
INSERT INTO tokens (coin, is_active) VALUES ('BTC', true), ('ETH', true), ('SOL', true)
ON CONFLICT (coin) DO NOTHING;

-- Create a view for latest orderbook data
CREATE VIEW latest_orderbook AS
SELECT DISTINCT ON (coin)
    coin, timestamp, best_bid, best_ask, bid_size, ask_size, spread_bps
FROM orderbook
ORDER BY coin, timestamp DESC;

-- Create a view for latest OI
CREATE VIEW latest_oi AS
SELECT DISTINCT ON (coin)
    coin, timestamp, value
FROM open_interest
ORDER BY coin, timestamp DESC;

-- Grant permissions (optional, for security)
-- CREATE USER collector_user WITH PASSWORD 'your_password';
-- GRANT CONNECT ON DATABASE hyperliquid_data TO collector_user;
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO collector_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO collector_user;

-- Show summary
SELECT 'Database setup complete!' as status;
SELECT tablename FROM pg_tables WHERE schemaname = 'public';
