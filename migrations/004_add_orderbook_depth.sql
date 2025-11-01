-- Add orderbook depth table to store 10 levels on each side at candle close
-- This aligns orderbook snapshots with candle timestamps

CREATE TABLE orderbook_depth (
    id BIGSERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,  -- '5m', '1h', '4h' - matches candle intervals
    timestamp BIGINT NOT NULL,      -- Unix timestamp in ms - matches candle timestamp
    side VARCHAR(4) NOT NULL,       -- 'bid' or 'ask'
    level INTEGER NOT NULL,         -- 1-10, where 1 is best bid/ask
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin, interval, timestamp, side, level)
);

-- Index for fast queries by coin, interval, and timestamp
CREATE INDEX idx_orderbook_depth_coin_interval_timestamp
ON orderbook_depth(coin, interval, timestamp DESC);

-- Index for querying specific side
CREATE INDEX idx_orderbook_depth_side
ON orderbook_depth(coin, interval, timestamp DESC, side);

-- Show completion
SELECT 'Orderbook depth table created successfully!' as status;
