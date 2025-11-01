-- Anomaly Metrics Snapshot Table
-- Stores computed metrics for real-time anomaly detection and filtering

CREATE TABLE IF NOT EXISTS metrics_snapshot (
    id SERIAL PRIMARY KEY,
    coin VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL, -- 5m, 1h, 4h
    timestamp BIGINT NOT NULL, -- Unix timestamp in milliseconds
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Core Price Metrics
    price DECIMAL(20, 8),
    price_change_pct DECIMAL(10, 4), -- % change from last candle
    price_deviation_from_ma DECIMAL(10, 4), -- % deviation from 20-period SMA
    rsi_14 DECIMAL(6, 2), -- RSI 14-period

    -- Volume Metrics
    volume DECIMAL(20, 2),
    volume_ratio_to_avg DECIMAL(10, 4), -- Current volume / 20-period avg

    -- Open Interest Metrics
    open_interest DECIMAL(20, 2),
    oi_change_pct DECIMAL(10, 4), -- % change from last period
    oi_to_volume_ratio DECIMAL(10, 4), -- OI change % / Volume ratio

    -- Funding Rate Metrics
    funding_rate DECIMAL(10, 8),
    funding_rate_zscore DECIMAL(10, 4), -- Z-score vs 24h mean

    -- Order Book Metrics
    orderbook_buy_depth DECIMAL(20, 2), -- Cumulative buy depth within 0.5%
    orderbook_sell_depth DECIMAL(20, 2), -- Cumulative sell depth within 0.5%
    buy_sell_ratio DECIMAL(10, 4),
    buy_sell_ratio_deviation DECIMAL(10, 4), -- Deviation from 1.0
    cumulative_imbalance_pct DECIMAL(10, 4), -- (Buy - Sell) / Total
    depth_weighted_price_skew DECIMAL(10, 4), -- Price-weighted depth skew
    spread_bps DECIMAL(10, 4), -- Current spread in basis points

    -- Advanced Anomaly Indicators
    multi_candle_acceleration DECIMAL(10, 4), -- (Current change - Prev change) / Prev change
    volatility_ratio DECIMAL(10, 4), -- Current ATR / 20-period avg ATR

    -- Composite Scoring
    anomaly_score DECIMAL(10, 4), -- Weighted composite of z-scores
    anomaly_flag VARCHAR(10), -- 'none', 'low', 'medium', 'high'

    -- Constraints
    UNIQUE(coin, interval, timestamp)
);

-- Indexes for fast filtering and querying
CREATE INDEX IF NOT EXISTS idx_metrics_coin_interval ON metrics_snapshot(coin, interval);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshot(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_anomaly_score ON metrics_snapshot(anomaly_score DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_anomaly_flag ON metrics_snapshot(anomaly_flag);
CREATE INDEX IF NOT EXISTS idx_metrics_coin_timestamp ON metrics_snapshot(coin, timestamp DESC);

-- View for latest metrics per token per interval
CREATE OR REPLACE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin,
    interval,
    timestamp,
    price,
    price_change_pct,
    price_deviation_from_ma,
    rsi_14,
    volume,
    volume_ratio_to_avg,
    open_interest,
    oi_change_pct,
    funding_rate,
    funding_rate_zscore,
    buy_sell_ratio,
    cumulative_imbalance_pct,
    anomaly_score,
    anomaly_flag,
    computed_at
FROM metrics_snapshot
ORDER BY coin, interval, timestamp DESC;

-- View for high anomaly tokens
CREATE OR REPLACE VIEW high_anomaly_tokens AS
SELECT
    coin,
    interval,
    timestamp,
    price,
    price_change_pct,
    volume_ratio_to_avg,
    oi_change_pct,
    funding_rate_zscore,
    cumulative_imbalance_pct,
    anomaly_score,
    anomaly_flag
FROM latest_metrics
WHERE anomaly_flag IN ('high', 'medium')
ORDER BY anomaly_score DESC;

COMMENT ON TABLE metrics_snapshot IS 'Real-time computed metrics for anomaly detection and trading signals';
COMMENT ON COLUMN metrics_snapshot.anomaly_score IS 'Weighted composite: 40% price/volume, 30% orderbook, 30% funding/OI';
COMMENT ON COLUMN metrics_snapshot.anomaly_flag IS 'Alert level: high (3+ metrics >1.5σ), medium (2), low (1), none (0)';
