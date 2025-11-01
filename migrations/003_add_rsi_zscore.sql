-- Migration 003: Add RSI z-score metric
-- Adds: rsi_zscore column to track extreme momentum shifts

-- Drop views first
DROP VIEW IF EXISTS high_anomaly_tokens CASCADE;
DROP VIEW IF EXISTS latest_metrics CASCADE;

-- Add rsi_zscore column
ALTER TABLE metrics_snapshot
    ADD COLUMN IF NOT EXISTS rsi_zscore DECIMAL(10,4);

-- Recreate views with new column
CREATE OR REPLACE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin,
    interval,
    timestamp,
    price,
    price_change_pct,
    price_change_24_pct,
    price_deviation_from_ema,
    rsi_14,
    rsi_zscore,
    volume,
    volume_ratio_to_avg,
    volume_normalized,
    open_interest,
    oi_change_pct,
    oi_to_volume_ratio,
    funding_rate,
    funding_rate_zscore,
    orderbook_buy_depth,
    orderbook_sell_depth,
    buy_sell_ratio,
    buy_sell_ratio_deviation,
    cumulative_imbalance_pct,
    depth_weighted_price_skew,
    spread_bps,
    multi_candle_acceleration,
    volatility_ratio,
    anomaly_score,
    anomaly_flag,
    directional_alignment,
    computed_at
FROM metrics_snapshot
ORDER BY coin, interval, timestamp DESC;

CREATE OR REPLACE VIEW high_anomaly_tokens AS
SELECT
    coin,
    interval,
    timestamp,
    price,
    price_change_pct,
    price_change_24_pct,
    price_deviation_from_ema,
    rsi_14,
    rsi_zscore,
    volume,
    volume_ratio_to_avg,
    volume_normalized,
    open_interest,
    oi_change_pct,
    funding_rate_zscore,
    cumulative_imbalance_pct,
    anomaly_score,
    anomaly_flag,
    directional_alignment
FROM latest_metrics
WHERE anomaly_score > 0.5
ORDER BY anomaly_score DESC;

-- Create index for new column
CREATE INDEX IF NOT EXISTS idx_metrics_rsi_zscore ON metrics_snapshot(rsi_zscore);
