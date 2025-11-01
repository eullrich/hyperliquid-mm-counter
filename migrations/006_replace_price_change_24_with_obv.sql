-- Replace price_change_24_pct with OBV (On-Balance Volume)
-- OBV is a cumulative volume flow indicator that's more useful for perp trading

-- Add OBV column first
ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS obv DECIMAL(30, 2);

-- Add index for OBV queries
CREATE INDEX IF NOT EXISTS idx_metrics_obv ON metrics_snapshot(coin, interval, obv);

-- Drop dependent views
DROP VIEW IF EXISTS high_anomaly_tokens CASCADE;
DROP VIEW IF EXISTS latest_metrics CASCADE;

-- Drop the old column (now that views are gone)
ALTER TABLE metrics_snapshot
DROP COLUMN IF EXISTS price_change_24_pct;

-- Recreate latest_metrics view with OBV instead of price_change_24_pct
CREATE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin, interval, timestamp, price, vwap, price_change_pct, obv, price_deviation_from_ema,
    rsi_14, rsi_zscore, volume, volume_ratio_to_avg, volume_normalized,
    open_interest, oi_change_pct, oi_to_volume_ratio,
    funding_rate, funding_rate_zscore,
    orderbook_buy_depth, orderbook_sell_depth, buy_sell_ratio,
    buy_sell_ratio_deviation, cumulative_imbalance_pct,
    depth_weighted_price_skew, spread_bps, multi_candle_acceleration,
    volatility_ratio, anomaly_score, anomaly_flag, directional_alignment
FROM metrics_snapshot
ORDER BY coin, interval, timestamp DESC;

-- Recreate high_anomaly_tokens view
CREATE VIEW high_anomaly_tokens AS
SELECT * FROM latest_metrics
WHERE anomaly_score >= 5
ORDER BY anomaly_score DESC;
