-- Migration 002: Update metrics with EMA, volume normalization, and directional alignment
-- Adds: price_deviation_from_ema, volume_normalized, directional_alignment
-- Renames: price_deviation_from_ma -> price_deviation_from_ema

-- Drop views first (they depend on columns)
DROP VIEW IF EXISTS high_anomaly_tokens CASCADE;
DROP VIEW IF EXISTS latest_metrics CASCADE;

-- Add new columns
ALTER TABLE metrics_snapshot
    ADD COLUMN IF NOT EXISTS price_deviation_from_ema DECIMAL(10,4),
    ADD COLUMN IF NOT EXISTS volume_normalized DECIMAL(15,4),
    ADD COLUMN IF NOT EXISTS directional_alignment VARCHAR(10);

-- Copy data from old column to new (if old column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name='metrics_snapshot' AND column_name='price_deviation_from_ma') THEN
        UPDATE metrics_snapshot
        SET price_deviation_from_ema = price_deviation_from_ma
        WHERE price_deviation_from_ema IS NULL;
    END IF;
END $$;

-- Drop old column if it exists
ALTER TABLE metrics_snapshot
    DROP COLUMN IF EXISTS price_deviation_from_ma CASCADE;

-- Recreate views with new columns

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

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_metrics_directional_alignment ON metrics_snapshot(directional_alignment);
CREATE INDEX IF NOT EXISTS idx_metrics_volume_normalized ON metrics_snapshot(volume_normalized);
