-- Replace anomaly detection with MM-counter signal detection
-- Drop old anomaly columns and add signal column

-- Drop views first
DROP VIEW IF EXISTS high_anomaly_tokens CASCADE;
DROP VIEW IF EXISTS latest_metrics CASCADE;

-- Drop old anomaly columns
ALTER TABLE metrics_snapshot
DROP COLUMN IF EXISTS anomaly_score CASCADE,
DROP COLUMN IF EXISTS anomaly_flag CASCADE,
DROP COLUMN IF EXISTS directional_alignment CASCADE,
DROP COLUMN IF EXISTS computed_at CASCADE;

-- Add signal column
ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS signal TEXT DEFAULT 'none';

-- Create index for signal filtering
CREATE INDEX IF NOT EXISTS idx_metrics_signal ON metrics_snapshot(coin, interval, signal) WHERE signal != 'none';

-- Recreate latest_metrics view with signal
CREATE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin, interval, timestamp, price, vwap, price_change_pct, obv, delta, cumulative_delta,
    price_deviation_from_ema, rsi_14, rsi_zscore, macd_line, macd_signal, macd_histogram,
    volume, volume_ratio_to_avg, volume_normalized,
    open_interest, oi_change_pct, oi_to_volume_ratio,
    funding_rate, funding_rate_zscore,
    orderbook_buy_depth, orderbook_sell_depth, buy_sell_ratio,
    buy_sell_ratio_deviation, cumulative_imbalance_pct,
    depth_weighted_price_skew, spread_bps, multi_candle_acceleration,
    volatility_ratio, signal
FROM metrics_snapshot
ORDER BY coin, interval, timestamp DESC;

-- Recreate high signals view (replaces high_anomaly_tokens)
CREATE VIEW active_signals AS
SELECT * FROM latest_metrics
WHERE signal != 'none';

COMMENT ON COLUMN metrics_snapshot.signal IS 'MM-counter trading signal: buy_dip, fade_pump, spoof_alert, exit_accum, or none';
COMMENT ON VIEW active_signals IS 'Tokens with active trading signals';
