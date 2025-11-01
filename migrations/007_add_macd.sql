-- Add MACD (Moving Average Convergence Divergence) columns

ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS macd_line DECIMAL(20, 8),
ADD COLUMN IF NOT EXISTS macd_signal DECIMAL(20, 8),
ADD COLUMN IF NOT EXISTS macd_histogram DECIMAL(20, 8);

-- Add index for MACD queries
CREATE INDEX IF NOT EXISTS idx_metrics_macd ON metrics_snapshot(coin, interval, macd_histogram);

-- Update views to include MACD
DROP VIEW IF EXISTS high_anomaly_tokens CASCADE;
DROP VIEW IF EXISTS latest_metrics CASCADE;

CREATE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin, interval, timestamp, price, vwap, price_change_pct, obv, price_deviation_from_ema,
    rsi_14, rsi_zscore, macd_line, macd_signal, macd_histogram,
    volume, volume_ratio_to_avg, volume_normalized,
    open_interest, oi_change_pct, oi_to_volume_ratio,
    funding_rate, funding_rate_zscore,
    orderbook_buy_depth, orderbook_sell_depth, buy_sell_ratio,
    buy_sell_ratio_deviation, cumulative_imbalance_pct,
    depth_weighted_price_skew, spread_bps, multi_candle_acceleration,
    volatility_ratio, anomaly_score, anomaly_flag, directional_alignment
FROM metrics_snapshot
ORDER BY coin, interval, timestamp DESC;

CREATE VIEW high_anomaly_tokens AS
SELECT * FROM latest_metrics
WHERE anomaly_score >= 5
ORDER BY anomaly_score DESC;
