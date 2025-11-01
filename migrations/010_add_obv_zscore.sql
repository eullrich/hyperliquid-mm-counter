-- Add OBV Z-Score column for relative intensity highlighting
ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS obv_zscore DOUBLE PRECISION DEFAULT 0;

COMMENT ON COLUMN metrics_snapshot.obv_zscore IS 'OBV Z-Score: normalized OBV relative to recent history for intensity highlighting';

-- Recreate views to include obv_zscore
DROP VIEW IF EXISTS latest_metrics CASCADE;
DROP VIEW IF EXISTS active_signals CASCADE;

CREATE VIEW latest_metrics AS
SELECT DISTINCT ON (coin, interval)
    coin, interval, timestamp, price, vwap, price_change_pct, obv, obv_zscore, delta, cumulative_delta,
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

CREATE VIEW active_signals AS
SELECT * FROM latest_metrics
WHERE signal != 'none';

COMMENT ON VIEW latest_metrics IS 'Latest metrics for each token and interval';
COMMENT ON VIEW active_signals IS 'Tokens with active MM-counter trading signals';
