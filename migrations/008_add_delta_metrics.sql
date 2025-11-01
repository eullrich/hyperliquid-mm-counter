-- Add delta and cumulative_delta columns to metrics_snapshot
-- Delta = candle buy volume - sell volume (approximated from price direction)
-- Cumulative Delta = running sum of delta over time

ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS delta DECIMAL(30, 2),
ADD COLUMN IF NOT EXISTS cumulative_delta DECIMAL(30, 2);

-- Create index for efficient querying
CREATE INDEX IF NOT EXISTS idx_metrics_delta ON metrics_snapshot(coin, interval, cumulative_delta);

-- Add comment explaining calculation method
COMMENT ON COLUMN metrics_snapshot.delta IS 'Per-candle buy/sell volume delta (+ if close>open, - if close<open)';
COMMENT ON COLUMN metrics_snapshot.cumulative_delta IS 'Running sum of delta showing cumulative buying/selling pressure';
