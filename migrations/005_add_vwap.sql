-- Add VWAP (Volume Weighted Average Price) column to metrics_snapshot

ALTER TABLE metrics_snapshot
ADD COLUMN IF NOT EXISTS vwap DECIMAL(20, 8);

-- Add index for VWAP queries
CREATE INDEX IF NOT EXISTS idx_metrics_vwap ON metrics_snapshot(coin, interval, vwap);
