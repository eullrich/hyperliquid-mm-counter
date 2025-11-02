"""
Configuration for Hyperliquid data collector

Rate Limits (per Hyperliquid API docs):
- REST: 1200 weight/minute
  - metaAndAssetCtxs: 20 weight
  - fundingHistory: 20 weight
- WebSocket: max 1000 subscriptions, max 2000 messages sent/minute

Our Usage (218 perpetual tokens):
- WebSocket: 872 subscriptions (218 tokens × 3 candle intervals [5m,1h,4h] + 218 orderbook) ✓
  Note: Orderbook is cached and snapshotted at candle close for perfect alignment
- REST:
  - OI fetch: 20 weight every 5 min = 4 weight/min ✓
  - Funding fetch: 218 tokens × 20 weight = 4360 weight spread over 262 seconds
    = 998 weight/min ✓
- Total: ~1002 weight/min (under 1200 limit) ✓

Storage Efficiency:
- Orderbook snapshots: ~69K rows/day (only at candle close, not continuous)
- vs continuous updates: ~3.1M rows/day (45x reduction in storage!) ✓
"""

# Database configuration
import os
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'hyperliquid_data'),
    'user': os.getenv('DB_USER', 'ericullrich'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Hyperliquid WebSocket
HYPERLIQUID_WS_URL = 'wss://api.hyperliquid.xyz/ws'

# Intervals to collect (removed 1d to stay under 1000 WebSocket subscription limit)
CANDLE_INTERVALS = ['15m', '1h', '4h']

# How often to snapshot orderbook (seconds)
ORDERBOOK_SNAPSHOT_INTERVAL = 60  # Every minute

# How often to update open interest (seconds)
OI_UPDATE_INTERVAL = 300  # Every 5 minutes

# How often to update funding rates (seconds)
FUNDING_UPDATE_INTERVAL = 300  # Every 5 minutes

# Delay between individual funding rate requests (seconds) - for rate limiting
FUNDING_REQUEST_DELAY = 1.2  # 1.2 seconds = max 50 requests/minute = 1000 weight/minute

# How often to send heartbeat (seconds)
HEARTBEAT_INTERVAL = 60  # Every minute

# WebSocket health check settings
WS_HEALTH_CHECK_INTERVAL = 300  # Check every 5 minutes
WS_STALE_THRESHOLD = 900  # Restart if no data for 15 minutes

# How often to update token categories from frontend (seconds)
CATEGORY_UPDATE_INTERVAL = 86400  # Once per day (24 hours)

# How often to compute anomaly metrics (seconds)
METRICS_UPDATE_INTERVAL = 300  # Every 5 minutes (aligns with candle close)

# Data pruning/retention settings
PRUNING_INTERVAL = 86400  # Run pruning once per day (24 hours)
RETENTION_DAYS = {
    '15m': 28,    # Keep 15m candles for 28 days (4 weeks)
    '1h': 28,     # Keep 1h candles for 28 days (4 weeks)
    '4h': 90,     # Keep 4h candles for 90 days
    'orderbook': 28,  # Keep orderbook aligned with shortest candle interval
    'oi': 90,     # Keep open interest for 90 days
    'funding': 90 # Keep funding rates for 90 days
}

# Bulk insert batch size
BATCH_SIZE = 100

# Reconnection settings
RECONNECT_DELAY = 5  # seconds (exponential backoff, capped at 5 minutes)
MAX_RECONNECT_ATTEMPTS = 0  # 0 = infinite retries (survives extended power outages)

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'collector.log')

# Service name for monitoring
SERVICE_NAME = 'hyperliquid_collector'
