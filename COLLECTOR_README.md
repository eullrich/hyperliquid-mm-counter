# Hyperliquid Data Collector

Continuous background data collection from Hyperliquid API via WebSocket.

## What It Collects

- **Candles (OHLCV)**: 5m, 1h, 4h, 1d intervals for all tokens
- **Orderbook**: Best bid/ask, sizes, spread (every minute)
- **Open Interest**: Current OI for all tokens (every 5 minutes)
- **Funding Rates**: 8-hour funding rates (every 5 minutes)

## Quick Start

### 1. Test the Collector (Run Once)

```bash
./manage_collector.sh test
```

This runs the collector in the foreground so you can see what's happening. Press Ctrl+C to stop.

Check the database to verify data is being collected:

```bash
psql hyperliquid_data -c "SELECT coin, interval, timestamp, close FROM candles ORDER BY timestamp DESC LIMIT 10;"
psql hyperliquid_data -c "SELECT coin, timestamp, best_bid, best_ask FROM orderbook ORDER BY timestamp DESC LIMIT 10;"
```

### 2. Install as Background Service

Once you've verified it works:

```bash
./manage_collector.sh install
```

This installs it as a macOS LaunchAgent that:
- Starts automatically when you log in
- Restarts automatically if it crashes
- Runs continuously in the background

### 3. Check Status

```bash
./manage_collector.sh status
```

Shows if the collector is running and displays recent logs.

### 4. View Live Logs

```bash
./manage_collector.sh logs
```

Tail the logs in real-time (Ctrl+C to exit).

### 5. Stop/Start/Restart

```bash
./manage_collector.sh stop
./manage_collector.sh start
./manage_collector.sh restart
```

### 6. Uninstall

```bash
./manage_collector.sh uninstall
```

Removes the background service (but keeps the database data).

## Preventing Laptop Sleep

The collector will stop when your laptop sleeps. To prevent this:

### Option 1: Prevent Sleep While Plugged In

```bash
./manage_collector.sh prevent-sleep
```

This configures your Mac to never sleep while plugged into power.

### Option 2: Use Caffeinate

Keep your Mac awake while running the collector:

```bash
caffeinate -s &
```

This prevents sleep until you kill the caffeinate process.

### Option 3: Manual Energy Settings

System Settings → Battery → Power Adapter → Set "Turn display off after" and "Prevent automatic sleeping on power adapter when the display is off"

## Logs

Logs are stored in:
- `/Users/ericullrich/Code/hedge-v4/logs/collector.log` - Main application log
- `/Users/ericullrich/Code/hedge-v4/logs/collector.stdout.log` - Standard output
- `/Users/ericullrich/Code/hedge-v4/logs/collector.stderr.log` - Error output

## Database

**Database Name**: `hyperliquid_data`

**Tables**:
- `candles` - OHLCV data
- `orderbook` - Orderbook snapshots
- `open_interest` - OI data
- `funding_rates` - Funding rates
- `tokens` - List of active tokens
- `collector_health` - Heartbeat/monitoring

**Connect to database**:
```bash
psql hyperliquid_data
```

**Example queries**:

```sql
-- Latest prices for all tokens
SELECT DISTINCT ON (coin)
    coin, close as price, timestamp
FROM candles
WHERE interval = '1h'
ORDER BY coin, timestamp DESC;

-- Trading volume last 24 hours
SELECT coin, SUM(volume) as volume_24h
FROM candles
WHERE interval = '1h'
  AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
GROUP BY coin
ORDER BY volume_24h DESC;

-- Check collector health
SELECT * FROM collector_health ORDER BY id DESC LIMIT 10;
```

## Configuration

Edit `config.py` to customize:
- Candle intervals to collect
- Database connection details
- Snapshot frequencies
- Logging level

## Troubleshooting

### Collector Not Starting

```bash
# Check logs
./manage_collector.sh logs

# Try running in foreground to see errors
./manage_collector.sh test
```

### Database Connection Errors

```bash
# Verify PostgreSQL is running
psql postgres -c "SELECT 1"

# Check database exists
psql postgres -c "\l" | grep hyperliquid_data
```

### No Data Being Collected

```bash
# Check if WebSocket is connected (look for "Subscribed to" messages)
./manage_collector.sh logs | grep "Subscribed"

# Verify tokens are fetched
psql hyperliquid_data -c "SELECT * FROM tokens;"
```

### Collector Keeps Crashing

```bash
# Check error logs
cat /Users/ericullrich/Code/hedge-v4/logs/collector.stderr.log

# Increase reconnection delay in config.py
```

## Moving to Cloud (Option A)

When ready to move to a cloud VM for 24/7 operation:

1. Set up a cloud VM (DigitalOcean, AWS, etc.)
2. Install PostgreSQL on the VM
3. Copy these files to the VM:
   - `collector.py`
   - `db.py`
   - `config.py`
   - `db_schema.sql`
4. Update `config.py` with VM database credentials
5. Run `python3 collector.py` on the VM
6. Use systemd (Linux) instead of launchd for auto-start

## Architecture

```
Hyperliquid API (WebSocket)
    ↓
collector.py (Python)
  ├─ WebSocket subscriptions (candles, orderbook)
  ├─ REST API polling (OI, funding)
  ├─ Bulk insert buffer (5 second flush)
  ├─ Heartbeat monitor (1 minute)
  └─ Auto-reconnect on failure
    ↓
PostgreSQL Database (local)
  ├─ candles table
  ├─ orderbook table
  ├─ open_interest table
  └─ funding_rates table
    ↓
Your Applications
  ├─ Dash UI (scanner)
  ├─ Analysis tools
  └─ Alerting system
```

## Next Steps

1. Let collector run for 24 hours to accumulate data
2. Build Dash UI to query and visualize the data
3. Create metrics calculator (correlation, z-score, RSI, etc.)
4. Add alerting system for trading signals
5. Consider migrating to cloud for 24/7 operation
