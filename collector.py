"""
Hyperliquid Data Collector - WebSocket-based continuous data collection
Collects candles, orderbook, open interest, and funding rates for all tokens
"""

import logging
import time
import signal
import sys
from datetime import datetime
from threading import Thread, Event
import requests

from hyperliquid.info import Info
from hyperliquid.utils import constants

from db import Database, DataWriter
from token_categories import get_token_category
from metrics_analyzer import MetricsAnalyzer
from config import (
    HYPERLIQUID_WS_URL, CANDLE_INTERVALS, ORDERBOOK_SNAPSHOT_INTERVAL,
    OI_UPDATE_INTERVAL, FUNDING_UPDATE_INTERVAL, FUNDING_REQUEST_DELAY,
    HEARTBEAT_INTERVAL, WS_HEALTH_CHECK_INTERVAL, WS_STALE_THRESHOLD,
    CATEGORY_UPDATE_INTERVAL, METRICS_UPDATE_INTERVAL, PRUNING_INTERVAL, RETENTION_DAYS, LOG_LEVEL,
    LOG_FILE, SERVICE_NAME, RECONNECT_DELAY, MAX_RECONNECT_ATTEMPTS
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HyperliquidCollector:
    """Main collector class that manages all data subscriptions"""

    def __init__(self):
        self.db = Database()
        self.writer = DataWriter(self.db)
        self.metrics_analyzer = MetricsAnalyzer(self.db)
        self.info = None
        self.running = Event()
        self.tokens = []
        self.reconnect_count = 0
        self.latest_orderbook = {}  # Cache latest orderbook for each token
        self.last_candle_time = {}  # Track last candle received per token
        self.last_candle_timestamp = {}  # Track last candle timestamp per coin+interval to detect NEW candles
        self.last_data_check = time.time()  # Track last health check

    def get_all_tokens(self):
        """Fetch list of all available perpetual tokens from Hyperliquid (excludes spot)"""
        try:
            url = "https://api.hyperliquid.xyz/info"
            response = requests.post(url, json={"type": "metaAndAssetCtxs"})
            response.raise_for_status()
            data = response.json()

            # Extract coin names from universe - filter for perps only
            if isinstance(data, list) and len(data) > 0:
                universe = data[0].get('universe', [])
                # Filter out spot tokens (they typically have ':' in the name like 'PURR:USDC')
                # Perp tokens are just the base symbol like 'BTC', 'ETH', 'SOL'
                perp_tokens = [
                    asset['name'] for asset in universe
                    if 'name' in asset and ':' not in asset['name']
                ]
                logger.info(f"Fetched {len(perp_tokens)} perpetual tokens from Hyperliquid (excluded spot)")
                return perp_tokens
            return []
        except Exception as e:
            logger.error(f"Error fetching tokens: {e}")
            # Fallback to known perp tokens
            return ['BTC', 'ETH', 'SOL', 'HYPE', 'PURR']

    def init_websocket(self):
        """Initialize WebSocket connection with Hyperliquid"""
        try:
            self.info = Info(constants.MAINNET_API_URL, skip_ws=False)
            logger.info("WebSocket connection initialized")
            self.reconnect_count = 0
            return True
        except Exception as e:
            logger.error(f"Error initializing WebSocket: {e}")
            return False

    def reconnect_websocket(self):
        """Attempt to reconnect WebSocket with exponential backoff"""
        attempt = 0
        max_wait_time = 300  # Cap exponential backoff at 5 minutes

        while True:
            attempt += 1

            # Check if we've exceeded max attempts (unless set to 0 for infinite)
            if MAX_RECONNECT_ATTEMPTS > 0 and attempt > MAX_RECONNECT_ATTEMPTS:
                logger.error(f"Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts")
                return False

            try:
                attempts_msg = f"{attempt}/{MAX_RECONNECT_ATTEMPTS}" if MAX_RECONNECT_ATTEMPTS > 0 else f"{attempt}"
                logger.info(f"Reconnection attempt {attempts_msg}")

                # Close existing connection if any
                try:
                    if self.info:
                        # The hyperliquid SDK doesn't expose a close method, so we just recreate
                        self.info = None
                except Exception as e:
                    logger.warning(f"Error closing old connection: {e}")

                # Wait with exponential backoff (capped at max_wait_time)
                wait_time = min(RECONNECT_DELAY * (2 ** (attempt - 1)), max_wait_time)
                logger.info(f"Waiting {wait_time}s before reconnection attempt...")
                time.sleep(wait_time)

                # Reinitialize WebSocket
                if not self.init_websocket():
                    logger.error(f"Reconnection attempt {attempt} failed to initialize WebSocket")
                    continue

                # Resubscribe to all data streams
                logger.info("Resubscribing to data streams...")
                self.subscribe_candles()
                self.subscribe_orderbook()

                # Reset health check timer
                self.last_data_check = time.time()

                logger.info(f"Successfully reconnected on attempt {attempt}")
                return True

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {e}")

    def subscribe_candles(self):
        """Subscribe to candle updates for all tokens and intervals"""
        for token in self.tokens:
            for interval in CANDLE_INTERVALS:
                try:
                    subscription = {
                        "type": "candle",
                        "coin": token,
                        "interval": interval
                    }
                    self.info.subscribe(subscription, self.handle_candle_update)
                    logger.info(f"Subscribed to candles: {token} {interval}")
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error subscribing to candles for {token} {interval}: {e}")

    def subscribe_orderbook(self):
        """Subscribe to orderbook updates for all tokens"""
        for token in self.tokens:
            try:
                subscription = {
                    "type": "l2Book",
                    "coin": token
                }
                self.info.subscribe(subscription, self.handle_orderbook_update)
                logger.info(f"Subscribed to orderbook: {token}")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error subscribing to orderbook for {token}: {e}")

    def handle_candle_update(self, data):
        """Handle incoming candle data from WebSocket and snapshot orderbook at candle close"""
        try:
            if isinstance(data, dict) and data.get('channel') == 'candle':
                candle_data = data.get('data', {})
                coin = candle_data.get('s')  # symbol
                interval = candle_data.get('i')  # interval
                timestamp = candle_data.get('t')  # timestamp

                # Track last candle time for health monitoring
                self.last_candle_time[coin] = time.time()

                # OHLCV data
                open_price = float(candle_data.get('o', 0))
                high = float(candle_data.get('h', 0))
                low = float(candle_data.get('l', 0))
                close = float(candle_data.get('c', 0))
                volume = float(candle_data.get('v', 0))

                # Check if this is a NEW candle (different timestamp than last seen)
                candle_key = f"{coin}_{interval}"
                is_new_candle = candle_key not in self.last_candle_timestamp or self.last_candle_timestamp[candle_key] != timestamp

                # Always update the candle data (ON CONFLICT DO UPDATE handles updates to current candle)
                self.writer.add_candle(coin, interval, timestamp, open_price, high, low, close, volume)
                logger.debug(f"Received candle: {coin} {interval} @ {timestamp} (new={is_new_candle})")

                # Only snapshot orderbook when we see a NEW candle (candle close)
                if is_new_candle and coin in self.latest_orderbook:
                    ob = self.latest_orderbook[coin]
                    self.writer.add_orderbook(
                        coin, timestamp,  # Use candle timestamp for alignment
                        ob['best_bid'], ob['best_ask'],
                        ob['bid_size'], ob['ask_size']
                    )
                    logger.debug(f"Snapshotted orderbook for {coin} at candle close {interval} @ {timestamp}")

                    # Also snapshot full orderbook depth (10 levels each side)
                    if 'bid_levels' in ob and 'ask_levels' in ob:
                        self.writer.add_orderbook_depth(
                            coin, interval, timestamp,
                            ob['bid_levels'], ob['ask_levels']
                        )

                # Update last seen timestamp for this coin+interval
                self.last_candle_timestamp[candle_key] = timestamp

        except Exception as e:
            logger.error(f"Error handling candle update: {e}")

    def handle_orderbook_update(self, data):
        """Cache orderbook updates (don't store yet, wait for candle close)"""
        try:
            if isinstance(data, dict) and data.get('channel') == 'l2Book':
                book_data = data.get('data', {})
                coin = book_data.get('coin')

                levels = book_data.get('levels', [[], []])
                bids = levels[0] if len(levels) > 0 else []
                asks = levels[1] if len(levels) > 1 else []

                if bids and asks:
                    # Extract best bid and ask for backward compatibility
                    best_bid = float(bids[0]['px']) if bids[0] else 0
                    bid_size = float(bids[0]['sz']) if bids[0] else 0
                    best_ask = float(asks[0]['px']) if asks[0] else 0
                    ask_size = float(asks[0]['sz']) if asks[0] else 0

                    # Extract top 10 levels from each side for depth analysis
                    bid_levels = []
                    for i, bid in enumerate(bids[:10]):  # Top 10 bids
                        if bid:
                            bid_levels.append({
                                'level': i + 1,
                                'price': float(bid['px']),
                                'size': float(bid['sz'])
                            })

                    ask_levels = []
                    for i, ask in enumerate(asks[:10]):  # Top 10 asks
                        if ask:
                            ask_levels.append({
                                'level': i + 1,
                                'price': float(ask['px']),
                                'size': float(ask['sz'])
                            })

                    # Cache the latest orderbook for this token
                    # Will be snapshotted when candle closes
                    self.latest_orderbook[coin] = {
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'bid_size': bid_size,
                        'ask_size': ask_size,
                        'bid_levels': bid_levels,
                        'ask_levels': ask_levels
                    }
                    logger.debug(f"Cached orderbook update for {coin} ({len(bid_levels)} bids, {len(ask_levels)} asks)")
        except Exception as e:
            logger.error(f"Error handling orderbook update: {e}")

    def fetch_open_interest(self):
        """Periodically fetch open interest for all tokens (REST API)"""
        while self.running.is_set():
            try:
                url = "https://api.hyperliquid.xyz/info"
                response = requests.post(url, json={"type": "metaAndAssetCtxs"})
                response.raise_for_status()
                data = response.json()

                timestamp = int(datetime.now().timestamp() * 1000)

                if isinstance(data, list) and len(data) > 1:
                    meta = data[0]['universe']  # Token metadata with names
                    contexts = data[1]  # Asset contexts with OI data

                    # Meta and contexts are parallel arrays - same index = same token
                    for i, (token_meta, ctx) in enumerate(zip(meta, contexts)):
                        coin = token_meta.get('name')
                        oi = ctx.get('openInterest')
                        if coin and oi:
                            self.writer.add_open_interest(coin, timestamp, float(oi))

                logger.info(f"Fetched open interest for {len(contexts)} tokens")
            except Exception as e:
                logger.error(f"Error fetching open interest: {e}")

            time.sleep(OI_UPDATE_INTERVAL)

    def fetch_funding_rates(self):
        """Periodically fetch funding rates for all tokens (REST API with rate limiting)"""
        while self.running.is_set():
            try:
                # Rate limit: max 1200 weight/minute
                # fundingHistory = 20 weight per request
                # So max 60 requests/minute = 1 request every 1 second
                # Using 1.2 second delay to be safe (50 requests/minute = 1000 weight)

                for token in self.tokens:
                    if not self.running.is_set():  # Check if we should stop
                        break

                    url = "https://api.hyperliquid.xyz/info"
                    response = requests.post(url, json={
                        "type": "fundingHistory",
                        "coin": token,
                        "startTime": int((datetime.now().timestamp() - 3600) * 1000)  # Last hour
                    })
                    response.raise_for_status()
                    data = response.json()

                    if data and len(data) > 0:
                        # Get latest funding rate
                        latest = data[-1]
                        timestamp = latest.get('time')
                        rate = latest.get('fundingRate')
                        if timestamp and rate:
                            self.writer.add_funding_rate(token, timestamp, float(rate))

                    # Rate limiting: 1.2 seconds between requests (stay under 1200 weight/min)
                    time.sleep(FUNDING_REQUEST_DELAY)

                logger.info(f"Fetched funding rates for {len(self.tokens)} tokens")
            except Exception as e:
                logger.error(f"Error fetching funding rates: {e}")

            # Wait before next full cycle
            time.sleep(FUNDING_UPDATE_INTERVAL)

    def flush_data_periodically(self):
        """Periodically flush buffered data to database"""
        while self.running.is_set():
            time.sleep(5)  # Flush every 5 seconds
            try:
                self.writer.flush()
            except Exception as e:
                logger.error(f"Error during periodic flush: {e}")

    def send_heartbeat(self):
        """Periodically send heartbeat to database"""
        while self.running.is_set():
            try:
                self.writer.write_heartbeat(SERVICE_NAME, 'running', f'Collecting data for {len(self.tokens)} tokens')
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

            time.sleep(HEARTBEAT_INTERVAL)

    def check_websocket_health(self):
        """Monitor WebSocket health and restart if stale"""
        while self.running.is_set():
            time.sleep(WS_HEALTH_CHECK_INTERVAL)

            try:
                current_time = time.time()

                # Check if we've received any candle data recently
                if self.last_candle_time:
                    most_recent = max(self.last_candle_time.values())
                    time_since_data = current_time - most_recent

                    if time_since_data > WS_STALE_THRESHOLD:
                        logger.error(f"WebSocket appears stale! No data for {time_since_data:.0f} seconds")
                        logger.error("Attempting to reconnect...")

                        # Attempt to reconnect
                        if not self.reconnect_websocket():
                            logger.error("All reconnection attempts failed, exiting...")
                            self.running.clear()
                            sys.exit(1)
                    else:
                        logger.info(f"WebSocket health OK - last data {time_since_data:.0f}s ago")
                else:
                    # No data received yet - might be starting up
                    startup_time = current_time - self.last_data_check
                    if startup_time > WS_STALE_THRESHOLD:
                        logger.error(f"WebSocket never received data after {startup_time:.0f}s")
                        logger.error("Attempting to reconnect...")

                        # Attempt to reconnect
                        if not self.reconnect_websocket():
                            logger.error("All reconnection attempts failed, exiting...")
                            self.running.clear()
                            sys.exit(1)

            except Exception as e:
                logger.error(f"Error in WebSocket health check: {e}")

    def compute_metrics_periodically(self):
        """Periodically compute anomaly detection metrics"""
        while self.running.is_set():
            time.sleep(METRICS_UPDATE_INTERVAL)  # Wait 5 minutes before first computation

            try:
                logger.info("Computing anomaly metrics for all tokens...")
                start_time = time.time()

                # Compute metrics for all tokens across all intervals
                self.metrics_analyzer.compute_all_metrics(self.tokens, CANDLE_INTERVALS)

                elapsed = time.time() - start_time
                logger.info(f"Metrics computation completed in {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Error computing metrics: {e}")

    def update_categories_periodically(self):
        """Periodically update token categories from Hyperliquid frontend"""
        while self.running.is_set():
            time.sleep(CATEGORY_UPDATE_INTERVAL)  # Wait 24 hours before first update

            try:
                logger.info("Updating token categories from Hyperliquid frontend...")

                # Import scraper locally to avoid circular import issues
                from category_updater import CategoryScraper, update_token_categories_file

                # Scrape categories
                scraper = CategoryScraper()
                categories = scraper.scrape()

                if categories:
                    # Update token_categories.py file
                    update_token_categories_file(categories)

                    # Reload the token_categories module to pick up changes
                    import importlib
                    import token_categories
                    importlib.reload(token_categories)

                    # Re-fetch all tokens and update database
                    self.tokens = self.get_all_tokens()
                    tokens_with_categories = [(token, token_categories.get_token_category(token)) for token in self.tokens]
                    self.writer.update_tokens(tokens_with_categories)

                    logger.info(f"Successfully updated categories for {len(self.tokens)} tokens")
                else:
                    logger.warning("No categories scraped, skipping update")

            except Exception as e:
                logger.error(f"Error updating categories: {e}")

    def prune_old_data(self):
        """Periodically prune old data based on retention settings"""
        while self.running.is_set():
            time.sleep(PRUNING_INTERVAL)  # Wait 24 hours before first prune

            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Prune candles by interval
                    for interval, days in RETENTION_DAYS.items():
                        if interval in ['5m', '1h', '4h']:
                            # Calculate cutoff timestamp (milliseconds)
                            cutoff_ts = int((datetime.now().timestamp() - (days * 86400)) * 1000)

                            cur.execute(
                                "DELETE FROM candles WHERE interval = %s AND timestamp < %s",
                                (interval, cutoff_ts)
                            )
                            deleted = cur.rowcount
                            if deleted > 0:
                                logger.info(f"Pruned {deleted} old {interval} candles (older than {days} days)")

                    # Prune orderbook (keep same as shortest candle interval)
                    orderbook_days = RETENTION_DAYS.get('orderbook', 2)
                    cutoff_ts = int((datetime.now().timestamp() - (orderbook_days * 86400)) * 1000)
                    cur.execute("DELETE FROM orderbook WHERE timestamp < %s", (cutoff_ts,))
                    deleted = cur.rowcount
                    if deleted > 0:
                        logger.info(f"Pruned {deleted} old orderbook snapshots (older than {orderbook_days} days)")

                    # Prune open interest
                    oi_days = RETENTION_DAYS.get('oi', 90)
                    cutoff_ts = int((datetime.now().timestamp() - (oi_days * 86400)) * 1000)
                    cur.execute("DELETE FROM open_interest WHERE timestamp < %s", (cutoff_ts,))
                    deleted = cur.rowcount
                    if deleted > 0:
                        logger.info(f"Pruned {deleted} old open interest records (older than {oi_days} days)")

                    # Prune funding rates
                    funding_days = RETENTION_DAYS.get('funding', 90)
                    cutoff_ts = int((datetime.now().timestamp() - (funding_days * 86400)) * 1000)
                    cur.execute("DELETE FROM funding_rates WHERE timestamp < %s", (cutoff_ts,))
                    deleted = cur.rowcount
                    if deleted > 0:
                        logger.info(f"Pruned {deleted} old funding rates (older than {funding_days} days)")

                    conn.commit()
                    logger.info("Data pruning completed successfully")

                    # Optional: Run VACUUM to reclaim space (can be slow)
                    # cur.execute("VACUUM ANALYZE candles, orderbook, open_interest, funding_rates")

            except Exception as e:
                logger.error(f"Error during data pruning: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    self.db.return_connection(conn)

    def start(self):
        """Start the data collector"""
        logger.info(f"Starting {SERVICE_NAME}...")

        # Fetch tokens
        self.tokens = self.get_all_tokens()
        if not self.tokens:
            logger.error("No tokens found, exiting")
            return

        # Assign categories to tokens
        tokens_with_categories = [(token, get_token_category(token)) for token in self.tokens]

        # Update tokens in database with categories
        self.writer.update_tokens(tokens_with_categories)

        # Initialize WebSocket
        if not self.init_websocket():
            logger.error("Failed to initialize WebSocket, exiting")
            return

        # Subscribe to data streams
        logger.info("Subscribing to data streams...")
        self.subscribe_candles()
        self.subscribe_orderbook()

        # Set running flag
        self.running.set()

        # Start background threads
        threads = [
            Thread(target=self.fetch_open_interest, daemon=True),
            Thread(target=self.fetch_funding_rates, daemon=True),
            Thread(target=self.flush_data_periodically, daemon=True),
            Thread(target=self.send_heartbeat, daemon=True),
            Thread(target=self.check_websocket_health, daemon=True),
            Thread(target=self.compute_metrics_periodically, daemon=True),
            Thread(target=self.update_categories_periodically, daemon=True),
            Thread(target=self.prune_old_data, daemon=True),
        ]

        for thread in threads:
            thread.start()

        logger.info(f"{SERVICE_NAME} started successfully")
        logger.info(f"Collecting data for {len(self.tokens)} tokens")
        logger.info(f"Intervals: {', '.join(CANDLE_INTERVALS)}")

        # Keep main thread alive
        try:
            while self.running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.stop()

    def stop(self):
        """Stop the data collector"""
        logger.info("Stopping collector...")
        self.running.clear()

        # Final flush
        try:
            self.writer.flush()
            self.writer.write_heartbeat(SERVICE_NAME, 'stopped', 'Collector stopped gracefully')
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        # Close database connections
        self.db.close_all()
        logger.info("Collector stopped")


def signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start collector
    collector = HyperliquidCollector()
    collector.start()
