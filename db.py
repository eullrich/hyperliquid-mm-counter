"""
Database connection and operations for Hyperliquid data collector
"""

import psycopg2
from psycopg2.extras import execute_batch
from psycopg2 import pool
import logging
from datetime import datetime
from config import DB_CONFIG

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager with connection pooling"""

    def __init__(self):
        self.connection_pool = None
        self._init_pool()

    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # min and max connections
                **DB_CONFIG
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise

    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool"""
        self.connection_pool.putconn(conn)

    def close_all(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")


class DataWriter:
    """Handles writing data to PostgreSQL"""

    def __init__(self, db: Database):
        self.db = db
        self.candle_buffer = []
        self.orderbook_buffer = []
        self.orderbook_depth_buffer = []
        self.oi_buffer = []
        self.funding_buffer = []

    def add_candle(self, coin, interval, timestamp, open_price, high, low, close, volume):
        """Add candle data to buffer"""
        self.candle_buffer.append((coin, interval, timestamp, open_price, high, low, close, volume))
        logger.debug(f"Added candle: {coin} {interval} @ {timestamp}")

    def add_orderbook(self, coin, timestamp, best_bid, best_ask, bid_size, ask_size):
        """Add orderbook snapshot to buffer"""
        # Calculate spread in basis points
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid) * 10000 if mid > 0 else 0

        self.orderbook_buffer.append((coin, timestamp, best_bid, best_ask, bid_size, ask_size, spread_bps))
        logger.debug(f"Added orderbook: {coin} @ {timestamp}")

    def add_orderbook_depth(self, coin, interval, timestamp, bid_levels, ask_levels):
        """Add orderbook depth levels to buffer (10 levels each side)"""
        # Add bid levels
        for level_data in bid_levels:
            self.orderbook_depth_buffer.append((
                coin, interval, timestamp, 'bid',
                level_data['level'], level_data['price'], level_data['size']
            ))

        # Add ask levels
        for level_data in ask_levels:
            self.orderbook_depth_buffer.append((
                coin, interval, timestamp, 'ask',
                level_data['level'], level_data['price'], level_data['size']
            ))

        logger.debug(f"Added orderbook depth: {coin} {interval} @ {timestamp} ({len(bid_levels)} bids, {len(ask_levels)} asks)")

    def add_open_interest(self, coin, timestamp, value):
        """Add open interest data to buffer"""
        self.oi_buffer.append((coin, timestamp, value))
        logger.debug(f"Added OI: {coin} @ {timestamp}")

    def add_funding_rate(self, coin, timestamp, rate):
        """Add funding rate to buffer"""
        self.funding_buffer.append((coin, timestamp, rate))
        logger.debug(f"Added funding: {coin} @ {timestamp}")

    def flush(self):
        """Write all buffered data to database"""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                # Insert candles
                if self.candle_buffer:
                    execute_batch(
                        cur,
                        """INSERT INTO candles (coin, interval, timestamp, open, high, low, close, volume)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (coin, interval, timestamp) DO UPDATE
                           SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                               close = EXCLUDED.close, volume = EXCLUDED.volume""",
                        self.candle_buffer
                    )
                    logger.info(f"Inserted {len(self.candle_buffer)} candles")
                    self.candle_buffer.clear()

                # Insert orderbook
                if self.orderbook_buffer:
                    execute_batch(
                        cur,
                        """INSERT INTO orderbook (coin, timestamp, best_bid, best_ask, bid_size, ask_size, spread_bps)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (coin, timestamp) DO UPDATE
                           SET best_bid = EXCLUDED.best_bid, best_ask = EXCLUDED.best_ask,
                               bid_size = EXCLUDED.bid_size, ask_size = EXCLUDED.ask_size,
                               spread_bps = EXCLUDED.spread_bps""",
                        self.orderbook_buffer
                    )
                    logger.info(f"Inserted {len(self.orderbook_buffer)} orderbook snapshots")
                    self.orderbook_buffer.clear()

                # Insert orderbook depth (10 levels each side)
                if self.orderbook_depth_buffer:
                    execute_batch(
                        cur,
                        """INSERT INTO orderbook_depth (coin, interval, timestamp, side, level, price, size)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (coin, interval, timestamp, side, level) DO UPDATE
                           SET price = EXCLUDED.price, size = EXCLUDED.size""",
                        self.orderbook_depth_buffer
                    )
                    logger.info(f"Inserted {len(self.orderbook_depth_buffer)} orderbook depth records")
                    self.orderbook_depth_buffer.clear()

                # Insert open interest
                if self.oi_buffer:
                    execute_batch(
                        cur,
                        """INSERT INTO open_interest (coin, timestamp, value)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (coin, timestamp) DO UPDATE
                           SET value = EXCLUDED.value""",
                        self.oi_buffer
                    )
                    logger.info(f"Inserted {len(self.oi_buffer)} OI records")
                    self.oi_buffer.clear()

                # Insert funding rates
                if self.funding_buffer:
                    execute_batch(
                        cur,
                        """INSERT INTO funding_rates (coin, timestamp, rate)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (coin, timestamp) DO UPDATE
                           SET rate = EXCLUDED.rate""",
                        self.funding_buffer
                    )
                    logger.info(f"Inserted {len(self.funding_buffer)} funding rates")
                    self.funding_buffer.clear()

                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error flushing data to database: {e}")
        finally:
            self.db.return_connection(conn)

    def write_heartbeat(self, service_name, status='running', message=''):
        """Write heartbeat to database"""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                timestamp = int(datetime.now().timestamp() * 1000)
                cur.execute(
                    """INSERT INTO collector_health (service_name, last_heartbeat, status, message)
                       VALUES (%s, %s, %s, %s)""",
                    (service_name, timestamp, status, message)
                )
                conn.commit()
                logger.debug(f"Heartbeat written: {service_name} - {status}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error writing heartbeat: {e}")
        finally:
            self.db.return_connection(conn)

    def update_tokens(self, tokens_with_categories):
        """
        Update tokens table with list of active tokens and their categories

        Args:
            tokens_with_categories: List of tuples (token, category)
        """
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """INSERT INTO tokens (coin, is_active, category)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (coin) DO UPDATE
                       SET is_active = EXCLUDED.is_active,
                           category = EXCLUDED.category""",
                    [(token, True, category) for token, category in tokens_with_categories]
                )
                conn.commit()
                logger.info(f"Updated {len(tokens_with_categories)} tokens with categories in database")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating tokens: {e}")
        finally:
            self.db.return_connection(conn)

    def get_active_tokens(self):
        """Get list of active tokens from database"""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT coin FROM tokens WHERE is_active = true")
                tokens = [row[0] for row in cur.fetchall()]
                return tokens
        except Exception as e:
            logger.error(f"Error fetching tokens: {e}")
            return []
        finally:
            self.db.return_connection(conn)
