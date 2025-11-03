#!/usr/bin/env python3
"""
Metrics Analyzer - Computes anomaly detection metrics for trading
Analyzes price, volume, OI, funding, and orderbook data to detect trading anomalies
"""

import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from db import Database

# Suppress pandas SQLAlchemy warnings (psycopg2 works fine)
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """Analyzes market data and computes anomaly metrics"""

    def __init__(self, db: Database):
        self.db = db

    def compute_ema(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate EMA (Exponential Moving Average) - better for fast-moving crypto markets"""
        if len(prices) < period:
            return float(prices.mean())

        ema = prices.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else float(prices.iloc[-1])

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def compute_macd(self, prices: pd.Series) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: (macd_line, signal_line, macd_histogram)
        """
        if len(prices) < 26:
            return (0.0, 0.0, 0.0)

        # Calculate EMAs
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()

        # MACD Line = EMA(12) - EMA(26)
        macd_line = ema_12 - ema_26

        # Signal Line = EMA(9) of MACD Line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # MACD Histogram = MACD Line - Signal Line
        macd_histogram = macd_line - signal_line

        macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        signal_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        histogram_val = float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0

        return (macd_val, signal_val, histogram_val)

    def compute_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.0

        high_low = highs - lows
        high_close = np.abs(highs - closes.shift())
        low_close = np.abs(lows - closes.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    def compute_zscore(self, value: float, series: pd.Series) -> float:
        """Calculate z-score of a value relative to series"""
        if len(series) < 2:
            return 0.0

        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return 0.0

        return (value - mean) / std

    def fetch_candle_data(self, coin: str, interval: str, lookback_periods: int = 50) -> pd.DataFrame:
        """Fetch recent candle data for analysis"""
        conn = self.db.get_connection()
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE coin = %s AND interval = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            df = pd.read_sql(query, conn, params=(coin, interval, lookback_periods))
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        finally:
            self.db.return_connection(conn)

    def fetch_oi_data(self, coin: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Fetch recent open interest data"""
        conn = self.db.get_connection()
        try:
            cutoff = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
            query = """
                SELECT timestamp, value as open_interest
                FROM open_interest
                WHERE coin = %s AND timestamp >= %s
                ORDER BY timestamp DESC
            """
            df = pd.read_sql(query, conn, params=(coin, cutoff))
            return df.sort_values('timestamp').reset_index(drop=True)
        finally:
            self.db.return_connection(conn)

    def fetch_funding_data(self, coin: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Fetch recent funding rate data"""
        conn = self.db.get_connection()
        try:
            cutoff = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
            query = """
                SELECT timestamp, rate as funding_rate
                FROM funding_rates
                WHERE coin = %s AND timestamp >= %s
                ORDER BY timestamp DESC
            """
            df = pd.read_sql(query, conn, params=(coin, cutoff))
            return df.sort_values('timestamp').reset_index(drop=True)
        finally:
            self.db.return_connection(conn)

    def fetch_orderbook_data(self, coin: str, interval: str) -> Optional[Dict]:
        """Fetch latest orderbook snapshot with 10-level depth"""
        conn = self.db.get_connection()
        try:
            # Get 10-level depth for the latest available snapshot for this interval
            depth_query = """
                SELECT side, SUM(size) as total_size, MAX(timestamp) as latest_ts
                FROM orderbook_depth
                WHERE coin = %s AND interval = %s
                GROUP BY side
                ORDER BY latest_ts DESC
                LIMIT 2
            """
            with conn.cursor() as cur:
                cur.execute(depth_query, (coin, interval))
                depth_rows = cur.fetchall()

                # Calculate deep book imbalance from 10 levels
                total_bid_depth = 0.0
                total_ask_depth = 0.0
                latest_depth_timestamp = None

                for side, size, ts in depth_rows:
                    if side == 'bid':
                        total_bid_depth = float(size)
                        latest_depth_timestamp = ts
                    elif side == 'ask':
                        total_ask_depth = float(size)
                        if not latest_depth_timestamp:
                            latest_depth_timestamp = ts

                # Get best bid/ask snapshot for the same or closest timestamp
                query = """
                    SELECT best_bid, best_ask, bid_size, ask_size, spread_bps
                    FROM orderbook
                    WHERE coin = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                cur.execute(query, (coin,))
                row = cur.fetchone()
                if not row:
                    return None

                result = {
                    'best_bid': float(row[0]),
                    'best_ask': float(row[1]),
                    'bid_size': float(row[2]),
                    'ask_size': float(row[3]),
                    'spread_bps': float(row[4]),
                    'deep_bid_size': total_bid_depth,
                    'deep_ask_size': total_ask_depth
                }

                return result
        finally:
            self.db.return_connection(conn)

    def compute_metrics(self, coin: str, interval: str) -> Optional[Dict]:
        """
        Compute all anomaly metrics for a given token and interval
        Returns dict of metrics ready for database insertion
        """
        try:
            # Fetch candle data
            candles = self.fetch_candle_data(coin, interval, lookback_periods=50)
            if len(candles) < 2:
                logger.warning(f"Not enough candle data for {coin} {interval}")
                return None

            current_candle = candles.iloc[-1]
            prev_candle = candles.iloc[-2] if len(candles) > 1 else current_candle

            # Basic price metrics
            price = float(current_candle['close'])
            price_change_pct = ((price - float(prev_candle['close'])) / float(prev_candle['close'])) * 100

            # On-Balance Volume (OBV) - cumulative volume flow indicator
            obv = 0.0
            obv_zscore = 0.0
            if len(candles) >= 2:
                # Calculate OBV: add volume on up days, subtract on down days
                obv_series = [0]  # Start at 0
                for i in range(1, len(candles)):
                    if candles.iloc[i]['close'] > candles.iloc[i-1]['close']:
                        obv_series.append(obv_series[-1] + candles.iloc[i]['volume'])
                    elif candles.iloc[i]['close'] < candles.iloc[i-1]['close']:
                        obv_series.append(obv_series[-1] - candles.iloc[i]['volume'])
                    else:
                        obv_series.append(obv_series[-1])
                obv = float(obv_series[-1])

                # Calculate OBV Z-Score (relative to recent history)
                if len(obv_series) >= 10:
                    obv_mean = np.mean(obv_series[-20:])  # Use last 20 periods
                    obv_std = np.std(obv_series[-20:])
                    if obv_std > 0:
                        obv_zscore = (obv - obv_mean) / obv_std

            # Delta and Cumulative Delta (Order Flow / Buying vs Selling Pressure)
            # Delta: approximated from price direction (close > open = buy pressure, close < open = sell pressure)
            delta = 0.0
            cumulative_delta = 0.0
            if len(candles) >= 2:
                # Calculate delta for each candle and cumulative sum
                delta_series = []
                for i in range(len(candles)):
                    candle = candles.iloc[i]
                    if candle['close'] > candle['open']:
                        # Bullish candle - buying pressure
                        delta_series.append(float(candle['volume']))
                    elif candle['close'] < candle['open']:
                        # Bearish candle - selling pressure
                        delta_series.append(-float(candle['volume']))
                    else:
                        # Neutral candle
                        delta_series.append(0.0)

                # Current candle delta
                delta = delta_series[-1]

                # Cumulative delta over all candles
                cumulative_delta = sum(delta_series)

            # VWAP calculation (Volume Weighted Average Price)
            # Adaptive periods: 4h=5 (20hrs), 1h=10 (10hrs), 15m=20 (5hrs)
            vwap_periods = {
                '15m': 20,  # 5 hours
                '1h': 10,   # 10 hours
                '4h': 5     # 20 hours
            }
            lookback = vwap_periods.get(interval, 20)

            vwap = 0.0
            if len(candles) >= lookback:
                recent_candles = candles.tail(lookback)
                typical_price = (recent_candles['high'] + recent_candles['low'] + recent_candles['close']) / 3
                vwap = (typical_price * recent_candles['volume']).sum() / recent_candles['volume'].sum()
                vwap = float(vwap) if not pd.isna(vwap) else price

            # Price deviation from EMA (better for fast-moving crypto)
            ema_20 = self.compute_ema(candles['close'], period=20)
            price_deviation_from_ema = ((price - ema_20) / ema_20) * 100 if ema_20 > 0 else 0

            # RSI and RSI z-score
            rsi_14 = self.compute_rsi(candles['close'], period=14)

            # MACD (Moving Average Convergence Divergence)
            macd_line, macd_signal, macd_histogram = self.compute_macd(candles['close'])

            # Compute RSI z-score over lookback period
            rsi_values = []
            if len(candles) >= 20:
                for i in range(max(0, len(candles) - 20), len(candles)):
                    if i >= 14:  # Need at least 14 periods for RSI
                        rsi_val = self.compute_rsi(candles['close'].iloc[:i+1], period=14)
                        rsi_values.append(rsi_val)

            rsi_zscore = 0.0
            if len(rsi_values) >= 2:
                rsi_series = pd.Series(rsi_values)
                rsi_zscore = self.compute_zscore(rsi_14, rsi_series)

            # Fetch OI data first (needed for volume normalization)
            oi_data = self.fetch_oi_data(coin, lookback_hours=24)
            open_interest = 0.0
            oi_change_pct = 0.0
            oi_to_volume_ratio = 0.0

            if len(oi_data) > 1:
                open_interest = float(oi_data.iloc[-1]['open_interest'])
                # Use average of last 20 OI readings as baseline
                avg_oi_20 = oi_data['open_interest'].tail(20).mean()
                oi_change_pct = ((open_interest - avg_oi_20) / avg_oi_20) * 100 if avg_oi_20 > 0 else 0

            # Volume metrics - normalized by OI (proxy for market cap in perp markets)
            volume = float(current_candle['volume'])
            avg_volume = candles['volume'].tail(20).mean()
            volume_ratio_to_avg = volume / avg_volume if avg_volume > 0 else 1.0

            # Volume normalized by open interest (measures relative trading activity)
            volume_normalized = (volume / open_interest * 100) if open_interest > 0 else 0
            avg_volume_normalized = avg_volume / open_interest * 100 if open_interest > 0 else 0

            if volume_ratio_to_avg > 0:
                oi_to_volume_ratio = oi_change_pct / volume_ratio_to_avg
            else:
                oi_to_volume_ratio = 0

            # Fetch funding data
            funding_data = self.fetch_funding_data(coin, lookback_hours=24)
            funding_rate = 0.0
            funding_rate_zscore = 0.0

            if len(funding_data) > 0:
                funding_rate = float(funding_data.iloc[-1]['funding_rate'])
                if len(funding_data) > 2:
                    funding_rate_zscore = self.compute_zscore(funding_rate, funding_data['funding_rate'])

            # Fetch orderbook data with 10-level depth
            orderbook = self.fetch_orderbook_data(coin, interval)
            orderbook_buy_depth = 0.0
            orderbook_sell_depth = 0.0
            buy_sell_ratio = 1.0
            buy_sell_ratio_deviation = 0.0
            cumulative_imbalance_pct = 0.0
            depth_weighted_price_skew = 0.0
            spread_bps = 0.0

            if orderbook:
                # Use best bid/ask for buy/sell ratio (1-level)
                orderbook_buy_depth = orderbook['bid_size']
                orderbook_sell_depth = orderbook['ask_size']
                total_depth = orderbook_buy_depth + orderbook_sell_depth

                if orderbook_sell_depth > 0:
                    buy_sell_ratio = orderbook_buy_depth / orderbook_sell_depth

                buy_sell_ratio_deviation = (buy_sell_ratio - 1.0) / 1.0

                # Use 10-level depth for cumulative imbalance (more accurate)
                deep_bid_size = orderbook.get('deep_bid_size', 0.0)
                deep_ask_size = orderbook.get('deep_ask_size', 0.0)
                total_deep_depth = deep_bid_size + deep_ask_size

                if total_deep_depth > 0:
                    cumulative_imbalance_pct = ((deep_bid_size - deep_ask_size) / total_deep_depth) * 100
                else:
                    # Fallback to 1-level if no depth data available yet
                    if total_depth > 0:
                        cumulative_imbalance_pct = ((orderbook_buy_depth - orderbook_sell_depth) / total_depth) * 100

                spread_bps = orderbook['spread_bps']

                # Simplified depth-weighted price skew
                mid_price = (orderbook['best_bid'] + orderbook['best_ask']) / 2
                if mid_price > 0 and total_depth > 0:
                    buy_weighted = (orderbook['best_bid'] * orderbook_buy_depth) / total_depth
                    sell_weighted = (orderbook['best_ask'] * orderbook_sell_depth) / total_depth
                    depth_weighted_price_skew = ((buy_weighted - sell_weighted) / mid_price) * 100

            # Multi-candle acceleration
            multi_candle_acceleration = 0.0
            if len(candles) > 2:
                prev_change = ((float(prev_candle['close']) - float(candles.iloc[-3]['close'])) /
                               float(candles.iloc[-3]['close'])) * 100
                if prev_change != 0:
                    multi_candle_acceleration = ((price_change_pct - prev_change) / abs(prev_change)) * 100

            # Volatility ratio
            volatility_ratio = 1.0
            if len(candles) >= 20:
                current_atr = self.compute_atr(candles['high'].tail(15),
                                              candles['low'].tail(15),
                                              candles['close'].tail(15), period=14)
                avg_atr = candles.apply(lambda row: self.compute_atr(
                    candles['high'].tail(len(candles)),
                    candles['low'].tail(len(candles)),
                    candles['close'].tail(len(candles)), period=14
                ), axis=1).mean() if len(candles) > 14 else current_atr

                if avg_atr > 0:
                    volatility_ratio = current_atr / avg_atr

            # Compute composite anomaly score with directional alignment
            # Weights: 40% price/volume, 30% orderbook, 30% funding/OI
            score_components = []

            # MM-Counter Signal Detection
            # Check for OBV divergence (price movement vs OBV movement)
            obv_divergence = False
            if len(candles) >= 5:
                # Check if price is up but OBV is flat/down over last 5 candles
                price_trend = candles['close'].iloc[-1] - candles['close'].iloc[-5]
                # Calculate OBV for last 5 candles to check trend
                obv_trend_start = 0.0
                obv_trend_end = 0.0
                for i in range(len(candles)-5, len(candles)):
                    if i > len(candles)-5 and candles.iloc[i]['close'] > candles.iloc[i-1]['close']:
                        obv_trend_end += candles.iloc[i]['volume']
                    elif i > len(candles)-5 and candles.iloc[i]['close'] < candles.iloc[i-1]['close']:
                        obv_trend_end -= candles.iloc[i]['volume']
                # Simple divergence check: price up + OBV down OR price down + OBV up
                obv_divergence = (price_trend > 0 and obv_trend_end < obv_trend_start * 0.5) or \
                                 (price_trend < 0 and obv_trend_end > obv_trend_start * 0.5)

            # Cumulative Delta flip detection
            cum_delta_positive_flip = cumulative_delta > 0 and delta > 0

            # Signal detection with 2-3 confirmations
            signal_type = 'none'
            signal_reasons = []

            # 🟢 Bullish: Buy Dip (Post-Shakeout)
            buy_dip_conditions = []
            # Tweak 1: Tightened EMA threshold from -5% to -3.5% for earlier entry
            if price_deviation_from_ema < -3.5:
                buy_dip_conditions.append(f"EMA {price_deviation_from_ema:.1f}%")
            # Tweak 2: Volume pre-filter - require 1.5x+ volume before rebound
            if obv > 0 and not obv_divergence and volume_ratio_to_avg > 1.5:  # OBV rebound with volume
                buy_dip_conditions.append(f"Vol {volume_ratio_to_avg:.1f}x + OBV")
            if cum_delta_positive_flip and volume_ratio_to_avg > 1.5:
                buy_dip_conditions.append("Cum Δ flip")
            if funding_rate < -0.0003:  # -0.03%
                buy_dip_conditions.append(f"Fund {funding_rate*100:.3f}%")

            if len(buy_dip_conditions) >= 2:
                signal_type = 'buy_dip'
                signal_reasons = buy_dip_conditions

            # 📈 Bullish: Imbalance Lead (Pre-Breakout)
            # Catches early momentum before FOMO chase based on MM absorption patterns
            imbalance_lead_conditions = []
            imbalance_spike = cumulative_imbalance_pct > 15  # Positive spike (MM absorbing)
            volume_confirm = volume_ratio_to_avg > 2.5  # Intent confirmation
            momentum_building = macd_histogram > 0  # Bullish momentum
            buy_pressure = buy_sell_ratio > 2.0  # Strong buy side

            # Build conditions for display
            if imbalance_spike:
                imbalance_lead_conditions.append(f"Imb +{cumulative_imbalance_pct:.0f}%")
            if volume_confirm:
                imbalance_lead_conditions.append(f"Vol {volume_ratio_to_avg:.1f}x")
            if momentum_building:
                imbalance_lead_conditions.append(f"MACD {macd_histogram:.2f}")
            if buy_pressure:
                imbalance_lead_conditions.append(f"B/S {buy_sell_ratio:.1f}")

            # Require 3 core conditions: imbalance + volume + momentum
            if imbalance_spike and volume_confirm and momentum_building and signal_type == 'none':
                signal_type = 'imbalance_lead'
                signal_reasons = imbalance_lead_conditions

            # 🔴 Bearish: Fade Pump (Reversal Trap)
            fade_pump_conditions = []
            # Tweak 1: Raised RSI threshold from 70 to 77 for extreme overbought (FOMO tops)
            if rsi_14 > 77:
                fade_pump_conditions.append(f"RSI {rsi_14:.0f}")
            if obv_divergence and price_change_pct > 0:
                fade_pump_conditions.append("OBV div")
            # Tweak 2: Hiked funding/OI bar to >0.07% AND >40% (was 0.05% and 30%)
            if funding_rate > 0.0007 and oi_change_pct > 40:  # 0.07% + 40% OI for crowded longs
                fade_pump_conditions.append(f"Fund {funding_rate*100:.2f}% & OI +{oi_change_pct:.0f}%")
            if buy_sell_ratio < 0.5:
                fade_pump_conditions.append(f"B/S {buy_sell_ratio:.2f}")

            if len(fade_pump_conditions) >= 2 and signal_type == 'none':
                signal_type = 'fade_pump'
                signal_reasons = fade_pump_conditions

            # ⚠️ Bearish: Liquidity Yank (Spoof Alert) - Short-only scalp signal
            # Tightened logic: Require ALL 3 core confirms (no "or") to reduce noise by 50%
            spoof_conditions = []
            vol_extreme = volume_ratio_to_avg > 4  # Raised from 3x to 4x (extreme MM wash/spoof vol)
            imbalance_deep = cumulative_imbalance_pct < -15  # Raised from -10% to -15% (deep yank)
            sell_pressure = buy_sell_ratio < 0.3  # Heavy sell skew (true pressure, not equilibrium)
            macd_fade = macd_histogram < -0.5  # MACD momentum fade (bearish hist drop)

            # Build conditions for display
            if vol_extreme:
                spoof_conditions.append(f"Vol {volume_ratio_to_avg:.1f}x")
            if imbalance_deep:
                spoof_conditions.append(f"Imb {cumulative_imbalance_pct:.0f}%")
            if sell_pressure:
                spoof_conditions.append(f"B/S {buy_sell_ratio:.2f}")
            if macd_fade:
                spoof_conditions.append(f"MACD {macd_histogram:.2f}")

            # Require ALL 3 core conditions (vol + imbalance + sell pressure) for high conviction
            if vol_extreme and imbalance_deep and sell_pressure and signal_type == 'none':
                signal_type = 'spoof_alert'
                signal_reasons = spoof_conditions

            # ⚪ Neutral: Exit/Accumulate
            exit_conditions = []
            if price < vwap and prev_candle['close'] > vwap:  # VWAP cross below
                exit_conditions.append("VWAP cross")
            if oi_change_pct < -20:
                exit_conditions.append(f"OI -{abs(oi_change_pct):.0f}%")
            if volume_ratio_to_avg < 1:
                exit_conditions.append(f"Vol {volume_ratio_to_avg:.1f}x")

            if len(exit_conditions) >= 2 and signal_type == 'none':
                signal_type = 'exit_accum'
                signal_reasons = exit_conditions

            # Format signal string
            if signal_type != 'none':
                signal_str = f"{signal_type}:{' + '.join(signal_reasons)}"
            else:
                signal_str = 'none'

            return {
                'coin': coin,
                'interval': interval,
                'timestamp': int(current_candle['timestamp']),
                'price': price,
                'vwap': vwap,
                'price_change_pct': price_change_pct,
                'obv': obv,
                'obv_zscore': obv_zscore,
                'delta': delta,
                'cumulative_delta': cumulative_delta,
                'price_deviation_from_ema': price_deviation_from_ema,
                'rsi_14': rsi_14,
                'rsi_zscore': rsi_zscore,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'volume': volume,
                'volume_ratio_to_avg': volume_ratio_to_avg,
                'volume_normalized': volume_normalized,
                'open_interest': open_interest,
                'oi_change_pct': oi_change_pct,
                'oi_to_volume_ratio': oi_to_volume_ratio,
                'funding_rate': funding_rate,
                'funding_rate_zscore': funding_rate_zscore,
                'orderbook_buy_depth': orderbook_buy_depth,
                'orderbook_sell_depth': orderbook_sell_depth,
                'buy_sell_ratio': buy_sell_ratio,
                'buy_sell_ratio_deviation': buy_sell_ratio_deviation,
                'cumulative_imbalance_pct': cumulative_imbalance_pct,
                'depth_weighted_price_skew': depth_weighted_price_skew,
                'spread_bps': spread_bps,
                'multi_candle_acceleration': multi_candle_acceleration,
                'volatility_ratio': volatility_ratio,
                'signal': signal_str
            }

        except Exception as e:
            logger.error(f"Error computing metrics for {coin} {interval}: {e}")
            return None

    def _convert_numpy_to_python(self, metrics: Dict) -> Dict:
        """Convert NumPy types to Python native types for PostgreSQL"""
        converted = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                converted[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif pd.isna(value):
                converted[key] = None
            else:
                converted[key] = value
        return converted

    def save_metrics(self, metrics: Dict):
        """Save computed metrics to database"""
        # Convert NumPy types to Python native types
        metrics = self._convert_numpy_to_python(metrics)

        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO metrics_snapshot (
                        coin, interval, timestamp, price, vwap, price_change_pct, obv, obv_zscore, delta, cumulative_delta,
                        price_deviation_from_ema, rsi_14, rsi_zscore, macd_line, macd_signal, macd_histogram,
                        volume, volume_ratio_to_avg, volume_normalized, open_interest, oi_change_pct,
                        oi_to_volume_ratio, funding_rate, funding_rate_zscore,
                        orderbook_buy_depth, orderbook_sell_depth, buy_sell_ratio,
                        buy_sell_ratio_deviation, cumulative_imbalance_pct,
                        depth_weighted_price_skew, spread_bps, multi_candle_acceleration,
                        volatility_ratio, signal
                    ) VALUES (
                        %(coin)s, %(interval)s, %(timestamp)s, %(price)s, %(vwap)s, %(price_change_pct)s, %(obv)s, %(obv_zscore)s,
                        %(delta)s, %(cumulative_delta)s,
                        %(price_deviation_from_ema)s, %(rsi_14)s, %(rsi_zscore)s, %(macd_line)s, %(macd_signal)s, %(macd_histogram)s,
                        %(volume)s, %(volume_ratio_to_avg)s, %(volume_normalized)s,
                        %(open_interest)s, %(oi_change_pct)s, %(oi_to_volume_ratio)s,
                        %(funding_rate)s, %(funding_rate_zscore)s, %(orderbook_buy_depth)s,
                        %(orderbook_sell_depth)s, %(buy_sell_ratio)s, %(buy_sell_ratio_deviation)s,
                        %(cumulative_imbalance_pct)s, %(depth_weighted_price_skew)s,
                        %(spread_bps)s, %(multi_candle_acceleration)s, %(volatility_ratio)s,
                        %(signal)s
                    )
                    ON CONFLICT (coin, interval, timestamp) DO UPDATE SET
                        price = EXCLUDED.price,
                        vwap = EXCLUDED.vwap,
                        price_change_pct = EXCLUDED.price_change_pct,
                        obv = EXCLUDED.obv,
                        obv_zscore = EXCLUDED.obv_zscore,
                        delta = EXCLUDED.delta,
                        cumulative_delta = EXCLUDED.cumulative_delta,
                        price_deviation_from_ema = EXCLUDED.price_deviation_from_ema,
                        rsi_14 = EXCLUDED.rsi_14,
                        rsi_zscore = EXCLUDED.rsi_zscore,
                        macd_line = EXCLUDED.macd_line,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram,
                        volume = EXCLUDED.volume,
                        volume_ratio_to_avg = EXCLUDED.volume_ratio_to_avg,
                        volume_normalized = EXCLUDED.volume_normalized,
                        open_interest = EXCLUDED.open_interest,
                        oi_change_pct = EXCLUDED.oi_change_pct,
                        oi_to_volume_ratio = EXCLUDED.oi_to_volume_ratio,
                        funding_rate = EXCLUDED.funding_rate,
                        funding_rate_zscore = EXCLUDED.funding_rate_zscore,
                        orderbook_buy_depth = EXCLUDED.orderbook_buy_depth,
                        orderbook_sell_depth = EXCLUDED.orderbook_sell_depth,
                        buy_sell_ratio = EXCLUDED.buy_sell_ratio,
                        buy_sell_ratio_deviation = EXCLUDED.buy_sell_ratio_deviation,
                        cumulative_imbalance_pct = EXCLUDED.cumulative_imbalance_pct,
                        depth_weighted_price_skew = EXCLUDED.depth_weighted_price_skew,
                        spread_bps = EXCLUDED.spread_bps,
                        multi_candle_acceleration = EXCLUDED.multi_candle_acceleration,
                        volatility_ratio = EXCLUDED.volatility_ratio,
                        signal = EXCLUDED.signal
                """
                cur.execute(query, metrics)
                conn.commit()
        finally:
            self.db.return_connection(conn)

    def compute_all_metrics(self, tokens: List[str], intervals: List[str] = ['15m', '1h', '4h']):
        """Compute metrics for all tokens and intervals"""
        total = len(tokens) * len(intervals)
        processed = 0
        saved = 0

        logger.info(f"Computing metrics for {len(tokens)} tokens across {len(intervals)} intervals...")

        for coin in tokens:
            for interval in intervals:
                metrics = self.compute_metrics(coin, interval)
                if metrics:
                    self.save_metrics(metrics)
                    saved += 1
                processed += 1

                if processed % 50 == 0:
                    logger.info(f"Progress: {processed}/{total} ({saved} saved)")

        logger.info(f"Metrics computation complete: {saved}/{total} saved successfully")


if __name__ == "__main__":
    # Test/standalone execution
    logging.basicConfig(level=logging.INFO)

    from config import DB_CONFIG
    db = Database()
    analyzer = MetricsAnalyzer(db)

    # Get tokens from database
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT coin FROM candles ORDER BY coin")
            tokens = [row[0] for row in cur.fetchall()]
    finally:
        db.return_connection(conn)

    # Compute metrics for all tokens
    analyzer.compute_all_metrics(tokens)
