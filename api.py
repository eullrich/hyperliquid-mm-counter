#!/usr/bin/env python3
"""
FastAPI backend for Hyperliquid Anomaly Metrics Dashboard
Provides REST endpoints for querying metrics data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
import logging
from datetime import datetime

from db import Database

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Hyperliquid Anomaly Metrics API",
    description="Real-time anomaly detection metrics for Hyperliquid perpetual tokens",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
db = Database()


@app.get("/")
async def root():
    """Serve the main dashboard page"""
    return FileResponse("static/index.html")


@app.get("/api/metrics")
async def get_metrics(
    interval: str = Query("15m", regex="^(15m|1h|4h)$"),
    anomaly_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    sort_by: str = Query("obv_zscore", regex="^(obv_zscore|price_change_pct|volume_ratio_to_avg|coin|signal)$"),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    """
    Get latest metrics for all tokens

    - **interval**: Candle interval (15m, 1h, 4h)
    - **anomaly_only**: Filter to only show anomalies
    - **limit**: Max number of results
    - **sort_by**: Column to sort by
    - **order**: Sort order (asc/desc)
    """
    conn = db.get_connection()
    try:
        # Build query
        where_clause = "WHERE interval = %s"
        params = [interval]

        if anomaly_only:
            where_clause += " AND signal != 'none'"

        order_clause = f"ORDER BY {sort_by} {order.upper()}"

        query = f"""
            SELECT
                coin,
                interval,
                timestamp,
                price,
                vwap,
                price_change_pct,
                obv,
                obv_zscore,
                delta,
                cumulative_delta,
                price_deviation_from_ema,
                rsi_14,
                rsi_zscore,
                macd_line,
                macd_signal,
                macd_histogram,
                volume,
                volume_ratio_to_avg,
                volume_normalized,
                open_interest,
                oi_change_pct,
                oi_to_volume_ratio,
                funding_rate,
                funding_rate_zscore,
                orderbook_buy_depth,
                orderbook_sell_depth,
                buy_sell_ratio,
                buy_sell_ratio_deviation,
                cumulative_imbalance_pct,
                depth_weighted_price_skew,
                spread_bps,
                multi_candle_acceleration,
                volatility_ratio,
                signal
            FROM latest_metrics
            {where_clause}
            {order_clause}
            LIMIT %s
        """

        params.append(limit)

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            # Convert to list of dicts
            results = []
            for row in rows:
                result = {}
                for col, val in zip(columns, row):
                    if isinstance(val, datetime):
                        result[col] = val.isoformat()
                    else:
                        result[col] = float(val) if isinstance(val, (int, float)) and col != 'timestamp' else val
                results.append(result)

            # Get candle count for this interval
            candle_count_query = """
                SELECT AVG(candle_count)::int as avg_candles
                FROM (
                    SELECT coin, COUNT(*) as candle_count
                    FROM candles
                    WHERE interval = %s
                    GROUP BY coin
                ) sub
            """
            cur.execute(candle_count_query, [interval])
            avg_candles = cur.fetchone()[0] or 0

            return {
                "interval": interval,
                "count": len(results),
                "candle_count": avg_candles,
                "data": results
            }

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/metrics/{coin}")
async def get_token_metrics(
    coin: str,
    interval: str = Query("5m", regex="^(5m|1h|4h)$")
):
    """
    Get detailed metrics for a specific token

    - **coin**: Token symbol (e.g., BTC, ETH)
    - **interval**: Candle interval
    """
    conn = db.get_connection()
    try:
        query = """
            SELECT
                coin,
                interval,
                timestamp,
                price,
                price_change_pct,
                price_deviation_from_ma,
                rsi_14,
                volume,
                volume_ratio_to_avg,
                open_interest,
                oi_change_pct,
                oi_to_volume_ratio,
                funding_rate,
                funding_rate_zscore,
                orderbook_buy_depth,
                orderbook_sell_depth,
                buy_sell_ratio,
                buy_sell_ratio_deviation,
                cumulative_imbalance_pct,
                depth_weighted_price_skew,
                spread_bps,
                multi_candle_acceleration,
                volatility_ratio,
                anomaly_score,
                anomaly_flag,
                directional_alignment
            FROM latest_metrics
            WHERE coin = %s AND interval = %s
        """

        with conn.cursor() as cur:
            cur.execute(query, (coin.upper(), interval))
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"No metrics found for {coin.upper()}")

            columns = [desc[0] for desc in cur.description]
            result = {}
            for col, val in zip(columns, row):
                if isinstance(val, datetime):
                    result[col] = val.isoformat()
                else:
                    result[col] = float(val) if isinstance(val, (int, float)) and col != 'timestamp' else val

            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching token metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/anomalies")
async def get_high_anomalies(
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get tokens with high anomaly scores across all intervals

    - **limit**: Max number of results
    """
    conn = db.get_connection()
    try:
        query = """
            SELECT
                coin,
                interval,
                timestamp,
                price,
                price_change_pct,
                volume_ratio_to_avg,
                oi_change_pct,
                funding_rate_zscore,
                cumulative_imbalance_pct,
                anomaly_score,
                anomaly_flag
            FROM high_anomaly_tokens
            LIMIT %s
        """

        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            results = []
            for row in rows:
                result = {}
                for col, val in zip(columns, row):
                    result[col] = float(val) if isinstance(val, (int, float)) and col != 'timestamp' else val
                results.append(result)

            return {
                "count": len(results),
                "data": results
            }

    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/tokens")
async def get_tokens():
    """Get list of all available tokens"""
    conn = db.get_connection()
    try:
        query = """
            SELECT DISTINCT coin
            FROM latest_metrics
            ORDER BY coin
        """

        with conn.cursor() as cur:
            cur.execute(query)
            tokens = [row[0] for row in cur.fetchall()]

            return {
                "count": len(tokens),
                "tokens": tokens
            }

    except Exception as e:
        logger.error(f"Error fetching tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    conn = db.get_connection()
    try:
        query = """
            SELECT
                COUNT(DISTINCT coin) as token_count,
                COUNT(*) as total_metrics,
                SUM(CASE WHEN signal LIKE 'buy_dip%' THEN 1 ELSE 0 END) as buy_dip_signals,
                SUM(CASE WHEN signal LIKE 'fade_pump%' OR signal LIKE 'spoof_alert%' THEN 1 ELSE 0 END) as bearish_signals,
                MAX(timestamp) as last_update
            FROM latest_metrics
            WHERE interval = '4h'
        """

        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()

            if row:
                return {
                    "token_count": row[0],
                    "total_metrics": row[1],
                    "buy_dip_signals": row[2],
                    "bearish_signals": row[3],
                    "last_update": row[4] if row[4] else None  # timestamp is already a bigint/int
                }

            return {
                "token_count": 0,
                "total_metrics": 0,
                "buy_dip_signals": 0,
                "bearish_signals": 0,
                "last_update": None
            }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/chart/compare")
async def get_chart_compare(
    coins: str = Query(..., description="Comma-separated list of token symbols"),
    metric: str = Query("anomaly_score", description="Metric to compare"),
    interval: str = Query("15m", regex="^(15m|1h|4h)$"),
    lookback_hours: int = Query(24, ge=1, le=168)
):
    """
    Get historical data for comparing the same metric across multiple tokens

    - **coins**: Comma-separated token symbols (e.g., BTC,ETH,SOL)
    - **metric**: Metric name to compare
    - **interval**: Candle interval (15m, 1h, 4h)
    - **lookback_hours**: Hours of history to fetch (max 168 = 7 days)
    """
    conn = db.get_connection()
    try:
        # Calculate timestamp cutoff
        from datetime import datetime, timedelta
        cutoff_dt = datetime.now() - timedelta(hours=lookback_hours)
        cutoff_ts = int(cutoff_dt.timestamp() * 1000)

        # Parse coins parameter
        coin_list = [c.strip().upper() for c in coins.split(',')]
        if not coin_list:
            raise HTTPException(status_code=400, detail="At least one coin must be specified")

        # Validate metric
        valid_metrics = [
            'price', 'vwap', 'price_change_pct', 'obv', 'delta', 'cumulative_delta',
            'price_deviation_from_ema', 'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
            'volume', 'volume_ratio_to_avg', 'volume_normalized',
            'open_interest', 'oi_change_pct', 'oi_to_volume_ratio',
            'funding_rate', 'funding_rate_zscore',
            'orderbook_buy_depth', 'orderbook_sell_depth',
            'buy_sell_ratio', 'buy_sell_ratio_deviation',
            'cumulative_imbalance_pct', 'depth_weighted_price_skew', 'spread_bps',
            'multi_candle_acceleration', 'volatility_ratio',
            'anomaly_score'
        ]
        if metric not in valid_metrics:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")

        query = f"""
            SELECT
                coin,
                timestamp,
                {metric}
            FROM metrics_snapshot
            WHERE coin = ANY(%s) AND interval = %s AND timestamp >= %s
            ORDER BY coin, timestamp ASC
        """

        with conn.cursor() as cur:
            cur.execute(query, (coin_list, interval, cutoff_ts))
            rows = cur.fetchall()

            if not rows:
                raise HTTPException(status_code=404, detail="No historical data found for specified coins")

            # Group data by coin
            data_by_coin = {}
            for row in rows:
                coin_name = row[0]
                timestamp = row[1]
                value = float(row[2]) if isinstance(row[2], (int, float)) else row[2]

                if coin_name not in data_by_coin:
                    data_by_coin[coin_name] = []

                data_by_coin[coin_name].append({
                    "timestamp": timestamp,
                    "value": value
                })

            return {
                "coins": list(data_by_coin.keys()),
                "metric": metric,
                "interval": interval,
                "lookback_hours": lookback_hours,
                "data": data_by_coin
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comparison chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/chart/{coin}")
async def get_chart_data(
    coin: str,
    interval: str = Query("15m", regex="^(15m|1h|4h)$"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics to fetch"),
    lookback_hours: int = Query(24, ge=1, le=168)
):
    """
    Get historical metrics data for charting a single token

    - **coin**: Token symbol (e.g., BTC, ETH)
    - **interval**: Candle interval (15m, 1h, 4h)
    - **metrics**: Comma-separated metric names (default: all metrics)
    - **lookback_hours**: Hours of history to fetch (max 168 = 7 days)
    """
    conn = db.get_connection()
    try:
        # Calculate timestamp cutoff
        from datetime import datetime, timedelta
        cutoff_dt = datetime.now() - timedelta(hours=lookback_hours)
        cutoff_ts = int(cutoff_dt.timestamp() * 1000)

        # Parse metrics parameter
        if metrics:
            metric_list = [m.strip() for m in metrics.split(',')]
            # Validate metrics exist in schema
            valid_metrics = [
                'price', 'vwap', 'price_change_pct', 'obv', 'delta', 'cumulative_delta',
                'price_deviation_from_ema', 'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                'volume', 'volume_ratio_to_avg', 'volume_normalized',
                'open_interest', 'oi_change_pct', 'oi_to_volume_ratio',
                'funding_rate', 'funding_rate_zscore',
                'orderbook_buy_depth', 'orderbook_sell_depth',
                'buy_sell_ratio', 'buy_sell_ratio_deviation',
                'cumulative_imbalance_pct', 'depth_weighted_price_skew', 'spread_bps',
                'multi_candle_acceleration', 'volatility_ratio',
                'anomaly_score', 'directional_alignment'
            ]
            metric_list = [m for m in metric_list if m in valid_metrics]
            if not metric_list:
                metric_list = ['price', 'anomaly_score']
            metric_columns = ', '.join(metric_list)
        else:
            # Default to all metrics
            metric_columns = """
                price, vwap, price_change_pct, obv, price_deviation_from_ema,
                rsi_14, macd_line, macd_signal, macd_histogram,
                volume, volume_ratio_to_avg, volume_normalized,
                open_interest, oi_change_pct, oi_to_volume_ratio,
                funding_rate, funding_rate_zscore,
                orderbook_buy_depth, orderbook_sell_depth,
                buy_sell_ratio, buy_sell_ratio_deviation,
                cumulative_imbalance_pct, depth_weighted_price_skew, spread_bps,
                multi_candle_acceleration, volatility_ratio,
                anomaly_score, directional_alignment
            """

        query = f"""
            SELECT
                timestamp,
                {metric_columns}
            FROM metrics_snapshot
            WHERE coin = %s AND interval = %s AND timestamp >= %s
            ORDER BY timestamp ASC
        """

        with conn.cursor() as cur:
            cur.execute(query, (coin.upper(), interval, cutoff_ts))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            if not rows:
                raise HTTPException(status_code=404, detail=f"No historical data found for {coin.upper()}")

            # Convert to list of dicts
            results = []
            for row in rows:
                result = {}
                for col, val in zip(columns, row):
                    if isinstance(val, datetime):
                        result[col] = val.isoformat()
                    else:
                        result[col] = float(val) if isinstance(val, (int, float)) and col != 'timestamp' else val
                results.append(result)

            return {
                "coin": coin.upper(),
                "interval": interval,
                "lookback_hours": lookback_hours,
                "data_points": len(results),
                "data": results
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chart data for {coin}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/orderbook-depth/{coin}")
async def get_orderbook_depth(
    coin: str,
    interval: str = Query("15m", regex="^(15m|1h|4h)$"),
    lookback_hours: int = Query(24, ge=1, le=168)
):
    """Get historical orderbook depth data (10 levels each side) for a single token"""
    conn = db.get_connection()
    try:
        # Calculate cutoff timestamp
        cutoff = int((datetime.now().timestamp() - (lookback_hours * 3600)) * 1000)

        with conn.cursor() as cur:
            # Fetch orderbook depth snapshots
            cur.execute("""
                SELECT timestamp, side, level, price, size
                FROM orderbook_depth
                WHERE coin = %s AND interval = %s AND timestamp >= %s
                ORDER BY timestamp ASC, side DESC, level ASC
            """, (coin, interval, cutoff))

            rows = cur.fetchall()

            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No orderbook depth data found for {coin} with interval {interval}"
                )

            # Group data by timestamp
            snapshots = {}
            for row in rows:
                timestamp, side, level, price, size = row
                if timestamp not in snapshots:
                    snapshots[timestamp] = {
                        'timestamp': timestamp,
                        'bids': [],
                        'asks': []
                    }

                level_data = {
                    'level': level,
                    'price': float(price),
                    'size': float(size)
                }

                if side == 'bid':
                    snapshots[timestamp]['bids'].append(level_data)
                else:
                    snapshots[timestamp]['asks'].append(level_data)

            # Convert to list and sort
            results = sorted(snapshots.values(), key=lambda x: x['timestamp'])

            return {
                "coin": coin,
                "interval": interval,
                "lookback_hours": lookback_hours,
                "snapshots": len(results),
                "data": results
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching orderbook depth for {coin}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.get("/api/orderbook-heatmap/{coin}")
async def get_orderbook_heatmap(
    coin: str,
    interval: str = Query("15m", regex="^(15m|1h|4h)$"),
    lookback_hours: int = Query(24, ge=1, le=168)
):
    """Get orderbook depth aggregated into price buckets for heatmap visualization"""
    conn = db.get_connection()
    try:
        # Calculate cutoff timestamp
        cutoff = int((datetime.now().timestamp() - (lookback_hours * 3600)) * 1000)

        with conn.cursor() as cur:
            # Fetch orderbook depth snapshots
            cur.execute("""
                SELECT timestamp, side, level, price, size
                FROM orderbook_depth
                WHERE coin = %s AND interval = %s AND timestamp >= %s
                ORDER BY timestamp ASC, side DESC, level ASC
            """, (coin, interval, cutoff))

            rows = cur.fetchall()

            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No orderbook heatmap data found for {coin} with interval {interval}"
                )

            # Group by timestamp first
            snapshots = {}
            for row in rows:
                timestamp, side, level, price, size = row
                if timestamp not in snapshots:
                    snapshots[timestamp] = {
                        'bids': [],
                        'asks': [],
                        'mid_price': None
                    }

                if side == 'bid':
                    snapshots[timestamp]['bids'].append({'price': float(price), 'size': float(size)})
                else:
                    snapshots[timestamp]['asks'].append({'price': float(price), 'size': float(size)})

            # Calculate mid prices and determine price range
            all_mid_prices = []
            for ts, snap in snapshots.items():
                if snap['bids'] and snap['asks']:
                    best_bid = max(snap['bids'], key=lambda x: x['price'])['price']
                    best_ask = min(snap['asks'], key=lambda x: x['price'])['price']
                    snap['mid_price'] = (best_bid + best_ask) / 2
                    all_mid_prices.append(snap['mid_price'])

            if not all_mid_prices:
                raise HTTPException(status_code=404, detail="No valid mid prices found")

            # Determine price range (±2% around average mid price)
            avg_mid = sum(all_mid_prices) / len(all_mid_prices)
            price_range_pct = 0.02  # 2%
            min_price = avg_mid * (1 - price_range_pct)
            max_price = avg_mid * (1 + price_range_pct)

            # Create price buckets (50 levels for smooth heatmap)
            num_buckets = 50
            price_step = (max_price - min_price) / num_buckets
            price_levels = [min_price + (i * price_step) for i in range(num_buckets + 1)]

            # Aggregate depth into buckets for each timestamp
            timestamps = sorted(snapshots.keys())
            bid_heatmap = []
            ask_heatmap = []

            for ts in timestamps:
                snap = snapshots[ts]
                bid_buckets = [0.0] * (num_buckets + 1)
                ask_buckets = [0.0] * (num_buckets + 1)

                # Aggregate bids into buckets
                for bid in snap['bids']:
                    bucket_idx = int((bid['price'] - min_price) / price_step)
                    if 0 <= bucket_idx <= num_buckets:
                        bid_buckets[bucket_idx] += bid['size']

                # Aggregate asks into buckets
                for ask in snap['asks']:
                    bucket_idx = int((ask['price'] - min_price) / price_step)
                    if 0 <= bucket_idx <= num_buckets:
                        ask_buckets[bucket_idx] += ask['size']

                bid_heatmap.append(bid_buckets)
                ask_heatmap.append(ask_buckets)

            return {
                "coin": coin,
                "interval": interval,
                "lookback_hours": lookback_hours,
                "timestamps": timestamps,
                "price_levels": price_levels,
                "bid_depths": bid_heatmap,
                "ask_depths": ask_heatmap,
                "avg_mid_price": avg_mid,
                "price_range": [min_price, max_price]
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching orderbook heatmap for {coin}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.return_connection(conn)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Hyperliquid Anomaly Metrics API")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")
    db.close_all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
