#!/usr/bin/env python3
"""
Query Metrics - Simple CLI tool to view anomaly metrics
"""

import sys
import argparse
from db import Database
from tabulate import tabulate


def query_latest_metrics(interval='5m', limit=20, anomaly_only=False):
    """Query latest metrics for all tokens"""
    db = Database()
    conn = db.get_connection()

    try:
        if anomaly_only:
            query = """
                SELECT
                    coin,
                    price,
                    ROUND(price_change_pct::numeric, 2) as price_chg,
                    ROUND(volume_ratio_to_avg::numeric, 2) as vol_ratio,
                    ROUND(rsi_14::numeric, 1) as rsi,
                    ROUND(oi_change_pct::numeric, 2) as oi_chg,
                    ROUND(funding_rate_zscore::numeric, 2) as fund_z,
                    ROUND(cumulative_imbalance_pct::numeric, 1) as book_imb,
                    ROUND(anomaly_score::numeric, 2) as score,
                    anomaly_flag as flag
                FROM latest_metrics
                WHERE interval = %s AND anomaly_flag != 'none'
                ORDER BY anomaly_score DESC
                LIMIT %s
            """
        else:
            query = """
                SELECT
                    coin,
                    price,
                    ROUND(price_change_pct::numeric, 2) as price_chg,
                    ROUND(volume_ratio_to_avg::numeric, 2) as vol_ratio,
                    ROUND(rsi_14::numeric, 1) as rsi,
                    ROUND(oi_change_pct::numeric, 2) as oi_chg,
                    ROUND(funding_rate_zscore::numeric, 2) as fund_z,
                    ROUND(cumulative_imbalance_pct::numeric, 1) as book_imb,
                    ROUND(anomaly_score::numeric, 2) as score,
                    anomaly_flag as flag
                FROM latest_metrics
                WHERE interval = %s
                ORDER BY anomaly_score DESC
                LIMIT %s
            """

        with conn.cursor() as cur:
            cur.execute(query, (interval, limit))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            print(f"\n{'='*100}")
            print(f"Latest Metrics - {interval} interval {'(Anomalies Only)' if anomaly_only else ''}")
            print(f"{'='*100}\n")

            if rows:
                print(tabulate(rows, headers=columns, tablefmt='grid'))
                print(f"\nShowing {len(rows)} tokens")
            else:
                print("No metrics found. Metrics may still be computing...")

    finally:
        db.return_connection(conn)
        db.close_all()


def query_token_detail(coin, interval='5m'):
    """Query detailed metrics for a specific token"""
    db = Database()
    conn = db.get_connection()

    try:
        query = """
            SELECT
                coin,
                interval,
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
                buy_sell_ratio,
                buy_sell_ratio_deviation,
                cumulative_imbalance_pct,
                spread_bps,
                multi_candle_acceleration,
                volatility_ratio,
                anomaly_score,
                anomaly_flag,
                computed_at
            FROM latest_metrics
            WHERE coin = %s AND interval = %s
        """

        with conn.cursor() as cur:
            cur.execute(query, (coin.upper(), interval))
            row = cur.fetchone()

            if row:
                columns = [desc[0] for desc in cur.description]
                print(f"\n{'='*80}")
                print(f"Detailed Metrics - {coin.upper()} ({interval})")
                print(f"{'='*80}\n")

                for col, val in zip(columns, row):
                    if isinstance(val, (int, float)):
                        print(f"{col:30s}: {val:>15.4f}")
                    else:
                        print(f"{col:30s}: {val}")
            else:
                print(f"\nNo metrics found for {coin.upper()} ({interval})")

    finally:
        db.return_connection(conn)
        db.close_all()


def query_high_anomalies(limit=10):
    """Query tokens with high anomaly scores across all intervals"""
    db = Database()
    conn = db.get_connection()

    try:
        query = """
            SELECT
                coin,
                interval,
                ROUND(price::numeric, 4) as price,
                ROUND(price_change_pct::numeric, 2) as price_chg,
                ROUND(volume_ratio_to_avg::numeric, 2) as vol_ratio,
                ROUND(rsi_14::numeric, 1) as rsi,
                ROUND(funding_rate_zscore::numeric, 2) as fund_z,
                ROUND(cumulative_imbalance_pct::numeric, 1) as book_imb,
                ROUND(anomaly_score::numeric, 2) as score,
                anomaly_flag as flag
            FROM high_anomaly_tokens
            LIMIT %s
        """

        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            print(f"\n{'='*100}")
            print(f"High Anomaly Tokens (Top {limit})")
            print(f"{'='*100}\n")

            if rows:
                print(tabulate(rows, headers=columns, tablefmt='grid'))
            else:
                print("No high anomaly tokens found")

    finally:
        db.return_connection(conn)
        db.close_all()


def main():
    parser = argparse.ArgumentParser(description='Query Hyperliquid anomaly metrics')
    parser.add_argument('--interval', '-i', default='5m', choices=['5m', '1h', '4h'],
                        help='Candle interval (default: 5m)')
    parser.add_argument('--limit', '-l', type=int, default=20,
                        help='Number of results to show (default: 20)')
    parser.add_argument('--anomalies', '-a', action='store_true',
                        help='Show only tokens with anomalies')
    parser.add_argument('--token', '-t', type=str,
                        help='Show detailed metrics for specific token (e.g., BTC)')
    parser.add_argument('--high', action='store_true',
                        help='Show high anomaly tokens across all intervals')

    args = parser.parse_args()

    try:
        if args.high:
            query_high_anomalies(args.limit)
        elif args.token:
            query_token_detail(args.token, args.interval)
        else:
            query_latest_metrics(args.interval, args.limit, args.anomalies)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
