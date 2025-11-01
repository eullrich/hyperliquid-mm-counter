#!/usr/bin/env python3
"""
One-time script to update existing tokens with categories
"""

import psycopg2
from token_categories import get_token_category
from config import DB_CONFIG

def update_all_categories():
    """Update categories for all existing tokens in database"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Get all tokens
            cur.execute("SELECT coin FROM tokens")
            tokens = [row[0] for row in cur.fetchall()]

            print(f"Updating categories for {len(tokens)} tokens...")

            # Update each token with its category
            for token in tokens:
                category = get_token_category(token)
                cur.execute(
                    "UPDATE tokens SET category = %s WHERE coin = %s",
                    (category, token)
                )
                print(f"  {token}: {category}")

            conn.commit()
            print(f"\n✓ Successfully updated {len(tokens)} tokens with categories")

            # Show summary
            cur.execute("""
                SELECT category, COUNT(*) as count
                FROM tokens
                WHERE is_active = true
                GROUP BY category
                ORDER BY count DESC
            """)

            print("\nCategory Summary:")
            print("-" * 40)
            for row in cur.fetchall():
                category, count = row
                print(f"  {category}: {count} tokens")

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update_all_categories()
