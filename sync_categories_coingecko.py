#!/usr/bin/env python3
"""
Fetch token categories from CoinGecko API and update token_categories.py
"""

import requests
import time
import json
from collections import defaultdict

# CoinGecko API (free tier - no key required)
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"

def fetch_hyperliquid_tokens():
    """Get all tokens from Hyperliquid"""
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "metaAndAssetCtxs"}

    response = requests.post(url, json=payload)
    data = response.json()

    # Extract token names
    tokens = []
    if isinstance(data, list) and len(data) > 0:
        universe = data[0].get('universe', [])
        tokens = [asset['name'] for asset in universe]

    print(f"Found {len(tokens)} tokens from Hyperliquid")
    return tokens

def search_coingecko_coin(symbol):
    """Search CoinGecko for a coin by symbol"""
    url = f"{COINGECKO_API_BASE}/search"
    params = {"query": symbol}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            coins = data.get('coins', [])

            # Try to find exact symbol match
            for coin in coins:
                if coin.get('symbol', '').upper() == symbol.upper():
                    return coin.get('id')

            # Return first result if no exact match
            if coins:
                return coins[0].get('id')

        return None
    except Exception as e:
        print(f"Error searching {symbol}: {e}")
        return None

def get_coin_details(coin_id):
    """Get detailed info including categories from CoinGecko"""
    url = f"{COINGECKO_API_BASE}/coins/{coin_id}"
    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'false',
        'community_data': 'false',
        'developer_data': 'false'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'symbol': data.get('symbol', '').upper(),
                'name': data.get('name'),
                'categories': data.get('categories', [])
            }
        return None
    except Exception as e:
        print(f"Error fetching details for {coin_id}: {e}")
        return None

def categorize_tokens(tokens):
    """Fetch categories for all tokens from CoinGecko"""
    token_categories = {}

    for i, symbol in enumerate(tokens):
        print(f"Processing {i+1}/{len(tokens)}: {symbol}")

        # Search for coin
        coin_id = search_coingecko_coin(symbol)
        if not coin_id:
            print(f"  ⚠️  Not found on CoinGecko")
            token_categories[symbol] = []
            continue

        # Get details
        details = get_coin_details(coin_id)
        if details:
            categories = details.get('categories', [])
            token_categories[symbol] = [cat for cat in categories if cat]  # Filter out None/empty
            print(f"  ✓ Categories: {', '.join(categories) if categories else 'None'}")
        else:
            token_categories[symbol] = []

        # Rate limiting - CoinGecko free tier: 10-30 calls/min
        # Use 3 second delay to be safe (20 calls/min)
        time.sleep(3)

    return token_categories

def map_coingecko_to_our_categories(coingecko_categories):
    """Map CoinGecko categories to our category schema"""

    # Mapping from CoinGecko category names to our categories
    category_mapping = {
        # Layer 1
        'Layer 1 (L1)': 'Layer 1',
        'Smart Contract Platform': 'Layer 1',
        'Proof of Stake (PoS)': 'Layer 1',
        'Proof of Work (PoW)': 'Layer 1',

        # Layer 2
        'Layer 2 (L2)': 'Layer 2',
        'Arbitrum Ecosystem': 'Layer 2',
        'Optimism Ecosystem': 'Layer 2',
        'Polygon Ecosystem': 'Layer 2',

        # DeFi
        'Decentralized Finance (DeFi)': 'DeFi',
        'Decentralized Exchange (DEX)': 'DeFi',
        'Lending/Borrowing': 'DeFi',
        'Yield Farming': 'DeFi',
        'Liquid Staking Derivatives': 'DeFi',
        'Stablecoins': 'DeFi',
        'Automated Market Maker (AMM)': 'DeFi',
        'Derivatives': 'DeFi',

        # Meme
        'Meme': 'Meme',
        'Meme Coins': 'Meme',
        'Dog': 'Meme',
        'Cat': 'Meme',

        # AI
        'Artificial Intelligence (AI)': 'AI',
        'AI Agents': 'AI',
        'Artificial Intelligence': 'AI',

        # Gaming
        'Gaming': 'Gaming',
        'Play To Earn': 'Gaming',
        'GameFi': 'Gaming',
        'Metaverse': 'Gaming',
        'Virtual Reality': 'Gaming',

        # NFT
        'Non-Fungible Tokens (NFT)': 'NFT',
        'NFT': 'NFT',
        'Collectibles': 'NFT',

        # Infrastructure
        'Oracle': 'Infrastructure',
        'Oracles': 'Infrastructure',
        'Data': 'Infrastructure',
        'Storage': 'Infrastructure',
        'Interoperability': 'Infrastructure',

        # Social
        'Social': 'Social',
        'Social Money': 'Social',

        # Privacy
        'Privacy': 'Privacy',
        'Privacy Coins': 'Privacy',
        'Zero Knowledge Proofs': 'Privacy',

        # Exchange
        'Exchange-based Tokens': 'Exchange',
        'Centralized Exchange (CEX)': 'Exchange',

        # RWA
        'Real World Assets (RWA)': 'RWA',
        'Tokenized Gold': 'RWA',

        # Solana
        'Solana Ecosystem': 'Solana',

        # Modular
        'Modular Blockchain': 'Modular',
    }

    our_categories = defaultdict(list)

    for symbol, cg_cats in coingecko_categories.items():
        mapped = False
        for cg_cat in cg_cats:
            if cg_cat in category_mapping:
                our_cat = category_mapping[cg_cat]
                our_categories[our_cat].append(symbol)
                mapped = True
                break  # Use first match

        if not mapped and cg_cats:
            # Put in "Other" if we have categories but no mapping
            our_categories['Other'].append(symbol)
        elif not mapped:
            # No categories at all - also goes to Other
            our_categories['Other'].append(symbol)

    return dict(our_categories)

def update_token_categories_file(categories):
    """Update token_categories.py with new data"""
    from datetime import datetime

    content = '''"""
Token category mappings for Hyperliquid tokens
Auto-generated from CoinGecko API
Last updated: {timestamp}
"""

# Category definitions
TOKEN_CATEGORIES = {{
{categories}
}}

def get_token_category(token: str) -> str:
    """
    Get the category for a given token symbol

    Args:
        token: Token symbol (e.g., 'BTC', 'ETH')

    Returns:
        Category name (e.g., 'Layer 1', 'DeFi', 'Meme')
    """
    for category, tokens in TOKEN_CATEGORIES.items():
        if token in tokens:
            return category
    return 'Other'

def get_all_categories() -> list:
    """Get list of all available categories"""
    return list(TOKEN_CATEGORIES.keys())

def get_tokens_by_category(category: str) -> list:
    """Get all tokens in a specific category"""
    return TOKEN_CATEGORIES.get(category, [])

def get_category_summary(tokens: list) -> dict:
    """
    Get a summary of how many tokens are in each category

    Args:
        tokens: List of token symbols

    Returns:
        Dict with category counts
    """
    summary = {{}}
    for token in tokens:
        category = get_token_category(token)
        summary[category] = summary.get(category, 0) + 1
    return summary
'''

    # Format categories
    categories_str = ""
    for cat in sorted(categories.keys()):
        tokens = sorted(categories[cat])
        categories_str += f"    '{cat}': [\n"

        # Format tokens in lines of ~10 tokens each
        for i in range(0, len(tokens), 10):
            chunk = tokens[i:i+10]
            categories_str += f"        {', '.join(repr(t) for t in chunk)},\n"

        categories_str += "    ],\n\n"

    # Fill in template
    content = content.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        categories=categories_str
    )

    # Backup old file
    import shutil
    shutil.copy('token_categories.py', 'token_categories.py.backup')
    print("\n✓ Backed up token_categories.py -> token_categories.py.backup")

    # Write new file
    with open('token_categories.py', 'w') as f:
        f.write(content)

    print(f"✓ Updated token_categories.py with {len(categories)} categories")
    print(f"✓ Total tokens categorized: {sum(len(v) for v in categories.values())}")

if __name__ == "__main__":
    print("="*60)
    print("CoinGecko Token Category Sync")
    print("="*60)
    print()

    # Step 1: Get Hyperliquid tokens
    print("Step 1: Fetching tokens from Hyperliquid...")
    tokens = fetch_hyperliquid_tokens()
    print()

    # Step 2: Get categories from CoinGecko
    print("Step 2: Fetching categories from CoinGecko...")
    print("(This will take a few minutes due to rate limiting)")
    print()
    coingecko_categories = categorize_tokens(tokens)

    # Save raw data
    with open('coingecko_categories_raw.json', 'w') as f:
        json.dump(coingecko_categories, f, indent=2)
    print(f"\n✓ Saved raw CoinGecko data to coingecko_categories_raw.json")

    # Step 3: Map to our categories
    print("\nStep 3: Mapping CoinGecko categories to our schema...")
    our_categories = map_coingecko_to_our_categories(coingecko_categories)

    # Step 4: Update file
    print("\nStep 4: Updating token_categories.py...")
    update_token_categories_file(our_categories)

    print("\n" + "="*60)
    print("Category Summary:")
    print("="*60)
    for cat in sorted(our_categories.keys()):
        print(f"{cat:20s}: {len(our_categories[cat]):3d} tokens")

    print("\n✓ Done! Categories updated from CoinGecko API")
