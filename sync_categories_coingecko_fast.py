#!/usr/bin/env python3
"""
Faster CoinGecko category sync using bulk endpoint
"""

import requests
import json
from collections import defaultdict
import time

def fetch_hyperliquid_tokens():
    """Get all tokens from Hyperliquid"""
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "metaAndAssetCtxs"}

    response = requests.post(url, json=payload)
    data = response.json()

    tokens = []
    if isinstance(data, list) and len(data) > 0:
        universe = data[0].get('universe', [])
        tokens = [asset['name'] for asset in universe]

    print(f"✓ Found {len(tokens)} tokens from Hyperliquid")
    return tokens

def fetch_all_coins_from_coingecko():
    """Fetch complete coin list with categories from CoinGecko"""
    print("Fetching complete coin list from CoinGecko...")
    print("(This may take a minute...)")

    url = "https://api.coingecko.com/api/v3/coins/list"
    params = {'include_platform': 'false'}

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return {}

    coins = response.json()
    print(f"✓ Fetched {len(coins)} coins from CoinGecko")

    # Create symbol -> id mapping
    symbol_to_id = {}
    for coin in coins:
        symbol = coin.get('symbol', '').upper()
        coin_id = coin.get('id')
        # Prefer coins with matching symbol
        if symbol and coin_id:
            if symbol not in symbol_to_id:
                symbol_to_id[symbol] = coin_id

    return symbol_to_id

def get_categories_batch(coin_ids, batch_size=50):
    """Get categories for coins in batches"""
    all_categories = {}

    total_batches = (len(coin_ids) + batch_size - 1) // batch_size

    for i in range(0, len(coin_ids), batch_size):
        batch = coin_ids[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} coins)...")

        for symbol, coin_id in batch:
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                params = {
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'false',
                    'community_data': 'false',
                    'developer_data': 'false'
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    categories = data.get('categories', [])
                    all_categories[symbol] = [cat for cat in categories if cat]
                    print(f"  {symbol:10s}: {', '.join(categories[:3]) if categories else 'No categories'}")
                elif response.status_code == 429:
                    print(f"  Rate limited! Waiting 60 seconds...")
                    time.sleep(60)
                else:
                    all_categories[symbol] = []

                time.sleep(3)  # Rate limiting

            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                all_categories[symbol] = []

    return all_categories

def map_coingecko_to_our_categories(coingecko_categories):
    """Map CoinGecko categories to our schema"""

    category_mapping = {
        'Layer 1 (L1)': 'Layer 1',
        'Smart Contract Platform': 'Layer 1',
        'Proof of Stake (PoS)': 'Layer 1',
        'Layer 2 (L2)': 'Layer 2',
        'Arbitrum Ecosystem': 'Layer 2',
        'Optimism Ecosystem': 'Layer 2',
        'Polygon Ecosystem': 'Layer 2',
        'Decentralized Finance (DeFi)': 'DeFi',
        'Decentralized Exchange (DEX)': 'DeFi',
        'Lending/Borrowing': 'DeFi',
        'Yield Farming': 'DeFi',
        'Liquid Staking Derivatives': 'DeFi',
        'Meme': 'Meme',
        'Memecoins': 'Meme',
        'Dog': 'Meme',
        'Cat': 'Meme',
        'Artificial Intelligence (AI)': 'AI',
        'AI Agents': 'AI',
        'Gaming': 'Gaming',
        'Play To Earn': 'Gaming',
        'GameFi': 'Gaming',
        'Metaverse': 'Gaming',
        'Non-Fungible Tokens (NFT)': 'NFT',
        'NFT': 'NFT',
        'Oracle': 'Infrastructure',
        'Oracles': 'Infrastructure',
        'Storage': 'Infrastructure',
        'Social': 'Social',
        'Privacy': 'Privacy',
        'Privacy Coins': 'Privacy',
        'Exchange-based Tokens': 'Exchange',
        'Real World Assets (RWA)': 'RWA',
        'Tokenized Gold': 'RWA',
        'Solana Ecosystem': 'Solana',
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
                break

        if not mapped:
            our_categories['Other'].append(symbol)

    return dict(our_categories)

def save_categories(categories, filename='token_categories.py'):
    """Save categories to Python file"""
    from datetime import datetime
    import shutil

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
    """Get the category for a given token symbol"""
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
    """Get a summary of how many tokens are in each category"""
    summary = {{}}
    for token in tokens:
        category = get_token_category(token)
        summary[category] = summary.get(category, 0) + 1
    return summary
'''

    categories_str = ""
    for cat in sorted(categories.keys()):
        tokens = sorted(categories[cat])
        categories_str += f"    '{cat}': [\n"
        for i in range(0, len(tokens), 10):
            chunk = tokens[i:i+10]
            categories_str += f"        {', '.join(repr(t) for t in chunk)},\n"
        categories_str += "    ],\n\n"

    content = content.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        categories=categories_str
    )

    # Backup
    try:
        shutil.copy(filename, f'{filename}.backup')
        print(f"\n✓ Backed up {filename} -> {filename}.backup")
    except:
        pass

    with open(filename, 'w') as f:
        f.write(content)

    print(f"✓ Updated {filename}")

if __name__ == "__main__":
    print("="*60)
    print("Fast CoinGecko Category Sync")
    print("="*60)
    print()

    # Get Hyperliquid tokens
    hl_tokens = fetch_hyperliquid_tokens()
    print()

    # Get CoinGecko coin list
    symbol_to_id = fetch_all_coins_from_coingecko()
    print()

    # Match tokens
    matched_tokens = []
    unmatched_tokens = []

    for token in hl_tokens:
        if token in symbol_to_id:
            matched_tokens.append((token, symbol_to_id[token]))
        else:
            unmatched_tokens.append(token)

    print(f"✓ Matched {len(matched_tokens)} tokens")
    if unmatched_tokens:
        print(f"⚠️  {len(unmatched_tokens)} unmatched: {', '.join(unmatched_tokens[:10])}{' ...' if len(unmatched_tokens) > 10 else ''}")
    print()

    # Get categories
    print(f"Fetching categories for {len(matched_tokens)} tokens...")
    print(f"Estimated time: ~{len(matched_tokens) * 3 / 60:.1f} minutes")
    print()

    categories_raw = get_categories_batch(matched_tokens)

    # Save raw data
    with open('coingecko_categories_raw.json', 'w') as f:
        json.dump(categories_raw, f, indent=2)
    print(f"\n✓ Saved raw data to coingecko_categories_raw.json")

    # Map to our categories
    our_categories = map_coingecko_to_our_categories(categories_raw)

    # Save
    save_categories(our_categories)

    print("\n" + "="*60)
    print("Category Summary:")
    print("="*60)
    for cat in sorted(our_categories.keys()):
        print(f"{cat:20s}: {len(our_categories[cat]):3d} tokens")

    print("\n✓ Done!")
