#!/usr/bin/env python3
"""
Fetch token categories from Hyperliquid frontend
Updates token_categories.py with the latest mappings
"""

import requests
import re
import json

def fetch_hyperliquid_categories():
    """Scrape categories from Hyperliquid frontend"""

    # Fetch the main app page
    url = "https://app.hyperliquid.xyz/trade"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        html = response.text

        # Look for category data in the JavaScript bundles
        # Hyperliquid loads their data in webpack chunks
        bundle_urls = re.findall(r'src="(/static/js/[^"]+\.js)"', html)

        categories = {}

        for bundle_path in bundle_urls[:10]:  # Check first 10 bundles
            bundle_url = f"https://app.hyperliquid.xyz{bundle_path}"
            print(f"Checking bundle: {bundle_path}")

            try:
                bundle_response = requests.get(bundle_url, timeout=10)
                bundle_js = bundle_response.text

                # Look for category definitions
                # Pattern: categories or groups with token arrays
                # This is a heuristic - may need adjustment based on their actual code

                # Try to find category objects
                category_patterns = [
                    r'"(Layer\s*[12]|AI|DeFi|Gaming|Meme|NFT|Infrastructure|Social|Privacy|Exchange|HIP-3|RWA|Data|Restaking|Modular|Agent|Solana|Trading)":\s*\[([^\]]+)\]',
                    r'categories:\s*{([^}]+)}',
                    r'tokenCategories:\s*{([^}]+)}',
                ]

                for pattern in category_patterns:
                    matches = re.findall(pattern, bundle_js, re.IGNORECASE)
                    if matches:
                        print(f"Found potential category data with pattern: {pattern[:50]}...")
                        print(f"Matches: {len(matches)}")

                        for match in matches[:5]:  # Show first 5
                            print(f"  {match}")

            except Exception as e:
                print(f"Error fetching bundle {bundle_path}: {e}")
                continue

        if not categories:
            print("\nCould not automatically extract categories from frontend.")
            print("The category data may be:")
            print("1. Loaded dynamically via API")
            print("2. Embedded in a different format")
            print("3. Obfuscated in the webpack bundle")
            print("\nRecommendation: Manually inspect the Hyperliquid frontend")
            print("or use their public documentation if available.")

        return categories

    except Exception as e:
        print(f"Error fetching Hyperliquid frontend: {e}")
        return {}

def check_api_for_categories():
    """Check if Hyperliquid API exposes category information"""

    api_url = "https://api.hyperliquid.xyz/info"

    # Try different request types
    request_types = [
        {"type": "metaAndAssetCtxs"},
        {"type": "spotMetaAndAssetCtxs"},
        {"type": "universe"},
        {"type": "meta"},
    ]

    for req_type in request_types:
        try:
            print(f"\nTrying API request: {req_type}")
            response = requests.post(api_url, json=req_type, timeout=10)
            data = response.json()

            # Check if response contains category information
            data_str = json.dumps(data)

            category_keywords = ['category', 'categories', 'tag', 'tags', 'group', 'sector']
            found_keywords = [kw for kw in category_keywords if kw.lower() in data_str.lower()]

            if found_keywords:
                print(f"Found keywords: {found_keywords}")
                print(f"Response preview: {json.dumps(data, indent=2)[:500]}...")
            else:
                print("No category data found in response")

        except Exception as e:
            print(f"Error with API request {req_type}: {e}")

def manual_browser_instructions():
    """Print instructions for manually extracting categories"""

    print("\n" + "="*60)
    print("MANUAL EXTRACTION INSTRUCTIONS")
    print("="*60)
    print("""
To manually extract categories from Hyperliquid frontend:

1. Open https://app.hyperliquid.xyz/trade in Chrome/Firefox
2. Open Developer Tools (F12)
3. Go to Console tab
4. Run this JavaScript:

   // Extract category data from the page
   let categories = {};

   // Method 1: Check React DevTools state
   // Install React DevTools extension and inspect component state

   // Method 2: Check for category in localStorage
   console.log(localStorage);

   // Method 3: Monitor network requests for category data
   // Go to Network tab, filter for "info" or "meta"
   // Look for responses containing category mappings

   // Method 4: Search in page source
   // Search for "Layer 1", "DeFi", "Meme" etc in the page source

5. Copy the category data and update token_categories.py

Alternatively, check if Hyperliquid has:
- GitHub repo with category definitions
- Public API documentation
- Discord/docs with category info
""")
    print("="*60)

if __name__ == "__main__":
    print("Attempting to fetch Hyperliquid token categories...\n")

    # Try API first (faster)
    print("Step 1: Checking API endpoints...")
    check_api_for_categories()

    # Try scraping frontend
    print("\n" + "="*60)
    print("Step 2: Scraping frontend bundles...")
    print("="*60)
    categories = fetch_hyperliquid_categories()

    if not categories:
        manual_browser_instructions()

    print("\nNote: The current token_categories.py file is manually maintained.")
    print("If categories are not available via API/scraping, you'll need to")
    print("update it manually by inspecting the Hyperliquid frontend.")
