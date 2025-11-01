#!/usr/bin/env python3
"""
Category Updater - Scrapes token categories from Hyperliquid frontend

This script fetches category data directly from the Hyperliquid frontend
since categories are not exposed via their API. It works by:
1. Fetching the main HTML from https://app.hyperliquid.xyz
2. Extracting the bundled JavaScript file path
3. Downloading and parsing the JS to find category definitions
4. Converting from category→[coins] to coin→[categories] format
5. Updating token_categories.py with the CATEGORY_MAP

Categories tracked: AI, Defi, Gaming, Layer 1, Layer 2, Meme, Pre-launch
"""

import re
import requests
from datetime import datetime
from typing import Dict, List, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CategoryScraper:
    """Scrapes category data from Hyperliquid frontend"""

    HYPERLIQUID_URL = "https://app.hyperliquid.xyz"

    # Category patterns to search for in the JS bundle
    CATEGORY_PATTERNS = {
        'AI': r'ai:\[([^\]]{50,500})\]',
        'Defi': r'defi:\[([^\]]+)\]',
        'Gaming': r'gaming:\[([^\]]+)\]',
        'Layer 1': r'layer1:\[([^\]]+)\]',
        'Layer 2': r'layer2:\[([^\]]+)\]',
        'Meme': r'meme:\[([^\]]+)\]',
        'Pre-launch': r'prelaunch:\[([^\]]+)\]',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_main_js_url(self) -> str:
        """Fetch the main HTML and extract the JS bundle URL"""
        try:
            logger.info(f"Fetching HTML from {self.HYPERLIQUID_URL}")
            response = self.session.get(self.HYPERLIQUID_URL, timeout=10)
            response.raise_for_status()

            # Look for main JS bundle (e.g., /static/js/main.abc123.js)
            js_pattern = r'/static/js/main\.[a-f0-9]+\.js'
            match = re.search(js_pattern, response.text)

            if match:
                js_path = match.group(0)
                js_url = f"{self.HYPERLIQUID_URL}{js_path}"
                logger.info(f"Found JS bundle: {js_path}")
                return js_url
            else:
                raise ValueError("Could not find main JS bundle in HTML")

        except Exception as e:
            logger.error(f"Error fetching main HTML: {e}")
            raise

    def fetch_js_bundle(self, js_url: str) -> str:
        """Download the JavaScript bundle"""
        try:
            logger.info(f"Downloading JS bundle from {js_url}")
            response = self.session.get(js_url, timeout=10)
            response.raise_for_status()
            logger.info(f"Downloaded {len(response.text)} bytes")
            return response.text

        except Exception as e:
            logger.error(f"Error fetching JS bundle: {e}")
            raise

    def extract_categories(self, js_content: str) -> Dict[str, List[str]]:
        """Extract category definitions from JS bundle"""
        categories = {}

        for category_name, pattern in self.CATEGORY_PATTERNS.items():
            try:
                match = re.search(pattern, js_content, re.IGNORECASE)
                if match:
                    # Extract the array content
                    array_content = match.group(1)

                    # Parse token symbols - look for quoted strings
                    token_pattern = r'["\']([A-Z0-9]+)["\']'
                    tokens = re.findall(token_pattern, array_content)

                    if tokens:
                        categories[category_name] = sorted(set(tokens))
                        logger.info(f"Found {len(tokens)} tokens in {category_name}: {tokens[:5]}...")
                    else:
                        logger.warning(f"No tokens found for {category_name}")
                else:
                    logger.warning(f"Pattern not found for {category_name}")

            except Exception as e:
                logger.error(f"Error extracting {category_name}: {e}")

        return categories

    def convert_to_token_map(self, categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Convert from category→[tokens] to token→[categories] format"""
        token_map = {}

        for category, tokens in categories.items():
            for token in tokens:
                if token not in token_map:
                    token_map[token] = []
                token_map[token].append(category)

        return token_map

    def scrape(self) -> Dict[str, List[str]]:
        """Main scraping method - returns categories dict"""
        try:
            # Step 1: Get JS bundle URL
            js_url = self.fetch_main_js_url()

            # Step 2: Download JS bundle
            js_content = self.fetch_js_bundle(js_url)

            # Step 3: Extract categories
            categories = self.extract_categories(js_content)

            if not categories:
                raise ValueError("No categories extracted from JS bundle")

            logger.info(f"Successfully extracted {len(categories)} categories")
            return categories

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raise


def update_token_categories_file(categories: Dict[str, List[str]]):
    """Update token_categories.py with new category data"""

    # Generate file content
    content = f'''"""
Token category mappings for Hyperliquid tokens
Auto-generated from Hyperliquid frontend
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Category definitions
TOKEN_CATEGORIES = {{
'''

    # Add each category
    for category, tokens in sorted(categories.items()):
        content += f"    '{category}': [\n"

        # Format tokens in rows of ~10
        for i in range(0, len(tokens), 10):
            chunk = tokens[i:i+10]
            content += f"        {', '.join(repr(t) for t in chunk)},\n"

        content += "    ],\n\n"

    content += '''
}

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
    summary = {}
    for token in tokens:
        category = get_token_category(token)
        summary[category] = summary.get(category, 0) + 1
    return summary
'''

    # Backup old file
    try:
        import shutil
        import os
        if os.path.exists('token_categories.py'):
            shutil.copy('token_categories.py', 'token_categories.py.backup')
            logger.info("Backed up token_categories.py -> token_categories.py.backup")
    except Exception as e:
        logger.warning(f"Could not backup old file: {e}")

    # Write new file
    with open('token_categories.py', 'w') as f:
        f.write(content)

    total_tokens = sum(len(tokens) for tokens in categories.values())
    logger.info(f"Updated token_categories.py with {len(categories)} categories, {total_tokens} tokens")

    # Print summary
    print("\n=== Category Update Summary ===")
    for category, tokens in sorted(categories.items()):
        print(f"{category}: {len(tokens)} tokens")
    print(f"\nTotal: {total_tokens} tokens in {len(categories)} categories")


def main():
    """Main entry point"""
    try:
        logger.info("Starting category scraper...")

        scraper = CategoryScraper()
        categories = scraper.scrape()

        if categories:
            update_token_categories_file(categories)
            logger.info("Category update completed successfully!")
            return 0
        else:
            logger.error("No categories scraped")
            return 1

    except Exception as e:
        logger.error(f"Category update failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
