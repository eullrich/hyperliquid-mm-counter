#!/usr/bin/env python3
"""
Helper script to update token categories from Hyperliquid UI inspection

INSTRUCTIONS:
1. Go to https://app.hyperliquid.xyz/trade
2. Open DevTools (F12) -> Console
3. Paste this JavaScript and run it:

// Copy this entire block into browser console
(() => {
  const categories = {};

  // Try to find category elements in the DOM
  const categoryElements = document.querySelectorAll('[data-category], .category-tag, .token-category');

  categoryElements.forEach(el => {
    const category = el.getAttribute('data-category') || el.textContent.trim();
    const tokenEl = el.closest('.token-item, [data-token]');
    if (tokenEl) {
      const token = tokenEl.getAttribute('data-token') || tokenEl.querySelector('.token-symbol')?.textContent.trim();
      if (token) {
        if (!categories[category]) categories[category] = [];
        categories[category].push(token);
      }
    }
  });

  // Output as Python dict format
  console.log('TOKEN_CATEGORIES = {');
  Object.entries(categories).forEach(([cat, tokens]) => {
    console.log(`    '${cat}': [`);
    console.log(`        '${tokens.join("', '")}'`);
    console.log('    ],');
  });
  console.log('}');

  // Also copy to clipboard
  const pythonDict = JSON.stringify(categories, null, 2);
  navigator.clipboard.writeText(pythonDict);
  console.log('\nJSON copied to clipboard! You can also use this format.');
})();

4. Copy the output from the console
5. Paste it below in the 'paste_categories_here' section
6. Run: python3 update_categories_from_ui.py

ALTERNATIVE METHOD (if the above doesn't work):
- Manually browse the tokens on Hyperliquid
- For each category visible on the UI, note down the tokens
- Format as Python dict and paste below
"""

# Paste the categories you extracted from Hyperliquid UI here as a Python dict
# Example format:
# UPDATED_CATEGORIES = {
#     'Layer 1': ['BTC', 'ETH', 'SOL', ...],
#     'DeFi': ['UNI', 'AAVE', ...],
#     ...
# }

UPDATED_CATEGORIES = {
    # PASTE YOUR EXTRACTED CATEGORIES HERE
    # Leave empty if using JSON format below
}

# OR paste JSON format here (will be converted):
UPDATED_CATEGORIES_JSON = """
{
}
"""

import json
import os

def update_token_categories():
    """Update token_categories.py with new data"""

    # Check if user provided updates
    if not UPDATED_CATEGORIES and UPDATED_CATEGORIES_JSON.strip() == "{\n}":
        print("ERROR: No categories provided!")
        print("\nPlease either:")
        print("1. Set UPDATED_CATEGORIES dict in this file")
        print("2. Set UPDATED_CATEGORIES_JSON string in this file")
        print("\nSee instructions at the top of this file.")
        return

    # Use JSON if provided
    categories = UPDATED_CATEGORIES
    if UPDATED_CATEGORIES_JSON.strip() != "{\n}":
        try:
            categories = json.loads(UPDATED_CATEGORIES_JSON)
        except json.JSONDecodeError as e:
            print(f"ERROR parsing JSON: {e}")
            return

    if not categories:
        print("No categories to update!")
        return

    # Generate new token_categories.py content
    content = '''"""
Token category mappings for Hyperliquid tokens
Auto-updated from Hyperliquid frontend
"""

# Category definitions (last updated: {timestamp})
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
    from datetime import datetime
    categories_str = ""
    for cat, tokens in categories.items():
        categories_str += f"    '{cat}': [\n"

        # Format tokens in lines of ~10 tokens each
        token_list = tokens if isinstance(tokens, list) else [tokens]
        for i in range(0, len(token_list), 10):
            chunk = token_list[i:i+10]
            categories_str += f"        {', '.join(repr(t) for t in chunk)},\n"

        categories_str += "    ],\n\n"

    # Fill in template
    content = content.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        categories=categories_str
    )

    # Backup old file
    if os.path.exists('token_categories.py'):
        import shutil
        shutil.copy('token_categories.py', 'token_categories.py.backup')
        print("Backed up old token_categories.py -> token_categories.py.backup")

    # Write new file
    with open('token_categories.py', 'w') as f:
        f.write(content)

    print(f"\nUpdated token_categories.py with {len(categories)} categories!")
    print(f"Total tokens: {sum(len(t) if isinstance(t, list) else 1 for t in categories.values())}")
    print("\nCategories updated:")
    for cat in categories.keys():
        print(f"  - {cat}")

if __name__ == "__main__":
    update_token_categories()
