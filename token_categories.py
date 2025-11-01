"""
Token category mappings for Hyperliquid tokens
Auto-generated from Hyperliquid frontend
Last updated: 2025-11-01 18:13:26
"""

# Category definitions
TOKEN_CATEGORIES = {
    'AI': [
        '0G', 'AI16Z', 'AIXBT', 'FET', 'GRASS', 'GRIFFAIN', 'IO', 'KAITO', 'NEAR', 'PROMPT',
        'RENDER', 'RNDR', 'TAO', 'VIRTUAL', 'VVV', 'WLD', 'ZEREBRO',
    ],

    'Defi': [
        'AAVE', 'ALT', 'BANANA', 'BNT', 'CAKE', 'COMP', 'CRV', 'DYDX', 'EIGEN', 'ENA',
        'ETHFI', 'FRAX', 'GMX', 'INJ', 'JTO', 'JUP', 'LDO', 'LINK', 'MAV', 'MET',
        'MORPHO', 'PENDLE', 'PUMP', 'PYTH', 'RESOLV', 'REZ', 'RLB', 'RSR', 'RUNE', 'SKY',
        'SNX', 'STBL', 'SUSHI', 'SYRUP', 'TRB', 'UMA', 'UNI', 'WCT',
    ],

    'Gaming': [
        'ACE', 'APE', 'BIGTIME', 'BLZ', 'GALA', 'GMT', 'HMSTR', 'ILV', 'IMX', 'MAVIA',
        'NOT', 'NXPC', 'SAND', 'SUPER', 'XAI', 'YGG',
    ],

    'Layer 1': [
        '0G', 'ADA', 'ALGO', 'APT', 'ATOM', 'AVAX', 'BCH', 'BERA', 'BNB', 'BSV',
        'BTC', 'CFX', 'DOT', 'DYM', 'ETC', 'ETH', 'FTM', 'HYPE', 'INIT', 'INJ',
        'IP', 'KAS', 'LTC', 'MINA', 'MON', 'NEAR', 'NEO', 'NTRN', 'OM', 'POLYX',
        'RUNE', 'S', 'SAGA', 'SEI', 'SOL', 'SUI', 'TIA', 'TON', 'TRX', 'XLM',
        'XPL', 'XRP', 'ZEN', 'ZETA',
    ],

    'Layer 2': [
        'ARB', 'BLAST', 'CELO', 'HEMI', 'IMX', 'LAYER', 'LINEA', 'MATIC', 'MEGA', 'MNT',
        'MOVE', 'OP', 'POL', 'SCR', 'STARK', 'STRK', 'ZK',
    ],

    'Meme': [
        'BOME', 'BRETT', 'CHILLGUY', 'DOGE', 'FARTCOIN', 'GOAT', 'HPOS', 'JELLY', 'MELANIA', 'MEME',
        'MEW', 'MOODENG', 'MYRO', 'PENGU', 'PEOPLE', 'PNUT', 'POPCAT', 'SHIA', 'SPX', 'TRUMP',
        'TST', 'TURBO', 'VINE', 'WIF', 'YZY',
    ],

    'Pre-launch': [
        'CC', 'MEGA', 'MON',
    ],


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
