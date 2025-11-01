-- Useful SQL queries for category-based analysis

-- 1. Get latest prices by category
SELECT
    t.category,
    t.coin,
    c.close as price,
    c.timestamp
FROM tokens t
JOIN LATERAL (
    SELECT close, timestamp
    FROM candles
    WHERE coin = t.coin AND interval = '1h'
    ORDER BY timestamp DESC
    LIMIT 1
) c ON true
WHERE t.is_active = true
ORDER BY t.category, t.coin;

-- 2. Average 24h volume by category
SELECT
    t.category,
    COUNT(DISTINCT t.coin) as num_tokens,
    AVG(vol.volume_24h) as avg_volume,
    SUM(vol.volume_24h) as total_volume
FROM tokens t
JOIN LATERAL (
    SELECT SUM(volume) as volume_24h
    FROM candles
    WHERE coin = t.coin
      AND interval = '1h'
      AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
) vol ON true
WHERE t.is_active = true
GROUP BY t.category
ORDER BY total_volume DESC;

-- 3. Get all DeFi tokens with latest price and spread
SELECT
    t.coin,
    c.close as price,
    o.best_bid,
    o.best_ask,
    o.spread_bps,
    c.timestamp
FROM tokens t
JOIN LATERAL (
    SELECT close, timestamp
    FROM candles
    WHERE coin = t.coin AND interval = '1h'
    ORDER BY timestamp DESC
    LIMIT 1
) c ON true
JOIN LATERAL (
    SELECT best_bid, best_ask, spread_bps
    FROM orderbook
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) o ON true
WHERE t.category = 'DeFi' AND t.is_active = true
ORDER BY t.coin;

-- 4. Compare meme coins vs layer 1 performance (24h change)
WITH price_changes AS (
    SELECT
        t.coin,
        t.category,
        first_value(c.close) OVER (PARTITION BY t.coin ORDER BY c.timestamp DESC) as current_price,
        first_value(c.close) OVER (PARTITION BY t.coin ORDER BY c.timestamp ASC) as price_24h_ago
    FROM tokens t
    JOIN candles c ON t.coin = c.coin
    WHERE c.interval = '1h'
      AND c.timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
      AND t.is_active = true
      AND t.category IN ('Meme', 'Layer 1')
)
SELECT DISTINCT
    coin,
    category,
    current_price,
    ((current_price - price_24h_ago) / price_24h_ago * 100) as change_24h_pct
FROM price_changes
ORDER BY category, change_24h_pct DESC;

-- 5. Top 10 tokens by volume in each category (last 24h)
WITH category_volumes AS (
    SELECT
        t.category,
        t.coin,
        SUM(c.volume) as volume_24h,
        ROW_NUMBER() OVER (PARTITION BY t.category ORDER BY SUM(c.volume) DESC) as rank
    FROM tokens t
    JOIN candles c ON t.coin = c.coin
    WHERE c.interval = '1h'
      AND c.timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
      AND t.is_active = true
    GROUP BY t.category, t.coin
)
SELECT category, coin, volume_24h
FROM category_volumes
WHERE rank <= 10
ORDER BY category, rank;

-- 6. Get funding rates by category (showing sentiment)
SELECT
    t.category,
    COUNT(DISTINCT t.coin) as num_tokens,
    AVG(f.rate) as avg_funding_rate,
    MIN(f.rate) as min_funding_rate,
    MAX(f.rate) as max_funding_rate
FROM tokens t
JOIN LATERAL (
    SELECT rate
    FROM funding_rates
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) f ON true
WHERE t.is_active = true
GROUP BY t.category
ORDER BY avg_funding_rate DESC;

-- 7. Open Interest by category
SELECT
    t.category,
    COUNT(DISTINCT t.coin) as num_tokens,
    SUM(oi.value) as total_oi
FROM tokens t
JOIN LATERAL (
    SELECT value
    FROM open_interest
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) oi ON true
WHERE t.is_active = true
GROUP BY t.category
ORDER BY total_oi DESC;

-- 8. List all tokens in a specific category with key metrics
SELECT
    t.coin,
    c.close as price,
    c.volume as volume_1h,
    o.spread_bps,
    f.rate as funding_rate,
    oi.value as open_interest
FROM tokens t
JOIN LATERAL (
    SELECT close, volume
    FROM candles
    WHERE coin = t.coin AND interval = '1h'
    ORDER BY timestamp DESC
    LIMIT 1
) c ON true
LEFT JOIN LATERAL (
    SELECT spread_bps
    FROM orderbook
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) o ON true
LEFT JOIN LATERAL (
    SELECT rate
    FROM funding_rates
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) f ON true
LEFT JOIN LATERAL (
    SELECT value
    FROM open_interest
    WHERE coin = t.coin
    ORDER BY timestamp DESC
    LIMIT 1
) oi ON true
WHERE t.category = 'AI' AND t.is_active = true
ORDER BY c.volume DESC;
