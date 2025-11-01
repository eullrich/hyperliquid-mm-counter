# Hyperliquid MM-Counter Trading Dashboard

Real-time trading dashboard for detecting market maker manipulation patterns on Hyperliquid perpetual markets. Tracks MM counter-trading signals using OBV, delta flow, funding rates, and technical indicators across 200+ tokens.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-green.svg)

## Features

- **Real-time WebSocket Data Collection** - Streams candle data, orderbook depth, funding rates, and open interest from Hyperliquid
- **MM Counter-Trading Signals** - Detects 4 signal types:
  - **Buy Dip** - Post-shakeout entry opportunities (MM dumps to shake weak hands)
  - **Fade Pump** - Reversal traps (engineered pumps before rug pulls)
  - **Spoof Alert** - Liquidity manipulation (MM spoofs breakout then pulls bids)
  - **Exit/Accum** - Cycle ending signals (trailing stop or DCA zones)
- **Signal Strength Visualization** - 2-4 confirmation bars based on multiple indicator convergence
- **OBV Z-Score Normalization** - Relative volume intensity highlighting across tokens
- **Interactive Dashboard** - Real-time filtering, search, and signal breakdown
- **Production-Ready Deployment** - Docker Compose orchestration with auto-restart

## Tech Stack

- **Backend**: Python 3.11, FastAPI, PostgreSQL 15
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Data Collection**: WebSockets, Hyperliquid Python SDK
- **Analysis**: NumPy, Pandas (technical indicators: RSI, MACD, EMA, OBV, Delta)
- **Deployment**: Docker, Docker Compose
- **Database**: PostgreSQL with time-series optimizations

## Architecture

```
┌─────────────────┐
│  Hyperliquid    │
│  WebSocket API  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Collector     │─────▶│  PostgreSQL  │
│  (WebSocket)    │      │   Database   │
└─────────────────┘      └──────┬───────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Metrics Analyzer│    │   FastAPI       │    │   Dashboard     │
│ (Every 15 min)  │    │   REST API      │    │  (Static HTML)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 2GB RAM minimum (4GB recommended)
- Port 8000 available

### 1. Clone Repository

```bash
git clone https://github.com/eullrich/hyperliquid-mm-counter.git
cd hyperliquid-mm-counter
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set your DB_PASSWORD
```

### 3. Deploy with Docker

```bash
docker compose up -d
```

### 4. Access Dashboard

Open http://localhost:8000 in your browser.

**Initial data collection takes ~15 minutes** for the first candles to close and signals to compute.

## Deployment Options

### Local Development

```bash
# Run collector only
docker compose up -d postgres collector

# Run metrics analyzer manually
python metrics_analyzer.py

# Access database
docker compose exec postgres psql -U hyperliquid -d hyperliquid_data
```

### Production (DigitalOcean/VPS)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed production deployment instructions.

**Recommended specs:**
- **$12/month droplet**: 2GB RAM, 1 vCPU, 50GB SSD
- **Ubuntu 24.04 LTS**
- **Docker + Docker Compose**

## Signal Detection Logic

### Buy Dip Signal
- Price below 20 EMA
- OBV > 0 (volume supporting upside)
- RSI < 30 (oversold)
- Funding rate < -0.01% (shorts paying longs)

### Fade Pump Signal
- Price above 20 EMA
- OBV < 0 (volume divergence)
- RSI > 70 (overbought)
- Funding rate > 0.01% (longs paying shorts)

### Spoof Alert
- Delta imbalance > 5% (orderbook skew)
- Price deviation from EMA > 2%
- Volume spike (2x+ average)

### Exit/Accum Signal
- Neutral zone (no strong directional bias)
- Low volatility (RSI 30-70)
- Use for trailing stops or DCA

## API Endpoints

### GET /api/metrics
Query latest metrics with filtering:
```bash
curl "http://localhost:8000/api/metrics?interval=5m&sort_by=signal&limit=50"
```

### GET /api/stats
Dashboard summary statistics:
```bash
curl http://localhost:8000/api/stats
```

### GET /api/health
Service health check:
```bash
curl http://localhost:8000/api/health
```

## Database Schema

- **candles** - OHLCV data (5m, 1h, 4h intervals)
- **orderbook_depth** - Best bid/ask snapshots
- **funding_rates** - 8-hour funding rates
- **open_interest** - Per-token OI
- **metrics_snapshot** - Computed signals and indicators

See [migrations/](migrations/) for full schema evolution.

## Configuration

Edit [config.py](config.py) for:
- Database connection (uses env vars in Docker)
- Candle intervals (default: 5m, 1h, 4h)
- Signal thresholds (RSI, funding, OBV)
- Data retention periods (default: 2 days for 5m, 90 days for 4h)

## Monitoring

### Check Service Status
```bash
docker compose ps
```

### View Logs
```bash
docker compose logs -f collector
docker compose logs -f metrics_analyzer
docker compose logs -f api
```

### Database Queries
```bash
# Recent signals
docker compose exec postgres psql -U hyperliquid -d hyperliquid_data -c \
  "SELECT coin, signal, price FROM latest_metrics WHERE signal != 'none' LIMIT 10;"

# Token count
docker compose exec postgres psql -U hyperliquid -d hyperliquid_data -c \
  "SELECT COUNT(DISTINCT coin) FROM candles;"
```

## Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Locally (without Docker)
```bash
# Start PostgreSQL first
createdb hyperliquid_data
psql hyperliquid_data < db_schema.sql

# Run collector
python collector.py

# Run API
uvicorn api:app --reload

# Run metrics analyzer
python metrics_analyzer.py
```

### Database Migrations
```bash
# Apply migrations manually
ls migrations/*.sql | sort | while read f; do
  psql hyperliquid_data < "$f"
done
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Supabase integration for serverless deployment
- [ ] Real-time dashboard updates (WebSocket)
- [ ] Additional signals (liquidation cascades, whale tracking)
- [ ] Mobile-responsive UI improvements
- [ ] Backtesting framework
- [ ] Trading bot integration (demo mode)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and informational purposes only. Not financial advice. Trading cryptocurrencies carries significant risk. Use at your own discretion.

## Support

- **Issues**: [GitHub Issues](https://github.com/eullrich/hyperliquid-mm-counter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/eullrich/hyperliquid-mm-counter/discussions)

## Acknowledgments

- [Hyperliquid](https://hyperliquid.xyz) for the excellent API
- Built with Claude Code
