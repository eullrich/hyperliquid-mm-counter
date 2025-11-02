# DigitalOcean Droplet Management

Quick reference for managing the production deployment.

## Droplet Information

- **IP Address**: `167.172.85.88`
- **Name**: `ubuntu-s-1vcpu-2gb-amd-sgp1-01`
- **Region**: Singapore (sgp1)
- **Size**: 2GB RAM, 1 vCPU, 50GB SSD ($12/month)
- **OS**: Ubuntu 24.04.3 LTS
- **User**: `root`

## Quick Access

### SSH Connection
```bash
ssh root@167.172.85.88
```

### Dashboard URL
http://167.172.85.88:8000

## Common Management Commands

### Check Service Status
```bash
ssh root@167.172.85.88 "cd /root/hyperliquid-mm-counter && docker compose ps"
```

### View Logs
```bash
# All services
ssh root@167.172.85.88 "cd /root/hyperliquid-mm-counter && docker compose logs -f"

# Specific service
ssh root@167.172.85.88 "docker logs hyperliquid_collector -f"
ssh root@167.172.85.88 "docker logs hyperliquid_metrics -f"
ssh root@167.172.85.88 "docker logs hyperliquid_api -f"
```

### Deploy Updates
```bash
# Pull latest code and rebuild
ssh root@167.172.85.88 "cd /root/hyperliquid-mm-counter && \
  git pull && \
  docker compose down && \
  docker compose up -d --build"
```

### Restart Services
```bash
# Restart all
ssh root@167.172.85.88 "cd /root/hyperliquid-mm-counter && docker compose restart"

# Restart specific service
ssh root@167.172.85.88 "docker restart hyperliquid_collector"
ssh root@167.172.85.88 "docker restart hyperliquid_metrics"
ssh root@167.172.85.88 "docker restart hyperliquid_api"
```

### Database Access
```bash
# Connect to PostgreSQL
ssh root@167.172.85.88 "docker exec -it hyperliquid_db psql -U hyperliquid -d hyperliquid_data"

# Run query
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c 'SELECT COUNT(*) FROM candles;'"

# Check latest signals
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c \"SELECT coin, signal, price FROM latest_metrics WHERE signal != 'none' LIMIT 10;\""
```

### Resource Monitoring
```bash
# Disk usage
ssh root@167.172.85.88 "df -h"

# Docker stats
ssh root@167.172.85.88 "docker stats --no-stream"

# Memory usage
ssh root@167.172.85.88 "free -h"
```

## File Locations

- **Project Directory**: `/root/hyperliquid-mm-counter`
- **Docker Volumes**: `/var/lib/docker/volumes/`
- **Logs**: `/root/hyperliquid-mm-counter/logs/`

## Service Ports

- **API**: 8000
- **PostgreSQL**: 5432

## Troubleshooting

### Service Won't Start
```bash
# Check logs for errors
ssh root@167.172.85.88 "docker compose logs --tail=50 [service_name]"

# Check if port is in use
ssh root@167.172.85.88 "lsof -i :8000"
```

### Database Issues
```bash
# Check PostgreSQL health
ssh root@167.172.85.88 "docker exec hyperliquid_db pg_isready -U hyperliquid"

# Check database size
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c \"SELECT pg_size_pretty(pg_database_size('hyperliquid_data'));\""
```

### Out of Disk Space
```bash
# Clean Docker images
ssh root@167.172.85.88 "docker system prune -a"

# Clean logs
ssh root@167.172.85.88 "cd /root/hyperliquid-mm-counter && truncate -s 0 logs/*.log"
```

## SSH Key Setup

If you need to add SSH keys for password-less access:

```bash
# Copy your local SSH key to droplet
ssh-copy-id root@167.172.85.88

# Or manually
cat ~/.ssh/id_ed25519.pub | ssh root@167.172.85.88 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

## Backup & Recovery

### Backup Database
```bash
ssh root@167.172.85.88 "docker exec hyperliquid_db pg_dump -U hyperliquid hyperliquid_data > /root/backup_$(date +%Y%m%d).sql"

# Download backup
scp root@167.172.85.88:/root/backup_*.sql ./
```

### Restore Database
```bash
# Upload backup
scp ./backup_20251101.sql root@167.172.85.88:/root/

# Restore
ssh root@167.172.85.88 "docker exec -i hyperliquid_db psql -U hyperliquid -d hyperliquid_data < /root/backup_20251101.sql"
```

## Environment Variables

Located in `/root/hyperliquid-mm-counter/.env`:
```bash
DB_PASSWORD=HyperLiquid2024!SecureDB
```

To update:
```bash
ssh root@167.172.85.88 "nano /root/hyperliquid-mm-counter/.env"
```

## Metrics

### Data Collection Stats
```bash
# Candle count
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c 'SELECT COUNT(*) FROM candles;'"

# Token count
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c 'SELECT COUNT(DISTINCT coin) FROM candles;'"

# Signal breakdown
ssh root@167.172.85.88 "docker exec hyperliquid_db psql -U hyperliquid -d hyperliquid_data -c \"SELECT signal, COUNT(*) FROM latest_metrics GROUP BY signal;\""
```

## Cost Monitoring

- **Monthly Cost**: $12 (2GB droplet)
- **Bandwidth**: Unlimited
- **Backups**: $1.20/month (optional)

Check DigitalOcean dashboard for current usage and billing.
