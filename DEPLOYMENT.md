# DigitalOcean Deployment Guide

## Prerequisites
- DigitalOcean account
- Domain name (optional, for custom URL)
- SSH key configured

## Option 1: Docker Compose (Recommended)

### 1. Create DigitalOcean Droplet
```bash
# Recommended: $12/month (2GB RAM, 1 vCPU, 50GB SSD)
# Select: Ubuntu 22.04 LTS
# Enable: Docker (from Marketplace)
```

### 2. SSH into Droplet
```bash
ssh root@YOUR_DROPLET_IP
```

### 3. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/hedge-v4.git
cd hedge-v4
```

### 4. Configure Environment
```bash
cp .env.example .env
nano .env  # Set DB_PASSWORD to a secure password
```

### 5. Start Services
```bash
docker-compose up -d
```

### 6. Check Status
```bash
docker-compose ps
docker-compose logs -f api
```

### 7. Access Dashboard
```
http://YOUR_DROPLET_IP:8000
```

## Option 2: Manual Setup

### 1. Install Dependencies
```bash
apt update && apt upgrade -y
apt install -y python3.11 python3-pip postgresql-15 nginx
```

### 2. Setup PostgreSQL
```bash
sudo -u postgres psql
CREATE DATABASE hyperliquid_data;
CREATE USER hyperliquid WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE hyperliquid_data TO hyperliquid;
\q
```

### 3. Run Migrations
```bash
cd /root/hedge-v4
for file in migrations/*.sql; do
    psql -U hyperliquid -d hyperliquid_data -f "$file"
done
```

### 4. Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### 5. Create Systemd Services
```bash
# See systemd/ directory for service files
cp systemd/*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable hyperliquid-collector hyperliquid-metrics hyperliquid-api
systemctl start hyperliquid-collector hyperliquid-metrics hyperliquid-api
```

### 6. Setup Nginx (Optional - for custom domain)
```bash
# See nginx/hyperliquid.conf for config
cp nginx/hyperliquid.conf /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/hyperliquid.conf /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## Monitoring

### Check Logs
```bash
# Docker
docker-compose logs -f api
docker-compose logs -f collector
docker-compose logs -f metrics_analyzer

# Systemd
journalctl -u hyperliquid-api -f
journalctl -u hyperliquid-collector -f
journalctl -u hyperliquid-metrics -f
```

### Database Metrics
```bash
psql -U hyperliquid -d hyperliquid_data -c "SELECT COUNT(*) FROM metrics_snapshot;"
psql -U hyperliquid -d hyperliquid_data -c "SELECT coin, signal FROM latest_metrics WHERE signal != 'none';"
```

## Security

1. **Firewall**
```bash
ufw allow 22    # SSH
ufw allow 8000  # API (or 80/443 if using nginx)
ufw enable
```

2. **SSL Certificate** (if using domain)
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com
```

3. **Change Default Passwords**
```bash
# Update .env file
nano .env
docker-compose down && docker-compose up -d
```

## Estimated Costs

**Droplet**: $12/month (2GB RAM)
- API: ~200MB RAM
- Collector: ~300MB RAM
- Metrics Analyzer: ~400MB RAM (when running)
- PostgreSQL: ~500MB RAM
- Total: ~1.4GB RAM (fits in 2GB droplet)

**Storage**: ~5GB/month (with pruning enabled)

**Total**: ~$12-15/month
