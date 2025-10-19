# PathwayLens 2.0 Deployment Guide

This guide provides comprehensive instructions for deploying PathwayLens 2.0 in various environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Docker Deployment](#docker-deployment)
6. [Local Development](#local-development)
7. [Production Deployment](#production-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Using the Setup Script

The easiest way to get started is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/your-org/pathwaylens.git
cd pathwaylens

# Make setup script executable
chmod +x setup.sh

# Full installation with Docker
./setup.sh --docker

# CLI-only installation
./setup.sh --cli-only

# Full local installation
./setup.sh --full --local
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PathwayLens
pip install -e ".[all]"

# Install frontend dependencies
cd frontend
npm install
cd ..

# Start services
./setup.sh --local
```

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **Node.js**: 18.0 or higher (for web interface)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB free space minimum
- **Network**: Internet connection for external API calls

### Recommended Requirements

- **Python**: 3.11 or higher
- **Node.js**: 20.0 or higher
- **Memory**: 16GB RAM or higher
- **Storage**: 50GB+ SSD storage
- **CPU**: 4+ cores

### External Dependencies

- **PostgreSQL**: 13 or higher (for production)
- **Redis**: 6.0 or higher (for job queue)
- **Docker**: 20.10 or higher (for containerized deployment)

## Installation Methods

### 1. CLI-Only Installation

For users who only need the command-line interface:

```bash
./setup.sh --cli-only
```

This installs:
- PathwayLens CLI
- Core analysis modules
- Pathway databases
- Basic visualization support

### 2. API-Only Installation

For users who need the REST API but not the web interface:

```bash
./setup.sh --api-only
```

This installs:
- PathwayLens API
- Database support
- Job queue system
- Authentication

### 3. Full Installation

For complete functionality including web interface:

```bash
./setup.sh --full
```

This installs:
- All CLI functionality
- REST API
- Web interface
- Job queue system
- Database support
- Authentication
- Visualization engine

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://pathwaylens:password@localhost:5432/pathwaylens
REDIS_URL=redis://localhost:6379

# API Configuration
SECRET_KEY=your-secret-key-change-this-in-production
API_HOST=0.0.0.0
API_PORT=8000

# External API Keys
NCBI_API_KEY=your-ncbi-key
STRING_API_TOKEN=your-string-token

# Storage Configuration
STORAGE_BACKEND=local
STORAGE_PATH=./data

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

### Configuration Files

#### API Configuration (`pathwaylens_api/config.py`)

```python
# API settings
API_VERSION = "v1"
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Job queue settings
CELERY_BROKER_URL = os.getenv("REDIS_URL")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL")
```

#### Frontend Configuration (`frontend/.env.local`)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=PathwayLens
NEXT_PUBLIC_VERSION=2.0.0
```

## Docker Deployment

### Using Docker Compose

The recommended way to deploy PathwayLens 2.0 is using Docker Compose:

```bash
# Navigate to docker directory
cd infra/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services Included

- **PostgreSQL**: Database
- **Redis**: Cache and job queue
- **API**: FastAPI backend
- **Frontend**: Next.js web interface
- **Worker**: Celery worker for background jobs
- **Beat**: Celery scheduler
- **Flower**: Celery monitoring
- **Nginx**: Reverse proxy

### Custom Docker Deployment

#### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash pathwaylens && \
    chown -R pathwaylens:pathwaylens /app
USER pathwaylens

EXPOSE 8000

CMD ["uvicorn", "pathwaylens_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

```dockerfile
FROM node:18-alpine AS base

FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
RUN mkdir .next
RUN chown nextjs:nodejs .next

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

## Local Development

### Prerequisites

1. Install Python 3.8+
2. Install Node.js 18+
3. Install PostgreSQL
4. Install Redis

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/pathwaylens.git
cd pathwaylens

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[all,dev]"

# Install frontend dependencies
cd frontend
npm install
cd ..

# Setup database
createdb pathwaylens
psql pathwaylens < pathwaylens_api/database_schema.sql

# Start Redis
redis-server

# Start API server
uvicorn pathwaylens_api.main:app --reload

# Start frontend (in another terminal)
cd frontend
npm run dev
```

### Development URLs

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Frontend**: http://localhost:3000
- **Flower**: http://localhost:5555

## Production Deployment

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/pathwaylens.git
cd pathwaylens/infra/docker

# Create production environment file
cp .env.example .env.production

# Edit production configuration
nano .env.production

# Start production services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Using Kubernetes

#### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pathwaylens
```

#### PostgreSQL

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: pathwaylens
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: pathwaylens
        - name: POSTGRES_USER
          value: pathwaylens
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
```

#### API Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pathwaylens-api
  namespace: pathwaylens
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pathwaylens-api
  template:
    metadata:
      labels:
        app: pathwaylens-api
    spec:
      containers:
      - name: api
        image: pathwaylens/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
```

### Using Cloud Platforms

#### AWS ECS

```yaml
# task-definition.json
{
  "family": "pathwaylens",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "pathwaylens-api",
      "image": "pathwaylens/api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/pathwaylens"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pathwaylens",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: pathwaylens-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/project/pathwaylens-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

## Monitoring and Maintenance

### Health Checks

#### API Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Database Health Check

```bash
# PostgreSQL
pg_isready -h localhost -p 5432 -U pathwaylens

# Redis
redis-cli ping
```

### Logging

#### API Logs

```bash
# Docker
docker-compose logs -f api

# Local
tail -f logs/api.log
```

#### Application Logs

```bash
# Docker
docker-compose logs -f worker

# Local
tail -f logs/worker.log
```

### Monitoring with Prometheus

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pathwaylens-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 5s

  - job_name: 'pathwaylens-worker'
    static_configs:
      - targets: ['localhost:5555']
    metrics_path: '/metrics'
```

#### Grafana Dashboard

Import the provided Grafana dashboard configuration:

```bash
# Import dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @grafana-dashboard.json
```

### Backup and Recovery

#### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U pathwaylens pathwaylens > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
psql -h localhost -U pathwaylens pathwaylens < backup_20240101_120000.sql
```

#### File Storage Backup

```bash
# Backup data directory
tar -czf data_backup_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Restore
tar -xzf data_backup_20240101_120000.tar.gz
```

### Updates and Upgrades

#### Docker Update

```bash
# Pull latest images
docker-compose pull

# Restart services
docker-compose up -d
```

#### Local Update

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -e ".[all]"

# Run database migrations
alembic upgrade head

# Restart services
systemctl restart pathwaylens-api
systemctl restart pathwaylens-worker
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

**Problem**: API cannot connect to PostgreSQL

**Solution**:
```bash
# Check PostgreSQL status
systemctl status postgresql

# Check connection
psql -h localhost -U pathwaylens -d pathwaylens

# Check environment variables
echo $DATABASE_URL
```

#### 2. Redis Connection Issues

**Problem**: Celery workers cannot connect to Redis

**Solution**:
```bash
# Check Redis status
systemctl status redis

# Test connection
redis-cli ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

#### 3. Frontend Build Issues

**Problem**: Frontend fails to build

**Solution**:
```bash
# Clear node modules
rm -rf frontend/node_modules
rm -rf frontend/.next

# Reinstall dependencies
cd frontend
npm install

# Rebuild
npm run build
```

#### 4. Memory Issues

**Problem**: Out of memory errors

**Solution**:
```bash
# Check memory usage
free -h

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization

#### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_jobs_user_id ON jobs(user_id);
CREATE INDEX CONCURRENTLY idx_jobs_status ON jobs(status);
CREATE INDEX CONCURRENTLY idx_jobs_created_at ON jobs(created_at);

-- Analyze tables
ANALYZE jobs;
ANALYZE job_results;
```

#### Redis Optimization

```bash
# Configure Redis for better performance
echo "maxmemory 2gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
systemctl restart redis
```

#### API Optimization

```python
# Enable connection pooling
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# Enable caching
CACHE_TTL = 3600  # 1 hour
```

### Getting Help

#### Logs and Debugging

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# View detailed logs
tail -f logs/api.log logs/worker.log
```

#### Community Support

- **GitHub Issues**: https://github.com/your-org/pathwaylens/issues
- **Documentation**: https://docs.pathwaylens.org
- **Discord**: https://discord.gg/pathwaylens

#### Professional Support

For enterprise support and consulting:

- **Email**: support@pathwaylens.org
- **Website**: https://pathwaylens.org/support

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying PathwayLens 2.0 in various environments. For additional help or questions, please refer to the documentation or contact support.

**Happy analyzing! 🧬**