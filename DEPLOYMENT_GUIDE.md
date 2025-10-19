# PathwayLens 2.0 Deployment Guide

## 🚀 Quick Start

PathwayLens 2.0 is now fully functional and ready for deployment. This guide covers all deployment options from development to production.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Node.js**: 18 or higher (for frontend)
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 50GB free space

### Required Services
- **PostgreSQL**: 15 or higher
- **Redis**: 7 or higher
- **Nginx**: For production reverse proxy

## 🔧 Installation Options

### Option 1: Easy Installation (Recommended)

```bash
# Install core features (lightweight, ~50MB)
pip install pathwaylens

# Install all features (full functionality, ~500MB)
pip install pathwaylens[all]

# Install specific features
pip install pathwaylens[analysis]  # Core analysis tools
pip install pathwaylens[viz]       # Visualization tools
pip install pathwaylens[pathways]  # Pathway databases

# Verify installation
pathwaylens --help
```

### Option 2: Development Installation

```bash
# Create and activate conda environment
conda create -n pathwaylens python=3.10
conda activate pathwaylens

# Install PathwayLens in editable mode
cd /path/to/PathwayLens
pip install -e ".[dev]"

# Verify installation
pathwaylens --help
```

### Option 2: Docker Compose (Recommended for Production)

```bash
# Clone and navigate to project
cd /path/to/PathwayLens

# Start all services
docker compose -f infra/docker/docker-compose.yml up -d

# Check service status
docker compose -f infra/docker/docker-compose.yml ps
```

### Option 3: Kubernetes (Enterprise Deployment)

```bash
# Apply Kubernetes configurations
kubectl apply -f infra/kubernetes/

# Check pod status
kubectl get pods -n pathwaylens
```

## 🧬 CLI Usage

The PathwayLens CLI is now available as a standalone command:

```bash
# Basic usage
pathwaylens --help

# Normalize gene IDs
pathwaylens normalize gene-ids --input genes.csv --species human

# Perform pathway analysis
pathwaylens analyze enrichment --input deseq2_results.csv --databases kegg,reactome

# Compare datasets
pathwaylens compare --input dataset1.csv dataset2.csv --species human

# Generate visualizations
pathwaylens visualize --input analysis_results.json --type enrichment_plot
```

## 🌐 Web Interface

### Development Mode
```bash
# Start API server
cd /path/to/PathwayLens
conda activate pathwaylens
uvicorn pathwaylens_api.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Production Mode
```bash
# Using Docker Compose
docker compose -f infra/docker/docker-compose.yml up -d

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# Flower (Celery monitoring): http://localhost:5555
```

## 🗄️ Database Setup

### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE pathwaylens;
CREATE USER pathwaylens WITH PASSWORD 'pathwaylens_password';
GRANT ALL PRIVILEGES ON DATABASE pathwaylens TO pathwaylens;
```

### Database Migration
```bash
# Run database migrations
cd /path/to/PathwayLens
conda activate pathwaylens
alembic upgrade head
```

## 🔄 Background Jobs

### Celery Worker
```bash
# Start Celery worker
celery -A pathwaylens_api.celery_app worker --loglevel=info

# Start Celery beat (scheduler)
celery -A pathwaylens_api.celery_app beat --loglevel=info

# Monitor with Flower
celery -A pathwaylens_api.celery_app flower --port=5555
```

## 📊 Monitoring and Logging

### Health Checks
- **API Health**: `GET http://localhost:8000/health`
- **Frontend Health**: `GET http://localhost:3000/api/health`
- **Database Health**: `GET http://localhost:8000/health/database`

### Logging
- **API Logs**: `infra/docker/logs/api.log`
- **Worker Logs**: `infra/docker/logs/worker.log`
- **Frontend Logs**: `infra/docker/logs/frontend.log`

### Metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## 🔒 Security Configuration

### Environment Variables
```bash
# Production environment
export PATHWAYLENS_ENV=production
export SECRET_KEY=your-secret-key-here
export DATABASE_URL=postgresql://user:pass@host:port/db
export REDIS_URL=redis://host:port
```

### SSL/TLS Setup
```bash
# Generate SSL certificates
mkdir -p infra/docker/ssl
openssl req -x509 -newkey rsa:4096 -keyout infra/docker/ssl/key.pem -out infra/docker/ssl/cert.pem -days 365 -nodes
```

## 🚀 Production Deployment

### 1. Environment Setup
```bash
# Set production environment variables
export PATHWAYLENS_ENV=production
export DATABASE_URL=postgresql://pathwaylens:secure_password@db_host:5432/pathwaylens
export REDIS_URL=redis://redis_host:6379
export SECRET_KEY=your-secure-secret-key
```

### 2. Database Setup
```bash
# Create production database
psql -h db_host -U postgres -c "CREATE DATABASE pathwaylens;"
psql -h db_host -U postgres -c "CREATE USER pathwaylens WITH PASSWORD 'secure_password';"
psql -h db_host -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE pathwaylens TO pathwaylens;"
```

### 3. Deploy with Docker Compose
```bash
# Start production services
docker compose -f infra/docker/docker-compose.yml up -d

# Verify deployment
curl http://localhost/health
```

### 4. Nginx Configuration
```nginx
# /etc/nginx/sites-available/pathwaylens
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🧪 Testing

### Run Test Suite
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=pathwaylens_core --cov=pathwaylens_cli --cov=pathwaylens_api
```

### System Integration Test
```bash
# Run comprehensive system test
python test_system.py
```

## 🔧 Troubleshooting

### Common Issues

1. **CLI Command Not Found**
   ```bash
   # Reinstall in editable mode
   pip install -e .
   
   # Verify installation
   which pathwaylens
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   psql -h localhost -U pathwaylens -d pathwaylens -c "SELECT 1;"
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connection
   redis-cli ping
   ```

4. **Docker Services Not Starting**
   ```bash
   # Check service logs
   docker compose -f infra/docker/docker-compose.yml logs
   
   # Restart services
   docker compose -f infra/docker/docker-compose.yml restart
   ```

### Performance Optimization

1. **Database Optimization**
   ```sql
   -- Add indexes for better performance
   CREATE INDEX CONCURRENTLY idx_jobs_status ON jobs(status);
   CREATE INDEX CONCURRENTLY idx_jobs_created_at ON jobs(created_at);
   ```

2. **Redis Configuration**
   ```bash
   # Optimize Redis memory usage
   redis-cli CONFIG SET maxmemory 2gb
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Blueprint**: `PATHWAYLENS_2.0_UPGRADE_BLUEPRINT.md`
- **Test Suite**: `tests/`
- **Configuration**: `infra/`

## 🎯 System Status

✅ **CLI**: Fully functional standalone command  
✅ **API Backend**: FastAPI with job management  
✅ **Frontend**: Next.js with modern UI  
✅ **Database**: PostgreSQL with proper schema  
✅ **Background Jobs**: Celery with Redis  
✅ **Docker**: Complete containerization  
✅ **Kubernetes**: Production-ready manifests  
✅ **CI/CD**: GitHub Actions workflows  
✅ **Monitoring**: Prometheus and Grafana  
✅ **Security**: JWT authentication and HTTPS  

## 🚀 Next Steps

1. **Customize Configuration**: Modify settings in `pathwaylens_core/utils/config.py`
2. **Add Custom Analyses**: Extend `pathwaylens_core/analysis/` modules
3. **Deploy to Cloud**: Use provided Kubernetes manifests
4. **Set Up Monitoring**: Configure Prometheus and Grafana
5. **Backup Strategy**: Implement database and file backups

---

**PathwayLens 2.0 is now fully deployed and ready for production use!** 🎉

For support or questions, please refer to the documentation or contact the development team.
