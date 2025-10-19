#!/bin/bash

# PathwayLens 2.0 Setup Script
# This script sets up the complete PathwayLens platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

# Function to check Node.js version
check_node_version() {
    if command_exists node; then
        NODE_VERSION=$(node --version | sed 's/v//')
        REQUIRED_VERSION="18.0.0"
        
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
            print_success "Node.js $NODE_VERSION found (>= $REQUIRED_VERSION required)"
            return 0
        else
            print_error "Node.js $NODE_VERSION found, but $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Node.js is not installed"
        return 1
    fi
}

# Function to check Docker
check_docker() {
    if command_exists docker; then
        print_success "Docker found"
        if command_exists docker-compose; then
            print_success "Docker Compose found"
            return 0
        else
            print_warning "Docker Compose not found, but Docker is available"
            return 1
        fi
    else
        print_warning "Docker not found - containerized deployment will not be available"
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PathwayLens with all features
    pip install -e ".[all]"
    
    print_success "Python dependencies installed"
}

# Function to install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    if [ -d "frontend" ]; then
        cd frontend
        npm install
        cd ..
        print_success "Node.js dependencies installed"
    else
        print_warning "Frontend directory not found, skipping Node.js dependencies"
    fi
}

# Function to setup database
setup_database() {
    print_status "Setting up database..."
    
    # Create database directories
    mkdir -p infra/docker/data
    mkdir -p infra/docker/logs
    
    # Set permissions
    chmod 755 infra/docker/data
    chmod 755 infra/docker/logs
    
    print_success "Database directories created"
}

# Function to create configuration files
create_config() {
    print_status "Creating configuration files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# PathwayLens 2.0 Configuration

# Database
DATABASE_URL=postgresql://pathwaylens:pathwaylens_password@localhost:5432/pathwaylens
REDIS_URL=redis://localhost:6379

# API
SECRET_KEY=your-secret-key-change-this-in-production
API_HOST=0.0.0.0
API_PORT=8000

# External APIs
NCBI_API_KEY=your-ncbi-key
STRING_API_TOKEN=your-string-token

# Storage
STORAGE_BACKEND=local
STORAGE_PATH=./data

# Development
DEBUG=true
LOG_LEVEL=INFO
EOF
        print_success "Configuration file created (.env)"
    else
        print_warning "Configuration file already exists (.env)"
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run tests
    python -m pytest tests/ -v
    
    print_success "Tests completed"
}

# Function to start services with Docker
start_docker_services() {
    print_status "Starting services with Docker..."
    
    if command_exists docker-compose; then
        cd infra/docker
        docker-compose up -d
        cd ../..
        print_success "Docker services started"
        print_status "Services available at:"
        print_status "  - API: http://localhost:8000"
        print_status "  - Frontend: http://localhost:3000"
        print_status "  - Flower (Celery): http://localhost:5555"
    else
        print_error "Docker Compose not available"
        return 1
    fi
}

# Function to start services locally
start_local_services() {
    print_status "Starting services locally..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Redis (if available)
    if command_exists redis-server; then
        print_status "Starting Redis..."
        redis-server --daemonize yes
    else
        print_warning "Redis not found, please install and start Redis manually"
    fi
    
    # Start PostgreSQL (if available)
    if command_exists pg_ctl; then
        print_status "Starting PostgreSQL..."
        # This would need to be customized based on your PostgreSQL setup
        print_warning "Please start PostgreSQL manually"
    else
        print_warning "PostgreSQL not found, please install and start PostgreSQL manually"
    fi
    
    print_status "To start the API server:"
    print_status "  source venv/bin/activate"
    print_status "  uvicorn pathwaylens_api.main:app --reload"
    
    print_status "To start the frontend:"
    print_status "  cd frontend"
    print_status "  npm run dev"
}

# Function to show usage
show_usage() {
    echo "PathwayLens 2.0 Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cli-only        Install CLI only (no web interface)"
    echo "  --api-only        Install API only (no frontend)"
    echo "  --full            Install everything (default)"
    echo "  --docker          Use Docker for deployment"
    echo "  --local           Use local installation (default)"
    echo "  --test            Run tests after installation"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --cli-only     # Install CLI only"
    echo "  $0 --docker       # Install with Docker"
    echo "  $0 --full --test  # Install everything and run tests"
}

# Main function
main() {
    echo "🧬 PathwayLens 2.0 Setup Script"
    echo "================================"
    echo ""
    
    # Parse command line arguments
    CLI_ONLY=false
    API_ONLY=false
    FULL_INSTALL=true
    USE_DOCKER=false
    USE_LOCAL=true
    RUN_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cli-only)
                CLI_ONLY=true
                FULL_INSTALL=false
                shift
                ;;
            --api-only)
                API_ONLY=true
                FULL_INSTALL=false
                shift
                ;;
            --full)
                FULL_INSTALL=true
                shift
                ;;
            --docker)
                USE_DOCKER=true
                USE_LOCAL=false
                shift
                ;;
            --local)
                USE_LOCAL=true
                USE_DOCKER=false
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check system requirements
    print_status "Checking system requirements..."
    
    if ! check_python_version; then
        print_error "Python requirements not met"
        exit 1
    fi
    
    if [ "$FULL_INSTALL" = true ] && ! check_node_version; then
        print_error "Node.js requirements not met"
        exit 1
    fi
    
    if [ "$USE_DOCKER" = true ] && ! check_docker; then
        print_error "Docker requirements not met"
        exit 1
    fi
    
    print_success "System requirements check passed"
    echo ""
    
    # Setup steps
    print_status "Starting PathwayLens 2.0 setup..."
    echo ""
    
    # Create virtual environment
    create_venv
    echo ""
    
    # Install Python dependencies
    if [ "$CLI_ONLY" = true ]; then
        print_status "Installing CLI dependencies only..."
        source venv/bin/activate
        pip install -e ".[analysis,pathways]"
    elif [ "$API_ONLY" = true ]; then
        print_status "Installing API dependencies only..."
        source venv/bin/activate
        pip install -e ".[api,jobs,database,auth]"
    else
        install_python_deps
    fi
    echo ""
    
    # Install Node.js dependencies (if full install)
    if [ "$FULL_INSTALL" = true ]; then
        install_node_deps
        echo ""
    fi
    
    # Setup database
    setup_database
    echo ""
    
    # Create configuration files
    create_config
    echo ""
    
    # Run tests (if requested)
    if [ "$RUN_TESTS" = true ]; then
        run_tests
        echo ""
    fi
    
    # Start services
    if [ "$USE_DOCKER" = true ]; then
        start_docker_services
    else
        start_local_services
    fi
    echo ""
    
    # Final instructions
    print_success "PathwayLens 2.0 setup completed!"
    echo ""
    print_status "Next steps:"
    
    if [ "$CLI_ONLY" = true ]; then
        print_status "  - Use the CLI: pathwaylens --help"
    elif [ "$API_ONLY" = true ]; then
        print_status "  - Start the API: uvicorn pathwaylens_api.main:app --reload"
        print_status "  - API docs: http://localhost:8000/docs"
    else
        print_status "  - Use the CLI: pathwaylens --help"
        print_status "  - Start the API: uvicorn pathwaylens_api.main:app --reload"
        print_status "  - Start the frontend: cd frontend && npm run dev"
        print_status "  - API docs: http://localhost:8000/docs"
        print_status "  - Web interface: http://localhost:3000"
    fi
    
    echo ""
    print_status "For more information, see the documentation in the docs/ directory"
}

# Run main function
main "$@"
