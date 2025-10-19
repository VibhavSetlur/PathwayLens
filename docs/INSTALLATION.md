# PathwayLens 2.0 Installation Guide

## Overview

PathwayLens 2.0 is a next-generation computational biology platform that provides comprehensive pathway analysis capabilities. This guide will help you install and configure PathwayLens on your system.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM
- **Storage**: 2 GB free space
- **Operating System**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.10 or higher
- **Memory**: 8 GB RAM or more
- **Storage**: 10 GB free space
- **CPU**: Multi-core processor

## Installation Methods

### Method 1: Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/pathwaylens/pathwaylens.git
cd pathwaylens

# Run the setup script
python setup.py

# Test the installation
pathwaylens --version
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/pathwaylens/pathwaylens.git
cd pathwaylens

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# Create configuration
pathwaylens config init
```

### Method 3: Docker Installation

```bash
# Build the Docker image
docker build -t pathwaylens:latest .

# Run PathwayLens
docker run -it pathwaylens:latest
```

## Configuration

### Initial Configuration

After installation, configure PathwayLens:

```bash
# Initialize configuration
pathwaylens config init

# View current configuration
pathwaylens config show

# Set specific values
pathwaylens config set databases.kegg.enabled true
pathwaylens config set analysis.significance_threshold 0.01
```

### Environment Variables

Create a `.env` file in your project directory:

```bash
# Database URLs
DATABASE_URL=postgresql://user:pass@localhost:5432/pathwaylens
REDIS_URL=redis://localhost:6379/0

# API Configuration
SECRET_KEY=your-secret-key-here
API_HOST=0.0.0.0
API_PORT=8000

# External APIs
NCBI_API_KEY=your-ncbi-key-here
STRING_API_TOKEN=your-string-token-here

# Storage
STORAGE_BACKEND=local
STORAGE_PATH=/path/to/storage

# Development
DEBUG=false
LOG_LEVEL=INFO
```

## Verification

### Test Installation

```bash
# Check version
pathwaylens --version

# Test CLI commands
pathwaylens info
pathwaylens info databases

# Test with sample data
pathwaylens normalize --help
pathwaylens analyze --help
```

### Test Database Connections

```bash
# Check database availability
pathwaylens info databases

# Test specific database
pathwaylens info databases --verbose
```

## Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version

# If version is too old, upgrade Python
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.10

# On macOS with Homebrew:
brew install python@3.10
```

#### 2. Permission Issues
```bash
# Install with user flag
pip install --user -e .

# Or use virtual environment
python -m venv pathwaylens-env
source pathwaylens-env/bin/activate  # On Windows: pathwaylens-env\Scripts\activate
pip install -e .
```

#### 3. Dependency Conflicts
```bash
# Create clean environment
python -m venv clean-env
source clean-env/bin/activate
pip install --upgrade pip
pip install -e .
```

#### 4. Network Issues
```bash
# Use alternative package index
pip install -i https://pypi.org/simple/ -e .

# Or use conda
conda create -n pathwaylens python=3.10
conda activate pathwaylens
pip install -e .
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look in `.pathwaylens/logs/` for error messages
2. **Verify configuration**: Run `pathwaylens config show` to check settings
3. **Test connectivity**: Run `pathwaylens info databases` to test database connections
4. **Check dependencies**: Ensure all required packages are installed

### Support

For additional help:
- **Documentation**: Check the `docs/` directory
- **Issues**: Report issues on GitHub
- **Community**: Join our discussion forum

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: `docs/QUICKSTART.md`
2. **Explore the CLI**: `pathwaylens --help`
3. **Try sample analyses**: Use the example data in `tests/data/`
4. **Configure your environment**: Set up your preferred databases and parameters

## Uninstallation

To remove PathwayLens:

```bash
# Remove the package
pip uninstall pathwaylens

# Remove configuration and cache
rm -rf ~/.pathwaylens

# Remove project directory
rm -rf /path/to/pathwaylens
```
