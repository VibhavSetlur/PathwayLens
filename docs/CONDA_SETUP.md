# Conda Environment Setup Guide

This guide explains how to set up a conda environment for PathwayLens 2.0 development and testing.

## Prerequisites

- **Miniconda** or **Anaconda** installed
  - Download from: https://docs.conda.io/en/latest/miniconda.html
  - Verify installation: `conda --version`

## Quick Start

### Option 1: Automated Setup (Recommended)

Use the provided setup script:

```bash
bash scripts/setup/conda_setup.sh
```

This will:
1. Create a conda environment named `pathwaylens`
2. Install all dependencies from `environment.yml`
3. Install PathwayLens in development mode
4. Verify the installation

### Option 2: Manual Setup

1. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate pathwaylens
   ```

3. **Install PathwayLens in development mode:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import pathwaylens_core; print('✓ Installation successful')"
   pathwaylens --version
   ```

## Environment Management

### Activating the Environment

```bash
conda activate pathwaylens
```

### Updating the Environment

If `environment.yml` is updated:

```bash
conda env update -f environment.yml --prune
```

### Deactivating the Environment

```bash
conda deactivate
```

### Removing the Environment

```bash
conda env remove -n pathwaylens
```

## Testing

Once the environment is set up, you can run tests:

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-e2e

# Run with coverage
make test-coverage
```

Or directly with pytest:

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=pathwaylens_core --cov=pathwaylens_cli --cov-report=html
```

## Code Quality Checks

```bash
# Linting
make lint

# Format code
make format

# Type checking
make type-check
```

## Troubleshooting

### Issue: Conda command not found

**Solution:** Ensure conda is in your PATH. Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# For Miniconda
export PATH="$HOME/miniconda3/bin:$PATH"

# For Anaconda
export PATH="$HOME/anaconda3/bin:$PATH"
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Issue: Package conflicts during installation

**Solution:** Try creating a fresh environment:

```bash
conda env remove -n pathwaylens
conda env create -f environment.yml
```

### Issue: pip packages fail to install

**Solution:** Ensure pip is up to date in the environment:

```bash
conda activate pathwaylens
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Import errors after installation

**Solution:** Verify the package is installed correctly:

```bash
conda activate pathwaylens
pip install -e . --force-reinstall
```

## Environment Structure

The `environment.yml` file includes:

- **Core dependencies:** Python, numpy, scipy, pandas, etc.
- **Bioinformatics tools:** biopython, networkx, etc.
- **Pathway analysis:** gseapy, enrichr, gprofiler, etc.
- **Visualization:** matplotlib, seaborn, plotly
- **Testing:** pytest, pytest-cov, hypothesis, etc.
- **Code quality:** black, flake8, mypy, etc.

## Alternative: Using pip only

If you prefer not to use conda, you can use pip with a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Next Steps

After setting up the environment:

1. **Run tests** to verify everything works
2. **Read the documentation** in `docs/`
3. **Check the contributing guide** in `docs/CONTRIBUTING.md`
4. **Set up pre-commit hooks:** `pre-commit install`

## Support

For issues or questions:
- Check the main README.md
- Review the troubleshooting section above
- Check existing GitHub issues



