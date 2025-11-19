# PathwayLens Installation Guide

## System Requirements

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **OS**: Linux, macOS, or Windows
- **RAM**: 4GB minimum (8GB+ recommended for large datasets)
- **Disk Space**: 500MB for installation + data

## Installation Methods

### Method 1: pip Install (Recommended)

```bash
# Create virtual environment (recommended)
python -m venv pathwaylens-env
source pathwaylens-env/bin/activate  # On Windows: pathwaylens-env\Scripts\activate

# Install PathwayLens
pip install pathwaylens

# Verify installation
python -c "import pathwaylens_core; print('✓ PathwayLens installed successfully!')"
```

### Method 2: Install from Source

```bash
# Clone repository
git clone https://github.com/yourusername/PathwayLens.git
cd PathwayLens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run validation
python validate_installation.py
```

### Method 3: Conda Install

```bash
# Create conda environment
conda create -n pathwaylens python=3.10
conda activate pathwaylens

# Install PathwayLens
pip install pathwaylens

# Verify
python validate_installation.py
```

## Validation

After installation, run the validation script:

```bash
python validate_installation.py
```

Expected output:
```
============================================================
PathwayLens Installation Validation
============================================================

[1/5] Checking Python version...
✓ Python 3.10.x

[2/5] Checking dependencies...
✓ numpy 1.26.x
✓ scipy 1.15.x
✓ pandas 2.3.x
...

✅ ALL CHECKS PASSED - PathwayLens is ready to use!
```

## Dependencies

PathwayLens will automatically install these required packages:

### Core Dependencies
- `numpy >= 1.20.0` - Numerical computing
- `scipy >= 1.7.0` - Scientific computing
- `pandas >= 1.3.0` - Data manipulation
- `pydantic >= 2.0.0` - Data validation

### Analysis Dependencies
- `statsmodels >= 0.13.0` - Statistical models
- `scikit-learn >= 1.0.0` - Machine learning

### Visualization Dependencies
- `plotly >= 5.15.0` - Interactive plotting
- `networkx >= 2.6.0` - Network analysis

### System Dependencies
- `psutil >= 5.8.0` - System monitoring
- `loguru >= 0.5.0` - Logging

## Troubleshooting

### Import Errors

If you see import errors:
```python
ImportError: No module named 'pathwaylens_core'
```

**Solution**:
1. Ensure you're in the correct virtual environment
2. Reinstall: `pip install --force-reinstall pathwaylens`
3. Check Python path: `which python` (should point to venv)

### Dependency Conflicts

If you have dependency version conflicts:

```bash
# Create fresh environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install pathwaylens
```

### Permission Errors

On Linux/macOS, if you get permission errors:

```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install pathwaylens

# OR use --user flag (not recommended)
pip install --user pathwaylens
```

### Windows-Specific Issues

On Windows, if you encounter build errors:

1. Install Visual C++ Build Tools
2. Use Anaconda (easier dependency management)

```bash
# Using Anaconda
conda create -n pathwaylens python=3.10
conda activate pathwaylens
pip install pathwaylens
```

## Offline Installation

For offline or air-gapped systems:

```bash
# On a system with internet, download packages
pip download pathwaylens -d pathwaylens-packages/

# Transfer pathwaylens-packages/ to offline system
# On offline system:
pip install --no-index --find-links pathwaylens-packages/ pathwaylens
```

## Updating

To update to the latest version:

```bash
pip install --upgrade pathwaylens
```

## Uninstallation

To remove PathwayLens:

```bash
pip uninstall pathwaylens
```

## Getting Help

If you encounter issues:

1. **Check Documentation**: [docs.pathwaylens.org](https://docs.pathwaylens.org)
2. **GitHub Issues**: [github.com/yourusername/PathwayLens/issues](https://github.com/yourusername/PathwayLens/issues)
3. **Email Support**: support@pathwaylens.org

## Next Steps

After successful installation:

1. **Run Example**: `python examples/basic_ora_example.py`
2. **Read Quick Start**: See `README.md`
3. **Explore Tutorials**: Check `examples/` directory
4. **Read Documentation**: Visit docs site

---

✅ **Installation Complete!** You're ready to use PathwayLens for research-grade pathway analysis.
