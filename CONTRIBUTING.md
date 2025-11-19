# Contributing to PathwayLens

Thank you for considering contributing to PathwayLens! This document provides guidelines for contributing.

## Code of Conduct

Be respectful and professional in all interactions.

## How to Contribute

### Reporting Bugs

Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

### Suggesting Enhancements

Open an issue describing:
- The enhancement
- Use case and motivation
- Proposed implementation (if applicable)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation
7. Commit with clear messages
8. Push to your fork
9. Open a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/PathwayLens.git
cd PathwayLens

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Coding Standards

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Add tests for new code
- Keep functions focused and small

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_statistical_utils.py

# Run with coverage
pytest tests/ --cov=pathwaylens_core
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing!
