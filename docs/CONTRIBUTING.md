# Contributing to PathwayLens 2.0

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style and Standards](#code-style-and-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Release Process](#release-process)

## Introduction

Thank you for your interest in contributing to PathwayLens 2.0! This document provides guidelines for contributing to the project, including development setup, code standards, testing, and the submission process.

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add new functionality and capabilities
- **Documentation**: Improve guides, API docs, and examples
- **Testing**: Add tests and improve test coverage
- **Performance**: Optimize code and improve efficiency
- **UI/UX**: Enhance user interface and experience

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher (for frontend development)
- Git
- Docker (optional, for containerized development)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/PathwayLens.git
   cd PathwayLens
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/pathwaylens/PathwayLens.git
   ```

### Development Branches

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature development branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical bug fixes

## Development Setup

### Backend Development

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**:
   ```bash
   pathwaylens db init
   ```

5. **Run development server**:
   ```bash
   pathwaylens api serve --reload
   ```

### Frontend Development

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

### Docker Development

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Run specific services**:
   ```bash
   docker-compose up backend frontend
   ```

## Code Style and Standards

### Python Code Style

We follow PEP 8 and use several tools for code quality:

#### Black (Code Formatting)

```bash
# Format code
black pathwaylens_core/ pathwaylens_cli/ pathwaylens_api/

# Check formatting
black --check pathwaylens_core/ pathwaylens_cli/ pathwaylens_api/
```

#### Flake8 (Linting)

```bash
# Run linting
flake8 pathwaylens_core/ pathwaylens_cli/ pathwaylens_api/

# Configuration in setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,node_modules
```

#### MyPy (Type Checking)

```bash
# Run type checking
mypy pathwaylens_core/ pathwaylens_cli/ pathwaylens_api/

# Configuration in mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### TypeScript/JavaScript Code Style

We use ESLint and Prettier for frontend code:

#### ESLint

```bash
# Run linting
npm run lint

# Fix issues
npm run lint:fix
```

#### Prettier

```bash
# Format code
npm run format

# Check formatting
npm run format:check
```

### Code Organization

#### Backend Structure

```
pathwaylens_core/
├── normalization/     # Data normalization and conversion
├── analysis/         # Pathway analysis engines
├── data/            # Database adapters and caching
├── comparison/      # Multi-dataset comparison
├── visualization/   # Visualization and reporting
├── multi_omics/     # Multi-omics analysis
├── plugins/         # Plugin system
└── utils/           # Utility functions
```

#### Frontend Structure

```
frontend/
├── src/
│   ├── components/  # Reusable UI components
│   ├── pages/       # Next.js pages
│   ├── hooks/       # Custom React hooks
│   ├── services/    # API services
│   ├── stores/      # State management
│   ├── types/       # TypeScript type definitions
│   └── utils/       # Utility functions
├── design-system/   # Design system components
└── public/          # Static assets
```

### Naming Conventions

#### Python

- **Classes**: PascalCase (e.g., `PathwayAnalyzer`)
- **Functions/Variables**: snake_case (e.g., `analyze_pathways`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_THRESHOLD`)
- **Private**: Leading underscore (e.g., `_internal_method`)

#### TypeScript/JavaScript

- **Classes**: PascalCase (e.g., `PathwayAnalyzer`)
- **Functions/Variables**: camelCase (e.g., `analyzePathways`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_THRESHOLD`)
- **Private**: Leading underscore (e.g., `_internalMethod`)

### Documentation Standards

#### Python Docstrings

Use Google-style docstrings:

```python
def analyze_pathways(genes: List[str], species: str) -> AnalysisResult:
    """Analyze pathways for a list of genes.
    
    Args:
        genes: List of gene identifiers
        species: Species code (e.g., 'human', 'mouse')
        
    Returns:
        AnalysisResult containing pathway analysis results
        
    Raises:
        ValidationError: If input data is invalid
        DatabaseError: If database connection fails
    """
    pass
```

#### TypeScript/JavaScript JSDoc

```typescript
/**
 * Analyzes pathways for a list of genes
 * @param genes - List of gene identifiers
 * @param species - Species code (e.g., 'human', 'mouse')
 * @returns Promise resolving to analysis results
 * @throws {ValidationError} If input data is invalid
 * @throws {DatabaseError} If database connection fails
 */
async function analyzePathways(genes: string[], species: string): Promise<AnalysisResult> {
  // Implementation
}
```

## Testing

### Backend Testing

#### Unit Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_analysis.py

# Run with coverage
pytest --cov=pathwaylens_core --cov-report=html

# Run specific test
pytest tests/unit/test_analysis.py::test_ora_analysis
```

#### Integration Tests

```bash
# Run integration tests
pytest tests/integration/

# Run with database
pytest tests/integration/ --database
```

#### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test fixtures and data
└── conftest.py    # Pytest configuration
```

#### Writing Tests

```python
import pytest
from pathwaylens_core.analysis.engine import AnalysisEngine

class TestAnalysisEngine:
    def test_ora_analysis(self, sample_genes):
        """Test ORA analysis with sample genes."""
        engine = AnalysisEngine()
        result = engine.analyze_ora(
            genes=sample_genes,
            species="human",
            databases=["kegg"]
        )
        
        assert result is not None
        assert len(result.pathways) > 0
        assert all(p.p_value < 0.05 for p in result.pathways)
    
    def test_invalid_species(self):
        """Test analysis with invalid species."""
        engine = AnalysisEngine()
        
        with pytest.raises(ValidationError):
            engine.analyze_ora(
                genes=["TP53"],
                species="invalid_species",
                databases=["kegg"]
            )
```

### Frontend Testing

#### Unit Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test
npm test -- --testNamePattern="PathwayAnalyzer"
```

#### Integration Tests

```bash
# Run integration tests
npm run test:integration
```

#### Writing Tests

```typescript
import { render, screen } from '@testing-library/react';
import { PathwayAnalyzer } from '../PathwayAnalyzer';

describe('PathwayAnalyzer', () => {
  it('renders analysis form', () => {
    render(<PathwayAnalyzer />);
    
    expect(screen.getByLabelText(/gene list/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/species/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze/i })).toBeInTheDocument();
  });
  
  it('submits analysis form', async () => {
    const mockAnalyze = jest.fn();
    render(<PathwayAnalyzer onAnalyze={mockAnalyze} />);
    
    // Fill form and submit
    // ... test implementation
    
    expect(mockAnalyze).toHaveBeenCalledWith({
      genes: ['TP53', 'BRCA1'],
      species: 'human'
    });
  });
});
```

### Test Data

Use the `tests/fixtures/` directory for test data:

```
tests/fixtures/
├── genes/          # Gene lists and identifiers
├── pathways/       # Pathway data
├── analysis/       # Analysis results
└── configs/        # Configuration files
```

## Documentation

### Documentation Structure

```
docs/
├── INSTALLATION.md     # Installation guide
├── QUICKSTART.md       # Quick start guide
├── USER_GUIDE.md       # Comprehensive user guide
├── CLI_REFERENCE.md    # CLI command reference
├── API_REFERENCE.md    # API documentation
├── CONTRIBUTING.md     # This file
├── CHANGELOG.md        # Release notes
└── examples/           # Usage examples
```

### Writing Documentation

1. **Use clear, concise language**
2. **Include code examples**
3. **Update documentation with code changes**
4. **Test all examples**
5. **Use consistent formatting**

### API Documentation

API documentation is automatically generated from code comments and schemas. Use descriptive docstrings and type hints:

```python
from pydantic import BaseModel
from typing import List, Optional

class AnalysisRequest(BaseModel):
    """Request model for pathway analysis.
    
    Attributes:
        genes: List of gene identifiers to analyze
        species: Species code for analysis
        databases: List of databases to use
        parameters: Optional analysis parameters
    """
    genes: List[str]
    species: str
    databases: List[str]
    parameters: Optional[dict] = None
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new pathway analysis feature"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Use the PR template
   - Provide a clear description
   - Link related issues
   - Request reviews from maintainers

### Commit Message Format

We use conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring
- **test**: Test changes
- **chore**: Maintenance tasks

#### Examples

```
feat(analysis): add GSEA analysis support
fix(api): resolve database connection timeout
docs(cli): update command examples
test(analysis): add unit tests for ORA engine
```

### Pull Request Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)

## Related Issues

Closes #123
```

### Code Review Process

1. **Automated checks** must pass
2. **At least one maintainer** must approve
3. **All conversations** must be resolved
4. **Documentation** must be updated
5. **Tests** must be added/updated

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `pyproject.toml` and `package.json`
2. **Update CHANGELOG.md** with release notes
3. **Create release branch** from `main`
4. **Run full test suite**
5. **Create GitHub release**
6. **Publish to PyPI** and npm
7. **Update documentation**

### Release Checklist

- [ ] Version updated
- [ ] CHANGELOG updated
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Release notes written
- [ ] GitHub release created
- [ ] Packages published

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow community standards

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat and support
- **Email**: pathwaylens@example.com

### Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributors
- **Documentation**: Code examples and guides

## Development Tools

### Recommended IDEs

- **VS Code**: With Python, TypeScript, and Docker extensions
- **PyCharm**: Professional Python IDE
- **WebStorm**: Professional JavaScript/TypeScript IDE

### Useful Extensions

- **Python**: Python, Pylance, Black Formatter
- **TypeScript**: TypeScript Importer, ESLint
- **Git**: GitLens, Git Graph
- **Docker**: Docker, Docker Compose

### Development Scripts

```bash
# Backend
make install-dev      # Install development dependencies
make test            # Run tests
make lint            # Run linting
make format          # Format code
make docs            # Generate documentation

# Frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run test         # Run tests
npm run lint         # Run linting
npm run format       # Format code
```

## License

By contributing to PathwayLens 2.0, you agree that your contributions will be licensed under the same license as the project.

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Ask in our Discord community
4. Contact maintainers directly

Thank you for contributing to PathwayLens 2.0!
