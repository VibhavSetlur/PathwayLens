# Contributing to PathwayLens

Thank you for your interest in contributing to PathwayLens! We welcome contributions from the community to make this tool better for everyone.

## Governance Model

PathwayLens is a community-driven project.
- **Maintainer:** Vibhav Setlur
- **Core Contributors:** Vibhav Setlur

We follow a benevolent dictator governance model where the maintainer has final say on design decisions, but we actively seek consensus and community input.

## How to Contribute

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PathwayLens.git
   cd PathwayLens
   ```

2. **Create a conda environment (recommended):**
   ```bash
   conda env create -f environment.yml
   conda activate pathwaylens
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Standards

### Test Coverage Requirements

- **All new features MUST include unit tests** with >80% coverage
- **Bug fixes MUST include regression tests** demonstrating the fix
- Run tests locally before submitting PR:
  ```bash
  pytest tests/ --cov=pathwaylens_core --cov=pathwaylens_cli --cov-report=term-missing
  ```
- Coverage must not decrease from the main branch

### Docstring Standards

All functions, classes, and modules must follow **NumPy-style docstrings**:

```python
def analyze_pathway(gene_list, database, species="human"):
    """
    Perform pathway enrichment analysis.
    
    Parameters
    ----------
    gene_list : list of str
        List of gene identifiers.
    database : DatabaseType
        Database to use for analysis.
    species : str, optional
        Species name, by default "human".
    
    Returns
    -------
    DatabaseResult
        Analysis results containing enriched pathways.
    
    Raises
    ------
    ValueError
        If gene_list is empty or database is invalid.
    
    Examples
    --------
    >>> result = analyze_pathway(['TP53', 'BRCA1'], DatabaseType.KEGG)
    >>> print(result.significant_pathways)
    """
```

### Pre-commit Hooks

We enforce code quality with pre-commit hooks:
- **Black**: Code formatting
- **Flake8**: Linting
- **mypy**: Type checking (where applicable)

These run automatically on commit. To run manually:
```bash
pre-commit run --all-files
```

### Scientific Quality Gates

For contributions affecting statistical methods:
1. **Provide references** to published algorithms
2. **Include validation tests** against known results
3. **Document assumptions** and limitations clearly
4. **Compare** with existing implementations where applicable

## Reporting Issues
If you find a bug, please open an issue on GitHub. Include:
- Steps to reproduce.
- Expected vs. actual behavior.
- System information (OS, Python version).

We love new ideas! Open an issue with the "Feature Request" label. Describe the use case and why it would be valuable.

### 3. Submitting Code
1.  Fork the repository.
2.  Create a new branch: `git checkout -b feature/my-new-feature`.
3.  Write your code and **add tests**.
4.  Run tests: `pytest`.
5.  Submit a Pull Request (PR).

### Code Style
- We use `black` for formatting and `flake8` for linting.
- Use type hints for all function signatures.
- Write docstrings in Google style.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/pathwaylens.git
cd pathwaylens

# Create virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
