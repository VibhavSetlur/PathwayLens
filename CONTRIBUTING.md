# Contributing to PathwayLens

Thank you for your interest in contributing to PathwayLens! We welcome contributions from the community to make this tool better for everyone.

## Governance Model

PathwayLens is a community-driven project.
- **Maintainer:** [Your Name/Organization]
- **Core Contributors:** [List of core contributors]

We follow a benevolent dictator governance model where the maintainer has final say on design decisions, but we actively seek consensus and community input.

## How to Contribute

### 1. Reporting Bugs
If you find a bug, please open an issue on GitHub. Include:
- Steps to reproduce.
- Expected vs. actual behavior.
- System information (OS, Python version).

### 2. Requesting Features
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
