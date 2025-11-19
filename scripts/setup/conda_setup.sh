#!/bin/bash
# PathwayLens 2.0 - Conda Environment Setup Script
# ================================================
# This script creates and configures a conda environment for PathwayLens
# Usage: bash scripts/setup/conda_setup.sh [environment_name]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get environment name from argument or use default
ENV_NAME="${1:-pathwaylens}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environment.yml"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PathwayLens 2.0 Conda Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${YELLOW}Checking conda installation...${NC}"
conda --version
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo -e "${YELLOW}Using existing environment.${NC}"
        echo -e "${GREEN}Activating environment...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate "${ENV_NAME}"
        
        echo -e "${YELLOW}Updating environment from environment.yml...${NC}"
        conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
        
        echo -e "${GREEN}Installing PathwayLens in development mode...${NC}"
        pip install -e "${PROJECT_ROOT}"
        
        echo -e "${GREEN}Setup complete!${NC}"
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Check if environment.yml exists
if [ ! -f "${ENV_FILE}" ]; then
    echo -e "${RED}Error: environment.yml not found at ${ENV_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}Creating conda environment '${ENV_NAME}' from environment.yml...${NC}"
conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"

echo -e "${GREEN}Environment created successfully!${NC}"
echo ""

# Activate environment
echo -e "${YELLOW}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PathwayLens in development mode
echo -e "${YELLOW}Installing PathwayLens in development mode...${NC}"
cd "${PROJECT_ROOT}"
pip install -e .

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import pathwaylens_core; print('✓ pathwaylens_core imported successfully')" || {
    echo -e "${RED}Error: Failed to import pathwaylens_core${NC}"
    exit 1
}

python -c "import pathwaylens_cli; print('✓ pathwaylens_cli imported successfully')" || {
    echo -e "${RED}Error: Failed to import pathwaylens_cli${NC}"
    exit 1
}

# Run basic tests to verify setup
echo -e "${YELLOW}Running basic import tests...${NC}"
python -m pytest tests/unit/test_normalization.py::test_imports -v 2>/dev/null || {
    echo -e "${YELLOW}Note: Some tests may require additional setup${NC}"
}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Environment name: ${ENV_NAME}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo ""
echo "To run tests:"
echo -e "  ${YELLOW}make test${NC}"
echo "  or"
echo -e "  ${YELLOW}pytest tests/ -v${NC}"
echo ""
echo "To run linting:"
echo -e "  ${YELLOW}make lint${NC}"
echo ""
echo "To install pre-commit hooks:"
echo -e "  ${YELLOW}pre-commit install${NC}"
echo ""



