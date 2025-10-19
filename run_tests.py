#!/usr/bin/env python3
"""
Comprehensive test runner for PathwayLens 2.0.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def run_unit_tests(verbose: bool = False) -> bool:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "--strict-markers"])
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False) -> bool:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "--strict-markers"])
    return run_command(cmd, "Integration Tests")


def run_e2e_tests(verbose: bool = False) -> bool:
    """Run end-to-end tests."""
    cmd = ["python", "-m", "pytest", "tests/e2e/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "--strict-markers"])
    return run_command(cmd, "End-to-End Tests")


def run_all_tests(verbose: bool = False) -> bool:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "--strict-markers"])
    return run_command(cmd, "All Tests")


def run_coverage_tests() -> bool:
    """Run tests with coverage reporting."""
    cmd = [
        "python", "-m", "pytest", "tests/",
        "--cov=pathwaylens_core",
        "--cov=pathwaylens_api", 
        "--cov=pathwaylens_cli",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ]
    return run_command(cmd, "Coverage Tests")


def run_linting() -> bool:
    """Run code linting."""
    # Run flake8
    flake8_success = run_command(
        ["python", "-m", "flake8", "pathwaylens_core/", "pathwaylens_api/", "pathwaylens_cli/"],
        "Flake8 Linting"
    )
    
    # Run black check
    black_success = run_command(
        ["python", "-m", "black", "--check", "pathwaylens_core/", "pathwaylens_api/", "pathwaylens_cli/"],
        "Black Formatting Check"
    )
    
    # Run isort check
    isort_success = run_command(
        ["python", "-m", "isort", "--check-only", "pathwaylens_core/", "pathwaylens_api/", "pathwaylens_cli/"],
        "Import Sorting Check"
    )
    
    return flake8_success and black_success and isort_success


def run_type_checking() -> bool:
    """Run type checking with mypy."""
    cmd = ["python", "-m", "mypy", "pathwaylens_core/", "pathwaylens_api/", "pathwaylens_cli/"]
    return run_command(cmd, "Type Checking")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="PathwayLens 2.0 Test Runner")
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e", "all", "coverage"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting checks"
    )
    parser.add_argument(
        "--type-check",
        action="store_true",
        help="Run type checking"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all checks (tests, linting, type checking)"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    success = True
    
    if args.full:
        # Run all checks
        success &= run_linting()
        success &= run_type_checking()
        success &= run_coverage_tests()
    else:
        # Run specific checks
        if args.lint:
            success &= run_linting()
        
        if args.type_check:
            success &= run_type_checking()
        
        # Run tests
        if args.test_type == "unit":
            success &= run_unit_tests(args.verbose)
        elif args.test_type == "integration":
            success &= run_integration_tests(args.verbose)
        elif args.test_type == "e2e":
            success &= run_e2e_tests(args.verbose)
        elif args.test_type == "coverage":
            success &= run_coverage_tests()
        else:  # all
            success &= run_all_tests(args.verbose)
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All checks passed successfully!")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
