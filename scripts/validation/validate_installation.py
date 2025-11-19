#!/usr/bin/env python3
"""
PathwayLens Installation Validation Script

Validates that PathwayLens is correctly installed and all components are functional.
"""

import sys
from typing import List, Tuple


def check_python_version() -> bool:
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check all required dependencies are installed"""
    required = {
        'numpy': '1.20.0',
        'scipy': '1.7.0',
        'pandas': '1.3.0',
        'pydantic': '2.0.0',
        'plotly': '5.15.0',
        'psutil': '5.8.0',
        'loguru': '0.5.0',
        'networkx': '2.6.0'
    }
    
    missing = []
    version_issues = []
    
    for package, min_version in required.items():
        try:
            mod = __import__(package)
            # Check version if available
            if hasattr(mod, '__version__'):
                current_version = mod.__version__
                print(f"✓ {package} {current_version}")
            else:
                print(f"✓ {package} (version unknown)")
        except ImportError:
            print(f"✗ {package} not installed")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_core_imports() -> Tuple[bool, List[str]]:
    """Check PathwayLens core modules can be imported"""
    core_modules = [
        'pathwaylens_core',
        'pathwaylens_core.analysis',
        'pathwaylens_core.data',
        'pathwaylens_core.visualization',
        'pathwaylens_core.normalization',
        'pathwaylens_core.comparison',
        'pathwaylens_core.io'
    ]
    
    failed = []
    for module in core_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0, failed


def check_new_modules() -> Tuple[bool, List[str]]:
    """Check newly added research-grade modules"""
    new_modules = [
        'pathwaylens_core.analysis.statistical_utils',
        'pathwaylens_core.analysis.diagnostics',
        'pathwaylens_core.schemas.provenance',
        'pathwaylens_core.utils.manifest_generator',
        'pathwaylens_core.io.output_manager',
        'pathwaylens_core.visualization.palettes',
        'pathwaylens_core.visualization.diagnostic_plots',
        'pathwaylens_core.exceptions'
    ]
    
    failed = []
    for module in new_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0, failed


def check_key_classes() -> Tuple[bool, List[str]]:
    """Check key classes can be instantiated"""
    checks = []
    
    try:
        from pathwaylens_core.analysis.statistical_utils import calculate_odds_ratio
        result = calculate_odds_ratio(10, 50, 100, 20000)
        assert result.odds_ratio > 0
        print("✓ StatisticalUtils.calculate_odds_ratio")
    except Exception as e:
        print(f"✗ StatisticalUtils: {e}")
        checks.append("StatisticalUtils")
    
    try:
        from pathwaylens_core.visualization.palettes import ColorPalette
        palette = ColorPalette.get_colorblind_safe_palette(8)
        assert len(palette) == 8
        print("✓ ColorPalette.get_colorblind_safe_palette")
    except Exception as e:
        print(f"✗ ColorPalette: {e}")
        checks.append("ColorPalette")
    
    try:
        from pathwaylens_core.exceptions import PathwayLensError
        err = PathwayLensError("test")
        print("✓ PathwayLensError")
    except Exception as e:
        print(f"✗ PathwayLensError: {e}")
        checks.append("PathwayLensError")
    
    try:
        from pathwaylens_core.io.output_manager import AnalysisOutputManager
        # Just check it can be imported and class exists
        assert AnalysisOutputManager is not None
        print("✓ AnalysisOutputManager")
    except Exception as e:
        print(f"✗ AnalysisOutputManager: {e}")
        checks.append("AnalysisOutputManager")
    
    return len(checks) == 0, checks


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("PathwayLens Installation Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Python version
    print("\n[1/5] Checking Python version...")
    if not check_python_version():
        all_passed = False
    
    # Dependencies
    print("\n[2/5] Checking dependencies...")
    passed, missing = check_dependencies()
    if not passed:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        all_passed = False
    
    # Core imports
    print("\n[3/5] Checking core modules...")
    passed, failed = check_core_imports()
    if not passed:
        print(f"\n  Failed modules: {', '.join(failed)}")
        all_passed = False
    
    # New research modules
    print("\n[4/5] Checking research-grade modules...")
    passed, failed = check_new_modules()
    if not passed:
        print(f"\n  Failed modules: {', '.join(failed)}")
        all_passed = False
    
    # Key classes
    print("\n[5/5] Checking key classes...")
    passed, failed = check_key_classes()
    if not passed:
        print(f"\n  Failed classes: {', '.join(failed)}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - PathwayLens is ready to use!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - See errors above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
