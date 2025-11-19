#!/usr/bin/env python3
"""
Comprehensive PathwayLens Test Script

Tests both CLI and Python API functionality to ensure everything works.
"""

import sys
import subprocess
import tempfile
from pathlib import Path


def test_cli_help():
    """Test CLI help command"""
    print("\n" + "="*60)
    print("Testing CLI Help Command")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pathwaylens_cli", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and "PathwayLens" in result.stdout:
        print("✓ CLI help works")
        return True
    else:
        print(f"✗ CLI help failed: {result.stderr}")
        return False


def test_cli_analyze_help():
    """Test CLI analyze subcommand help"""
    print("\n" + "="*60)
    print("Testing CLI Analyze Help")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pathwaylens_cli", "analyze", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and "analyze" in result.stdout.lower():
        print("✓ CLI analyze help works")
        return True
    else:
        print(f"✗ CLI analyze help failed: {result.stderr}")
        return False


def test_python_api_imports():
    """Test Python API imports"""
    print("\n" + "="*60)
    print("Testing Python API Imports")
    print("="*60)
    
    try:
        # Core imports
        from pathwaylens_core.analysis import ORAEngine, GSEAEngine
        from pathwaylens_core.data import DatabaseManager
        from pathwaylens_core.analysis.schemas import DatabaseType
        
        # New research modules
        from pathwaylens_core.analysis import statistical_utils
        from pathwaylens_core.io import AnalysisOutputManager
        from pathwaylens_core.visualization.palettes import ColorPalette
        from pathwaylens_core.exceptions import PathwayLensError
        
        print("✓ All Python API imports successful")
        return True
    except ImportError as e:
        print(f"✗ Python API import failed: {e}")
        return False


def test_statistical_functions():
    """Test statistical utility functions"""
    print("\n" + "="*60)
    print("Testing Statistical Functions")
    print("="*60)
    
    try:
        from pathwaylens_core.analysis.statistical_utils import (
            calculate_odds_ratio,
            calculate_fold_enrichment,
            calculate_cohens_h
        )
        
        # Test odds ratio calculation
        result = calculate_odds_ratio(10, 50, 100, 20000)
        assert result.odds_ratio > 0, "Odds ratio should be positive"
        assert result.odds_ratio_ci_lower > 0, "CI lower should be positive"
        
        # Test fold enrichment
        fold_enrich = calculate_fold_enrichment(10, 50, 100, 20000)
        assert fold_enrich > 0, "Fold enrichment should be positive"
        
        # Test effect size
        effect_size = calculate_cohens_h(10, 50, 100, 20000)
        assert isinstance(effect_size, float), "Effect size should be float"
        
        print(f"✓ Odds Ratio: {result.odds_ratio:.2f} "
              f"(95% CI: {result.odds_ratio_ci_lower:.2f}-{result.odds_ratio_ci_upper:.2f})")
        print(f"✓ Fold Enrichment: {fold_enrich:.2f}x")
        print(f"✓ Effect Size (Cohen's h): {effect_size:.3f}")
        return True
    except Exception as e:
        print(f"✗ Statistical functions failed: {e}")
        return False


def test_output_manager():
    """Test output manager functionality"""
    print("\n" + "="*60)
    print("Testing Output Manager")
    print("="*60)
    
    try:
        from pathwaylens_core.io.output_manager import AnalysisOutputManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisOutputManager(
                base_dir=tmpdir,
                analysis_id="test_001"
            )
            
            # Create directory structure
            dirs = manager.create_directory_structure()
            
            assert "results" in dirs, "Results directory should exist"
            assert "figures" in dirs, "Figures directory should exist"
            assert "methods" in dirs, "Methods directory should exist"
            
            print(f"✓ Created {len(dirs)} directories")
            print(f"✓ Output directory: {manager.output_dir}")
            return True
    except Exception as e:
        print(f"✗ Output manager failed: {e}")
        return False


def test_color_palettes():
    """Test color palette functionality"""
    print("\n" + "="*60)
    print("Testing Color Palettes")
    print("="*60)
    
    try:
        from pathwaylens_core.visualization.palettes import ColorPalette
        
        # Test colorblind-safe palette
        palette = ColorPalette.get_colorblind_safe_palette(8)
        assert len(palette) == 8, "Should return 8 colors"
        assert all(c.startswith('#') for c in palette), "Should be hex colors"
        
        # Test publication palette
        nature = ColorPalette.get_publication_palette("nature")
        assert len(nature) > 0, "Should return Nature palette"
        
        print(f"✓ Colorblind-safe palette: {len(palette)} colors")
        print(f"✓ Nature palette: {len(nature)} colors")
        print(f"✓ First color: {palette[0]}")
        return True
    except Exception as e:
        print(f"✗ Color palettes failed: {e}")
        return False


def test_exception_hierarchy():
    """Test custom exception hierarchy"""
    print("\n" + "="*60)
    print("Testing Exception Hierarchy")
    print("="*60)
    
    try:
        from pathwaylens_core.exceptions import (
            PathwayLensError,
            AnalysisError,
            ORAError,
            VisualizationError
        )
        
        # Create exceptions
        base_err = PathwayLensError("Base error")
        analysis_err = AnalysisError("Analysis error")
        ora_err = ORAError("ORA error")
        
        # Check inheritance
        assert isinstance(analysis_err, PathwayLensError)
        assert isinstance(ora_err, AnalysisError)
        assert isinstance(ora_err, PathwayLensError)
        
        print("✓ Exception hierarchy correct")
        print("✓ PathwayLensError (base)")
        print("  ├── AnalysisError")
        print("  │   ├── ORAError")
        print("   │   └── GSEAError")
        return True
    except Exception as e:
        print(f"✗ Exception hierarchy failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" PathwayLens Comprehensive Validation")
    print("="*70)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("CLI Analyze Help", test_cli_analyze_help),
        ("Python API Imports", test_python_api_imports),
        ("Statistical Functions", test_statistical_functions),
        ("Output Manager", test_output_manager),
        ("Color Palettes", test_color_palettes),
        ("Exception Hierarchy", test_exception_hierarchy),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print(" Test Summary")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - PathwayLens is fully functional!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print(f"❌ {total - passed} tests failed")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
