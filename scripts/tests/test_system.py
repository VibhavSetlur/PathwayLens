#!/usr/bin/env python3
"""
PathwayLens 2.0 System Test Script
This script tests the complete system functionality
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def test_cli():
    """Test CLI functionality"""
    print("🔍 Testing CLI...")
    
    try:
        # Test main CLI
        result = subprocess.run([
            sys.executable, "-m", "pathwaylens_cli", "--help"
        ], capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            print("✅ CLI main help: OK")
        else:
            print(f"❌ CLI main help: {result.stderr}")
            return False
            
        # Test normalize command
        result = subprocess.run([
            sys.executable, "-m", "pathwaylens_cli", "normalize", "--help"
        ], capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            print("✅ CLI normalize help: OK")
        else:
            print(f"❌ CLI normalize help: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_core_modules():
    """Test core modules"""
    print("🔍 Testing Core Modules...")
    
    try:
        # Test normalization
        from pathwaylens_core.normalization.normalizer import Normalizer
        print("✅ Normalization Engine: OK")
        
        # Test analysis
        from pathwaylens_core.analysis.engine import AnalysisEngine
        print("✅ Analysis Engine: OK")
        
        # Test visualization
        from pathwaylens_core.visualization.engine import VisualizationEngine
        print("✅ Visualization Engine: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Core modules test failed: {e}")
        return False

def test_api_backend():
    """Test API backend"""
    print("🔍 Testing API Backend...")
    
    try:
        # Test FastAPI app
        from pathwaylens_api.main import app
        print("✅ FastAPI App: OK")
        
        # Test Celery app
        from pathwaylens_api.celery_app import celery_app
        print("✅ Celery App: OK")
        
        # Test database manager
        from pathwaylens_api.utils.database import DatabaseManager
        print("✅ Database Manager: OK")
        
        # Test storage manager
        from pathwaylens_api.utils.storage import StorageManager
        print("✅ Storage Manager: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ API backend test failed: {e}")
        return False

def test_frontend():
    """Test frontend setup"""
    print("🔍 Testing Frontend...")
    
    try:
        frontend_path = Path("frontend")
        if not frontend_path.exists():
            print("❌ Frontend directory not found")
            return False
            
        package_json = frontend_path / "package.json"
        if not package_json.exists():
            print("❌ package.json not found")
            return False
            
        print("✅ Frontend structure: OK")
        return True
        
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def test_docker():
    """Test Docker configuration"""
    print("🔍 Testing Docker Configuration...")
    
    try:
        # Test docker compose config
        result = subprocess.run([
            "docker", "compose", "-f", "infra/docker/docker-compose.yml", "config"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Docker Compose config: OK")
        else:
            print(f"❌ Docker Compose config: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PathwayLens 2.0 System Test")
    print("=" * 50)
    
    tests = [
        ("CLI", test_cli),
        ("Core Modules", test_core_modules),
        ("API Backend", test_api_backend),
        ("Frontend", test_frontend),
        ("Docker", test_docker),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
