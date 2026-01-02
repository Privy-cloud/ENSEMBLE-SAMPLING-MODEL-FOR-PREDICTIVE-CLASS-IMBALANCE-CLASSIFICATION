"""
Installation and Setup Verification Script
Verifies that all components are correctly installed and configured.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python version too old: {version.major}.{version.minor}.{version.micro}")
        print("    Required: Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'imblearn', 
        'xgboost', 'matplotlib', 'seaborn', 'requests', 'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    required_files = [
        'README.md',
        'USAGE.md',
        'requirements.txt',
        'main.py',
        'quick_start.py',
        'generate_demo_data.py',
        'src/__init__.py',
        'src/data_loader.py',
        'src/ensemble_sampler.py',
        'src/model_trainer.py',
        'notebooks/exploratory_data_analysis.ipynb',
        '.gitignore'
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_present = False
    
    return all_present

def check_modules():
    """Check if custom modules can be imported"""
    print("\nChecking custom modules...")
    sys.path.insert(0, 'src')
    
    try:
        from src.data_loader import BankMarketingDataLoader
        print("  ✓ data_loader module")
    except Exception as e:
        print(f"  ✗ data_loader module - {e}")
        return False
    
    try:
        from src.ensemble_sampler import EnsembleSampler
        print("  ✓ ensemble_sampler module")
    except Exception as e:
        print(f"  ✗ ensemble_sampler module - {e}")
        return False
    
    try:
        from src.model_trainer import ModelTrainer
        print("  ✓ model_trainer module")
    except Exception as e:
        print(f"  ✗ model_trainer module - {e}")
        return False
    
    return True

def main():
    """Run all verification checks"""
    print("=" * 70)
    print("INSTALLATION VERIFICATION")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Custom Modules", check_modules)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ All verification checks passed!")
        print("\nYou can now run:")
        print("  - python quick_start.py       # Quick demo")
        print("  - python main.py              # Full pipeline")
        print("  - python main.py --compare-all # Compare all strategies")
        return 0
    else:
        print("\n❌ Some verification checks failed.")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
