#!/usr/bin/env python3
"""
Complete Phase 1 Environment Verification Script

Run this after activation to verify everything is working:
  python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_python():
    """Verify Python version"""
    print(f"✓ Python Version: {sys.version.split()[0]}")
    expected = "3.11"
    if expected in sys.version:
        print(f"✓ Python 3.11.x detected (correct)")
        return True
    else:
        print(f"✗ Expected Python 3.11.x, got {sys.version}")
        return False

def check_pytorch():
    """Verify PyTorch installation and GPU"""
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        
        if "cu121" in torch.__version__:
            print(f"✓ CUDA variant detected (cu121)")
        else:
            print(f"⚠ Warning: Expected CUDA 12.1 variant, got {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA Available: True")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Name: {gpu_name}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"✓ Compute Capability: {props.major}.{props.minor}")
            
            total_memory = props.total_memory / 1e9
            print(f"✓ GPU Memory: {total_memory:.1f} GB")
            
            return True
        else:
            print(f"✗ CUDA Not Available!")
            print(f"  Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
            
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False

def check_packages():
    """Verify ML packages"""
    packages = [
        'torch',
        'transformers',
        'xformers',
        'peft',
        'accelerate',
        'huggingface_hub',
        'tqdm',
        'numpy',
        'PIL',
    ]
    
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} (missing)")
            missing.append(pkg)
    
    return len(missing) == 0

def check_aiprod():
    """Verify AIPROD packages"""
    packages = [
        'aiprod_core',
        'aiprod_pipelines',
        'aiprod_trainer',
    ]
    
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} (missing)")
            all_ok = False
    
    return all_ok

def check_models():
    """Check for model files"""
    model_dirs = [
        Path("models/aiprod2"),
        Path("models/aiprod-sovereign"),
    ]
    
    for model_dir in model_dirs:
        if model_dir.exists():
            files = list(model_dir.glob("*.safetensors"))
            print(f"✓ {model_dir}: {len(files)} model files")
        else:
            print(f"⚠ {model_dir}: Not found (need to download)")

def check_infrastructure():
    """Check for created tools"""
    files = [
        "examples/quickstart.py",
        "scripts/monitor_gpu.py",
        "activate.bat",
    ]
    
    all_ok = True
    for file in files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_ok = False
    
    return all_ok

def check_vram():
    """Check current VRAM usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✓ VRAM Allocated: {allocated:.2f} GB")
            print(f"✓ VRAM Reserved: {reserved:.2f} GB")
            print(f"✓ VRAM Total: {total:.1f} GB")
            print(f"✓ VRAM Available: {total - reserved:.2f} GB")
    except Exception as e:
        print(f"⚠ Could not check VRAM: {e}")

def run_simple_test():
    """Run a simple GPU test"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ Skipping GPU test (CUDA not available)")
            return False
        
        print("\nRunning simple GPU test...")
        x = torch.randn(100, 100).cuda()
        y = x @ x.t()
        print(f"✓ GPU computation successful")
        print(f"✓ Result shape: {y.shape}")
        print(f"✓ GPU is working!")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def main():
    """Run all checks"""
    print_header("AIPROD Phase 1 - Complete Environment Verification")
    
    results = {}
    
    print_header("1. Python Environment")
    results['python'] = check_python()
    
    print_header("2. PyTorch & GPU")
    results['pytorch'] = check_pytorch()
    
    print_header("3. ML Packages")
    print("Checking core packages:")
    results['packages'] = check_packages()
    
    print_header("4. AIPROD Packages")
    print("Checking AIPROD installation:")
    results['aiprod'] = check_aiprod()
    
    print_header("5. Development Tools")
    print("Checking infrastructure files:")
    results['infrastructure'] = check_infrastructure()
    
    print_header("6. Model Files")
    print("Checking downloaded models:")
    check_models()
    
    print_header("7. GPU Memory Status")
    check_vram()
    
    print_header("8. GPU Computation Test")
    results['gpu_test'] = run_simple_test()
    
    # Summary
    print_header("SUMMARY")
    
    all_ok = all(results.values())
    
    print("Results:")
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:25} {status}")
    
    print()
    
    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour environment is ready. Next steps:")
        print("  1. Download models from HuggingFace (see DEVELOPMENT_GUIDE.md)")
        print("  2. Run: python examples/quickstart.py --prompt 'Your description'")
        print("  3. Monitor GPU: python scripts/monitor_gpu.py")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  - GPU not available: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  - Models missing: Download from HuggingFace (see DEVELOPMENT_GUIDE.md)")
        print("  - Tools missing: Check if files exist in examples/ and scripts/")
        return 1

if __name__ == "__main__":
    sys.exit(main())
