#!/usr/bin/env python3
"""
Setup verification script for InternVideo2.5 Video Analyzer
"""

import sys
import subprocess
import importlib
import torch

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        print(f"✅ GPU Available: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        print(f"   GPU Count: {gpu_count}")

        if gpu_memory < 6.0:
            print("⚠️  Warning: Low GPU memory, model may run slowly")

        return True
    else:
        print("❌ No GPU available - model will run very slowly on CPU")
        return False

def check_package(package_name, import_name=None, optional=False):
    """Check if a package is installed"""
    try:
        importlib.import_module(import_name or package_name)
        if optional:
            print(f"✅ {package_name} (optional)")
        else:
            print(f"✅ {package_name}")
        return True
    except ImportError:
        if optional:
            print(f"⚠️  {package_name} (optional) - Not installed (OK)")
            return True  # Don't fail for optional packages
        else:
            print(f"❌ {package_name} - Not installed")
            return False

def main():
    """Main setup check"""
    print("🔍 InternVideo2.5 Setup Verification")
    print("=" * 50)

    checks_passed = 0
    total_checks = 0

    # Check Python version
    total_checks += 1
    if check_python_version():
        checks_passed += 1

    print()

    # Check GPU
    total_checks += 1
    if check_gpu():
        checks_passed += 1

    print()

    # Check required packages
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("einops", "einops"),
        ("timm", "timm"),
        # Optional packages
        ("flash-attn", "flash_attn"),
        ("decord", "decord"),
        ("opencv-python", "cv2"),
        ("imageio", "imageio"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("huggingface_hub", "huggingface_hub"),
    ]

    print("Checking required packages:")
    for package_name, import_name in packages:
        total_checks += 1
        # Check if it's an optional package (after the comment)
        optional = package_name == "flash-attn"
        if check_package(package_name, import_name, optional):
            checks_passed += 1

    print()
    print("=" * 50)
    print(f"Setup Status: {checks_passed}/{total_checks} checks passed")

    if checks_passed == total_checks:
        print("🎉 All dependencies installed! Ready to run video analysis.")
        print("\nTo start analysis, run:")
        print("  python run_analysis.py")
        return True
    else:
        print("⚠️  Some dependencies missing.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")

        if not torch.cuda.is_available():
            print("\n⚠️  Note: Running on CPU will be very slow.")
            print("   GPU acceleration is highly recommended.")

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)