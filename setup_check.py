#!/usr/bin/env python3
"""
Setup verification script for InternVideo2.5 Video Analyzer
"""

import sys
import subprocess
import importlib
import torch
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("- Python 3.8+ required")
        return False
    else:
        print("+ Python version OK")
        return True

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        print(f"+ GPU Available: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        print(f"   GPU Count: {gpu_count}")

        if gpu_memory < 6.0:
            print("! Warning: Low GPU memory, model may run slowly")

        return True
    else:
        print("- No GPU available - model will run very slowly on CPU")
        return False

def check_model_files():
    """Check if the InternVideo2.5 model files are present"""
    model_path = "./models/OpenGVLab/InternVideo2_5_Chat_8B"

    # Critical files needed for model to work
    critical_files = [
        "config.json",
        "tokenizer.model",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]

    print("Checking model files:")

    if not os.path.exists(model_path):
        print(f"- Model directory not found: {model_path}")
        print("   Please run: python download_models.py")
        return False

    missing_files = []
    total_size = 0
    found_files = []

    for filename in critical_files:
        file_path = os.path.join(model_path, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            found_files.append(filename)
            print(f"+ {filename}")
        else:
            missing_files.append(filename)
            print(f"- {filename} (missing)")

    if missing_files:
        print(f"\n!  Model incomplete: {len(missing_files)} critical files missing")
        print("   Please run: python download_models.py")
        print("   Choose option 2 to download the complete model")
        return False
    else:
        # Format size for display
        def format_size(bytes_size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.1f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.1f} TB"

        print(f"+ All {len(found_files)} critical model files present")
        print(f"   Total model size: {format_size(total_size)}")
        print(f"   Location: {os.path.abspath(model_path)}")
        return True

def check_package(package_name, import_name=None, optional=False):
    """Check if a package is installed"""
    try:
        importlib.import_module(import_name or package_name)
        if optional:
            print(f"+ {package_name} (optional)")
        else:
            print(f"+ {package_name}")
        return True
    except ImportError:
        if optional:
            print(f"!  {package_name} (optional) - Not installed (OK)")
            return True  # Don't fail for optional packages
        else:
            print(f"- {package_name} - Not installed")
            return False

def main():
    """Main setup check"""
    print("InternVideo2.5 Setup Verification")
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

    # Check model files
    total_checks += 1
    if check_model_files():
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
        print(" All checks passed! Ready to run video analysis.")
        print("\n Setup Summary:")
        print("  + Python environment OK")
        print("  + GPU acceleration available")
        print("  + All required packages installed")
        print("  + Model files downloaded and complete")

        print("\n To start video analysis, run:")
        print("  python video_analyzer.py")

        print("\n Make sure your video file is ready:")
        print("  - Expected video path: ./car.mp4")
        print("  - Or modify VIDEO_PATH in video_analyzer.py")

        return True
    else:
        print("!  Some checks failed. Setup incomplete.")

        # Check what specifically failed
        if not os.path.exists("./models/OpenGVLab/InternVideo2_5_Chat_8B"):
            print("\n- Model files missing:")
            print("   Run: python download_models.py")
            print("   Choose option 2 for complete model download")
        else:
            print("\n📦 To install missing packages, run:")
            print("   pip install -r requirements.txt")

        if not torch.cuda.is_available():
            print("\n!  Warning: No GPU detected.")
            print("   Video analysis will be extremely slow on CPU.")
            print("   GPU with CUDA is strongly recommended.")

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)