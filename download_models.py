#!/usr/bin/env python3
"""
Download InternVideo2.5 model from Hugging Face Hub
This script automatically downloads and sets up the required model files.
"""

import os
import sys
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("❌ huggingface_hub not found. Installing...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

# Configuration
MODEL_REPO = "OpenGVLab/InternVideo2.5-Chat-8B"
MODEL_PATH = "./models/OpenGVLab/InternVideo2_5_Chat_8B"
BASE_FILES_TO_DOWNLOAD = [
    "config.json",
    "configuration_internlm2.py",
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_intern_vit.py",
    "modeling_internlm2.py",
    "modeling_internvl_chat_hico2.py",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenization_internlm2.py",
    "README.md"
]
MODEL_WEIGHTS = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "tokenizer.model"
]

def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def download_with_progress(repo_id, filename, local_path):
    """Download a single file with progress indication"""
    try:
        print(f"  📥 Downloading {filename}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        return file_path
    except Exception as e:
        print(f"  ❌ Failed to download {filename}: {e}")
        return None

def check_disk_space(required_gb):
    """Check if there's enough disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("./")
        free_gb = free // (1024**3)

        if free_gb < required_gb:
            print(f"❌ Insufficient disk space!")
            print(f"   Required: {required_gb} GB")
            print(f"   Available: {free_gb} GB")
            return False
        else:
            print(f"✅ Disk space check passed: {free_gb} GB available")
            return True
    except:
        print("⚠️  Could not check disk space, proceeding anyway...")
        return True

def download_model_code_only():
    """Download only the model code files (no weights)"""
    print("=" * 60)
    print("🔧 Downloading InternVideo2.5 Model Code")
    print("=" * 60)

    # Create model directory
    os.makedirs(MODEL_PATH, exist_ok=True)

    try:
        print(f"📂 Target directory: {os.path.abspath(MODEL_PATH)}")

        # Download configuration and code files
        downloaded_files = []
        total_size = 0

        for filename in BASE_FILES_TO_DOWNLOAD:
            file_path = download_with_progress(MODEL_REPO, filename, MODEL_PATH)
            if file_path and os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                downloaded_files.append(filename)

        if downloaded_files:
            print(f"\n✅ Model code download completed!")
            print(f"   Files downloaded: {len(downloaded_files)}")
            print(f"   Total size: {format_size(total_size)}")
            print(f"   Location: {os.path.abspath(MODEL_PATH)}")
            return True
        else:
            print("❌ No files were downloaded!")
            return False

    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

def download_full_model():
    """Download the complete model including weights"""
    print("=" * 60)
    print("🤖 Downloading Complete InternVideo2.5 Model")
    print("=" * 60)

    # Check disk space (model is ~15GB)
    if not check_disk_space(20):  # 20GB to be safe
        return False

    try:
        print(f"📂 Target directory: {os.path.abspath(MODEL_PATH)}")
        print("⏳ This will download ~15GB of model weights...")
        print("   This may take 10-30 minutes depending on your internet speed.")

        response = input("\n🚀 Continue with full model download? (y/N): ")
        if response.lower() != 'y':
            print("❌ Download cancelled.")
            return False

        # Use snapshot_download for the full model
        print("\n🔄 Downloading model files...")
        downloaded_path = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✅ Full model download completed!")
        print(f"   Location: {os.path.abspath(downloaded_path)}")
        return True

    except RepositoryNotFoundError:
        print(f"❌ Model repository not found: {MODEL_REPO}")
        return False
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

def verify_installation():
    """Verify that the model files are present"""
    print("\n" + "=" * 60)
    print("🔍 Verifying Model Installation")
    print("=" * 60)

    required_files = BASE_FILES_TO_DOWNLOAD + MODEL_WEIGHTS
    missing_files = []

    for filename in required_files:
        file_path = os.path.join(MODEL_PATH, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {filename} ({format_size(size)})")
        else:
            missing_files.append(filename)
            print(f"❌ {filename} (missing)")

    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        print("   Run this script again with option 2 to download missing files")
        return False
    else:
        print(f"\n🎉 All required files are present!")
        return True

def main():
    """Main function with menu options"""
    print("🤖 InternVideo2.5 Model Downloader")
    print("=" * 40)

    while True:
        print("\n📋 Choose an option:")
        print("1. Download model code only (fast, ~50MB)")
        print("2. Download complete model with weights (slow, ~15GB)")
        print("3. Verify existing installation of model files")
        print("4. Exit")

        choice = input("\n👉 Enter your choice (1-4): ").strip()

        if choice == "1":
            success = download_model_code_only()
            if success:
                print("\n🎯 Model code downloaded successfully!")
                print("   ⚠️  IMPORTANT: This is NOT enough for video analysis!")
                print("   🔧 What you can do:")
                print("      - Run this script again and choose option 2")
                print("      - Test your setup with 'python setup_check.py'")
                print("   🚀 For full functionality, choose option 2")
        elif choice == "2":
            success = download_full_model()
            if success:
                print("\n🎯 Complete model downloaded successfully!")
                print("   You can now run the video analysis!")
        elif choice == "3":
            verify_installation()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

        if choice in ["1", "2", "3"]:
            input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)