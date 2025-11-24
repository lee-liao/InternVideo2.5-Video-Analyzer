#!/usr/bin/env python3
"""
Simple launcher for video analysis
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_analyzer import main

    if __name__ == "__main__":
        print("Starting video analysis...")
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install requirements with: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)