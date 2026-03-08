#!/usr/bin/env python3

"""
Quick start script untuk menjalankan Genetic Algorithm Influencer Optimizer
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import numpy
        import matplotlib
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("  Genetic Algorithm - Influencer Selection Optimizer")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStarting GUI application...")
    print("Please wait while the interface loads...")
    print()
    
    # Import and run GUI
    try:
        from gui_app import main as run_gui
        run_gui()
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
