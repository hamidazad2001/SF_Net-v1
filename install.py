#!/usr/bin/env python3
"""
Setup script for installing SF_Net package in development mode.
This allows changes to the code to be immediately reflected without reinstallation.
"""

import os
import subprocess
import sys

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Install the package in development mode
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    
    print("\nSF_Net installed successfully in development mode.")
    print("You can now import modules using 'from src.models import ...' etc.")
    print("\nExamples:")
    print("- To train: python scripts/train.py --gin_config config/film_net-Style.gin --base_folder ./Checkpoint --label test_run")
    print("- To evaluate: python scripts/eval.py")

if __name__ == "__main__":
    main() 