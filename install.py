#!/usr/bin/env python
# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to install the SF_Net package in development mode."""

import os
import subprocess
import sys

def setup_pythonpath():
    """Set up PYTHONPATH to include the project root directory."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    if root_dir not in pythonpath:
        if pythonpath:
            os.environ['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{root_dir}"
        else:
            os.environ['PYTHONPATH'] = root_dir
            
    print(f"PYTHONPATH set to include: {root_dir}")
    
    # Create a batch file to set PYTHONPATH for future sessions
    if os.name == 'nt':  # Windows
        batch_file = os.path.join(root_dir, 'set_pythonpath.bat')
        with open(batch_file, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'set PYTHONPATH=%PYTHONPATH%;{root_dir}\n')
            f.write(f'echo PYTHONPATH updated to include {root_dir}\n')
        print(f"Created batch file {batch_file}")
        print(f"Run 'set_pythonpath.bat' before running the scripts in future terminal sessions")
    else:  # Unix-like
        script_file = os.path.join(root_dir, 'set_pythonpath.sh')
        with open(script_file, 'w') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'export PYTHONPATH=$PYTHONPATH:{root_dir}\n')
            f.write(f'echo PYTHONPATH updated to include {root_dir}\n')
        os.chmod(script_file, 0o755)
        print(f"Created script {script_file}")
        print(f"Run 'source set_pythonpath.sh' before running the scripts in future terminal sessions")

def install_package():
    """Install the package in development mode."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Installing SF_Net from {root_dir} in development mode...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=root_dir)
        print("Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed with error: {e}")
        return False

def main():
    """Main function to set up SF_Net."""
    print("SF_Net Setup")
    print("============")
    print("1. Try pip installation (recommended if possible)")
    print("2. Just set up PYTHONPATH (alternative if pip installation fails)")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == '1':
        success = install_package()
        if not success:
            print("\nPip installation failed. Would you like to set up PYTHONPATH instead?")
            fallback = input("Enter 'y' for yes or any other key to exit: ").strip().lower()
            if fallback == 'y':
                setup_pythonpath()
    elif choice == '2':
        setup_pythonpath()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main() 