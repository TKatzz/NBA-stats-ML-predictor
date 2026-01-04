#!/usr/bin/env python3
"""
Quick launcher for the Streamlit web application
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "web", "app.py")
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

