#!/usr/bin/env python3
import os
import sys
import subprocess

# Set up the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run streamlit
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main_app.py"])