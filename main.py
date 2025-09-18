#!/usr/bin/env python3
"""
GPU-Accelerated command-line interface for Whimper live transcription
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whimper import main

if __name__ == "__main__":
    exit(main())