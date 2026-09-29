"""
Top-level entry point for the FSOC PAT Simulation Demonstration.

Usage:
  py demo.py --preview                      # Interactive live GUI preview
  py demo.py --preview --save-video         # Live preview + save MP4 demo video
  py demo.py --headless                     # Headless fast execution
  py demo.py --stage 3                      # Run specific act (1 to 5)
  py demo.py --benchmark                    # Run algorithm benchmark comparison
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scripts.demo_simulation import main

if __name__ == "__main__":
    main()
