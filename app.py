#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Entry Point
=======================

Run the template-based affine video stabiliser GUI.

Usage:
    uv run python app.py
    uv run python app.py --video 01_input/input.mp4
    uv run python app.py --nogui  # batch mode
"""

import sys
from pathlib import Path

# Add 02_code to path for module import
sys.path.insert(0, str(Path(__file__).parent / '02_code'))

from affine_template_matching import main

if __name__ == '__main__':
    main()
