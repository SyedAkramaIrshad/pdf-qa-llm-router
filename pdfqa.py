#!/usr/bin/env python3
"""PDF QA System - Simple entry point.

Usage:
    python pdfqa.py config
    python pdfqa.py ask document.pdf "What is this about?"
    python pdfqa.py ask document.pdf -i  # Interactive mode
    python pdfqa.py index document.pdf
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.main import cli

if __name__ == "__main__":
    cli()
