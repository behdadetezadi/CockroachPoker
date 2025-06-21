"""
Test suite for Cockroach Poker AI Research Platform

Run tests using:
    python -m pytest tests/
    python -m pytest tests/test_game.py -v
    python -m pytest tests/ --cov=src
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

__version__ = "1.0.0"