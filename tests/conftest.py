"""
Shared test fixtures and configuration.
"""
import sys
import os

# Ensure project root is on PYTHONPATH for all tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
