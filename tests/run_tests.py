#!/usr/bin/env python3
"""Run all tests and print a coverage-style summary."""
import sys, os, unittest

# Make sure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

loader = unittest.TestLoader()
suite = loader.discover(os.path.dirname(__file__), pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
