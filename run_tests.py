#!/usr/bin/env python3
"""
Test runner script for product recommender system.

This script provides a convenient way to run tests with different configurations
and generate comprehensive test reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    result = subprocess.run(command, shell=True, capture_output=False)
    return result.returncode


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for product recommender system")
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run only fast tests (exclude slow/integration tests)"
    )
    parser.add_argument(
        "--unit-only", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration-only", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--specific-test", 
        type=str, 
        help="Run a specific test file or test function"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = "python -m pytest"
    
    # Add verbosity
    if args.verbose:
        pytest_cmd += " -v"
    
    # Add coverage
    if args.coverage:
        pytest_cmd += " --cov=flipkart --cov=utils --cov-report=term-missing --cov-report=html:htmlcov"
    
    # Add test selection
    if args.fast:
        pytest_cmd += " -m 'not slow'"
    elif args.unit_only:
        pytest_cmd += " -m unit"
    elif args.integration_only:
        pytest_cmd += " -m integration"
    
    # Add specific test
    if args.specific_test:
        pytest_cmd += f" {args.specific_test}"
    
    # Add test discovery path
    pytest_cmd += " tests/"
    
    print("Product Recommender System - Test Runner")
    print("=" * 50)
    
    # Run the tests
    exit_code = run_command(pytest_cmd, "Running pytest")
    
    if args.coverage and exit_code == 0:
        print(f"\nCoverage report generated in: htmlcov/index.html")
    
    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print('='*60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
