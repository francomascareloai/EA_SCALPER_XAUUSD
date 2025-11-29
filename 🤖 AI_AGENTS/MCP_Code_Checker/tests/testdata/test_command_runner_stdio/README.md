# Test Data Directory

This directory contains test data for the command runner STDIO tests.

## Structure

- `test_scripts/` - Sample Python scripts for testing subprocess execution
- `expected_outputs/` - Expected output files for comparison tests
- `environment_configs/` - Environment configuration files for testing

## Usage

Tests in `test_command_runner_stdio.py` use this test data to verify:
- Python subprocess STDIO isolation works correctly
- Environment variable isolation functions properly  
- File-based STDIO redirection operates as expected
- Timeout and error handling behaves correctly
