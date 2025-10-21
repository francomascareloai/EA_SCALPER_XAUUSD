#!/usr/bin/env python3
"""
Sleep script for testing MCP subprocess execution.

CRITICAL: This script must be called with python -u flag to prevent timeout issues!
The -u flag forces unbuffered output, ensuring immediate communication with parent process.
"""

import sys
import time


def main():
    """Sleep for specified seconds with immediate output."""
    
    # Get sleep duration from command line
    sleep_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    
    # Basic validation
    if not 0 <= sleep_seconds <= 300:
        print(f"Error: Sleep duration must be between 0-300 seconds", flush=True)
        sys.exit(1)
    
    # Sleep with timing measurement
    start_time = time.time()
    time.sleep(sleep_seconds)
    actual_time = time.time() - start_time
    
    # Output results (flush=True is critical for MCP communication)
    print("Sleep operation completed successfully:", flush=True)
    print(f"  Requested: {sleep_seconds} seconds", flush=True)
    print(f"  Actual: {actual_time:.3f} seconds", flush=True)
    print(f"  Precision: {abs(actual_time - sleep_seconds):.3f} seconds difference", flush=True)


if __name__ == "__main__":
    main()
