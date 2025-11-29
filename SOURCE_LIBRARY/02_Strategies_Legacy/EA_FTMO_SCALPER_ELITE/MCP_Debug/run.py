#!/usr/bin/env python
"""
Development entry point for the MetaTrader 5 MCP server.
This script allows running the server directly from the project root.
"""

import os
import sys

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcp_metatrader5_server.cli import main

if __name__ == "__main__":
    sys.exit(main())
