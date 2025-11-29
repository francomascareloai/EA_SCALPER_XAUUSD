"""
MetaTrader 5 MCP Server - Command Line Interface

This module provides a command-line interface for the MetaTrader 5 MCP server.
"""

import argparse
import logging
import os
import sys
import subprocess
from importlib.metadata import version

from mcp_metatrader5_server.main import mcp

logger = logging.getLogger("mt5-mcp-server.cli")

def get_version():
    """Get the package version."""
    try:
        return version("mcp-metatrader5-server")
    except Exception:
        return "0.1.0"  # Default version if not installed

def setup_environment():
    """Setup environment variables for autonomous agent integration"""
    # Set default MT5 paths if not already set
    if not os.environ.get("MT5_PATH"):
        # Common MT5 installation paths
        possible_paths = [
            "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
            "C:\\Users\\Admin\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\MQL5"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["MT5_PATH"] = path
                break
    
    # Set default timeout values for autonomous operation
    os.environ.setdefault("MT5_CONNECT_TIMEOUT", "30")
    os.environ.setdefault("MT5_RETRY_ATTEMPTS", "3")
    os.environ.setdefault("MT5_RETRY_DELAY", "5")
    
    # Set logging level for autonomous mode
    if os.environ.get("MT5_AUTONOMOUS_MODE") == "true":
        logging.getLogger().setLevel(logging.INFO)
        logger.info("🤖 Environment configured for autonomous agent operation")

def main():
    """Main entry point for the CLI."""
    # Setup environment before parsing arguments
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="MetaTrader 5 MCP Server - A Model Context Protocol server for MetaTrader 5"
    )
    
    parser.add_argument(
        "--version", "-v", action="store_true", help="Show version and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dev command
    dev_parser = subparsers.add_parser("dev", help="Run the server in development mode")
    dev_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to"
    )
    dev_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to"
    )
    dev_parser.add_argument(
        "--broker", type=str, default="default", help="Broker configuration to use (roboforex, default)"
    )
    dev_parser.add_argument(
        "--config-path", type=str, help="Path to broker configuration file"
    )
    dev_parser.add_argument(
        "--enable-autonomous", action="store_true", help="Enable autonomous agent mode with enhanced error handling"
    )
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install the server for Claude Desktop")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"mcp-metatrader5-server version {get_version()}")
        return 0
    
    if args.command == "dev":
        # Set environment variables for development mode
        os.environ["MT5_MCP_DEV_MODE"] = "true"
        os.environ["MT5_BROKER"] = args.broker
        
        # Set broker-specific environment variables
        if args.broker == "roboforex":
            os.environ["MT5_SERVER"] = os.environ.get("MT5_SERVER", "RoboForex-Demo")
            if args.config_path:
                os.environ["MT5_CONFIG_PATH"] = args.config_path
            else:
                # Use default RoboForex config path
                config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "roboforex_config.json")
                os.environ["MT5_CONFIG_PATH"] = os.path.abspath(config_path)
        
        # Enable autonomous mode if requested
        if args.enable_autonomous:
            os.environ["MT5_AUTONOMOUS_MODE"] = "true"
            logger.info("🤖 Autonomous agent mode enabled")
        
        logger.info(f"🚀 Starting MT5 MCP Server with broker: {args.broker}")
        logger.info(f"🌐 Server address: {args.host}:{args.port}")
        
        try:
            # Use uvicorn directly
            import uvicorn
            logger.info(f"📡 Starting server at {args.host}:{args.port}")
            uvicorn.run(
                "mcp_metatrader5_server.main:mcp",
                host=args.host,
                port=args.port,
                reload=True,
                log_level="info"
            )
            return 0
        except ImportError:
            # If uvicorn is not available, try using the command line
            cmd = [sys.executable, "-m", "uvicorn", "mcp_metatrader5_server.main:mcp", 
                   f"--host={args.host}", f"--port={args.port}", "--reload"]
            logger.info(f"Running command: {' '.join(cmd)}")
            return subprocess.call(cmd)
    elif args.command == "install":
        # Install for Claude Desktop
        try:
            cmd = [sys.executable, "--with", "mcp-metatrader5-server", "fastmcp", "install", "src\mcp_metatrader5_server\server.py"]
            return subprocess.call(cmd)
        except ImportError:
            logger.error("Failed to install MCP server for Claude Desktop")
            return 1
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
