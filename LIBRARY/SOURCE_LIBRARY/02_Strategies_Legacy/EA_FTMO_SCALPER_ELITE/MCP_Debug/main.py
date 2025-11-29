"""
MetaTrader 5 MCP Server - Main Entry Point

This is the main entry point for the MetaTrader 5 MCP server.
"""

import logging
import os
from fastmcp import FastMCP
from fastmcp.prompts.base import UserMessage, AssistantMessage

# Import server modules
from mt5_server import mcp
import mt5_server_market_data
import mt5_server_trading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mt5-mcp-server")

# Add prompts to help AI assistants use the MetaTrader 5 API

@mcp.prompt()
def connect_to_mt5(account: int, password: str, server: str) -> list:
    """
    Prompt for connecting to MetaTrader 5 and logging in to a trading account.
    
    Args:
        account: Trading account number
        password: Trading account password
        server: Trading server name
    """
    return [
        UserMessage("I need to connect to my MetaTrader 5 account and start trading."),
        AssistantMessage(
            "I'll help you connect to your MetaTrader 5 account. First, we need to initialize the MT5 terminal and then log in to your account."
        ),
        AssistantMessage(
            f"Let me do that for you. I'll use your account number {account}, connect to the {server} server, and log in."
        ),
        UserMessage("Great, please proceed with the connection."),
    ]

@mcp.prompt()
def analyze_market_data(symbol: str, timeframe: int) -> list:
    """
    Prompt for analyzing market data for a specific symbol and timeframe.
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "BTCUSD", etc.)
        timeframe: Timeframe (e.g., 1 for M1, 5 for M5, 15 for M15, etc.)
    """
    return [
        UserMessage(f"I want to analyze the market data for {symbol} on the {timeframe} timeframe."),
        AssistantMessage(
            f"I'll help you analyze the market data for {symbol} on the {timeframe} timeframe. "
            f"First, let me fetch the recent price data."
        ),
        UserMessage("Please show me the recent price movements and any significant patterns."),
    ]

@mcp.prompt()
def place_trade(symbol: str, order_type: str, volume: float) -> list:
    """
    Prompt for placing a trade for a specific symbol.
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "BTCUSD", etc.)
        order_type: Type of order (e.g., "buy", "sell", etc.)
        volume: Trade volume in lots
    """
    return [
        UserMessage(f"I want to place a {order_type} order for {symbol} with a volume of {volume} lots."),
        AssistantMessage(
            f"I'll help you place a {order_type} order for {symbol} with a volume of {volume} lots. "
            f"First, let me check the current market conditions and prepare the order."
        ),
        UserMessage("Please proceed with placing the order."),
    ]

@mcp.prompt()
def manage_positions() -> list:
    """
    Prompt for managing open positions.
    """
    return [
        UserMessage("I want to check and manage my open positions."),
        AssistantMessage(
            "I'll help you manage your open positions. Let me first fetch all your current open positions."
        ),
        UserMessage("Please show me the details of my open positions and any recommendations."),
    ]

@mcp.prompt()
def analyze_trading_history(days: int) -> list:
    """
    Prompt for analyzing trading history.
    
    Args:
        days: Number of days to analyze
    """
    return [
        UserMessage(f"I want to analyze my trading history for the past {days} days."),
        AssistantMessage(
            f"I'll help you analyze your trading history for the past {days} days. "
            f"Let me fetch your historical orders and deals."
        ),
        UserMessage("Please show me my performance statistics and any patterns in my trading."),
    ]

# Add context to help AI assistants understand how to use the MetaTrader 5 API

@mcp.resource("mt5://getting_started")
def getting_started() -> str:
    """
    Resource providing getting started information for the MetaTrader 5 API.
    """
    with open("docs/getting_started.md", "r") as file:
        return file.read()

@mcp.resource("mt5://trading_guide")
def trading_guide() -> str:
    """
    Resource providing a guide for trading with the MetaTrader 5 API.
    """
    with open("docs/trading_guide.md", "r") as file:
        return file.read()

@mcp.resource("mt5://market_data_guide")
def market_data_guide() -> str:
    """
    Resource providing a guide for accessing market data with the MetaTrader 5 API.
    """
    with open("docs/market_data_guide.md", "r") as file:
        return file.read()

# Run the server
if __name__ == "__main__":
    # Check if running in development mode
    dev_mode = os.environ.get("MT5_MCP_DEV_MODE", "false").lower() == "true"
    
    if dev_mode:
        # Run in development mode
        import uvicorn
        uvicorn.run("main:mcp.app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Run with FastMCP CLI
        print("Run the server with: fastmcp dev main.py")
        print("Or install it with: fastmcp install main.py")
