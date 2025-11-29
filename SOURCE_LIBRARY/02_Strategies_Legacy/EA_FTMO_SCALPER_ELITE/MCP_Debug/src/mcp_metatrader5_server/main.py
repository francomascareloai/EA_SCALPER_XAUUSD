"""
MetaTrader 5 MCP Server - Main Entry Point

This is the main entry point for the MetaTrader 5 MCP server.
"""

import logging
import os
from fastmcp import FastMCP
from fastmcp.prompts.base import UserMessage, AssistantMessage

# Import server modules
from mcp_metatrader5_server.server import mcp
import mcp_metatrader5_server.market_data
import mcp_metatrader5_server.trading

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
    return """
# Getting Started with MetaTrader 5 API

This MCP server provides access to the MetaTrader 5 API for trading and market data analysis.

## Basic Workflow

1. **Initialize the MT5 terminal**:
   - Use the `initialize()` tool to connect to the MT5 terminal.

2. **Log in to your trading account**:
   - Use the `login(account, password, server)` tool to log in to your trading account.

3. **Access market data**:
   - Use tools like `get_symbols()`, `copy_rates_from_pos()`, etc. to access market data.

4. **Place trades**:
   - Use the `order_send()` tool to place trades.

5. **Manage positions**:
   - Use tools like `positions_get()` to manage your open positions.

6. **Analyze trading history**:
   - Use tools like `history_orders_get()` and `history_deals_get()` to analyze your trading history.

7. **Shut down the connection**:
   - Use the `shutdown()` tool to close the connection to the MT5 terminal.

## Example: Connecting to MT5 and Getting Market Data

```python
# Initialize MT5
initialize()

# Log in to your trading account
login(account=123456, password="your_password", server="your_server")

# Get available symbols
symbols = get_symbols()

# Get recent price data for EURUSD
rates = copy_rates_from_pos(symbol="EURUSD", timeframe=15, start_pos=0, count=100)

# Shut down the connection
shutdown()
```

## Example: Placing a Trade

```python
# Initialize and log in
initialize()
login(account=123456, password="your_password", server="your_server")

# Create an order request
request = OrderRequest(
    action=mt5.TRADE_ACTION_DEAL,
    symbol="EURUSD",
    volume=0.1,
    type=mt5.ORDER_TYPE_BUY,
    price=1.1,
    deviation=20,
    magic=123456,
    comment="Buy order",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC
)

# Send the order
result = order_send(request)

# Shut down the connection
shutdown()
```
"""

@mcp.resource("mt5://trading_guide")
def trading_guide() -> str:
    """
    Resource providing a guide for trading with the MetaTrader 5 API.
    """
    return """
# Trading Guide for MetaTrader 5 API

This guide provides information on how to place and manage trades using the MetaTrader 5 API.

## Order Types

- **Market Orders**:
  - `ORDER_TYPE_BUY`: Buy at market price
  - `ORDER_TYPE_SELL`: Sell at market price

- **Pending Orders**:
  - `ORDER_TYPE_BUY_LIMIT`: Buy at specified price (lower than current price)
  - `ORDER_TYPE_SELL_LIMIT`: Sell at specified price (higher than current price)
  - `ORDER_TYPE_BUY_STOP`: Buy at specified price (higher than current price)
  - `ORDER_TYPE_SELL_STOP`: Sell at specified price (lower than current price)
  - `ORDER_TYPE_BUY_STOP_LIMIT`: Buy stop limit order
  - `ORDER_TYPE_SELL_STOP_LIMIT`: Sell stop limit order

## Trade Actions

- `TRADE_ACTION_DEAL`: Place a market order
- `TRADE_ACTION_PENDING`: Place a pending order
- `TRADE_ACTION_SLTP`: Modify stop loss and take profit levels
- `TRADE_ACTION_MODIFY`: Modify an existing order
- `TRADE_ACTION_REMOVE`: Remove a pending order
- `TRADE_ACTION_CLOSE_BY`: Close a position by an opposite one

## Example: Placing a Market Buy Order

```python
from mcp_metatrader5_server.server import OrderRequest

# Create an order request
request = OrderRequest(
    action=mt5.TRADE_ACTION_DEAL,
    symbol="EURUSD",
    volume=0.1,
    type=mt5.ORDER_TYPE_BUY,
    price=mt5.symbol_info_tick("EURUSD").ask,
    deviation=20,
    magic=123456,
    comment="Buy order",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC
)

# Send the order
result = order_send(request)
```

## Example: Placing a Pending Order

```python
from mcp_metatrader5_server.server import OrderRequest

# Create a pending order request
request = OrderRequest(
    action=mt5.TRADE_ACTION_PENDING,
    symbol="EURUSD",
    volume=0.1,
    type=mt5.ORDER_TYPE_BUY_LIMIT,
    price=1.08,  # Price to buy at
    sl=1.07,     # Stop loss
    tp=1.09,     # Take profit
    deviation=20,
    magic=123456,
    comment="Buy limit order",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC
)

# Send the order
result = order_send(request)
```

## Example: Modifying an Existing Position

```python
from mcp_metatrader5_server.server import OrderRequest

# Get the position
position = positions_get_by_ticket(ticket=123456)

# Create a request to modify stop loss and take profit
request = OrderRequest(
    action=mt5.TRADE_ACTION_SLTP,
    symbol=position.symbol,
    sl=1.07,     # New stop loss
    tp=1.09,     # New take profit
    position=position.ticket
)

# Send the order
result = order_send(request)
```

## Example: Closing a Position

```python
from mcp_metatrader5_server.server import OrderRequest

# Get the position
position = positions_get_by_ticket(ticket=123456)

# Create a request to close the position
request = OrderRequest(
    action=mt5.TRADE_ACTION_DEAL,
    symbol=position.symbol,
    volume=position.volume,
    type=mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
    price=mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
    position=position.ticket,
    deviation=20,
    magic=123456,
    comment="Close position",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC
)

# Send the order
result = order_send(request)
```
"""

@mcp.resource("mt5://market_data_guide")
def market_data_guide() -> str:
    """
    Resource providing a guide for accessing market data with the MetaTrader 5 API.
    """
    return """
# Market Data Guide for MetaTrader 5 API

This guide provides information on how to access and analyze market data using the MetaTrader 5 API.

## Timeframes

- `TIMEFRAME_M1`: 1 minute
- `TIMEFRAME_M5`: 5 minutes
- `TIMEFRAME_M15`: 15 minutes
- `TIMEFRAME_M30`: 30 minutes
- `TIMEFRAME_H1`: 1 hour
- `TIMEFRAME_H4`: 4 hours
- `TIMEFRAME_D1`: 1 day
- `TIMEFRAME_W1`: 1 week
- `TIMEFRAME_MN1`: 1 month

## Accessing Price Data

### Getting Bars (Candlesticks)

```python
# Get the last 100 bars for EURUSD on the H1 timeframe
rates = copy_rates_from_pos(symbol="EURUSD", timeframe=60, start_pos=0, count=100)

# Get bars for EURUSD on the D1 timeframe from a specific date
from datetime import datetime
rates = copy_rates_from_date(
    symbol="EURUSD",
    timeframe=1440,
    date_from=datetime(2023, 1, 1),
    count=100
)

# Get bars for EURUSD on the M15 timeframe within a date range
rates = copy_rates_range(
    symbol="EURUSD",
    timeframe=15,
    date_from=datetime(2023, 1, 1),
    date_to=datetime(2023, 1, 31)
)
```

### Getting Ticks

```python
# Get the last 1000 ticks for EURUSD
ticks = copy_ticks_from_pos(symbol="EURUSD", start_pos=0, count=1000)

# Get ticks for EURUSD from a specific date
ticks = copy_ticks_from_date(
    symbol="EURUSD",
    date_from=datetime(2023, 1, 1),
    count=1000
)

# Get ticks for EURUSD within a date range
ticks = copy_ticks_range(
    symbol="EURUSD",
    date_from=datetime(2023, 1, 1),
    date_to=datetime(2023, 1, 2)
)
```

## Analyzing Price Data

Once you have the price data, you can analyze it using pandas and numpy:

```python
import pandas as pd
import numpy as np

# Convert rates to a pandas DataFrame
df = pd.DataFrame(rates)

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')

# Calculate moving averages
df['sma_20'] = df['close'].rolling(window=20).mean()
df['sma_50'] = df['close'].rolling(window=50).mean()

# Calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['close'])

# Calculate MACD
def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

df['macd'], df['signal'], df['histogram'] = calculate_macd(df['close'])
```

## Getting Symbol Information

```python
# Get all available symbols
symbols = get_symbols()

# Get symbols by group
forex_symbols = get_symbols_by_group("*USD*")

# Get information about a specific symbol
symbol_info = get_symbol_info("EURUSD")

# Get the latest tick for a symbol
tick = get_symbol_info_tick("EURUSD")
```
"""
