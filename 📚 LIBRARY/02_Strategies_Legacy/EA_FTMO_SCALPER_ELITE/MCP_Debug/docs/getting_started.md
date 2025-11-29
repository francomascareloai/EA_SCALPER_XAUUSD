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
