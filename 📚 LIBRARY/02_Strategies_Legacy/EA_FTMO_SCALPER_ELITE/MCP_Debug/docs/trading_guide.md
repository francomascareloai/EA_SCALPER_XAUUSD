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
from mt5_server import OrderRequest

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
from mt5_server import OrderRequest

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
from mt5_server import OrderRequest

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
from mt5_server import OrderRequest

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
