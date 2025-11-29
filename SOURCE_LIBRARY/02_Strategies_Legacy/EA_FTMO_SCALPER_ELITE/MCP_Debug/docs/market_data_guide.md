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
