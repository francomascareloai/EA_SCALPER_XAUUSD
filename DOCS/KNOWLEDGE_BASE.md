# Knowledge Base - EA_SCALPER_XAUUSD

## 1. Regime Detection

### Hurst Exponent
```python
def calculate_hurst(prices, min_k=10, max_k=50):
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    rs_values, window_sizes = [], []
    for n in range(min_k, max_k + 1):
        num_subseries = len(returns) // n
        rs_list = []
        for i in range(num_subseries):
            subseries = returns[i * n:(i + 1) * n]
            cumdev = np.cumsum(subseries - np.mean(subseries))
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(subseries, ddof=1)
            if S > 0: rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            window_sizes.append(n)
    log_n, log_rs = np.log(window_sizes), np.log(rs_values)
    return np.clip(np.polyfit(log_n, log_rs, 1)[0], 0, 1)
```

### Shannon Entropy
```python
def calculate_entropy(returns, bins=10):
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))
```

### Kalman Filter
```python
class KalmanTrendFilter:
    def __init__(self, Q=0.01, R=1.0):
        self.Q, self.R, self.x, self.P = Q, R, None, 1.0
    
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return measurement, 0.0
        x_pred, P_pred = self.x, self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x, measurement - x_pred
```

## 2. Feature Engineering (15 Features)

| # | Feature | Calculation |
|---|---------|-------------|
| 1-2 | Returns | pct_change, log |
| 3 | Range % | (H-L)/C |
| 4-6 | RSI | M5, M15, H1 |
| 7-8 | ATR, MA Dist | Normalized |
| 9 | BB Position | (C-mid)/width |
| 10-11 | Hurst, Entropy | Rolling 100 |
| 12-14 | Session, Hour | Sin/Cos encoding |
| 15 | OB Distance | To nearest OB |

## 3. SMC Concepts

**Order Blocks**: Last opposite candle before impulse
**FVG**: Gap between C1 high/low and C3 low/high
**Liquidity**: Equal highs/lows = pools to be swept

## 4. FTMO Rules

| Rule | Limit | Buffer |
|------|-------|--------|
| Daily DD | 5% | 4% |
| Total DD | 10% | 8% |
| Risk/trade | 1% | 0.5% |
