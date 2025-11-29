# Trading Frameworks Overview: VectorBT + TensorTrade

## 1. VectorBT - High-Performance Backtesting

Repository: polakowo/vectorbt
Focus: Ultra-fast vectorized backtesting with NumPy/Pandas

### Key Features

- **Vectorized Operations**: Extremely fast backtesting using NumPy
- **10,000+ strategies** in seconds
- **Built-in technical indicators**
- **Portfolio analysis and visualization**

### Example: Simple Backtest

```python
import vectorbt as vbt

price = vbt.YFData.download('BTC-USD').get('Close')

# Buy & Hold
pf = vbt.Portfolio.from_holding(price, init_cash=100)
pf.total_profit()  # Returns profit

# SMA Crossover
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
pf.total_profit()
```

### Example: Hyperparameter Optimization

```python
# Test 10,000 window combinations
windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits)
pf.total_return().vbt.heatmap()  # Visualize results
```

### Portfolio Statistics

```python
pf.stats()
# Returns: Start, End, Total Return, Max Drawdown, Win Rate,
# Best/Worst Trade, Sharpe Ratio, Calmar Ratio, etc.
```

### Why VectorBT for Trading

1. **Speed**: 100x faster than event-driven backtesting
2. **Optimization**: Test thousands of parameter combinations
3. **Visualization**: Built-in plotting for analysis
4. **Realistic**: Supports fees, slippage, position sizing

---

## 2. TensorTrade - Reinforcement Learning Trading

Repository: tensortrade-org/tensortrade
Focus: Deep RL for algorithmic trading

### Key Features

- **OpenAI Gym compatible** trading environments
- **Modular architecture**: Exchanges, rewards, actions
- **Multi-asset support**
- **Integrates with TensorFlow/Keras**

### Architecture Components

1. **Exchange**: Simulated or live market data
2. **Feature Pipeline**: Data preprocessing
3. **Action Scheme**: How agent can trade
4. **Reward Scheme**: How agent is rewarded
5. **Trading Agent**: RL algorithm (PPO, A2C, etc.)

### Example: Creating Environment

```python
import tensortrade as tt

# Define exchange with market data
exchange = tt.exchanges.simulated.fbm_exchange(
    base_price=100,
    time_frame='1h',
    n_steps=4000
)

# Create feature pipeline
features = [
    'close', 'volume', 'rsi', 'macd'
]

# Build environment
env = tt.environments.default.create(
    exchange=exchange,
    action_scheme=tt.actions.SimpleOrders(),
    reward_scheme=tt.rewards.RiskAdjustedReturns()
)
```

### Training Agent

```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
```

### Why TensorTrade for Trading

1. **Reinforcement Learning**: Learn optimal policies from data
2. **Flexibility**: Customize every component
3. **Scalability**: Distribute training across machines
4. **Modern**: Uses latest RL algorithms

---

## Comparison

| Feature | VectorBT | TensorTrade |
|---------|----------|-------------|
| Approach | Vectorized rules | Reinforcement Learning |
| Speed | Very fast | Slower (training) |
| Use Case | Strategy optimization | Learning optimal policies |
| Complexity | Simple rules | Complex adaptive behavior |
| Best For | Testing indicators | Discovering new strategies |

## Combined Workflow

1. **VectorBT**: Fast exploration of indicator combinations
2. **TensorTrade**: Train RL agent on promising signals
3. **Deploy**: Use learned policy for live trading

```python
# Find best indicators with VectorBT
best_params = vectorbt_optimization()

# Use as features for TensorTrade
env = create_env_with_features(best_params)
agent = train_rl_agent(env)

# Deploy
agent.trade_live()
```
