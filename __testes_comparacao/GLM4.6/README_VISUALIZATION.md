# 🎨 Advanced Visualization & Interactive Dashboard

This directory contains the sophisticated visualization and real-time monitoring system for the EA Scalper XAUUSD project.

## 📊 Features Overview

### 🔄 Real-time Data Collection
- **Live Market Data**: Real-time price feeds with bid/ask spreads
- **Trade Execution Tracking**: Monitor all trade executions in real-time
- **Equity Monitoring**: Track account equity, balance, and margin levels
- **Performance Metrics**: Calculate and display key performance indicators

### 📈 Advanced Charting
- **Price Charts**: Interactive candlestick/line charts with technical indicators
- **Equity Curves**: Real-time equity and balance tracking
- **Performance Heatmaps**: Hourly performance visualization
- **Trade Distribution**: P&L distribution histograms
- **Risk Radar**: Multi-dimensional performance metrics

### 🚨 Alert System
- **Price Alerts**: Configurable price threshold notifications
- **Trade Alerts**: Real-time trade execution notifications
- **Risk Alerts**: Drawdown and margin level warnings
- **Performance Alerts**: Target achievement notifications

### 🌐 Interactive Dashboard
- **WebSocket Connectivity**: Real-time bi-directional communication
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Live Updates**: Automatic data refresh without page reload
- **Interactive Controls**: Symbol selection, chart refresh, connection management

## 🚀 Quick Start

### 1. Basic Dashboard Generation

```python
from src.advanced_visualization import VisualizationSystem

# Initialize visualization system
viz_system = VisualizationSystem()

# Start real-time monitoring
viz_system.start_monitoring(["XAUUSD"])

# Generate static dashboard
dashboard_path = viz_system.generate_dashboard("XAUUSD", "dashboard.html")
print(f"Dashboard generated: {dashboard_path}")
```

### 2. Interactive Dashboard with WebSocket

```python
from src.interactive_dashboard import RealTimeDashboard
import asyncio

async def start_dashboard():
    # Create interactive dashboard
    dashboard = RealTimeDashboard(port=8765)

    # Create web client
    client_path = dashboard.create_web_client("interactive_dashboard.html")

    # Start server
    await dashboard.start_server()

# Run the dashboard
asyncio.run(start_dashboard())
```

### 3. Real-time Data Simulation

```python
from src.advanced_visualization import RealTimeDataCollector

# Initialize data collector
collector = RealTimeDataCollector()

# Simulate market data
collector.add_tick("XAUUSD", 2650.45, 2650.75, 50)
collector.add_trade({
    'symbol': 'XAUUSD',
    'type': 'BUY',
    'volume': 0.01,
    'profit': 25.50
})

# Calculate metrics
metrics = collector.calculate_real_time_metrics()
print(f"Win Rate: {metrics['win_rate'].value:.1f}%")
```

## 📁 File Structure

```
src/
├── advanced_visualization.py     # Core visualization components
├── interactive_dashboard.py      # WebSocket-based real-time dashboard
└── ...

templates/
└── (dashboard HTML templates)

generated/
├── dashboard.html               # Static dashboard
├── interactive_dashboard.html   # Interactive client
└── dashboard_data.json         # Exported dashboard data
```

## 🎯 Component Details

### RealTimeDataCollector
- **Purpose**: Collects and manages real-time trading data
- **Features**: Price history, trade tracking, equity monitoring
- **Methods**: `add_tick()`, `add_trade()`, `calculate_real_time_metrics()`

### ChartGenerator
- **Purpose**: Generates various types of trading charts
- **Charts**: Price, equity, performance heatmap, distribution, radar
- **Methods**: `create_price_chart()`, `create_equity_curve()`, etc.

### AlertManager
- **Purpose**: Manages real-time trading alerts
- **Features**: Price, trade, risk, and performance alerts
- **Methods**: `add_alert()`, `add_price_alert()`, `add_risk_alert()`

### RealTimeDashboard
- **Purpose**: WebSocket-based real-time dashboard server
- **Features**: Real-time updates, client management, data broadcasting
- **Methods**: `start_server()`, `handle_client()`, `_broadcast()`

## 🔧 Configuration Options

### Dashboard Customization
```python
# Update intervals
dashboard.update_interval = 1.0  # seconds

# Data history limits
collector.price_history = deque(maxlen=1000)
collector.trades_history = deque(maxlen=500)

# Alert thresholds
alert_manager.add_risk_alert('drawdown', current_drawdown, 10.0)
alert_manager.add_risk_alert('margin_level', margin_level, 150.0)
```

### Chart Styling
```python
# Custom chart colors
chart_colors = {
    'price': '#3b82f6',
    'equity': '#10b981',
    'profit': '#10b981',
    'loss': '#ef4444',
    'background': '#1f2937'
}
```

### Alert Configuration
```python
# Alert severity levels
alert_types = {
    'info': 'ℹ️',
    'warning': '⚠️',
    'critical': '🚨'
}
```

## 📊 Available Metrics

### Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of total profits to total losses
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Mean profit/loss per trade

### Market Metrics
- **Bid/Ask Spread**: Current market spread
- **Price Change**: 24-hour price movement
- **Volatility**: Annualized volatility measure
- **Volume**: Trading volume information

### Risk Metrics
- **Margin Level**: Current margin utilization
- **Free Margin**: Available margin for trading
- **Drawdown**: Current drawdown percentage
- **Risk Score**: Overall risk assessment

## 🌐 Web Interface

### Accessing the Dashboard

1. **Static Dashboard**:
   ```bash
   # Open in browser
   open dashboard.html
   ```

2. **Interactive Dashboard**:
   ```bash
   # Start the server first
   python src/interactive_dashboard.py

   # Then open the client
   open interactive_dashboard.html
   ```

### Interface Features

- **Connection Status**: Real-time connection indicator
- **Market Overview**: Current market prices and spreads
- **Performance Metrics**: Key performance indicators
- **Interactive Charts**: Multiple chart types with real-time updates
- **Alerts Panel**: Live alert notifications
- **Activity Log**: System activity and connection events

## 🔍 API Reference

### WebSocket Messages

#### Client → Server
```json
{
    "type": "subscribe",
    "subscription": "all"
}

{
    "type": "request_chart",
    "chart_type": "price",
    "symbol": "XAUUSD"
}
```

#### Server → Client
```json
{
    "type": "market_update",
    "data": {
        "symbol": "XAUUSD",
        "bid": 2650.45,
        "ask": 2650.75,
        "spread": 0.30
    }
}

{
    "type": "alert",
    "data": {
        "type": "trade",
        "message": "Trade executed: BUY 0.01 XAUUSD",
        "severity": "info"
    }
}
```

### REST API Endpoints

#### Export Dashboard Data
```python
# Get all dashboard data
dashboard_data = dashboard.dashboard_renderer.export_dashboard_data()

# Save to file
with open("dashboard_export.json", "w") as f:
    json.dump(dashboard_data, f, indent=2, default=str)
```

## 🛠️ Development Guide

### Adding New Chart Types

```python
def create_custom_chart(self, data: List[Dict]) -> ChartData:
    """Create custom chart type"""
    labels = [d['timestamp'] for d in data]
    values = [d['value'] for d in data]

    datasets = [{
        'label': 'Custom Data',
        'data': values,
        'borderColor': '#8b5cf6',
        'backgroundColor': 'rgba(139, 92, 246, 0.1)'
    }]

    return ChartData(
        labels=labels,
        datasets=datasets,
        title="Custom Chart",
        chart_type="line"
    )
```

### Custom Alert Types

```python
def add_custom_alert(self, condition: bool, message: str):
    """Add custom alert based on condition"""
    if condition:
        return self.add_alert(
            'custom',
            message,
            'warning',
            {'condition': condition, 'timestamp': datetime.now()}
        )
```

### Extending Metrics

```python
def calculate_custom_metric(self) -> DashboardMetric:
    """Calculate custom performance metric"""
    # Custom calculation logic
    value = self.calculate_custom_value()
    change = self.calculate_change()

    return DashboardMetric(
        'Custom Metric',
        value,
        change,
        (change / abs(value)) * 100 if value != 0 else 0,
        'positive' if value > 0 else 'negative',
        datetime.now(),
        'custom'
    )
```

## 🚨 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**:
   - Check if server is running on correct port
   - Verify firewall settings
   - Check browser console for errors

2. **Charts Not Updating**:
   - Ensure WebSocket connection is established
   - Check data collector is receiving data
   - Verify chart initialization

3. **Alerts Not Working**:
   - Check alert manager configuration
   - Verify alert thresholds
   - Check alert callback functions

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor WebSocket messages
dashboard.debug_mode = True
```

## 📈 Performance Optimization

### Data Management
- Use deques with appropriate maxlen for memory efficiency
- Implement data aggregation for long-running sessions
- Cache frequently accessed calculations

### WebSocket Optimization
- Implement message batching for high-frequency updates
- Use compression for large data transfers
- Implement client-side rate limiting

### Chart Optimization
- Limit data points for performance
- Use chart.js animation settings appropriately
- Implement lazy loading for complex charts

## 🔒 Security Considerations

- WebSocket connections should use WSS in production
- Implement authentication for dashboard access
- Validate all incoming data
- Rate limit alert notifications
- Sanitize user inputs

## 📚 Dependencies

### Required Packages
- `websockets`: For WebSocket server functionality
- `asyncio`: For asynchronous operations
- `statistics`: For statistical calculations
- `json`: For data serialization
- `datetime`: For timestamp handling

### Optional Packages
- `numpy`: For advanced numerical computations
- `pandas`: For data analysis and manipulation
- `plotly`: For alternative charting library

## 🌟 Future Enhancements

### Planned Features
- [ ] Multi-timeframe chart support
- [ ] Advanced technical indicators
- [ ] Portfolio-level monitoring
- [ ] Mobile app interface
- [ ] Historical data analysis
- [ ] Machine learning predictions
- [ ] Social trading integration
- [ ] Custom alert rules engine

### Performance Improvements
- [ ] Real-time data compression
- [ ] Distributed architecture support
- [ ] Database integration for persistence
- [ ] Caching layer implementation
- [ ] Load balancing for multiple clients

---

**Note**: This visualization system is designed to work with the EA Scalper XAUUSD optimization framework. For best results, ensure all components are properly configured and integrated.