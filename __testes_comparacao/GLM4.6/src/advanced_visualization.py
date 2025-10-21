#!/usr/bin/env python3
"""
Advanced Real-time Visualization and Dashboard System
Implements interactive charts, real-time monitoring, and comprehensive analytics
"""

import json
import time
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

@dataclass
class ChartData:
    """Chart data structure"""
    labels: List[str]
    datasets: List[Dict[str, Any]]
    title: str
    chart_type: str  # 'line', 'bar', 'candlestick', 'heatmap', 'scatter'

@dataclass
class DashboardMetric:
    """Dashboard metric structure"""
    name: str
    value: float
    change: float
    change_percent: float
    status: str  # 'positive', 'negative', 'neutral'
    timestamp: datetime
    category: str

class RealTimeDataCollector:
    """Collects and manages real-time trading data"""

    def __init__(self):
        self.price_history = deque(maxlen=1000)
        self.trades_history = deque(maxlen=500)
        self.equity_history = deque(maxlen=2000)
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.real_time_updates = deque(maxlen=50)
        self.performance_metrics = {}
        self.risk_metrics = {}

    def add_tick(self, symbol: str, bid: float, ask: float, volume: int):
        """Add real-time tick data"""
        tick = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'mid_price': (bid + ask) / 2,
            'spread': ask - bid,
            'volume': volume
        }
        self.price_history.append(tick)
        self.real_time_updates.append({
            'type': 'tick',
            'data': tick,
            'timestamp': datetime.now()
        })

    def add_trade(self, trade_data: Dict[str, Any]):
        """Add completed trade"""
        trade = {
            **trade_data,
            'timestamp': datetime.now()
        }
        self.trades_history.append(trade)
        self.real_time_updates.append({
            'type': 'trade',
            'data': trade,
            'timestamp': datetime.now()
        })

    def update_equity(self, equity: float, balance: float, margin: float):
        """Update account equity"""
        equity_update = {
            'timestamp': datetime.now(),
            'equity': equity,
            'balance': balance,
            'margin': margin,
            'free_margin': balance - margin,
            'margin_level': (equity / margin * 100) if margin > 0 else 100
        }
        self.equity_history.append(equity_update)

    def calculate_real_time_metrics(self) -> Dict[str, DashboardMetric]:
        """Calculate real-time performance metrics"""
        if not self.trades_history:
            return {}

        # Recent trades (last 50)
        recent_trades = list(self.trades_history)[-50:]

        # Win rate
        wins = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        win_rate = (wins / len(recent_trades)) * 100 if recent_trades else 0

        # Average win/loss
        profits = [t.get('profit', 0) for t in recent_trades if t.get('profit', 0) > 0]
        losses = [t.get('profit', 0) for t in recent_trades if t.get('profit', 0) < 0]

        avg_win = statistics.mean(profits) if profits else 0
        avg_loss = statistics.mean(losses) if losses else 0

        # Profit factor
        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Current drawdown
        if self.equity_history:
            equity_curve = [e['equity'] for e in self.equity_history]
            peak = max(equity_curve)
            current = equity_curve[-1]
            drawdown = ((peak - current) / peak) * 100 if peak > 0 else 0
        else:
            drawdown = 0

        # Sharpe ratio (simplified)
        if len(recent_trades) >= 10:
            returns = [t.get('profit', 0) for t in recent_trades]
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            sharpe_ratio = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        metrics = {
            'win_rate': DashboardMetric(
                'Win Rate', win_rate, 0, 0,
                'positive' if win_rate >= 60 else 'negative' if win_rate < 40 else 'neutral',
                datetime.now(), 'performance'
            ),
            'profit_factor': DashboardMetric(
                'Profit Factor', profit_factor, 0, 0,
                'positive' if profit_factor >= 1.5 else 'negative' if profit_factor < 1.0 else 'neutral',
                datetime.now(), 'performance'
            ),
            'drawdown': DashboardMetric(
                'Drawdown', drawdown, 0, 0,
                'negative' if drawdown > 10 else 'positive' if drawdown < 5 else 'neutral',
                datetime.now(), 'risk'
            ),
            'sharpe_ratio': DashboardMetric(
                'Sharpe Ratio', sharpe_ratio, 0, 0,
                'positive' if sharpe_ratio >= 1.0 else 'negative' if sharpe_ratio < 0.5 else 'neutral',
                datetime.now(), 'performance'
            ),
            'avg_win': DashboardMetric(
                'Avg Win', avg_win, 0, 0,
                'positive',
                datetime.now(), 'performance'
            ),
            'avg_loss': DashboardMetric(
                'Avg Loss', abs(avg_loss), 0, 0,
                'negative',
                datetime.now(), 'risk'
            )
        }

        return metrics

    def get_market_data(self, symbol: str = None) -> Dict[str, Any]:
        """Get current market data"""
        if not self.price_history:
            return {}

        recent_prices = list(self.price_history)[-100:]

        if symbol:
            symbol_prices = [p for p in recent_prices if p.get('symbol') == symbol]
            if symbol_prices:
                latest = symbol_prices[-1]
                prices = [p['mid_price'] for p in symbol_prices]
            else:
                latest = recent_prices[-1]
                prices = [p['mid_price'] for p in recent_prices]
        else:
            latest = recent_prices[-1]
            prices = [p['mid_price'] for p in recent_prices]

        # Technical indicators
        sma_20 = statistics.mean(prices[-20:]) if len(prices) >= 20 else 0
        sma_50 = statistics.mean(prices[-50:]) if len(prices) >= 50 else 0

        # Volatility (20-period standard deviation)
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = statistics.stdev(returns[-20:]) * math.sqrt(252) if len(returns) >= 20 else 0
        else:
            volatility = 0

        # Price change
        price_change = ((latest['mid_price'] - prices[0]) / prices[0] * 100) if prices and prices[0] > 0 else 0

        return {
            'symbol': symbol or latest.get('symbol', 'UNKNOWN'),
            'bid': latest['bid'],
            'ask': latest['ask'],
            'mid_price': latest['mid_price'],
            'spread': latest['spread'],
            'volume': latest['volume'],
            'sma_20': sma_20,
            'sma_50': sma_50,
            'volatility': volatility,
            'price_change_24h': price_change,
            'timestamp': latest['timestamp'].isoformat()
        }

class ChartGenerator:
    """Generates various types of trading charts"""

    def __init__(self, data_collector: RealTimeDataCollector):
        self.data_collector = data_collector

    def create_price_chart(self, symbol: str = None, periods: int = 100) -> ChartData:
        """Create price chart with moving averages"""
        prices = list(self.data_collector.price_history)[-periods:]

        if symbol:
            prices = [p for p in prices if p.get('symbol') == symbol]

        if not prices:
            return ChartData([], [], "Price Chart", "line")

        labels = [p['timestamp'].strftime('%H:%M:%S') for p in prices]
        mid_prices = [p['mid_price'] for p in prices]
        sma_20 = []
        sma_50 = []

        # Calculate moving averages
        for i in range(len(prices)):
            if i >= 19:
                sma_20.append(statistics.mean(mid_prices[i-19:i+1]))
            else:
                sma_20.append(mid_prices[i])

            if i >= 49:
                sma_50.append(statistics.mean(mid_prices[i-49:i+1]))
            else:
                sma_50.append(mid_prices[i])

        datasets = [
            {
                'label': 'Price',
                'data': mid_prices,
                'borderColor': '#3b82f6',
                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                'fill': False,
                'tension': 0.1
            },
            {
                'label': 'SMA 20',
                'data': sma_20,
                'borderColor': '#f59e0b',
                'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                'fill': False,
                'tension': 0.1
            },
            {
                'label': 'SMA 50',
                'data': sma_50,
                'borderColor': '#ef4444',
                'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                'fill': False,
                'tension': 0.1
            }
        ]

        return ChartData(
            labels=labels,
            datasets=datasets,
            title=f"{symbol or 'Symbol'} Price Chart",
            chart_type="line"
        )

    def create_equity_curve(self) -> ChartData:
        """Create equity curve chart"""
        equity_data = list(self.data_collector.equity_history)

        if not equity_data:
            return ChartData([], [], "Equity Curve", "line")

        labels = [e['timestamp'].strftime('%H:%M:%S') for e in equity_data]
        equity = [e['equity'] for e in equity_data]
        balance = [e['balance'] for e in equity_data]

        datasets = [
            {
                'label': 'Equity',
                'data': equity,
                'borderColor': '#10b981',
                'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                'fill': True,
                'tension': 0.1
            },
            {
                'label': 'Balance',
                'data': balance,
                'borderColor': '#6366f1',
                'backgroundColor': 'rgba(99, 102, 241, 0.1)',
                'fill': False,
                'tension': 0.1
            }
        ]

        return ChartData(
            labels=labels,
            datasets=datasets,
            title="Account Equity Curve",
            chart_type="line"
        )

    def create_performance_heatmap(self) -> ChartData:
        """Create performance heatmap (hourly performance)"""
        trades = list(self.data_collector.trades_history)

        if not trades:
            return ChartData([], [], "Performance Heatmap", "heatmap")

        # Group trades by hour
        hourly_performance = defaultdict(list)
        for trade in trades:
            hour = trade['timestamp'].hour
            profit = trade.get('profit', 0)
            hourly_performance[hour].append(profit)

        # Calculate average profit per hour
        hours = list(range(24))
        performance_data = []
        for hour in hours:
            if hour in hourly_performance:
                avg_profit = statistics.mean(hourly_performance[hour])
                performance_data.append(avg_profit)
            else:
                performance_data.append(0)

        # Create heatmap data format
        heatmap_data = []
        for i, hour in enumerate(hours):
            heatmap_data.append({
                'x': hour,
                'y': 0,
                'v': performance_data[i]
            })

        datasets = [
            {
                'label': 'Profit by Hour',
                'data': heatmap_data,
                'backgroundColor': lambda x: '#10b981' if x['v'] > 0 else '#ef4444' if x['v'] < 0 else '#6b7280'
            }
        ]

        return ChartData(
            labels=[str(h) for h in hours],
            datasets=datasets,
            title="Hourly Performance Heatmap",
            chart_type="heatmap"
        )

    def create_trade_distribution(self) -> ChartData:
        """Create trade profit/loss distribution chart"""
        trades = list(self.data_collector.trades_history)

        if not trades:
            return ChartData([], [], "Trade Distribution", "bar")

        profits = [t.get('profit', 0) for t in trades]

        # Create bins for histogram
        min_profit = min(profits) if profits else -100
        max_profit = max(profits) if profits else 100
        bins = 20
        bin_size = (max_profit - min_profit) / bins

        bin_counts = [0] * bins
        bin_labels = []

        for i in range(bins):
            bin_start = min_profit + i * bin_size
            bin_end = bin_start + bin_size
            bin_labels.append(f"{bin_start:.1f}")

            for profit in profits:
                if bin_start <= profit < bin_end:
                    bin_counts[i] += 1

        datasets = [
            {
                'label': 'Trade Count',
                'data': bin_counts,
                'backgroundColor': [
                    '#10b981' if i >= len(bin_counts)//2 else '#ef4444'
                    for i in range(len(bin_counts))
                ],
                'borderColor': '#374151',
                'borderWidth': 1
            }
        ]

        return ChartData(
            labels=bin_labels,
            datasets=datasets,
            title="Trade P&L Distribution",
            chart_type="bar"
        )

    def create_risk_metrics_chart(self) -> ChartData:
        """Create risk metrics radar chart"""
        metrics = self.data_collector.calculate_real_time_metrics()

        if not metrics:
            return ChartData([], [], "Risk Metrics", "radar")

        # Normalize metrics to 0-100 scale
        radar_data = {
            'Win Rate': min(metrics.get('win_rate', DashboardMetric('', 0, 0, 0, '', datetime.now(), '')).value, 100),
            'Profit Factor': min(metrics.get('profit_factor', DashboardMetric('', 0, 0, 0, '', datetime.now(), '')).value * 20, 100),
            'Sharpe Ratio': min(abs(metrics.get('sharpe_ratio', DashboardMetric('', 0, 0, 0, '', datetime.now(), '')).value) * 10, 100),
            'Low Drawdown': max(100 - metrics.get('drawdown', DashboardMetric('', 0, 0, 0, '', datetime.now(), '')).value, 0),
            'Consistency': random.uniform(60, 90)  # Placeholder
        }

        datasets = [
            {
                'label': 'Current Performance',
                'data': list(radar_data.values()),
                'borderColor': '#3b82f6',
                'backgroundColor': 'rgba(59, 130, 246, 0.2)',
                'pointBackgroundColor': '#3b82f6',
                'pointBorderColor': '#fff',
                'pointHoverBackgroundColor': '#fff',
                'pointHoverBorderColor': '#3b82f6'
            }
        ]

        return ChartData(
            labels=list(radar_data.keys()),
            datasets=datasets,
            title="Performance Radar",
            chart_type="radar"
        )

class DashboardRenderer:
    """Renders the trading dashboard with real-time updates"""

    def __init__(self, data_collector: RealTimeDataCollector):
        self.data_collector = data_collector
        self.chart_generator = ChartGenerator(data_collector)

    def generate_dashboard_html(self, symbol: str = "XAUUSD") -> str:
        """Generate complete dashboard HTML"""

        # Get real-time data
        market_data = self.data_collector.get_market_data(symbol)
        metrics = self.data_collector.calculate_real_time_metrics()

        # Generate charts
        price_chart = self.chart_generator.create_price_chart(symbol)
        equity_chart = self.chart_generator.create_equity_curve()
        performance_heatmap = self.chart_generator.create_performance_heatmap()
        trade_distribution = self.chart_generator.create_trade_distribution()
        risk_radar = self.chart_generator.create_risk_metrics_chart()

        # Convert charts to JSON
        charts_json = {
            'price_chart': asdict(price_chart),
            'equity_chart': asdict(equity_chart),
            'performance_heatmap': asdict(performance_heatmap),
            'trade_distribution': asdict(trade_distribution),
            'risk_radar': asdict(risk_radar)
        }

        # Metrics to JSON
        metrics_json = {k: asdict(v) for k, v in metrics.items()}

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EA Scalper XAUUSD - Real-time Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes pulse-green {
            0%, 100% { background-color: rgba(16, 185, 129, 0.1); }
            50% { background-color: rgba(16, 185, 129, 0.3); }
        }
        @keyframes pulse-red {
            0%, 100% { background-color: rgba(239, 68, 68, 0.1); }
            50% { background-color: rgba(239, 68, 68, 0.3); }
        }
        .status-positive { animation: pulse-green 2s infinite; }
        .status-negative { animation: pulse-red 2s infinite; }
        .chart-container { position: relative; height: 300px; }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <header class="mb-8">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-bold text-blue-400">EA Scalper Dashboard</h1>
                    <p class="text-gray-400">Real-time Trading Performance Monitor</p>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-bold text-green-400" id="current-time">--:--:--</div>
                    <div class="text-gray-400" id="current-date">----</div>
                </div>
            </div>
        </header>

        <!-- Market Overview -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Market Overview - {symbol}</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Bid</div>
                        <div class="text-lg font-bold text-yellow-400">{bid}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Ask</div>
                        <div class="text-lg font-bold text-yellow-400">{ask}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Spread</div>
                        <div class="text-lg font-bold text-orange-400">{spread}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Change 24h</div>
                        <div class="text-lg font-bold {change_color}">{change_24h:.2f}%</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Volatility</div>
                        <div class="text-lg font-bold text-purple-400">{volatility:.2f}%</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Volume</div>
                        <div class="text-lg font-bold text-cyan-400">{volume}</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Performance Metrics -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Performance Metrics</h2>
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    {metrics_cards}
                </div>
            </div>
        </section>

        <!-- Charts Grid -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Price Chart -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Price Chart</h3>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>

            <!-- Equity Curve -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Equity Curve</h3>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>

            <!-- Performance Heatmap -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Hourly Performance</h3>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>

            <!-- Trade Distribution -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Trade Distribution</h3>
                <div class="chart-container">
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>
        </section>

        <!-- Risk Radar -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Performance Radar</h3>
                <div class="chart-container" style="height: 400px;">
                    <canvas id="riskRadar"></canvas>
                </div>
            </div>
        </section>

        <!-- Live Activity Feed -->
        <section>
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Live Activity Feed</h3>
                <div id="activity-feed" class="space-y-2 max-h-60 overflow-y-auto">
                    <div class="text-gray-400">Waiting for live data...</div>
                </div>
            </div>
        </section>
    </div>

    <script>
        // Chart data
        const chartsData = {charts_json};
        const metricsData = {metrics_json};

        // Update time
        function updateTime() {{
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
            document.getElementById('current-date').textContent = now.toLocaleDateString();
        }}
        setInterval(updateTime, 1000);
        updateTime();

        // Create metric cards
        function createMetricCards() {{
            const container = document.querySelector('.grid.grid-cols-2.md\\:grid-cols-3.lg\\:grid-cols-6');
            if (!container) return;

            container.innerHTML = '';
            Object.entries(metricsData).forEach(([key, metric]) => {{
                const statusClass = metric.status === 'positive' ? 'status-positive border-green-500' :
                                   metric.status === 'negative' ? 'status-negative border-red-500' :
                                   'border-gray-500';
                const statusIcon = metric.status === 'positive' ? '↑' :
                                  metric.status === 'negative' ? '↓' : '→';
                const valueColor = metric.status === 'positive' ? 'text-green-400' :
                                  metric.status === 'negative' ? 'text-red-400' : 'text-gray-400';

                const card = document.createElement('div');
                card.className = `bg-gray-700 rounded p-3 border-2 ${{statusClass}} transition-all duration-300`;
                card.innerHTML = `
                    <div class="text-sm text-gray-400">${{metric.name}}</div>
                    <div class="text-lg font-bold ${{valueColor}}">${{metric.value.toFixed(2)}}</div>
                    <div class="text-xs text-gray-500">${{statusIcon}} ${{metric.change_percent.toFixed(2)}}%</div>
                `;
                container.appendChild(card);
            }});
        }}

        // Initialize charts
        function initCharts() {{
            // Price Chart
            const priceCtx = document.getElementById('priceChart').getContext('2d');
            new Chart(priceCtx, {{
                type: 'line',
                data: {{
                    labels: chartsData.price_chart.labels,
                    datasets: chartsData.price_chart.datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }},
                        y: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }}
                    }}
                }}
            }});

            // Equity Chart
            const equityCtx = document.getElementById('equityChart').getContext('2d');
            new Chart(equityCtx, {{
                type: 'line',
                data: {{
                    labels: chartsData.equity_chart.labels,
                    datasets: chartsData.equity_chart.datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }},
                        y: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }}
                    }}
                }}
            }});

            // Performance Chart (Bar)
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            new Chart(performanceCtx, {{
                type: 'bar',
                data: {{
                    labels: chartsData.performance_heatmap.labels,
                    datasets: [{{
                        label: 'Profit by Hour',
                        data: chartsData.performance_heatmap.datasets[0].data.map(d => d.v),
                        backgroundColor: chartsData.performance_heatmap.datasets[0].backgroundColor
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }},
                        y: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }}
                    }}
                }}
            }});

            // Distribution Chart
            const distributionCtx = document.getElementById('distributionChart').getContext('2d');
            new Chart(distributionCtx, {{
                type: 'bar',
                data: {{
                    labels: chartsData.trade_distribution.labels,
                    datasets: chartsData.trade_distribution.datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }},
                        y: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }}
                    }}
                }}
            }});

            // Risk Radar
            const radarCtx = document.getElementById('riskRadar').getContext('2d');
            new Chart(radarCtx, {{
                type: 'radar',
                data: {{
                    labels: chartsData.risk_radar.labels,
                    datasets: chartsData.risk_radar.datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#fff' }} }}
                    }},
                    scales: {{
                        r: {{
                            angleLines: {{ color: '#374151' }},
                            grid: {{ color: '#374151' }},
                            pointLabels: {{ color: '#9ca3af' }},
                            ticks: {{ color: '#9ca3af', backdropColor: 'transparent' }}
                        }}
                    }}
                }}
            }});
        }}

        // Simulate live updates
        function simulateLiveUpdates() {{
            const feed = document.getElementById('activity-feed');
            const activities = [
                {{ type: 'trade', message: 'Trade executed: BUY 0.01 XAUUSD @ 2650.45', class: 'text-green-400' }},
                {{ type: 'trade', message: 'Trade closed: Profit +$15.30', class: 'text-green-400' }},
                {{ type: 'signal', message: 'New signal detected: OVERSOLD', class: 'text-blue-400' }},
                {{ type: 'risk', message: 'Stop loss adjusted to 2648.20', class: 'text-yellow-400' }},
                {{ type: 'performance', message: 'Daily target: 78% achieved', class: 'text-purple-400' }}
            ];

            setInterval(() => {{
                const activity = activities[Math.floor(Math.random() * activities.length)];
                const time = new Date().toLocaleTimeString();
                const item = document.createElement('div');
                item.className = `${{activity.class}} text-sm`;
                item.innerHTML = `<span class="text-gray-500">${{time}}</span> - ${{activity.message}}`;

                feed.insertBefore(item, feed.firstChild);

                // Keep only last 10 items
                while (feed.children.length > 10) {{
                    feed.removeChild(feed.lastChild);
                }}
            }}, 5000);
        }}

        // Initialize everything
        document.addEventListener('DOMContentLoaded', () => {{
            createMetricCards();
            initCharts();
            simulateLiveUpdates();
        }});
    </script>
</body>
</html>
        """

        # Format template variables
        metrics_cards = ""
        for key, metric in metrics.items():
            status_class = "bg-green-900" if metric.status == 'positive' else "bg-red-900" if metric.status == 'negative' else "bg-gray-700"
            value_color = "text-green-400" if metric.status == 'positive' else "text-red-400" if metric.status == 'negative' else "text-gray-400"

            metrics_cards += f"""
                    <div class="{status_class} bg-opacity-30 rounded-lg p-4 border border-opacity-30 border-{metric.status}-500">
                        <div class="text-sm text-gray-300 mb-1">{metric.name}</div>
                        <div class="text-2xl font-bold {value_color}">{metric.value:.2f}</div>
                        <div class="text-xs text-gray-400 mt-1">{metric.change_percent:+.2f}%</div>
                    </div>
            """

        # Replace template placeholders using replace instead of format to avoid conflicts
        html_content = html_template
        html_content = html_content.replace('{symbol}', symbol)
        html_content = html_content.replace('{bid}', str(market_data.get('bid', 0)))
        html_content = html_content.replace('{ask}', str(market_data.get('ask', 0)))
        html_content = html_content.replace('{spread}', str(market_data.get('spread', 0)))
        html_content = html_content.replace('{change_color}', "text-green-400" if market_data.get('price_change_24h', 0) > 0 else "text-red-400")
        html_content = html_content.replace('{change_24h:.2f}', f"{market_data.get('price_change_24h', 0):.2f}")
        html_content = html_content.replace('{volatility:.2f}', f"{market_data.get('volatility', 0):.2f}")
        html_content = html_content.replace('{volume}', str(market_data.get('volume', 0)))
        html_content = html_content.replace('{metrics_cards}', metrics_cards)
        html_content = html_content.replace('{charts_json}', json.dumps(charts_json, default=str))
        html_content = html_content.replace('{metrics_json}', json.dumps(metrics_json, default=str))

        return html_content

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export all dashboard data for API usage"""
        return {
            'timestamp': datetime.now().isoformat(),
            'market_data': self.data_collector.get_market_data(),
            'metrics': {k: asdict(v) for k, v in self.data_collector.calculate_real_time_metrics().items()},
            'price_chart': asdict(self.chart_generator.create_price_chart()),
            'equity_chart': asdict(self.chart_generator.create_equity_curve()),
            'performance_heatmap': asdict(self.chart_generator.create_performance_heatmap()),
            'trade_distribution': asdict(self.chart_generator.create_trade_distribution()),
            'risk_radar': asdict(self.chart_generator.create_risk_metrics_chart()),
            'recent_trades': [dict(t) if hasattr(t, '__dict__') else t for t in list(self.data_collector.trades_history)[-10:]],
            'real_time_updates': [dict(u) if hasattr(u, '__dict__') else u for u in list(self.data_collector.real_time_updates)[-20:]]
        }

class VisualizationSystem:
    """Main visualization system coordinator"""

    def __init__(self):
        self.data_collector = RealTimeDataCollector()
        self.dashboard_renderer = DashboardRenderer(self.data_collector)
        self.is_running = False

    def start_monitoring(self, symbols: List[str] = ["XAUUSD"]):
        """Start real-time monitoring simulation"""
        self.is_running = True

        def simulate_data():
            """Simulate real-time market data"""
            base_price = 2650.0

            while self.is_running:
                for symbol in symbols:
                    # Simulate price movement
                    price_change = random.uniform(-2.0, 2.0)
                    base_price += price_change

                    bid = base_price - random.uniform(0.1, 0.3)
                    ask = base_price + random.uniform(0.1, 0.3)
                    volume = random.randint(10, 100)

                    self.data_collector.add_tick(symbol, bid, ask, volume)

                # Simulate occasional trades
                if random.random() < 0.1:  # 10% chance per iteration
                    trade = {
                        'symbol': random.choice(symbols),
                        'type': random.choice(['BUY', 'SELL']),
                        'volume': random.uniform(0.01, 0.1),
                        'open_price': base_price,
                        'close_price': base_price + random.uniform(-5.0, 5.0),
                        'profit': random.uniform(-50, 100),
                        'duration_minutes': random.randint(1, 60)
                    }
                    self.data_collector.add_trade(trade)

                # Update equity
                current_equity = 10000 + random.uniform(-500, 1000)
                current_balance = 10000 + random.uniform(-200, 500)
                current_margin = random.uniform(500, 1500)

                self.data_collector.update_equity(current_equity, current_balance, current_margin)

                time.sleep(1)  # Update every second

        # Start simulation in background
        import threading
        thread = threading.Thread(target=simulate_data, daemon=True)
        thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False

    def generate_dashboard(self, symbol: str = "XAUUSD", output_path: str = "dashboard.html") -> str:
        """Generate complete dashboard HTML file"""
        html_content = self.dashboard_renderer.generate_dashboard_html(symbol)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def create_interactive_report(self, optimization_results: List[Dict[str, Any]],
                                backtest_results: List[Dict[str, Any]]) -> str:
        """Create interactive optimization report"""

        report_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EA Optimization Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-blue-400 mb-2">EA Optimization Report</h1>
            <p class="text-gray-400">Generated on {timestamp}</p>
        </header>

        <section class="mb-8">
            <h2 class="text-2xl font-bold mb-4 text-blue-300">Optimization Results</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full bg-gray-800 rounded-lg">
                    <thead class="bg-gray-700">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Strategy</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Total Return</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Sharpe Ratio</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Max Drawdown</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Win Rate</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Profit Factor</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-600">
                        {optimization_rows}
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <h2 class="text-2xl font-bold mb-4 text-blue-300">Backtest Results</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {backtest_cards}
            </div>
        </section>
    </div>
</body>
</html>
        """

        # Generate optimization rows
        optimization_rows = ""
        for result in optimization_results[:10]:  # Top 10 results
            status = "🟢 Excellent" if result.get('sharpe_ratio', 0) > 1.5 else "🟡 Good" if result.get('sharpe_ratio', 0) > 1.0 else "🔴 Needs Improvement"

            optimization_rows += f"""
                        <tr class="hover:bg-gray-700">
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">{result.get('strategy_name', 'Unknown')}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{result.get('total_return', 0):.2f}%</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{result.get('sharpe_ratio', 0):.3f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{result.get('max_drawdown', 0):.2f}%</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{result.get('win_rate', 0):.1f}%</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{result.get('profit_factor', 0):.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">{status}</td>
                        </tr>
            """

        # Generate backtest cards
        backtest_cards = ""
        for i, result in enumerate(backtest_results[:4]):
            backtest_cards += f"""
                <div class="bg-gray-800 rounded-lg p-6">
                    <h3 class="text-lg font-bold mb-2 text-blue-300">Backtest {i+1}</h3>
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-gray-400">Duration:</span>
                            <span>{result.get('duration_hours', 0)} hours</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Total Trades:</span>
                            <span>{result.get('total_trades', 0)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Profit:</span>
                            <span class="text-green-400">${result.get('total_profit', 0):.2f}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Success Rate:</span>
                            <span>{result.get('success_rate', 0):.1f}%</span>
                        </div>
                    </div>
                </div>
            """

        return report_html.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            optimization_rows=optimization_rows,
            backtest_cards=backtest_cards
        )

# Example usage and testing
if __name__ == "__main__":
    print("🎨 Initializing Advanced Visualization System...")

    viz_system = VisualizationSystem()

    # Start monitoring
    print("▶️ Starting real-time monitoring...")
    viz_system.start_monitoring(["XAUUSD"])

    # Let it run for a few seconds to generate data
    time.sleep(5)

    # Generate dashboard
    print("📊 Generating trading dashboard...")
    dashboard_path = viz_system.generate_dashboard("XAUUSD", "trading_dashboard.html")
    print(f"✅ Dashboard generated: {dashboard_path}")

    # Export data
    print("📤 Exporting dashboard data...")
    dashboard_data = viz_system.dashboard_renderer.export_dashboard_data()

    # Save exported data
    with open("dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    print("✅ Dashboard data exported: dashboard_data.json")

    # Stop monitoring
    viz_system.stop_monitoring()
    print("⏹️ Monitoring stopped")

    print("\n🎯 Visualization Features:")
    print("• Real-time price charts with technical indicators")
    print("• Interactive equity curve visualization")
    print("• Performance heatmaps and distribution charts")
    print("• Risk metrics radar charts")
    print("• Live dashboard with automatic updates")
    print("• Export functionality for API integration")
    print("• Responsive design for all devices")
    print("• Professional dark theme interface")

    print("\n🌟 Advanced visualization system ready!")