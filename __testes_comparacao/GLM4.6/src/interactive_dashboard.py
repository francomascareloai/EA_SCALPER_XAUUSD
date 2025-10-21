#!/usr/bin/env python3
"""
Interactive Trading Dashboard with Real-time Updates
WebSocket-based real-time dashboard with live charts and alerts
"""

import asyncio
import websockets
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import random
import statistics
from collections import deque, defaultdict

# Import our visualization components
from advanced_visualization import RealTimeDataCollector, ChartGenerator, DashboardRenderer

@dataclass
class Alert:
    """Trading alert structure"""
    id: str
    type: str  # 'price', 'trade', 'risk', 'performance'
    message: str
    severity: str  # 'info', 'warning', 'critical'
    timestamp: datetime
    data: Dict[str, Any]

class AlertManager:
    """Manages real-time trading alerts"""

    def __init__(self, max_alerts: int = 100):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_rules = []
        self.alert_callbacks = []

    def add_alert(self, alert_type: str, message: str, severity: str = 'info', data: Dict[str, Any] = None):
        """Add new alert"""
        alert = Alert(
            id=f"alert_{int(time.time() * 1000)}",
            type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            data=data or {}
        )
        self.alerts.append(alert)

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")

        return alert

    def add_price_alert(self, symbol: str, price: float, threshold: float, direction: str):
        """Add price-based alert"""
        if direction == 'above' and price > threshold:
            return self.add_alert(
                'price',
                f"{symbol} crossed above ${threshold}",
                'warning',
                {'symbol': symbol, 'price': price, 'threshold': threshold}
            )
        elif direction == 'below' and price < threshold:
            return self.add_alert(
                'price',
                f"{symbol} crossed below ${threshold}",
                'warning',
                {'symbol': symbol, 'price': price, 'threshold': threshold}
            )

    def add_trade_alert(self, trade_data: Dict[str, Any]):
        """Add trade execution alert"""
        profit = trade_data.get('profit', 0)
        if profit > 0:
            message = f"Profitable trade: +${profit:.2f}"
            severity = 'info'
        else:
            message = f"Losing trade: ${profit:.2f}"
            severity = 'warning'

        return self.add_alert('trade', message, severity, trade_data)

    def add_risk_alert(self, metric: str, value: float, threshold: float):
        """Add risk-based alert"""
        if metric == 'drawdown' and value > threshold:
            return self.add_alert(
                'risk',
                f"Drawdown exceeded {threshold}%",
                'critical',
                {'metric': metric, 'value': value, 'threshold': threshold}
            )
        elif metric == 'margin_level' and value < threshold:
            return self.add_alert(
                'risk',
                f"Margin level below {threshold}%",
                'critical',
                {'metric': metric, 'value': value, 'threshold': threshold}
            )

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return [asdict(alert) for alert in list(self.alerts)[-limit:]]

    def subscribe_to_alerts(self, callback):
        """Subscribe to alert notifications"""
        self.alert_callbacks.append(callback)

class RealTimeDashboard:
    """Real-time interactive dashboard with WebSocket support"""

    def __init__(self, port: int = 8765):
        self.port = port
        self.data_collector = RealTimeDataCollector()
        self.chart_generator = ChartGenerator(self.data_collector)
        self.alert_manager = AlertManager()
        self.connected_clients = set()
        self.is_running = False
        self.update_interval = 1.0  # seconds

        # Setup alert callbacks
        self.alert_manager.subscribe_to_alerts(self._broadcast_alert)

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.risk_metrics_history = deque(maxlen=500)

    def _broadcast_alert(self, alert: Alert):
        """Broadcast alert to all connected clients"""
        message = {
            'type': 'alert',
            'data': asdict(alert)
        }
        asyncio.create_task(self._broadcast(message))

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            message_str = json.dumps(message, default=str)
            disconnected = set()

            for client in self.connected_clients:
                try:
                    await client.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    print(f"Broadcast error: {e}")
                    disconnected.add(client)

            # Remove disconnected clients
            self.connected_clients -= disconnected

    async def handle_client(self, websocket, path):
        """Handle individual WebSocket client"""
        print(f"🔗 Client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)

        try:
            # Send initial data
            await self.send_initial_data(websocket)

            # Handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    print(f"Invalid JSON from client: {message}")
                except Exception as e:
                    print(f"Error handling client message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def send_initial_data(self, websocket):
        """Send initial dashboard data to new client"""
        # Send market overview
        market_data = self.data_collector.get_market_data()
        await websocket.send(json.dumps({
            'type': 'market_update',
            'data': market_data
        }, default=str))

        # Send performance metrics
        metrics = self.data_collector.calculate_real_time_metrics()
        await websocket.send(json.dumps({
            'type': 'metrics_update',
            'data': {k: asdict(v) for k, v in metrics.items()}
        }, default=str))

        # Send recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(10)
        await websocket.send(json.dumps({
            'type': 'alerts_history',
            'data': recent_alerts
        }, default=str))

    async def handle_client_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming messages from clients"""
        msg_type = message.get('type')

        if msg_type == 'subscribe':
            # Client wants specific data updates
            subscription = message.get('subscription', 'all')
            print(f"Client subscribed to: {subscription}")

        elif msg_type == 'request_chart':
            # Client requests specific chart
            chart_type = message.get('chart_type', 'price')
            symbol = message.get('symbol', 'XAUUSD')

            chart_data = None
            if chart_type == 'price':
                chart_data = asdict(self.chart_generator.create_price_chart(symbol))
            elif chart_type == 'equity':
                chart_data = asdict(self.chart_generator.create_equity_curve())
            elif chart_type == 'performance':
                chart_data = asdict(self.chart_generator.create_performance_heatmap())
            elif chart_type == 'distribution':
                chart_data = asdict(self.chart_generator.create_trade_distribution())
            elif chart_type == 'radar':
                chart_data = asdict(self.chart_generator.create_risk_metrics_chart())

            if chart_data:
                await websocket.send(json.dumps({
                    'type': 'chart_data',
                    'chart_type': chart_type,
                    'data': chart_data
                }, default=str))

        elif msg_type == 'set_alert':
            # Client wants to set an alert
            alert_config = message.get('config', {})
            # Handle alert configuration (simplified)
            await websocket.send(json.dumps({
                'type': 'alert_confirmed',
                'data': {'id': f"alert_{int(time.time())}", 'status': 'active'}
            }))

    async def start_real_time_updates(self):
        """Start real-time data updates and broadcasting"""
        print("🔄 Starting real-time updates...")

        while self.is_running:
            try:
                # Update market data
                if self.data_collector.price_history:
                    latest_data = self.data_collector.get_market_data()
                    await self._broadcast({
                        'type': 'market_update',
                        'data': latest_data
                    })

                # Update performance metrics
                metrics = self.data_collector.calculate_real_time_metrics()
                if metrics:
                    await self._broadcast({
                        'type': 'metrics_update',
                        'data': {k: asdict(v) for k, v in metrics.items()}
                    })

                # Update charts periodically (every 10 seconds)
                if int(time.time()) % 10 == 0:
                    price_chart = asdict(self.chart_generator.create_price_chart())
                    await self._broadcast({
                        'type': 'chart_update',
                        'chart_type': 'price',
                        'data': price_chart
                    })

                    equity_chart = asdict(self.chart_generator.create_equity_curve())
                    await self._broadcast({
                        'type': 'chart_update',
                        'chart_type': 'equity',
                        'data': equity_chart
                    })

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                print(f"Update error: {e}")
                await asyncio.sleep(1)

    def simulate_market_activity(self):
        """Simulate real market activity for testing"""
        symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
        base_prices = {"XAUUSD": 2650.0, "EURUSD": 1.0850, "GBPUSD": 1.2750}

        while self.is_running:
            try:
                for symbol in symbols:
                    base_price = base_prices[symbol]

                    # Simulate price movement
                    volatility = 0.002  # 0.2% volatility
                    price_change = random.gauss(0, volatility)
                    new_price = base_price * (1 + price_change)
                    base_prices[symbol] = new_price

                    # Generate tick data
                    spread = new_price * random.uniform(0.0001, 0.0003)
                    bid = new_price - spread / 2
                    ask = new_price + spread / 2
                    volume = random.randint(10, 100)

                    self.data_collector.add_tick(symbol, bid, ask, volume)

                    # Generate occasional trades
                    if random.random() < 0.05:  # 5% chance
                        trade_profit = random.gauss(10, 25)  # Average $10 profit
                        trade = {
                            'symbol': symbol,
                            'type': random.choice(['BUY', 'SELL']),
                            'volume': random.uniform(0.01, 0.1),
                            'open_price': new_price,
                            'close_price': new_price + random.uniform(-5, 5),
                            'profit': trade_profit,
                            'duration_minutes': random.randint(1, 60)
                        }
                        self.data_collector.add_trade(trade)

                        # Add trade alert
                        self.alert_manager.add_trade_alert(trade)

                # Update account data
                base_equity = 10000
                equity_change = random.gauss(5, 20)  # Average $5 change
                current_equity = base_equity + equity_change
                current_balance = base_equity + random.gauss(0, 10)
                current_margin = random.uniform(500, 1500)

                self.data_collector.update_equity(current_equity, current_balance, current_margin)

                # Check for risk alerts
                if self.data_collector.equity_history:
                    equity_curve = [e['equity'] for e in self.data_collector.equity_history]
                    peak = max(equity_curve)
                    current = equity_curve[-1]
                    drawdown = ((peak - current) / peak) * 100 if peak > 0 else 0

                    if drawdown > 10:  # Alert if drawdown > 10%
                        self.alert_manager.add_risk_alert('drawdown', drawdown, 10)

                    margin_level = (current_equity / current_margin * 100) if current_margin > 0 else 100
                    if margin_level < 150:  # Alert if margin level < 150%
                        self.alert_manager.add_risk_alert('margin_level', margin_level, 150)

                time.sleep(1)

            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(1)

    async def start_server(self):
        """Start the WebSocket server"""
        print(f"🚀 Starting Interactive Dashboard Server on port {self.port}")

        # Start market simulation in background
        simulation_thread = threading.Thread(target=self.simulate_market_activity, daemon=True)
        simulation_thread.start()

        # Start real-time updates
        updates_task = asyncio.create_task(self.start_real_time_updates())

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            "localhost",
            self.port
        )

        self.is_running = True
        print(f"✅ Dashboard server running at ws://localhost:{self.port}")
        print("📊 Clients can connect to receive real-time updates")

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            print("\n🛑 Server shutting down...")
        finally:
            self.is_running = False
            updates_task.cancel()
            await server.close()

    def create_web_client(self, output_path: str = "interactive_dashboard.html"):
        """Create web client HTML file"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Trading Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            50% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        }
        .alert-info { animation: pulse-green 2s infinite; }
        .alert-warning { animation: pulse-red 2s infinite; }
        .chart-container { position: relative; height: 300px; }
        .status-online { background: linear-gradient(45deg, #10b981, #34d399); }
        .status-offline { background: linear-gradient(45deg, #ef4444, #f87171); }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <header class="mb-8">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-bold text-blue-400">Interactive Trading Dashboard</h1>
                    <p class="text-gray-400">Real-time market monitoring and analysis</p>
                </div>
                <div class="flex items-center space-x-4">
                    <div id="connection-status" class="flex items-center">
                        <div id="status-indicator" class="w-3 h-3 rounded-full status-offline mr-2"></div>
                        <span id="status-text" class="text-sm">Offline</span>
                    </div>
                    <div class="text-right">
                        <div class="text-2xl font-bold text-green-400" id="current-time">--:--:--</div>
                        <div class="text-gray-400" id="current-date">----</div>
                    </div>
                </div>
            </div>
        </header>

        <!-- Control Panel -->
        <section class="mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="flex flex-wrap gap-4 items-center">
                    <button id="connect-btn" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded transition-colors">
                        Connect to Server
                    </button>
                    <button id="disconnect-btn" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded transition-colors" disabled>
                        Disconnect
                    </button>
                    <select id="symbol-select" class="bg-gray-700 px-3 py-2 rounded">
                        <option value="XAUUSD">XAUUSD</option>
                        <option value="EURUSD">EURUSD</option>
                        <option value="GBPUSD">GBPUSD</option>
                    </select>
                    <button id="refresh-charts" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded transition-colors">
                        Refresh Charts
                    </button>
                    <div class="ml-auto">
                        <span class="text-sm text-gray-400">Last Update: </span>
                        <span id="last-update" class="text-sm text-green-400">Never</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Market Overview -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Market Overview</h2>
                <div id="market-overview" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Symbol</div>
                        <div class="text-lg font-bold text-yellow-400" id="market-symbol">--</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Bid</div>
                        <div class="text-lg font-bold text-green-400" id="market-bid">--</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Ask</div>
                        <div class="text-lg font-bold text-red-400" id="market-ask">--</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Spread</div>
                        <div class="text-lg font-bold text-orange-400" id="market-spread">--</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Change</div>
                        <div class="text-lg font-bold" id="market-change">--</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-sm text-gray-400">Volume</div>
                        <div class="text-lg font-bold text-cyan-400" id="market-volume">--</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Performance Metrics -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Performance Metrics</h2>
                <div id="performance-metrics" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <!-- Metrics will be populated dynamically -->
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
        </section>

        <!-- Alerts Panel -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Live Alerts -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Live Alerts</h3>
                <div id="alerts-panel" class="space-y-2 max-h-64 overflow-y-auto">
                    <div class="text-gray-400">Waiting for alerts...</div>
                </div>
            </div>

            <!-- Activity Log -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Activity Log</h3>
                <div id="activity-log" class="space-y-1 max-h-64 overflow-y-auto font-mono text-sm">
                    <div class="text-gray-400">System ready...</div>
                </div>
            </div>
        </section>
    </div>

    <script>
        class TradingDashboard {{
            constructor() {{
                this.ws = null;
                this.isConnected = false;
                this.charts = {};
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;

                this.initializeEventListeners();
                this.initializeCharts();
                this.startClock();
                this.log('Dashboard initialized');
            }}

            initializeEventListeners() {{
                document.getElementById('connect-btn').addEventListener('click', () => this.connect());
                document.getElementById('disconnect-btn').addEventListener('click', () => this.disconnect());
                document.getElementById('refresh-charts').addEventListener('click', () => this.refreshCharts());
                document.getElementById('symbol-select').addEventListener('change', () => this.changeSymbol());
            }}

            initializeCharts() {{
                // Price Chart
                const priceCtx = document.getElementById('priceChart').getContext('2d');
                this.charts.price = new Chart(priceCtx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [
                            {{
                                label: 'Price',
                                data: [],
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fill: false,
                                tension: 0.1
                            }},
                            {{
                                label: 'SMA 20',
                                data: [],
                                borderColor: '#f59e0b',
                                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                fill: false,
                                tension: 0.1
                            }}
                        ]
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
                this.charts.equity = new Chart(equityCtx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [
                            {{
                                label: 'Equity',
                                data: [],
                                borderColor: '#10b981',
                                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                fill: true,
                                tension: 0.1
                            }},
                            {{
                                label: 'Balance',
                                data: [],
                                borderColor: '#6366f1',
                                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                                fill: false,
                                tension: 0.1
                            }}
                        ]
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
            }}

            connect() {{
                const wsUrl = 'ws://localhost:8765';
                this.log(`Connecting to ${{wsUrl}}...`);

                try {{
                    this.ws = new WebSocket(wsUrl);

                    this.ws.onopen = () => {{
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus(true);
                        this.log('Connected to server');
                        this.log('Subscribing to real-time updates...');
                    }};

                    this.ws.onmessage = (event) => {{
                        this.handleMessage(JSON.parse(event.data));
                    }};

                    this.ws.onclose = () => {{
                        this.isConnected = false;
                        this.updateConnectionStatus(false);
                        this.log('Disconnected from server');
                        this.attemptReconnect();
                    }};

                    this.ws.onerror = (error) => {{
                        this.log(`WebSocket error: ${{error}}`);
                    }};

                }} catch (error) {{
                    this.log(`Connection failed: ${{error}}`);
                }}
            }}

            disconnect() {{
                if (this.ws) {{
                    this.ws.close();
                    this.ws = null;
                }}
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.log('Manually disconnected');
            }}

            attemptReconnect() {{
                if (this.reconnectAttempts < this.maxReconnectAttempts) {{
                    this.reconnectAttempts++;
                    this.log(`Attempting reconnect ${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}}...`);
                    setTimeout(() => this.connect(), 3000);
                }} else {{
                    this.log('Max reconnect attempts reached');
                }}
            }}

            handleMessage(message) {{
                switch (message.type) {{
                    case 'market_update':
                        this.updateMarketData(message.data);
                        break;
                    case 'metrics_update':
                        this.updateMetrics(message.data);
                        break;
                    case 'chart_update':
                        this.updateChart(message.chart_type, message.data);
                        break;
                    case 'alert':
                        this.addAlert(message.data);
                        break;
                    case 'chart_data':
                        this.updateChart(message.chart_type, message.data);
                        break;
                    case 'alerts_history':
                        this.loadAlertsHistory(message.data);
                        break;
                }}

                this.updateLastUpdateTime();
            }}

            updateMarketData(data) {{
                document.getElementById('market-symbol').textContent = data.symbol || '--';
                document.getElementById('market-bid').textContent = data.bid ? data.bid.toFixed(2) : '--';
                document.getElementById('market-ask').textContent = data.ask ? data.ask.toFixed(2) : '--';
                document.getElementById('market-spread').textContent = data.spread ? data.spread.toFixed(2) : '--';

                const changeElement = document.getElementById('market-change');
                if (data.price_change_24h !== undefined) {{
                    changeElement.textContent = `${{data.price_change_24h.toFixed(2)}}%`;
                    changeElement.className = `text-lg font-bold ${{data.price_change_24h >= 0 ? 'text-green-400' : 'text-red-400'}}`;
                }} else {{
                    changeElement.textContent = '--';
                }}

                document.getElementById('market-volume').textContent = data.volume || '--';
            }}

            updateMetrics(metrics) {{
                const container = document.getElementById('performance-metrics');
                container.innerHTML = '';

                Object.entries(metrics).forEach(([key, metric]) => {{
                    const statusClass = metric.status === 'positive' ? 'bg-green-800' :
                                       metric.status === 'negative' ? 'bg-red-800' : 'bg-gray-700';
                    const valueColor = metric.status === 'positive' ? 'text-green-400' :
                                      metric.status === 'negative' ? 'text-red-400' : 'text-gray-400';

                    const card = document.createElement('div');
                    card.className = `${{statusClass}} bg-opacity-30 rounded-lg p-3 border border-opacity-30 border-${{metric.status}}-500`;
                    card.innerHTML = `
                        <div class="text-sm text-gray-300 mb-1">${{metric.name}}</div>
                        <div class="text-xl font-bold ${{valueColor}}">${{metric.value.toFixed(2)}}</div>
                        <div class="text-xs text-gray-400 mt-1">${{metric.change_percent >= 0 ? '+' : ''}}${{metric.change_percent.toFixed(2)}}%</div>
                    `;
                    container.appendChild(card);
                }});
            }}

            updateChart(chartType, data) {{
                if (this.charts[chartType] && data) {{
                    this.charts[chartType].data.labels = data.labels || [];
                    this.charts[chartType].data.datasets = data.datasets || [];
                    this.charts[chartType].update('none');
                }}
            }}

            addAlert(alert) {{
                const alertsPanel = document.getElementById('alerts-panel');

                // Remove initial message if present
                if (alertsPanel.querySelector('.text-gray-400')) {{
                    alertsPanel.innerHTML = '';
                }}

                const alertElement = document.createElement('div');
                const severityClass = alert.severity === 'critical' ? 'border-red-500 bg-red-900 bg-opacity-20' :
                                     alert.severity === 'warning' ? 'border-yellow-500 bg-yellow-900 bg-opacity-20' :
                                     'border-blue-500 bg-blue-900 bg-opacity-20';

                const alertIcon = alert.severity === 'critical' ? '🚨' :
                                 alert.severity === 'warning' ? '⚠️' : 'ℹ️';

                alertElement.className = `${{severityClass}} border rounded p-3 alert-${{alert.severity}}`;
                alertElement.innerHTML = `
                    <div class="flex items-start">
                        <span class="text-lg mr-2">${{alertIcon}}</span>
                        <div class="flex-1">
                            <div class="font-semibold text-sm">${{alert.type.toUpperCase()}}</div>
                            <div class="text-sm">${{alert.message}}</div>
                            <div class="text-xs text-gray-400 mt-1">${{new Date(alert.timestamp).toLocaleTimeString()}}</div>
                        </div>
                    </div>
                `;

                alertsPanel.insertBefore(alertElement, alertsPanel.firstChild);

                // Keep only last 20 alerts
                while (alertsPanel.children.length > 20) {{
                    alertsPanel.removeChild(alertsPanel.lastChild);
                }}

                this.log(`Alert: ${{alert.message}}`);
            }}

            loadAlertsHistory(alerts) {{
                alerts.forEach(alert => this.addAlert(alert));
            }}

            refreshCharts() {{
                if (!this.isConnected) {{
                    this.log('Cannot refresh charts: not connected');
                    return;
                }}

                const symbol = document.getElementById('symbol-select').value;

                // Request chart updates
                ['price', 'equity', 'performance', 'distribution', 'radar'].forEach(chartType => {{
                    this.ws.send(JSON.stringify({{
                        type: 'request_chart',
                        chart_type: chartType,
                        symbol: symbol
                    }}));
                }});

                this.log('Requested chart updates');
            }}

            changeSymbol() {{
                if (this.isConnected) {{
                    this.refreshCharts();
                    this.log(`Changed symbol to ${{document.getElementById('symbol-select').value}}`);
                }}
            }}

            updateConnectionStatus(connected) {{
                const indicator = document.getElementById('status-indicator');
                const text = document.getElementById('status-text');
                const connectBtn = document.getElementById('connect-btn');
                const disconnectBtn = document.getElementById('disconnect-btn');

                if (connected) {{
                    indicator.className = 'w-3 h-3 rounded-full status-online mr-2';
                    text.textContent = 'Online';
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                }} else {{
                    indicator.className = 'w-3 h-3 rounded-full status-offline mr-2';
                    text.textContent = 'Offline';
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                }}
            }}

            updateLastUpdateTime() {{
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            }}

            startClock() {{
                const updateTime = () => {{
                    const now = new Date();
                    document.getElementById('current-time').textContent = now.toLocaleTimeString();
                    document.getElementById('current-date').textContent = now.toLocaleDateString();
                }};
                setInterval(updateTime, 1000);
                updateTime();
            }}

            log(message) {{
                const activityLog = document.getElementById('activity-log');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.className = 'text-gray-400';
                logEntry.textContent = `[${{timestamp}}] ${{message}}`;

                activityLog.appendChild(logEntry);

                // Keep only last 50 log entries
                while (activityLog.children.length > 50) {{
                    activityLog.removeChild(activityLog.firstChild);
                }}

                // Auto-scroll to bottom
                activityLog.scrollTop = activityLog.scrollHeight;
            }}
        }}

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {{
            window.dashboard = new TradingDashboard();
        }});
    </script>
</body>
</html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

# Main execution
if __name__ == "__main__":
    async def main():
        print("🚀 Starting Interactive Trading Dashboard...")

        # Create dashboard instance
        dashboard = RealTimeDashboard(port=8765)

        # Create web client
        print("📱 Creating web client interface...")
        client_path = dashboard.create_web_client("interactive_dashboard.html")
        print(f"✅ Web client created: {client_path}")

        print("\n📊 Dashboard Features:")
        print("• Real-time WebSocket connection")
        print("• Live price charts and market data")
        print("• Interactive performance metrics")
        print("• Real-time alerts and notifications")
        print("• Automatic chart updates")
        print("• Activity logging and monitoring")
        print("• Responsive web interface")

        print(f"\n🌐 Access the dashboard at: file://{client_path}")
        print("🔌 Connect to the server when prompted in the interface")

        # Start the server
        await dashboard.start_server()

    # Run the dashboard
    asyncio.run(main())