#!/usr/bin/env python3
"""
Advanced Stress Testing Framework for EA Validation
Implements comprehensive validation scenarios and robustness testing
"""

import json
import time
import random
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import copy

@dataclass
class StressTestResult:
    """Stress test result structure"""
    test_name: str
    test_category: str
    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    max_drawdown: float
    max_consecutive_losses: int
    sharpe_ratio: float
    profit_factor: float
    recovery_factor: float
    avg_trade_duration: float
    volatility_exposure: float
    correlation_score: float
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]

@dataclass
class MarketScenario:
    """Market scenario definition"""
    name: str
    description: str
    volatility_multiplier: float
    trend_strength: float  # -1 to 1
    gap_probability: float
    news_impact: float
    liquidity_factor: float
    duration_hours: int
    characteristics: Dict[str, Any]

class MarketScenarioGenerator:
    """Generates various market stress scenarios"""

    def __init__(self):
        self.scenarios = self._create_base_scenarios()

    def _create_base_scenarios(self) -> List[MarketScenario]:
        """Create comprehensive market stress scenarios"""
        scenarios = [
            # High Volatility Scenarios
            MarketScenario(
                name="Extreme Volatility",
                description="Sudden market panic with extreme price movements",
                volatility_multiplier=3.0,
                trend_strength=0.0,
                gap_probability=0.15,
                news_impact=0.8,
                liquidity_factor=0.6,
                duration_hours=4,
                characteristics={
                    "price_jumps": True,
                    "spread_widening": True,
                    "slippage_high": True,
                    "order_rejections": True
                }
            ),
            MarketScenario(
                name="Flash Crash",
                description="Rapid downward price movement followed by recovery",
                volatility_multiplier=5.0,
                trend_strength=-0.8,
                gap_probability=0.3,
                news_impact=1.0,
                liquidity_factor=0.3,
                duration_hours=2,
                characteristics={
                    "rapid_decline": True,
                    "partial_recovery": True,
                    "order_delays": True,
                    "margin_calls": True
                }
            ),

            # Low Liquidity Scenarios
            MarketScenario(
                name="Low Liquidity",
                description="Thin market conditions with wide spreads",
                volatility_multiplier=1.5,
                trend_strength=0.2,
                gap_probability=0.05,
                news_impact=0.3,
                liquidity_factor=0.2,
                duration_hours=8,
                characteristics={
                    "wide_spreads": True,
                    "partial_fills": True,
                    "increased_slippage": True,
                    "reduced_volume": True
                }
            ),
            MarketScenario(
                name="Market Closure",
                description="Weekend/holiday gap scenarios",
                volatility_multiplier=2.0,
                trend_strength=0.5,
                gap_probability=0.8,
                news_impact=0.6,
                liquidity_factor=0.1,
                duration_hours=48,
                characteristics={
                    "gap_opening": True,
                    "price_gaps": True,
                    "no_trading_periods": True,
                    "weekend_effect": True
                }
            ),

            # Trend Scenarios
            MarketScenario(
                name="Strong Trend",
                description="Persistent directional market movement",
                volatility_multiplier=1.2,
                trend_strength=0.9,
                gap_probability=0.02,
                news_impact=0.4,
                liquidity_factor=0.8,
                duration_hours=12,
                characteristics={
                    "trend_persistence": True,
                    "momentum_effect": True,
                    "low_correlation": True,
                    "steady_movement": True
                }
            ),
            MarketScenario(
                name="Range Bound",
                description="Sideways market with clear support/resistance",
                volatility_multiplier=0.8,
                trend_strength=0.1,
                gap_probability=0.01,
                news_impact=0.2,
                liquidity_factor=0.9,
                duration_hours=24,
                characteristics={
                    "range_trading": True,
                    "support_resistance": True,
                    "mean_reversion": True,
                    "low_trend": True
                }
            ),

            # Economic Events
            MarketScenario(
                name="News Release",
                description="High-impact economic data release",
                volatility_multiplier=4.0,
                trend_strength=0.3,
                gap_probability=0.2,
                news_impact=1.0,
                liquidity_factor=0.5,
                duration_hours=1,
                characteristics={
                    "spike_volatility": True,
                    "rapid_movements": True,
                    "spread_explosion": True,
                    "order_cancellations": True
                }
            ),
            MarketScenario(
                name="Central Bank Announcement",
                description="Monetary policy decision impact",
                volatility_multiplier=2.5,
                trend_strength=0.6,
                gap_probability=0.1,
                news_impact=0.9,
                liquidity_factor=0.7,
                duration_hours=6,
                characteristics={
                    "policy_impact": True,
                    "currency_reaction": True,
                    "volatility_persist": True,
                    "trend_change": True
                }
            ),

            # Extreme Conditions
            MarketScenario(
                name="Black Swan",
                description="Unprecedented market event",
                volatility_multiplier=10.0,
                trend_strength=-0.5,
                gap_probability=0.5,
                news_impact=1.0,
                liquidity_factor=0.1,
                duration_hours=24,
                characteristics={
                    "extreme_movements": True,
                    "market_halts": True,
                    "counterparty_risk": True,
                    "systemic_stress": True
                }
            ),
            MarketScenario(
                name="Multiple Timeframes",
                description="Conflicting signals across timeframes",
                volatility_multiplier=1.5,
                trend_strength=0.0,
                gap_probability=0.05,
                news_impact=0.4,
                liquidity_factor=0.8,
                duration_hours=16,
                characteristics={
                    "timeframe_conflict": True,
                    "signal_divergence": True,
                    "choppy_market": True,
                    "whipsaw_patterns": True
                }
            )
        ]

        return scenarios

    def get_scenario(self, name: str) -> Optional[MarketScenario]:
        """Get specific scenario by name"""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None

    def get_all_scenarios(self) -> List[MarketScenario]:
        """Get all available scenarios"""
        return self.scenarios.copy()

class StressTestEngine:
    """Main stress testing engine"""

    def __init__(self):
        self.scenario_generator = MarketScenarioGenerator()
        self.test_results = []
        self.benchmark_results = {}

    def generate_market_data(self, scenario: MarketScenario, base_price: float = 2650.0,
                           duration_hours: int = None) -> List[Dict[str, Any]]:
        """Generate market data for specific stress scenario"""
        if duration_hours is None:
            duration_hours = scenario.duration_hours

        # Generate price ticks for the scenario
        ticks_per_hour = 60  # 1 tick per minute
        total_ticks = duration_hours * ticks_per_hour

        market_data = []
        current_price = base_price
        current_volatility = 0.001  # Base volatility (0.1%)

        # Apply scenario parameters
        volatility = current_volatility * scenario.volatility_multiplier

        for i in range(total_ticks):
            timestamp = datetime.now() + timedelta(minutes=i)

            # Generate price movement
            price_change = self._generate_price_change(
                current_price, volatility, scenario.trend_strength, i, total_ticks
            )

            # Apply gap if scenario requires
            if random.random() < scenario.gap_probability / 60:  # Per minute probability
                gap_size = random.gauss(0, volatility * 5)
                price_change += gap_size

            current_price += price_change

            # Ensure positive price
            current_price = max(current_price, 1.0)

            # Generate spread based on liquidity
            base_spread = current_price * 0.0001  # 1 pip base spread
            spread_multiplier = 1.0 / scenario.liquidity_factor
            spread = base_spread * spread_multiplier * random.uniform(0.8, 1.5)

            # Add news impact spikes
            if random.random() < scenario.news_impact / 100:
                news_impact = random.gauss(0, volatility * 3)
                current_price += news_impact
                spread *= random.uniform(1.5, 3.0)

            bid = current_price - spread / 2
            ask = current_price + spread / 2

            # Generate volume based on liquidity
            base_volume = 50
            volume = int(base_volume * scenario.liquidity_factor * random.uniform(0.5, 2.0))

            tick_data = {
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'mid_price': current_price,
                'spread': spread,
                'volume': volume,
                'scenario': scenario.name,
                'volatility': volatility,
                'liquidity_factor': scenario.liquidity_factor
            }

            market_data.append(tick_data)

        return market_data

    def _generate_price_change(self, current_price: float, volatility: float,
                             trend_strength: float, step: int, total_steps: int) -> float:
        """Generate realistic price change with trend"""
        # Random walk with trend
        random_component = random.gauss(0, volatility)
        trend_component = trend_strength * volatility * 0.1

        # Add some mean reversion for realism
        mean_reversion = -0.01 * random_component if abs(random_component) > volatility * 2 else 0

        # Add momentum effect
        momentum = 0
        if step > 0:
            momentum = 0.05 * random_component  # Small momentum effect

        total_change = (random_component + trend_component + mean_reversion + momentum) * current_price

        return total_change

    def simulate_ea_performance(self, market_data: List[Dict[str, Any]],
                              strategy_params: Dict[str, Any] = None) -> StressTestResult:
        """Simulate EA performance under stress conditions"""
        if strategy_params is None:
            strategy_params = self._get_default_strategy_params()

        start_time = market_data[0]['timestamp']
        end_time = market_data[-1]['timestamp']

        # Trading simulation
        trades = []
        equity_curve = []
        current_equity = 10000.0
        max_equity = current_equity
        current_position = None
        consecutive_losses = 0
        max_consecutive_losses = 0

        # Simulate trading
        for i, tick in enumerate(market_data):
            # Simple trend-following strategy for stress testing
            if len(market_data) > 20 and i > 0:
                recent_prices = [t['mid_price'] for t in market_data[max(0, i-20):i]]
                if len(recent_prices) >= 5:
                    sma_short = statistics.mean(recent_prices[-5:])
                else:
                    sma_short = recent_prices[-1] if recent_prices else current_price

                if len(recent_prices) >= 20:
                    sma_long = statistics.mean(recent_prices)
                else:
                    sma_long = recent_prices[-1] if recent_prices else current_price
            else:
                # Not enough data for trading signals
                continue

                current_price = tick['mid_price']
                spread = tick['spread']

                # Trading signals
                if current_position is None:
                    if sma_short > sma_long * 1.001:  # Uptrend
                        # Enter long
                        entry_price = ask = current_price + spread/2
                        current_position = {
                            'type': 'LONG',
                            'entry_price': entry_price,
                            'entry_time': tick['timestamp'],
                            'volume': 0.01,
                            'stop_loss': entry_price * 0.98,  # 2% SL
                            'take_profit': entry_price * 1.04  # 4% TP
                        }
                    elif sma_short < sma_long * 0.999:  # Downtrend
                        # Enter short
                        entry_price = bid = current_price - spread/2
                        current_position = {
                            'type': 'SHORT',
                            'entry_price': entry_price,
                            'entry_time': tick['timestamp'],
                            'volume': 0.01,
                            'stop_loss': entry_price * 1.02,  # 2% SL
                            'take_profit': entry_price * 0.96  # 4% TP
                        }

                elif current_position is not None:
                    # Check exit conditions
                    should_exit = False
                    exit_price = current_price
                    exit_reason = ""

                    if current_position['type'] == 'LONG':
                        if current_price <= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price >= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            should_exit = True
                            exit_reason = "Take Profit"
                        elif sma_short < sma_long * 0.998:  # Trend change
                            should_exit = True
                            exit_reason = "Trend Change"

                    else:  # SHORT
                        if current_price >= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price <= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            should_exit = True
                            exit_reason = "Take Profit"
                        elif sma_short > sma_long * 1.002:  # Trend change
                            should_exit = True
                            exit_reason = "Trend Change"

                    if should_exit or i == len(market_data) - 1:  # Force exit at end
                        # Calculate profit/loss
                        if current_position['type'] == 'LONG':
                            profit = (exit_price - current_position['entry_price']) * 100  # Simplified
                        else:
                            profit = (current_position['entry_price'] - exit_price) * 100

                        # Account for spread
                        profit -= spread * 50  # Spread cost

                        trade = {
                            'entry_time': current_position['entry_time'],
                            'exit_time': tick['timestamp'],
                            'type': current_position['type'],
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'exit_reason': exit_reason,
                            'duration_minutes': (tick['timestamp'] - current_position['entry_time']).total_seconds() / 60
                        }

                        trades.append(trade)
                        current_equity += profit

                        # Track consecutive losses
                        if profit < 0:
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            consecutive_losses = 0

                        max_equity = max(max_equity, current_equity)
                        current_position = None

            equity_curve.append({
                'timestamp': tick['timestamp'],
                'equity': current_equity,
                'price': tick['mid_price']
            })

        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = total_trades - winning_trades

        total_profit = sum(t['profit'] for t in trades)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        sharpe_ratio = self._calculate_sharpe_ratio(trades)
        profit_factor = self._calculate_profit_factor(trades)
        recovery_factor = abs(total_profit / max_drawdown) if max_drawdown != 0 else 0

        avg_trade_duration = statistics.mean([t['duration_minutes'] for t in trades]) if trades else 0

        # Calculate volatility exposure
        price_changes = [market_data[i]['mid_price'] - market_data[i-1]['mid_price']
                        for i in range(1, len(market_data))]
        volatility_exposure = statistics.stdev(price_changes) if len(price_changes) > 1 else 0

        # Score the performance (0-100)
        score = self._calculate_performance_score(
            total_trades, winning_trades, total_profit, max_drawdown,
            sharpe_ratio, profit_factor, max_consecutive_losses
        )

        # Determine if test passed (score >= 60)
        passed = score >= 60 and total_trades > 0 and max_drawdown < 30

        return StressTestResult(
            test_name=market_data[0]['scenario'] if market_data else "Unknown",
            test_category=self._categorize_scenario(market_data[0]['scenario'] if market_data else ""),
            start_time=start_time,
            end_time=end_time,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit=total_profit,
            max_drawdown=max_drawdown,
            max_consecutive_losses=max_consecutive_losses,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            avg_trade_duration=avg_trade_duration,
            volatility_exposure=volatility_exposure,
            correlation_score=0.0,  # Would need benchmark data
            passed=passed,
            score=score,
            details={
                'strategy_params': strategy_params,
                'market_conditions': market_data[0] if market_data else {},
                'trades': trades[:10],  # First 10 trades for analysis
                'equity_curve': equity_curve
            }
        )

    def _get_default_strategy_params(self) -> Dict[str, Any]:
        """Get default strategy parameters for testing"""
        return {
            'sma_short': 5,
            'sma_long': 20,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'volume': 0.01,
            'max_positions': 1
        }

    def _calculate_max_drawdown(self, equity_curve: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not equity_curve:
            return 0.0

        max_equity = equity_curve[0]['equity']
        max_drawdown = 0.0

        for point in equity_curve:
            current_equity = point['equity']
            max_equity = max(max_equity, current_equity)

            drawdown = ((max_equity - current_equity) / max_equity) * 100
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_sharpe_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio from trades"""
        if len(trades) < 2:
            return 0.0

        returns = [t['profit'] for t in trades]
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe ratio (assuming 252 trading days)
        sharpe = (avg_return / std_return) * math.sqrt(252)
        return sharpe

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_performance_score(self, total_trades: int, winning_trades: int,
                                   total_profit: float, max_drawdown: float,
                                   sharpe_ratio: float, profit_factor: float,
                                   max_consecutive_losses: int) -> float:
        """Calculate overall performance score (0-100)"""
        score = 0.0

        # Win rate component (20 points)
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            score += min(win_rate / 2, 20)  # Max 20 points for 40%+ win rate

        # Profit component (20 points)
        score += min(total_profit / 50, 20)  # Max 20 points for $1000+ profit

        # Drawdown component (20 points)
        drawdown_penalty = min(max_drawdown / 2, 20)
        score += max(0, 20 - drawdown_penalty)  # Less penalty for lower drawdown

        # Sharpe ratio component (15 points)
        score += min(abs(sharpe_ratio) * 5, 15)  # Max 15 points for 3.0+ Sharpe

        # Profit factor component (15 points)
        if profit_factor != float('inf'):
            score += min(profit_factor * 5, 15)  # Max 15 points for 3.0+ profit factor
        else:
            score += 15  # Perfect score for infinite profit factor

        # Consistency component (10 points)
        consistency_score = max(0, 10 - max_consecutive_losses)  # Penalty for consecutive losses
        score += consistency_score

        return min(score, 100.0)  # Cap at 100

    def _categorize_scenario(self, scenario_name: str) -> str:
        """Categorize stress scenario by type"""
        if 'volatility' in scenario_name.lower() or 'crash' in scenario_name.lower():
            return 'Volatility Stress'
        elif 'liquidity' in scenario_name.lower() or 'closure' in scenario_name.lower():
            return 'Liquidity Stress'
        elif 'trend' in scenario_name.lower() or 'range' in scenario_name.lower():
            return 'Market Regime'
        elif 'news' in scenario_name.lower() or 'announcement' in scenario_name.lower():
            return 'Event Risk'
        elif 'black' in scenario_name.lower() or 'swan' in scenario_name.lower():
            return 'Extreme Event'
        else:
            return 'General Stress'

    def run_stress_test(self, scenario: MarketScenario, strategy_params: Dict[str, Any] = None) -> StressTestResult:
        """Run single stress test scenario"""
        print(f"🧪 Running stress test: {scenario.name}")

        # Generate market data for scenario
        market_data = self.generate_market_data(scenario)

        # Simulate EA performance
        result = self.simulate_ea_performance(market_data, strategy_params)

        print(f"   ✓ Completed: Score {result.score:.1f}/100, Status: {'PASSED' if result.passed else 'FAILED'}")

        return result

    def run_comprehensive_stress_test(self, strategy_params: Dict[str, Any] = None) -> List[StressTestResult]:
        """Run all stress test scenarios"""
        print("🚀 Starting Comprehensive Stress Testing")
        print("=" * 50)

        scenarios = self.scenario_generator.get_all_scenarios()
        results = []

        for scenario in scenarios:
            result = self.run_stress_test(scenario, strategy_params)
            results.append(result)
            self.test_results.append(result)

            # Small delay between tests
            time.sleep(0.1)

        print("\n📊 Stress Testing Summary:")
        print("-" * 30)

        passed_tests = len([r for r in results if r.passed])
        total_tests = len(results)
        avg_score = statistics.mean([r.score for r in results])

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Average Score: {avg_score:.1f}/100")

        return results

    def generate_stress_test_report(self, results: List[StressTestResult] = None) -> str:
        """Generate comprehensive stress test report"""
        if results is None:
            results = self.test_results

        if not results:
            return "No stress test results available"

        # Calculate overall statistics
        passed_tests = len([r for r in results if r.passed])
        total_tests = len(results)
        avg_score = statistics.mean([r.score for r in results])

        # Group by category
        category_results = defaultdict(list)
        for result in results:
            category_results[result.test_category].append(result)

        # Generate HTML report
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EA Stress Test Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-passed {{ background: linear-gradient(45deg, #10b981, #34d399); }}
        .status-failed {{ background: linear-gradient(45deg, #ef4444, #f87171); }}
        .score-excellent {{ color: #10b981; }}
        .score-good {{ color: #f59e0b; }}
        .score-poor {{ color: #ef4444; }}
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-blue-400 mb-2">EA Stress Test Report</h1>
            <p class="text-gray-400">Comprehensive validation under extreme market conditions</p>
            <div class="text-sm text-gray-500 mt-2">Generated: {timestamp}</div>
        </header>

        <!-- Executive Summary -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-4 text-blue-300">Executive Summary</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div class="bg-gray-700 rounded p-4">
                        <div class="text-sm text-gray-400 mb-1">Total Tests</div>
                        <div class="text-2xl font-bold text-yellow-400">{total_tests}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-4">
                        <div class="text-sm text-gray-400 mb-1">Passed Tests</div>
                        <div class="text-2xl font-bold text-green-400">{passed_tests}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-4">
                        <div class="text-sm text-gray-400 mb-1">Success Rate</div>
                        <div class="text-2xl font-bold text-blue-400">{success_rate:.1f}%</div>
                    </div>
                    <div class="bg-gray-700 rounded p-4">
                        <div class="text-sm text-gray-400 mb-1">Average Score</div>
                        <div class="text-2xl font-bold text-purple-400">{avg_score:.1f}/100</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Score Distribution Chart -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Score Distribution</h2>
                <div style="height: 300px;">
                    <canvas id="scoreChart"></canvas>
                </div>
            </div>
        </section>

        <!-- Category Performance -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Performance by Category</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {category_cards}
                </div>
            </div>
        </section>

        <!-- Detailed Results -->
        <section>
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Detailed Test Results</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-gray-700 rounded-lg">
                        <thead class="bg-gray-600">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Test Name</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Category</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Score</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Status</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Trades</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Profit</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Max DD</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Sharpe</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-600">
                            {result_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Recommendations -->
        <section class="mt-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4 text-blue-300">Recommendations</h2>
                <div class="space-y-4">
                    {recommendations}
                </div>
            </div>
        </section>
    </div>

    <script>
        // Score Distribution Chart
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: {score_labels},
                datasets: [{{
                    label: 'Test Scores',
                    data: {score_data},
                    backgroundColor: {score_colors},
                    borderColor: '#374151',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{ color: '#9ca3af' }},
                        grid: {{ color: '#374151' }}
                    }},
                    x: {{
                        ticks: {{ color: '#9ca3af' }},
                        grid: {{ color: '#374151' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        # Prepare data for template
        score_labels = json.dumps([r.test_name for r in results])
        score_data = json.dumps([r.score for r in results])
        score_colors = json.dumps([
            '#10b981' if r.score >= 80 else '#f59e0b' if r.score >= 60 else '#ef4444'
            for r in results
        ])

        # Generate category cards
        category_cards = ""
        for category, cat_results in category_results.items():
            avg_score = statistics.mean([r.score for r in cat_results])
            passed = len([r for r in cat_results if r.passed])
            total = len(cat_results)

            status_color = "text-green-400" if avg_score >= 70 else "text-yellow-400" if avg_score >= 50 else "text-red-400"

            category_cards += f"""
                <div class="bg-gray-700 rounded p-4">
                    <div class="text-sm text-gray-400 mb-1">{category}</div>
                    <div class="text-xl font-bold {status_color}">{avg_score:.1f}/100</div>
                    <div class="text-xs text-gray-500 mt-1">{passed}/{total} tests passed</div>
                </div>
            """

        # Generate result rows
        result_rows = ""
        for result in results:
            status_class = "status-passed" if result.passed else "status-failed"
            score_class = "score-excellent" if result.score >= 80 else "score-good" if result.score >= 60 else "score-poor"
            status_text = "PASSED" if result.passed else "FAILED"

            result_rows += f"""
                <tr class="hover:bg-gray-600">
                    <td class="px-6 py-4 text-sm">{result.test_name}</td>
                    <td class="px-6 py-4 text-sm">{result.test_category}</td>
                    <td class="px-6 py-4 text-sm font-bold {score_class}">{result.score:.1f}</td>
                    <td class="px-6 py-4 text-sm">
                        <span class="{status_class} text-white px-2 py-1 rounded text-xs">
                            {status_text}
                        </span>
                    </td>
                    <td class="px-6 py-4 text-sm">{result.total_trades}</td>
                    <td class="px-6 py-4 text-sm ${'text-green-400' if result.total_profit >= 0 else 'text-red-400'}">
                        ${result.total_profit:.2f}
                    </td>
                    <td class="px-6 py-4 text-sm text-red-400">{result.max_drawdown:.1f}%</td>
                    <td class="px-6 py-4 text-sm">{result.sharpe_ratio:.2f}</td>
                </tr>
            """

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=total_tests,
            passed_tests=passed_tests,
            success_rate=(passed_tests/total_tests)*100,
            avg_score=avg_score,
            category_cards=category_cards,
            result_rows=result_rows,
            score_labels=score_labels,
            score_data=score_data,
            score_colors=score_colors,
            recommendations=recommendations
        )

    def _generate_recommendations(self, results: List[StressTestResult]) -> str:
        """Generate recommendations based on stress test results"""
        recommendations = []

        # Analyze weak areas
        failed_tests = [r for r in results if not r.passed]
        avg_drawdown = statistics.mean([r.max_drawdown for r in results])
        avg_consecutive_losses = statistics.mean([r.max_consecutive_losses for r in results])

        if len(failed_tests) > len(results) * 0.3:
            recommendations.append("""
                <div class="bg-red-900 bg-opacity-20 border border-red-500 rounded p-4">
                    <h4 class="font-bold text-red-400 mb-2">🚨 High Failure Rate Detected</h4>
                    <p class="text-sm">The strategy failed in more than 30% of stress tests. Consider:</p>
                    <ul class="text-sm mt-2 list-disc list-inside text-gray-300">
                        <li>Reviewing risk management parameters</li>
                        <li>Implementing more conservative position sizing</li>
                        <li>Adding market condition filters</li>
                        <li>Improving stop-loss mechanisms</li>
                    </ul>
                </div>
            """)

        if avg_drawdown > 20:
            recommendations.append("""
                <div class="bg-yellow-900 bg-opacity-20 border border-yellow-500 rounded p-4">
                    <h4 class="font-bold text-yellow-400 mb-2">⚠️ Excessive Drawdown Risk</h4>
                    <p class="text-sm">Average maximum drawdown exceeds 20%. Recommendations:</p>
                    <ul class="text-sm mt-2 list-disc list-inside text-gray-300">
                        <li>Implement dynamic position sizing based on volatility</li>
                        <li>Add maximum drawdown limits</li>
                        <li>Consider portfolio-level risk management</li>
                        <li>Test more conservative stop-loss levels</li>
                    </ul>
                </div>
            """)

        if avg_consecutive_losses > 5:
            recommendations.append("""
                <div class="bg-orange-900 bg-opacity-20 border border-orange-500 rounded p-4">
                    <h4 class="font-bold text-orange-400 mb-2">📉 High Consecutive Losses</h4>
                    <p class="text-sm">Strategy shows vulnerability to losing streaks. Consider:</p>
                    <ul class="text-sm mt-2 list-disc list-inside text-gray-300">
                        <li>Implementing cooldown periods after losses</li>
                        <li>Adding trend filters to avoid counter-trend trading</li>
                        <li>Dynamic risk adjustment based on recent performance</li>
                        <li>Multi-strategy diversification</li>
                    </ul>
                </div>
            """)

        # Performance-based recommendations
        high_volatility_failures = [r for r in failed_tests if 'volatility' in r.test_name.lower() or 'crash' in r.test_name.lower()]
        if len(high_volatility_failures) > 2:
            recommendations.append("""
                <div class="bg-purple-900 bg-opacity-20 border border-purple-500 rounded p-4">
                    <h4 class="font-bold text-purple-400 mb-2">📊 Volatility Sensitivity</h4>
                    <p class="text-sm">Strategy struggles in high-volatility environments. Suggestions:</p>
                    <ul class="text-sm mt-2 list-disc list-inside text-gray-300">
                        <li>Add volatility filters to strategy execution</li>
                        <li>Implement adaptive position sizing</li>
                        <li>Consider volatility-based stop-loss adjustments</li>
                        <li>Test different approach for ranging vs trending markets</li>
                    </ul>
                </div>
            """)

        if len(recommendations) == 0:
            recommendations.append("""
                <div class="bg-green-900 bg-opacity-20 border border-green-500 rounded p-4">
                    <h4 class="font-bold text-green-400 mb-2">✅ Excellent Performance</h4>
                    <p class="text-sm">The strategy demonstrates robust performance across stress test scenarios. Maintain current risk management practices and consider gradual position size increases after further validation.</p>
                </div>
            """)

        return "".join(recommendations)

# Main execution and testing
if __name__ == "__main__":
    print("🧪 EA Stress Testing Framework")
    print("=" * 50)
    print("Comprehensive validation under extreme market conditions\n")

    # Initialize stress test engine
    stress_engine = StressTestEngine()

    # Run comprehensive stress testing
    results = stress_engine.run_comprehensive_stress_test()

    # Generate report
    print("\n📄 Generating stress test report...")
    report_html = stress_engine.generate_stress_test_report(results)

    with open("stress_test_report.html", "w", encoding='utf-8') as f:
        f.write(report_html)

    print("✅ Report generated: stress_test_report.html")

    # Save results as JSON
    results_json = [asdict(r) for r in results]
    with open("stress_test_results.json", "w", encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, default=str)

    print("✅ Results saved: stress_test_results.json")

    # Summary
    print("\n🎯 Stress Testing Summary:")
    print("-" * 30)
    print(f"Total Scenarios: {len(results)}")
    print(f"Passed: {len([r for r in results if r.passed])}")
    print(f"Failed: {len([r for r in results if not r.passed])}")
    print(f"Success Rate: {(len([r for r in results if r.passed])/len(results)*100):.1f}%")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"Average Score: {avg_score:.1f}/100")

    # Worst case scenarios
    worst_results = sorted(results, key=lambda x: x.score)[:3]
    print(f"\n⚠️ Worst Performing Scenarios:")
    for i, result in enumerate(worst_results, 1):
        print(f"   {i}. {result.test_name}: {result.score:.1f}/100")

    # Best case scenarios
    best_results = sorted(results, key=lambda x: x.score, reverse=True)[:3]
    print(f"\n🏆 Best Performing Scenarios:")
    for i, result in enumerate(best_results, 1):
        print(f"   {i}. {result.test_name}: {result.score:.1f}/100")

    print("\n🌟 Stress testing framework ready for EA validation!")