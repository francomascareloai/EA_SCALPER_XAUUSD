#!/usr/bin/env python3
"""
Comprehensive Validation System Demo
Shows stress testing and robustness validation capabilities
"""

import json
import time
from datetime import datetime, timedelta
from src.stress_testing_framework import StressTestEngine, MarketScenario
from src.validation_engine import RobustnessValidator

def create_sample_strategy_data():
    """Create realistic sample strategy data for validation"""
    print("📊 Creating sample strategy data...")

    # Generate realistic trade history
    trades = []
    base_equity = 10000
    current_equity = base_equity

    # Simulate 6 months of trading
    for day in range(180):
        # Daily trading activity (0-5 trades per day)
        num_trades = random.randint(0, 5)

        for trade in range(num_trades):
            # Simulate realistic trade outcomes
            trade_type = random.choice(['LONG', 'SHORT'])
            base_profit = random.gauss(15, 25)  # Average $15 profit with variance

            # Add some strategy-specific patterns
            if day % 7 < 2:  # Weekends - higher volatility
                base_profit *= random.uniform(0.8, 1.5)

            # Ensure some losing trades for realism
            if random.random() < 0.35:  # 35% loss rate
                base_profit = -abs(base_profit * random.uniform(0.5, 1.5))

            # Trade duration correlates with profit magnitude
            duration = random.uniform(10, 120) * (1 + abs(base_profit) / 50)

            trade_data = {
                'timestamp': datetime.now() - timedelta(days=180-day, hours=random.randint(0, 23)),
                'type': trade_type,
                'profit': base_profit,
                'duration_minutes': duration,
                'entry_price': 2650 + random.uniform(-50, 50),
                'exit_price': 2650 + random.uniform(-50, 50),
                'volume': 0.01
            }

            trades.append(trade_data)
            current_equity += base_profit

    # Generate historical market data
    historical_data = []
    base_price = 2650

    for day in range(500):  # 2 years of data
        # Simulate market movement with trends and volatility
        daily_change = random.gauss(0, 15)  # Average daily change
        trend_component = math.sin(day / 50) * 5  # Cyclical trend
        base_price += daily_change + trend_component

        # Add some volatility clustering
        if day % 30 < 5:  # High volatility periods
            base_price += random.gauss(0, 10)

        historical_data.append({
            'date': datetime.now() - timedelta(days=500-day),
            'open': base_price,
            'high': base_price + random.uniform(0, 20),
            'low': base_price - random.uniform(0, 20),
            'close': base_price,
            'volume': random.randint(5000, 50000),
            'spread': base_price * random.uniform(0.0001, 0.0005)
        })

    # Define parameter ranges for optimization
    parameter_ranges = {
        'sma_short': (3, 15),
        'sma_long': (15, 50),
        'rsi_period': (10, 20),
        'rsi_overbought': (70, 80),
        'rsi_oversold': (20, 30),
        'stop_loss_pct': (1.0, 3.0),
        'take_profit_pct': (2.0, 6.0),
        'volume_size': (0.01, 0.1)
    }

    strategy_data = {
        'trades': trades,
        'historical_data': historical_data,
        'parameter_ranges': parameter_ranges,
        'strategy_name': 'XAUUSD Scalper Pro',
        'timeframe': 'M5',
        'initial_balance': base_equity,
        'final_balance': current_equity,
        'total_trades': len(trades)
    }

    print(f"   ✓ Generated {len(trades)} trades over 180 days")
    print(f"   ✓ Generated {len(historical_data)} days of market data")
    print(f"   ✓ Initial: ${base_equity:.2f} → Final: ${current_equity:.2f}")
    print(f"   ✓ Total return: ${current_equity - base_equity:.2f} ({((current_equity - base_equity) / base_equity * 100):.1f}%)")

    return strategy_data

def demo_stress_testing():
    """Demonstrate stress testing capabilities"""
    print("\n🧪 Stress Testing Demo")
    print("=" * 50)

    # Initialize stress test engine
    stress_engine = StressTestEngine()

    # Show available scenarios
    scenarios = stress_engine.scenario_generator.get_all_scenarios()
    print(f"\n📋 Available Stress Scenarios ({len(scenarios)} total):")

    categories = {}
    for scenario in scenarios:
        category = stress_engine._categorize_scenario(scenario.name)
        if category not in categories:
            categories[category] = []
        categories[category].append(scenario.name)

    for category, scenario_names in categories.items():
        print(f"\n   {category}:")
        for name in scenario_names:
            print(f"     • {name}")

    # Run selective stress tests for demo
    print(f"\n🎯 Running Key Stress Scenarios...")
    key_scenarios = [
        "Extreme Volatility",
        "Flash Crash",
        "Low Liquidity",
        "Strong Trend",
        "News Release"
    ]

    selected_results = []
    for scenario_name in key_scenarios:
        scenario = stress_engine.scenario_generator.get_scenario(scenario_name)
        if scenario:
            result = stress_engine.run_stress_test(scenario)
            selected_results.append(result)

    # Generate quick summary
    passed = len([r for r in selected_results if r.passed])
    avg_score = sum(r.score for r in selected_results) / len(selected_results)

    print(f"\n📊 Stress Test Summary:")
    print(f"   Scenarios Tested: {len(selected_results)}")
    print(f"   Passed: {passed}/{len(selected_results)}")
    print(f"   Average Score: {avg_score:.1f}/100")

    return selected_results

def demo_monte_carlo(strategy_data):
    """Demonstrate Monte Carlo simulation"""
    print("\n🎲 Monte Carlo Simulation Demo")
    print("=" * 50)

    from src.validation_engine import MonteCarloSimulator

    # Create Monte Carlo simulator
    mc_sim = MonteCarloSimulator(num_simulations=500)  # Reduced for demo

    print(f"📈 Running {mc_sim.num_simulations} Monte Carlo simulations...")

    # Run simulation
    mc_results = mc_sim.run_monte_carlo(strategy_data['trades'])

    # Display results
    print(f"\n📊 Monte Carlo Results:")
    print(f"   Success Rate: {mc_results.get('success_rate', 0):.1f}%")
    print(f"   Value at Risk (95%): ${mc_results.get('risk_assessment', {}).get('var_95', 0):.2f}")
    print(f"   Max Drawdown (95%): {mc_results.get('risk_assessment', {}).get('max_drawdown_95', 0):.1f}%")

    probabilities = mc_results.get('probability_thresholds', {})
    print(f"\n📈 Probability Analysis:")
    print(f"   Probability of Positive Return: {probabilities.get('profit_positive', 0):.1f}%")
    print(f"   Probability of >$500 Profit: {probabilities.get('profit_500', 0):.1f}%")
    print(f"   Probability of <20% Drawdown: {probabilities.get('drawdown_under_20', 0):.1f}%")
    print(f"   Probability of Sharpe >1.0: {probabilities.get('sharpe_above_1', 0):.1f}%")

    return mc_results

def demo_walk_forward_analysis(strategy_data):
    """Demonstrate walk-forward analysis"""
    print("\n🚶 Walk-Forward Analysis Demo")
    print("=" * 50)

    from src.validation_engine import WalkForwardOptimizer

    # Create walk-forward optimizer
    wf_optimizer = WalkForwardOptimizer(window_size=100, step_size=25)  # Smaller for demo

    print(f"📈 Running walk-forward optimization...")
    print(f"   Window size: {wf_optimizer.window_size} periods")
    print(f"   Step size: {wf_optimizer.step_size} periods")

    # Run analysis
    wf_results = wf_optimizer.run_walk_forward_analysis(
        strategy_data['historical_data'],
        strategy_data['parameter_ranges']
    )

    # Display results
    print(f"\n📊 Walk-Forward Results:")
    print(f"   Periods Analyzed: {wf_results.get('periods_analyzed', 0)}")
    print(f"   Average Return: {wf_results.get('average_return', 0):.2f}")
    print(f"   Return Stability: {wf_results.get('return_stability', 0):.1f}%")
    print(f"   Consistency Rate: {wf_results.get('consistency_rate', 0):.1f}%")
    print(f"   Parameter Stability: {wf_results.get('parameter_stability', 0):.1f}%")

    return wf_results

def generate_comprehensive_report(stress_results, mc_results, wf_results, validation_result):
    """Generate comprehensive validation report"""
    print("\n📄 Generating Comprehensive Validation Report...")

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive EA Validation Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-passed {{ background: linear-gradient(45deg, #10b981, #34d399); }}
        .status-failed {{ background: linear-gradient(45deg, #ef4444, #f87171); }}
        .status-warning {{ background: linear-gradient(45deg, #f59e0b, #fbbf24); }}
        .metric-excellent {{ color: #10b981; }}
        .metric-good {{ color: #f59e0b; }}
        .metric-poor {{ color: #ef4444; }}
        .section-card {{ transition: transform 0.2s ease-in-out; }}
        .section-card:hover {{ transform: translateY(-2px); }}
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8 text-center">
            <h1 class="text-5xl font-bold text-blue-400 mb-2">EA Validation Report</h1>
            <p class="text-gray-400 text-lg">Comprehensive Testing & Analysis Results</p>
            <div class="text-sm text-gray-500 mt-4">Generated: {timestamp}</div>
        </header>

        <!-- Executive Summary -->
        <section class="mb-8">
            <div class="bg-gradient-to-r from-blue-800 to-purple-800 rounded-lg p-6 section-card">
                <h2 class="text-3xl font-bold mb-6 text-white">Executive Summary</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div class="bg-gray-800 bg-opacity-50 rounded-lg p-4 text-center">
                        <div class="text-sm text-gray-300 mb-2">Overall Score</div>
                        <div class="text-3xl font-bold {overall_score_color}">{overall_score:.1f}/100</div>
                        <div class="text-xs text-gray-400 mt-1">{overall_status}</div>
                    </div>
                    <div class="bg-gray-800 bg-opacity-50 rounded-lg p-4 text-center">
                        <div class="text-sm text-gray-300 mb-2">Stress Tests</div>
                        <div class="text-3xl font-bold text-green-400">{stress_passed}/{stress_total}</div>
                        <div class="text-xs text-gray-400 mt-1">{stress_success_rate:.1f}% passed</div>
                    </div>
                    <div class="bg-gray-800 bg-opacity-50 rounded-lg p-4 text-center">
                        <div class="text-sm text-gray-300 mb-2">Monte Carlo</div>
                        <div class="text-3xl font-bold text-yellow-400">{mc_success_rate:.1f}%</div>
                        <div class="text-xs text-gray-400 mt-1">success rate</div>
                    </div>
                    <div class="bg-gray-800 bg-opacity-50 rounded-lg p-4 text-center">
                        <div class="text-sm text-gray-300 mb-2">Risk Level</div>
                        <div class="text-3xl font-bold {risk_color}">{risk_level}</div>
                        <div class="text-xs text-gray-400 mt-1">overall risk</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Validation Components -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <!-- Stress Testing Results -->
            <div class="bg-gray-800 rounded-lg p-6 section-card">
                <h3 class="text-xl font-bold mb-4 text-blue-300">🧪 Stress Testing</h3>
                <div class="space-y-3">
                    {stress_test_summary}
                </div>
                <div class="mt-4 pt-4 border-t border-gray-700">
                    <div class="text-sm text-gray-400">Worst Case:</div>
                    <div class="text-sm font-bold text-red-400">{worst_scenario}</div>
                </div>
            </div>

            <!-- Monte Carlo Results -->
            <div class="bg-gray-800 rounded-lg p-6 section-card">
                <h3 class="text-xl font-bold mb-4 text-blue-300">🎲 Monte Carlo</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Success Rate:</span>
                        <span class="font-bold text-green-400">{mc_success_rate:.1f}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">VaR (95%):</span>
                        <span class="font-bold text-yellow-400">${mc_var:.0f}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Max DD (95%):</span>
                        <span class="font-bold text-red-400">{mc_max_dd:.1f}%</span>
                    </div>
                </div>
                <div class="mt-4 pt-4 border-t border-gray-700">
                    <div class="text-sm text-gray-400">Reliability:</div>
                    <div class="text-sm font-bold {mc_reliability_color}">{mc_reliability}</div>
                </div>
            </div>

            <!-- Walk-Forward Results -->
            <div class="bg-gray-800 rounded-lg p-6 section-card">
                <h3 class="text-xl font-bold mb-4 text-blue-300">🚶 Walk-Forward</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Consistency:</span>
                        <span class="font-bold text-green-400">{wf_consistency:.1f}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Param Stability:</span>
                        <span class="font-bold text-yellow-400">{wf_stability:.1f}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Periods:</span>
                        <span class="font-bold text-blue-400">{wf_periods}</span>
                    </div>
                </div>
                <div class="mt-4 pt-4 border-t border-gray-700">
                    <div class="text-sm text-gray-400">Robustness:</div>
                    <div class="text-sm font-bold {wf_robustness_color}">{wf_robustness}</div>
                </div>
            </div>
        </div>

        <!-- Detailed Metrics -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-4 text-blue-300">📊 Detailed Validation Metrics</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {validation_metrics}
                </div>
            </div>
        </section>

        <!-- Risk Assessment -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-4 text-blue-300">⚠️ Risk Assessment</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="text-lg font-semibold mb-3 text-red-400">Risk Factors Identified:</h3>
                        <ul class="space-y-2">
                            {risk_factors}
                        </ul>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold mb-3 text-yellow-400">Risk Metrics:</h3>
                        <div class="space-y-2">
                            {risk_metrics}
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Recommendations -->
        <section class="mb-8">
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-4 text-blue-300">💡 Recommendations</h2>
                <div class="space-y-4">
                    {recommendations}
                </div>
            </div>
        </section>

        <!-- Charts Section -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Performance Distribution</h3>
                <canvas id="performanceChart" height="200"></canvas>
            </div>
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-bold mb-4 text-blue-300">Risk vs Return</h3>
                <canvas id="riskChart" height="200"></canvas>
            </div>
        </section>
    </div>

    <script>
        // Performance Distribution Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {{
            type: 'bar',
            data: {{
                labels: ['Stress Tests', 'Monte Carlo', 'Walk-Forward', 'Overall'],
                datasets: [{{
                    label: 'Score',
                    data: [{stress_avg_score}, {mc_success_rate}, {wf_consistency}, {overall_score}],
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.6)',
                        'rgba(16, 185, 129, 0.6)',
                        'rgba(245, 158, 11, 0.6)',
                        'rgba(139, 92, 246, 0.6)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(139, 92, 246, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
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

        // Risk vs Return Chart
        const riskCtx = document.getElementById('riskChart').getContext('2d');
        new Chart(riskCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Test Scenarios',
                    data: {risk_return_data},
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    pointRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Risk Score',
                            color: '#9ca3af'
                        }},
                        ticks: {{ color: '#9ca3af' }},
                        grid: {{ color: '#374151' }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Return Score',
                            color: '#9ca3af'
                        }},
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

    # Prepare template data
    overall_score = validation_result.overall_score
    overall_status = "PASSED" if validation_result.passed else "FAILED"
    overall_score_color = "text-green-400" if overall_score >= 80 else "text-yellow-400" if overall_score >= 60 else "text-red-400"

    # Stress test data
    stress_passed = len([r for r in stress_results if r.passed])
    stress_total = len(stress_results)
    stress_success_rate = (stress_passed / stress_total) * 100 if stress_total > 0 else 0
    stress_avg_score = sum(r.score for r in stress_results) / stress_total if stress_total > 0 else 0
    worst_scenario = min(stress_results, key=lambda x: x.score).test_name if stress_results else "N/A"

    # Monte Carlo data
    mc_success_rate = mc_results.get('success_rate', 0)
    mc_var = mc_results.get('risk_assessment', {}).get('var_95', 0)
    mc_max_dd = mc_results.get('risk_assessment', {}).get('max_drawdown_95', 0)
    mc_reliability = "HIGH" if mc_success_rate >= 70 else "MEDIUM" if mc_success_rate >= 50 else "LOW"
    mc_reliability_color = "text-green-400" if mc_success_rate >= 70 else "text-yellow-400" if mc_success_rate >= 50 else "text-red-400"

    # Walk-forward data
    wf_consistency = wf_results.get('consistency_rate', 0)
    wf_stability = wf_results.get('parameter_stability', 0) * 100
    wf_periods = wf_results.get('periods_analyzed', 0)
    wf_robustness = "HIGH" if wf_consistency >= 70 and wf_stability >= 70 else "MEDIUM" if wf_consistency >= 50 else "LOW"
    wf_robustness_color = "text-green-400" if wf_robustness == "HIGH" else "text-yellow-400" if wf_robustness == "MEDIUM" else "text-red-400"

    # Risk assessment
    risk_assessment = validation_result.risk_assessment
    risk_level = risk_assessment.get('overall_risk', 'MEDIUM')
    risk_color = "text-green-400" if risk_level == "LOW" else "text-yellow-400" if risk_level == "MEDIUM" else "text-red-400"

    # Generate HTML sections
    stress_test_summary = ""
    for result in stress_results[:3]:  # Top 3 results
        status_color = "text-green-400" if result.passed else "text-red-400"
        stress_test_summary += f"""
            <div class="flex justify-between items-center">
                <span class="text-sm text-gray-300">{result.test_name}</span>
                <span class="text-sm font-bold {status_color}">{result.score:.0f}</span>
            </div>
        """

    # Validation metrics
    validation_metrics = ""
    for metric in validation_result.metrics[:6]:  # Top 6 metrics
        status_color = "text-green-400" if metric.passed else "text-red-400"
        validation_metrics += f"""
            <div class="bg-gray-700 rounded p-3">
                <div class="text-sm text-gray-400 mb-1">{metric.name}</div>
                <div class="text-lg font-bold {status_color}">{metric.value:.1f}</div>
                <div class="text-xs text-gray-500">Benchmark: {metric.benchmark:.1f}</div>
            </div>
        """

    # Risk factors
    risk_factors = ""
    for factor in risk_assessment.get('risk_factors', [])[:5]:
        risk_factors += f"<li class='text-sm text-gray-300 flex items-start'><span class='text-red-400 mr-2'>⚠️</span>{factor}</li>"

    if not risk_factors:
        risk_factors = "<li class='text-sm text-green-300'>✅ No significant risk factors identified</li>"

    # Risk metrics
    risk_metrics = f"""
        <div class="flex justify-between">
            <span class="text-gray-400">Risk Score:</span>
            <span class="font-bold text-yellow-400">{risk_assessment.get('risk_score', 0):.0f}/100</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">VaR (95%):</span>
            <span class="font-bold text-red-400">${mc_var:.0f}</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Max DD Risk:</span>
            <span class="font-bold text-orange-400">{mc_max_dd:.1f}%</span>
        </div>
    """

    # Recommendations
    recommendations = ""
    for i, rec in enumerate(validation_result.recommendations[:5], 1):
        rec_type = "success" if "excellent" in rec.lower() or "passes" in rec.lower() else "warning" if "risk" in rec.lower() or "reduce" in rec.lower() else "info"
        icon = "✅" if rec_type == "success" else "⚠️" if rec_type == "warning" else "💡"
        color = "text-green-400" if rec_type == "success" else "text-yellow-400" if rec_type == "warning" else "text-blue-400"

        recommendations += f"""
            <div class="bg-gray-700 bg-opacity-50 rounded p-3 flex items-start">
                <span class="text-lg mr-3">{icon}</span>
                <span class="text-sm {color}">{rec}</span>
            </div>
        """

    # Risk vs Return data for chart
    risk_return_data = []
    for result in stress_results:
        risk_return_data.append({
            'x': 100 - result.score,  # Risk as inverse of score
            'y': result.total_profit if hasattr(result, 'total_profit') else result.score
        })

    return html_template.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        overall_score=overall_score,
        overall_status=overall_status,
        overall_score_color=overall_score_color,
        stress_passed=stress_passed,
        stress_total=stress_total,
        stress_success_rate=stress_success_rate,
        stress_avg_score=stress_avg_score,
        worst_scenario=worst_scenario,
        mc_success_rate=mc_success_rate,
        mc_var=mc_var,
        mc_max_dd=mc_max_dd,
        mc_reliability=mc_reliability,
        mc_reliability_color=mc_reliability_color,
        wf_consistency=wf_consistency,
        wf_stability=wf_stability,
        wf_periods=wf_periods,
        wf_robustness=wf_robustness,
        wf_robustness_color=wf_robustness_color,
        risk_level=risk_level,
        risk_color=risk_color,
        stress_test_summary=stress_test_summary,
        validation_metrics=validation_metrics,
        risk_factors=risk_factors,
        risk_metrics=risk_metrics,
        recommendations=recommendations,
        risk_return_data=json.dumps(risk_return_data)
    )

def main():
    """Main validation demo function"""
    print("🔍 Comprehensive EA Validation System Demo")
    print("=" * 60)
    print("Advanced stress testing, Monte Carlo simulation, and robustness analysis\n")

    # Create sample strategy data
    strategy_data = create_sample_strategy_data()

    # 1. Stress Testing
    stress_results = demo_stress_testing()

    # 2. Monte Carlo Simulation
    mc_results = demo_monte_carlo(strategy_data)

    # 3. Walk-Forward Analysis
    wf_results = demo_walk_forward_analysis(strategy_data)

    # 4. Comprehensive Validation
    print("\n🎯 Running Comprehensive Validation...")
    validator = RobustnessValidator()
    validation_result = validator.validate_strategy_robustness(strategy_data)

    # Save detailed results
    all_results = {
        'strategy_data': strategy_data,
        'stress_results': [r.__dict__ for r in stress_results],
        'monte_carlo_results': mc_results,
        'walk_forward_results': wf_results,
        'validation_result': validation_result.__dict__
    }

    with open("comprehensive_validation_results.json", "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate comprehensive report
    print("\n📄 Generating comprehensive validation report...")
    report_html = generate_comprehensive_report(stress_results, mc_results, wf_results, validation_result)

    with open("comprehensive_validation_report.html", "w", encoding='utf-8') as f:
        f.write(report_html)

    print("✅ Report generated: comprehensive_validation_report.html")
    print("✅ Results saved: comprehensive_validation_results.json")

    # Final Summary
    print("\n🎯 Comprehensive Validation Summary")
    print("=" * 50)

    # Overall assessment
    if validation_result.overall_score >= 80:
        assessment = "🏆 EXCELLENT"
        color = "🟢"
    elif validation_result.overall_score >= 60:
        assessment = "✅ GOOD"
        color = "🟡"
    else:
        assessment = "⚠️ NEEDS IMPROVEMENT"
        color = "🔴"

    print(f"Overall Assessment: {color} {assessment}")
    print(f"Overall Score: {validation_result.overall_score:.1f}/100")
    print(f"Status: {'PASSED' if validation_result.passed else 'FAILED'}")

    print(f"\n📊 Component Results:")
    print(f"   Stress Tests: {len([r for r in stress_results if r.passed])}/{len(stress_results)} passed")
    print(f"   Monte Carlo: {mc_results.get('success_rate', 0):.1f}% success rate")
    print(f"   Walk-Forward: {wf_results.get('consistency_rate', 0):.1f}% consistency")
    print(f"   Risk Level: {validation_result.risk_assessment.get('overall_risk', 'UNKNOWN')}")

    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(validation_result.recommendations[:3], 1):
        print(f"   {i}. {rec}")

    print(f"\n🌟 Validation system demonstrates comprehensive EA testing capabilities!")
    print(f"   📈 Stress Testing: {len(stress_results)} market scenarios")
    print(f"   🎲 Monte Carlo: {mc_results.get('success_rate', 0):.1f}% reliability")
    print(f"   🚶 Walk-Forward: {wf_results.get('periods_analyzed', 0)} periods analyzed")
    print(f"   🔍 Risk Assessment: {len(validation_result.risk_assessment.get('risk_factors', []))} factors evaluated")

if __name__ == "__main__":
    import random
    import math

    main()