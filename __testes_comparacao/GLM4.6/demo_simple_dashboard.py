#!/usr/bin/env python3
"""
Simple Demo Script for Advanced Visualization System
Shows the static dashboard capabilities without external dependencies
"""

import time
import json
from src.advanced_visualization import VisualizationSystem

def demo_static_dashboard():
    """Demonstrate static dashboard generation"""
    print("🎨 EA Scalper Visualization System Demo")
    print("=" * 60)
    print("This demo showcases the advanced visualization capabilities")
    print("of the EA Scalper XAUUSD optimization system.\n")

    # Initialize visualization system
    viz_system = VisualizationSystem()

    # Start monitoring with simulated data
    print("▶️ Starting real-time monitoring...")
    viz_system.start_monitoring(["XAUUSD", "EURUSD", "GBPUSD"])

    # Let it run for a few seconds to generate data
    print("⏳ Generating simulated market data...")
    for i in range(5):
        time.sleep(1)
        print(f"   Collecting data... {i+1}/5")

    # Generate dashboard
    print("\n📊 Generating trading dashboard...")
    dashboard_path = viz_system.generate_dashboard("XAUUSD", "demo_dashboard.html")
    print(f"✅ Dashboard generated: {dashboard_path}")

    # Export data
    print("📤 Exporting dashboard data...")
    dashboard_data = viz_system.dashboard_renderer.export_dashboard_data()

    # Save exported data with better formatting
    with open("demo_dashboard_data.json", "w", encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, default=str, ensure_ascii=False)
    print("✅ Dashboard data exported: demo_dashboard_data.json")

    # Generate sample optimization report
    print("\n📈 Generating optimization report...")
    sample_optimization = [
        {
            "strategy_name": "Balanced Strategy",
            "total_return": 15.75,
            "sharpe_ratio": 1.85,
            "max_drawdown": 8.2,
            "win_rate": 68.5,
            "profit_factor": 2.1,
            "total_trades": 245,
            "avg_win": 28.50,
            "avg_loss": -15.20
        },
        {
            "strategy_name": "Aggressive Strategy",
            "total_return": 28.40,
            "sharpe_ratio": 1.45,
            "max_drawdown": 15.8,
            "win_rate": 62.3,
            "profit_factor": 1.8,
            "total_trades": 380,
            "avg_win": 45.80,
            "avg_loss": -22.30
        },
        {
            "strategy_name": "Conservative Strategy",
            "total_return": 9.25,
            "sharpe_ratio": 2.15,
            "max_drawdown": 4.1,
            "win_rate": 75.2,
            "profit_factor": 2.8,
            "total_trades": 156,
            "avg_win": 18.90,
            "avg_loss": -8.75
        },
        {
            "strategy_name": "Scalper Pro",
            "total_return": 22.60,
            "sharpe_ratio": 1.95,
            "max_drawdown": 6.8,
            "win_rate": 71.8,
            "profit_factor": 2.4,
            "total_trades": 520,
            "avg_win": 12.40,
            "avg_loss": -6.20
        },
        {
            "strategy_name": "Swing Trader",
            "total_return": 18.90,
            "sharpe_ratio": 1.75,
            "max_drawdown": 9.5,
            "win_rate": 66.4,
            "profit_factor": 2.0,
            "total_trades": 89,
            "avg_win": 125.60,
            "avg_loss": -58.30
        }
    ]

    sample_backtest = [
        {"duration_hours": 24, "total_trades": 48, "total_profit": 125.50, "success_rate": 68.8, "max_dd": 3.2},
        {"duration_hours": 168, "total_trades": 312, "total_profit": 892.30, "success_rate": 71.2, "max_dd": 5.8},
        {"duration_hours": 720, "total_trades": 1450, "total_profit": 4250.80, "success_rate": 69.5, "max_dd": 8.4},
        {"duration_hours": 2160, "total_trades": 4320, "total_profit": 12680.40, "success_rate": 70.1, "max_dd": 11.2}
    ]

    report_html = viz_system.create_interactive_report(sample_optimization, sample_backtest)
    with open("demo_optimization_report.html", "w", encoding='utf-8') as f:
        f.write(report_html)
    print("✅ Optimization report generated: demo_optimization_report.html")

    # Generate additional symbol dashboards
    print("\n📊 Generating multi-symbol dashboards...")
    for symbol in ["EURUSD", "GBPUSD"]:
        symbol_path = viz_system.generate_dashboard(symbol, f"demo_dashboard_{symbol.lower()}.html")
        print(f"✅ {symbol} dashboard: {symbol_path}")

    # Stop monitoring
    viz_system.stop_monitoring()
    print("\n⏹️ Monitoring stopped")

    return dashboard_path

def show_dashboard_features():
    """Show dashboard features and capabilities"""
    print("\n🎯 Dashboard Features Overview")
    print("=" * 50)

    features = [
        "📊 Real-time Price Charts",
        "   • Live price updates with technical indicators",
        "   • Moving averages (SMA 20, SMA 50)",
        "   • Support for multiple timeframes",
        "",
        "💹 Equity Curve Monitoring",
        "   • Real-time equity and balance tracking",
        "   • Drawdown visualization",
        "   • Growth performance metrics",
        "",
        "🎯 Performance Analytics",
        "   • Win rate and profit factor",
        "   • Sharpe ratio calculation",
        "   • Risk-adjusted returns",
        "",
        "📈 Advanced Visualizations",
        "   • Performance heatmaps (hourly analysis)",
        "   • Trade distribution histograms",
        "   • Risk metrics radar charts",
        "",
        "🚨 Real-time Alerts",
        "   • Price threshold notifications",
        "   • Trade execution alerts",
        "   • Risk level warnings",
        "   • Performance target achievements",
        "",
        "🌐 Interactive Features",
        "   • WebSocket connectivity (with websockets)",
        "   • Live data streaming",
        "   • Responsive design",
        "   • Multi-device support"
    ]

    for feature in features:
        print(feature)

def show_data_structure():
    """Show the data structure used in the visualization"""
    print("\n📋 Data Structure Overview")
    print("=" * 50)

    print("🔍 RealTimeDataCollector:")
    print("   • price_history: Market tick data (bid/ask/spread)")
    print("   • trades_history: Completed trade information")
    print("   • equity_history: Account equity over time")
    print("   • metrics_history: Performance metrics tracking")
    print("   • real_time_updates: Live update queue")

    print("\n📊 ChartGenerator:")
    print("   • Price charts with technical indicators")
    print("   • Equity curve visualization")
    print("   • Performance heatmaps")
    print("   • Trade distribution analysis")
    print("   • Risk metrics radar charts")

    print("\n🎨 DashboardRenderer:")
    print("   • HTML template rendering")
    print("   • Real-time data integration")
    print("   • Interactive chart generation")
    print("   • Export functionality")

    print("\n📈 Available Metrics:")
    metrics = [
        "Win Rate: Percentage of profitable trades",
        "Profit Factor: Ratio of profits to losses",
        "Sharpe Ratio: Risk-adjusted performance",
        "Maximum Drawdown: Largest peak-to-trough decline",
        "Average Win/Loss: Mean profit per trade type",
        "Volatility: Price movement measurement",
        "Success Rate: Overall trading success percentage"
    ]

    for metric in metrics:
        print(f"   • {metric}")

def main():
    """Main demo function"""
    # Static dashboard demo
    dashboard_path = demo_static_dashboard()

    # Show features
    show_dashboard_features()

    # Show data structure
    show_data_structure()

    # Final summary
    print("\n📋 Demo Summary")
    print("=" * 30)

    generated_files = [
        "✅ Main Dashboard: demo_dashboard.html",
        "✅ Optimization Report: demo_optimization_report.html",
        "✅ Dashboard Data: demo_dashboard_data.json",
        "✅ EURUSD Dashboard: demo_dashboard_eurusd.html",
        "✅ GBPUSD Dashboard: demo_dashboard_gbpusd.html"
    ]

    for file_info in generated_files:
        print(file_info)

    print("\n🔧 To view the dashboards:")
    print("   Open any .html file in your web browser")

    print("\n🎯 Key Features Demonstrated:")
    achievements = [
        "✅ Real-time data collection and processing",
        "✅ Advanced chart generation with technical indicators",
        "✅ Interactive performance metrics calculation",
        "✅ Professional HTML dashboard rendering",
        "✅ Multi-symbol support",
        "✅ Optimization report generation",
        "✅ Data export capabilities"
    ]

    for achievement in achievements:
        print(achievement)

    print("\n🌟 Advanced visualization system demo completed!")
    print("\n📈 Next Steps:")
    print("1. Open the generated HTML files to explore the dashboards")
    print("2. Review the exported JSON data structure")
    print("3. Check the optimization report for strategy comparison")
    print("4. Customize the templates for your specific needs")

if __name__ == "__main__":
    main()