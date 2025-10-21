#!/usr/bin/env python3
"""
Demo Script for Advanced Visualization System
Shows how to use the visualization components
"""

import time
import json
import asyncio
from src.advanced_visualization import VisualizationSystem
from src.interactive_dashboard import RealTimeDashboard

def demo_static_dashboard():
    """Demonstrate static dashboard generation"""
    print("🎨 Static Dashboard Demo")
    print("=" * 50)

    # Initialize visualization system
    viz_system = VisualizationSystem()

    # Start monitoring with simulated data
    print("▶️ Starting real-time monitoring...")
    viz_system.start_monitoring(["XAUUSD", "EURUSD"])

    # Let it run for a few seconds to generate data
    print("⏳ Generating simulated data...")
    time.sleep(3)

    # Generate dashboard
    print("📊 Generating trading dashboard...")
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
    print("📈 Generating optimization report...")
    sample_optimization = [
        {
            "strategy_name": "Balanced Strategy",
            "total_return": 15.75,
            "sharpe_ratio": 1.85,
            "max_drawdown": 8.2,
            "win_rate": 68.5,
            "profit_factor": 2.1
        },
        {
            "strategy_name": "Aggressive Strategy",
            "total_return": 28.40,
            "sharpe_ratio": 1.45,
            "max_drawdown": 15.8,
            "win_rate": 62.3,
            "profit_factor": 1.8
        },
        {
            "strategy_name": "Conservative Strategy",
            "total_return": 9.25,
            "sharpe_ratio": 2.15,
            "max_drawdown": 4.1,
            "win_rate": 75.2,
            "profit_factor": 2.8
        }
    ]

    sample_backtest = [
        {"duration_hours": 24, "total_trades": 48, "total_profit": 125.50, "success_rate": 68.8},
        {"duration_hours": 168, "total_trades": 312, "total_profit": 892.30, "success_rate": 71.2},
        {"duration_hours": 720, "total_trades": 1450, "total_profit": 4250.80, "success_rate": 69.5}
    ]

    report_html = viz_system.create_interactive_report(sample_optimization, sample_backtest)
    with open("demo_optimization_report.html", "w", encoding='utf-8') as f:
        f.write(report_html)
    print("✅ Optimization report generated: demo_optimization_report.html")

    # Stop monitoring
    viz_system.stop_monitoring()
    print("⏹️ Monitoring stopped")

    return dashboard_path

async def demo_interactive_dashboard():
    """Demonstrate interactive dashboard"""
    print("\n🌐 Interactive Dashboard Demo")
    print("=" * 50)

    # Create interactive dashboard
    dashboard = RealTimeDashboard(port=8766)  # Different port to avoid conflicts

    # Create web client
    print("📱 Creating web client interface...")
    client_path = dashboard.create_web_client("demo_interactive_dashboard.html")
    print(f"✅ Web client created: {client_path}")

    print("\n📊 Interactive Dashboard Features:")
    print("• Real-time WebSocket connection")
    print("• Live price charts and market data")
    print("• Interactive performance metrics")
    print("• Real-time alerts and notifications")
    print("• Automatic chart updates")
    print("• Activity logging and monitoring")

    print(f"\n🌐 To test the interactive dashboard:")
    print(f"1. Open in browser: file://{client_path}")
    print(f"2. Start server: python3 -c 'import asyncio; from demo_dashboard import run_interactive_server; asyncio.run(run_interactive_server())'")
    print(f"3. Click 'Connect to Server' in the interface")

    # Start server for a few seconds to demonstrate
    print("\n🚀 Starting server for 10 seconds demonstration...")

    # Start simulation
    import threading
    simulation_thread = threading.Thread(target=dashboard.simulate_market_activity, daemon=True)
    simulation_thread.start()

    # Run server briefly
    try:
        server_task = asyncio.create_task(dashboard.start_server())
        await asyncio.wait_for(server_task, timeout=10.0)
    except asyncio.TimeoutError:
        print("\n⏰ Demo period ended")
        dashboard.is_running = False
        if dashboard.connected_clients:
            print(f"🔌 Disconnected {len(dashboard.connected_clients)} clients")

async def run_interactive_server():
    """Run interactive dashboard server"""
    dashboard = RealTimeDashboard(port=8766)
    await dashboard.start_server()

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
        "   • WebSocket connectivity",
        "   • Live data streaming",
        "   • Responsive design",
        "   • Multi-device support"
    ]

    for feature in features:
        print(feature)

def main():
    """Main demo function"""
    print("🎨 EA Scalper Visualization System Demo")
    print("=" * 60)
    print("This demo showcases the advanced visualization capabilities")
    print("of the EA Scalper XAUUSD optimization system.\n")

    # Show features
    show_dashboard_features()

    # Static dashboard demo
    dashboard_path = demo_static_dashboard()

    # Interactive dashboard demo
    try:
        asyncio.run(demo_interactive_dashboard())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")

    # Final summary
    print("\n📋 Demo Summary")
    print("=" * 30)
    print(f"✅ Static Dashboard: {dashboard_path}")
    print("✅ Interactive Client: demo_interactive_dashboard.html")
    print("✅ Dashboard Data: demo_dashboard_data.json")
    print("✅ Optimization Report: demo_optimization_report.html")

    print("\n🔧 To run full interactive dashboard:")
    print("   python3 src/interactive_dashboard.py")

    print("\n🎯 Next Steps:")
    print("1. Open the generated HTML files in your browser")
    print("2. Start the interactive server for real-time updates")
    print("3. Customize charts and metrics for your needs")
    print("4. Integrate with your EA trading system")

    print("\n🌟 Advanced visualization system demo completed!")

if __name__ == "__main__":
    main()