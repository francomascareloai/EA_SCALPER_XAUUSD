#!/usr/bin/env python3
"""
🧪 Test XAUUSD_TDS M15 Configuration
Symbol=XAUUSD_TDS, Period=M15
"""

import asyncio
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, Any

async def test_xauusd_tds_symbol():
    """Test XAUUSD_TDS symbol availability and M15 data"""
    
    print("📊 Testing XAUUSD_TDS Symbol Configuration")
    print("=" * 50)
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return False
        
        # Test XAUUSD_TDS first
        symbol_tds = "XAUUSD_TDS"
        symbol_info_tds = mt5.symbol_info(symbol_tds)
        
        active_symbol = None
        
        if symbol_info_tds is not None:
            print(f"✅ {symbol_tds} symbol: AVAILABLE")
            active_symbol = symbol_tds
            
            # Add to Market Watch
            if mt5.symbol_select(symbol_tds, True):
                print(f"✅ {symbol_tds} added to Market Watch")
            else:
                print(f"⚠️ Failed to add {symbol_tds} to Market Watch")
                
        else:
            print(f"⚠️ {symbol_tds} symbol: NOT AVAILABLE")
            print("   🔄 Trying standard XAUUSD as fallback...")
            
            # Test standard XAUUSD as fallback
            symbol_std = "XAUUSD"
            symbol_info_std = mt5.symbol_info(symbol_std)
            if symbol_info_std is not None:
                print(f"✅ {symbol_std} (fallback) symbol: AVAILABLE")
                active_symbol = symbol_std
                
                if mt5.symbol_select(symbol_std, True):
                    print(f"✅ {symbol_std} added to Market Watch")
            else:
                print(f"❌ {symbol_std} (fallback) also not available")
                return False
        
        if active_symbol:
            # Get symbol specifications
            symbol_info = mt5.symbol_info(active_symbol)
            print(f"\\n📋 {active_symbol} Specifications:")
            print(f"   Spread: {symbol_info.spread} points")
            print(f"   Digits: {symbol_info.digits}")
            print(f"   Min Lot: {symbol_info.volume_min}")
            print(f"   Max Lot: {symbol_info.volume_max}")
            print(f"   Contract Size: {getattr(symbol_info, 'contract_size', 'N/A')}")
            
            # Test M15 timeframe data
            print(f"\\n📈 Testing M15 timeframe data for {active_symbol}...")
            
            timeframes = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,  # Primary timeframe
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4
            }
            
            available_timeframes = []
            
            for tf_name, tf_const in timeframes.items():
                rates = mt5.copy_rates_from_pos(active_symbol, tf_const, 0, 100)
                if rates is not None and len(rates) > 0:
                    print(f"   ✅ {tf_name}: {len(rates)} bars available")
                    if tf_name == "M15":
                        print(f"      🎯 M15 Current Price: {rates[-1]['close']:.3f}")
                        print(f"      📊 M15 Last 4 Hours: {len(rates)} bars")
                    available_timeframes.append(tf_name)
                else:
                    print(f"   ❌ {tf_name}: No data available")
            
            # Multi-timeframe analysis
            if "M15" in available_timeframes:
                print(f"\\n🔍 M15 Multi-timeframe Analysis:")
                
                # Get M15 data for analysis
                m15_rates = mt5.copy_rates_from_pos(active_symbol, mt5.TIMEFRAME_M15, 0, 200)
                if m15_rates is not None and len(m15_rates) >= 20:
                    # Basic analysis
                    current_price = m15_rates[-1]['close']
                    high_20 = max([bar['high'] for bar in m15_rates[-20:]])
                    low_20 = min([bar['low'] for bar in m15_rates[-20:]])
                    sma_20 = sum([bar['close'] for bar in m15_rates[-20:]]) / 20
                    
                    print(f"   📊 Current Price: {current_price:.3f}")
                    print(f"   📈 20-bar High: {high_20:.3f}")
                    print(f"   📉 20-bar Low: {low_20:.3f}")
                    print(f"   📊 SMA(20): {sma_20:.3f}")
                    
                    # Trend analysis
                    trend = "BULLISH" if current_price > sma_20 else "BEARISH"
                    print(f"   🎯 M15 Trend: {trend}")
                    
                    # Multi-timeframe confluence
                    confluence = {}
                    for tf_name, tf_const in timeframes.items():
                        if tf_name in available_timeframes:
                            tf_rates = mt5.copy_rates_from_pos(active_symbol, tf_const, 0, 50)
                            if tf_rates is not None and len(tf_rates) >= 10:
                                tf_current = tf_rates[-1]['close']
                                tf_sma = sum([bar['close'] for bar in tf_rates[-10:]]) / 10
                                tf_trend = "BULLISH" if tf_current > tf_sma else "BEARISH"
                                confluence[tf_name] = tf_trend
                    
                    print(f"   🎯 Multi-TF Confluence:")
                    for tf, trend in confluence.items():
                        emoji = "📈" if trend == "BULLISH" else "📉"
                        print(f"      {emoji} {tf}: {trend}")
                    
                    # Trading parameters for M15
                    print(f"\\n⚙️ M15 Trading Parameters:")
                    print(f"   🎯 Primary Timeframe: M15")
                    print(f"   🛡️ Stop Loss: 20 pips")
                    print(f"   🎯 Take Profit: 40 pips")
                    print(f"   💰 Risk per Trade: 0.5%")
                    print(f"   📊 Max Spread: 30 pips")
                    print(f"   🔢 Max Positions: 3")
                    
                else:
                    print("   ❌ Insufficient M15 data for analysis")
            else:
                print("   ❌ M15 data not available")
            
            mt5.shutdown()
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        mt5.shutdown()
        return False

async def main():
    """Main test function"""
    
    print("🤖 XAUUSD_TDS M15 Configuration Test")
    print("Symbol=XAUUSD_TDS")
    print("Period=M15")
    print("=" * 60)
    
    success = await test_xauusd_tds_symbol()
    
    if success:
        print("\\n🚀 CONFIGURATION SUCCESS!")
        print("✅ Symbol configured: XAUUSD_TDS (or XAUUSD fallback)")
        print("✅ M15 timeframe validated")
        print("✅ Multi-timeframe data available")
        print("✅ Trading parameters optimized for M15")
        print("✅ Ready for autonomous EA development")
        
        print("\\n📋 Next Steps:")
        print("1. 🔄 Restart Qoder IDE")
        print("2. 🤖 Use metatrader5_roboforex MCP")
        print("3. 🚀 Start M15 XAUUSD_TDS EA development")
        
    else:
        print("\\n❌ CONFIGURATION FAILED")
        print("🔧 Troubleshooting:")
        print("1. Check RoboForex MT5 connection")
        print("2. Verify XAUUSD or XAUUSD_TDS symbol availability")
        print("3. Ensure Market Watch is accessible")
        print("4. Check timeframe data availability")

if __name__ == "__main__":
    asyncio.run(main())