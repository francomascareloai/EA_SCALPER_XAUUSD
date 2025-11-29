#!/usr/bin/env python3
"""
🤖 Setup RoboForex MetaTrader 5 for Autonomous EA Development
Configuração especializada para RoboForex com validação FTMO
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "mcp-metatrader5-server"))

from roboforex_mt5_connector import RoboForexMT5Connector

class RoboForexSetup:
    """Setup and configuration manager for RoboForex MT5"""
    
    def __init__(self):
        self.connector = RoboForexMT5Connector()
        self.credentials_file = "config/roboforex_credentials.json"
        
    def save_credentials(self, login: int, password: str, server: str = "RoboForex-Demo"):
        """
        Save RoboForex credentials securely
        
        Args:
            login: Account login number
            password: Account password  
            server: Server name (default: RoboForex-Demo)
        """
        # NOTE: do not hard-code real credentials in this file or commit them to version control.
        # Pass them explicitly (or load from environment variables) and store only in the local
        # JSON file below, which is ignored by .git.
        credentials = {
            "login": login,
            "password": password,
            "server": server,
            "broker": "RoboForex",
            "setup_date": "2025-08-22",
            "notes": "Demo account for FTMO challenge preparation"
        }
        
        # Create config directory if it doesn't exist
        config_dir = Path("mcp-metatrader5-server/config")
        config_dir.mkdir(exist_ok=True)
        
        # Save credentials
        with open(config_dir / "roboforex_credentials.json", 'w') as f:
            json.dump(credentials, f, indent=2)
        
        print(f"✅ Credentials saved for account: {login}")
        print(f"🏢 Server: {server}")
        print(f"⚠️ Keep credentials secure!")
    
    def load_credentials(self) -> Optional[Dict[str, Any]]:
        """Load saved credentials"""
        try:
            with open(f"mcp-metatrader5-server/{self.credentials_file}", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ No saved credentials found")
            return None
        except json.JSONDecodeError:
            print("❌ Invalid credentials file")
            return None
    
    async def test_connection(self, login: int = None, password: str = None, server: str = None):
        """
        Test connection to RoboForex MT5
        
        Args:
            login: Account login (if None, loads from saved credentials)
            password: Account password (if None, loads from saved credentials)
            server: Server name (if None, loads from saved credentials)
        """
        print("🔌 Testing RoboForex MT5 Connection...")
        print("=" * 50)
        
        # Load credentials if not provided
        if not all([login, password, server]):
            credentials = self.load_credentials()
            if not credentials:
                print("❌ No credentials provided or found")
                return False
            
            login = login or credentials.get("login")
            password = password or credentials.get("password")
            server = server or credentials.get("server", "RoboForex-Demo")
        
        # Test connection
        if await self.connector.connect(login, password, server):
            print("✅ Connection successful!")
            
            # Setup XAUUSD symbol
            print("\\n📊 Setting up XAUUSD symbol...")
            if await self.connector.setup_xauusd_symbol():
                print("✅ XAUUSD configured successfully")
            else:
                print("❌ Failed to configure XAUUSD")
            
            # Get trading conditions
            print("\\n📋 Getting RoboForex trading conditions...")
            conditions = await self.connector.get_roboforex_trading_conditions()
            if conditions:
                self.print_trading_conditions(conditions)
            
            # Test connection quality
            print("\\n📡 Testing connection quality...")
            quality = await self.connector.test_connection_quality()
            if quality:
                self.print_connection_quality(quality)
            
            # Validate FTMO compliance
            print("\\n✅ Validating FTMO compliance...")
            compliance = await self.connector.validate_ftmo_compliance()
            if compliance:
                self.print_ftmo_compliance(compliance)
            
            # Disconnect
            self.connector.disconnect()
            return True
        else:
            print("❌ Connection failed!")
            return False
    
    def print_trading_conditions(self, conditions: Dict[str, Any]):
        """Print trading conditions in a formatted way"""
        print("💰 RoboForex Trading Conditions:")
        print(f"  🏢 Broker: {conditions.get('broker', 'N/A')}")
        print(f"  🖥️ Server: {conditions.get('server', 'N/A')}")
        print(f"  📊 Leverage: 1:{conditions.get('leverage', 'N/A')}")
        print(f"  💱 Currency: {conditions.get('currency', 'N/A')}")
        print(f"  📱 Account Type: {conditions.get('account_type', 'N/A')}")
        
        if 'xauusd' in conditions:
            xau = conditions['xauusd']
            print("  \\n🥇 XAUUSD Specifications:")
            print(f"    📈 Spread: {xau.get('spread', 'N/A')} points")
            print(f"    📏 Min Lot: {xau.get('min_lot', 'N/A')}")
            print(f"    📏 Max Lot: {xau.get('max_lot', 'N/A')}")
            print(f"    💎 Contract Size: {xau.get('contract_size', 'N/A')}")
            print(f"    💰 Tick Value: ${xau.get('tick_value', 'N/A')}")
            print(f"    🔢 Digits: {xau.get('digits', 'N/A')}")
    
    def print_connection_quality(self, quality: Dict[str, Any]):
        """Print connection quality metrics"""
        print("📡 Connection Quality Report:")
        print(f"  🔌 Status: {quality.get('connection_status', 'N/A')}")
        
        if 'latency' in quality:
            lat = quality['latency']
            print(f"  ⚡ Average Latency: {lat.get('average', 'N/A')}")
            print(f"  📊 Symbol Info: {lat.get('symbol_info', 'N/A')}")
            print(f"  📈 Tick Data: {lat.get('tick_data', 'N/A')}")
            print(f"  👤 Account Info: {lat.get('account_info', 'N/A')}")
        
        if 'data_quality' in quality:
            data = quality['data_quality']
            print(f"  ✅ Data Quality:")
            print(f"    Symbol Info: {'✅' if data.get('symbol_info_available') else '❌'}")
            print(f"    Tick Data: {'✅' if data.get('tick_data_available') else '❌'}")
            print(f"    Account Data: {'✅' if data.get('account_data_available') else '❌'}")
    
    def print_ftmo_compliance(self, compliance: Dict[str, Any]):
        """Print FTMO compliance status"""
        print("✅ FTMO Compliance Check:")
        print(f"  🎯 Overall Compliant: {'✅' if compliance.get('ftmo_compliant') else '❌'}")
        print(f"  📊 Leverage OK: {'✅' if compliance.get('leverage_check') else '❌'}")
        print(f"  💱 Currency OK: {'✅' if compliance.get('currency_check') else '❌'}")
        print(f"  🔄 Netting OK: {'✅' if compliance.get('netting_check') else '❌'}")
        print(f"  🚫 Hedging Prohibited: {'✅' if not compliance.get('hedging_prohibited') else '❌'}")
        print(f"  💰 Account Balance: ${compliance.get('account_balance', 'N/A')}")
        print(f"  🧪 Demo Account: {'✅' if compliance.get('server_type') else '❌'}")
    
    def generate_mcp_config(self, login: int, server: str = "RoboForex-Demo"):
        """Generate MCP configuration for RoboForex"""
        
        config = {
            "mcpServers": {
                "metatrader5_roboforex": {
                    "command": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe",
                    "args": [
                        "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/run.py",
                        "dev",
                        "--host", "127.0.0.1",
                        "--port", "8000",
                        "--broker", "roboforex"
                    ],
                    "env": {
                        "PYTHONPATH": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/src",
                        "MT5_BROKER": "RoboForex",
                        "MT5_SERVER": server,
                        "MT5_LOGIN": str(login),
                        "MT5_CONFIG_PATH": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/config/roboforex_config.json"
                    }
                }
            }
        }
        
        # Save configuration
        with open("mcp_config_roboforex.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ MCP configuration generated: mcp_config_roboforex.json")
        print("📋 Copy this configuration to your Qoder IDE settings")
    
    async def autonomous_agent_setup(self):
        """Setup for autonomous agent development"""
        print("🤖 RoboForex Setup for Autonomous EA Agent")
        print("=" * 60)
        
        print("📋 Setup Steps:")
        print("1. 📝 Create RoboForex demo account")
        print("2. 💾 Save credentials")
        print("3. 🔌 Test connection")
        print("4. 📊 Validate XAUUSD setup")
        print("5. ✅ Verify FTMO compliance")
        print("6. 🚀 Generate MCP configuration")
        
        print("\\n⚠️ Instructions:")
        print("1. Visit: https://www.roboforex.com/demo-account/")
        print("2. Create a demo account")
        print("3. Download and install RoboForex MT5")
        print("4. Use your credentials to test this setup")
        
        return True

async def main():
    """Main setup function"""
    
    setup = RoboForexSetup()
    
    print("🤖 RoboForex MT5 Setup for XAUUSD EA Development")
    print("=" * 60)
    
    # Show autonomous agent setup guide
    await setup.autonomous_agent_setup()
    
    print("\\n" + "=" * 60)
    print("🚀 Setup ready for autonomous EA development!")
    print("📝 Edit credentials in this script and run test_connection()")
    print("✅ Follow the FTMO compliance guidelines")
    print("💡 Use demo account for testing before live trading")

if __name__ == "__main__":
    asyncio.run(main())