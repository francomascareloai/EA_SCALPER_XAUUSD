#!/usr/bin/env python3
"""
RoboForex MetaTrader 5 Connector for MCP Server
Specialized connector for RoboForex broker integration
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import MetaTrader5 as mt5

logger = logging.getLogger("roboforex_mt5_connector")

class RoboForexMT5Connector:
    """Specialized connector for RoboForex MetaTrader 5"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize RoboForex MT5 Connector
        
        Args:
            config_path: Path to RoboForex configuration file
        """
        self.config_path = config_path or "config/roboforex_config.json"
        self.config = self._load_config()
        self.is_connected = False
        self.account_info = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load RoboForex specific configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default RoboForex configuration"""
        return {
            "broker_config": {
                "name": "RoboForex",
                "server_name": "RoboForex-Demo"
            },
            "connection_settings": {
                "timeout": 10000,
                "retry_attempts": 3,
                "retry_delay": 5000
            },
            "trading_settings": {
                "execution_mode": "market",
                "max_slippage": 10
            }
        }
    
    async def connect(self, login: int, password: str, server: str = None) -> bool:
        """
        Connect to RoboForex MetaTrader 5
        
        Args:
            login: Account login number
            password: Account password
            server: Server name (defaults to configured server)
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Use configured server if not provided
            if server is None:
                server = self.config["connection_settings"].get("server", "RoboForex-Demo")
            
            logger.info(f"🔌 Connecting to RoboForex MT5 server: {server}")
            
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"❌ Failed to initialize MT5: {mt5.last_error()}")
                return False
            
            # Attempt login with retry logic
            for attempt in range(self.config["connection_settings"]["retry_attempts"]):
                logger.info(f"🔐 Login attempt {attempt + 1}/{self.config['connection_settings']['retry_attempts']}")
                
                if mt5.login(login, password, server):
                    self.is_connected = True
                    self.account_info = mt5.account_info()
                    
                    logger.info(f"✅ Successfully connected to RoboForex MT5")
                    logger.info(f"📊 Account: {login}")
                    logger.info(f"🏢 Server: {server}")
                    logger.info(f"💰 Balance: {self.account_info.balance}")
                    logger.info(f"💱 Currency: {self.account_info.currency}")
                    
                    # Validate broker
                    if not self._validate_broker():
                        logger.warning("⚠️ Broker validation failed, but connection established")
                    
                    return True
                else:
                    error_code = mt5.last_error()
                    logger.warning(f"❌ Login attempt {attempt + 1} failed: {error_code}")
                    
                    if attempt < self.config["connection_settings"]["retry_attempts"] - 1:
                        sleep_time = self.config["connection_settings"]["retry_delay"] / 1000
                        logger.info(f"⏱️ Waiting {sleep_time} seconds before retry...")
                        time.sleep(sleep_time)
            
            logger.error("❌ All login attempts failed")
            mt5.shutdown()
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def _validate_broker(self) -> bool:
        """Validate that we're connected to RoboForex"""
        if not self.account_info:
            return False
        
        company = self.account_info.company.lower()
        expected_company = "roboforex"
        
        if expected_company in company:
            logger.info(f"✅ Broker validation passed: {self.account_info.company}")
            return True
        else:
            logger.warning(f"⚠️ Unexpected broker: {self.account_info.company}")
            return False
    
    async def setup_xauusd_symbol(self) -> bool:
        """Setup XAUUSD_TDS symbol for trading"""
        try:
            symbol = "XAUUSD_TDS"
            
            # Check if symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"❌ Symbol {symbol} not found on RoboForex")
                # Try fallback to standard XAUUSD
                logger.info("🔄 Trying fallback to standard XAUUSD...")
                symbol = "XAUUSD"
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"❌ Fallback symbol {symbol} also not found")
                    return False
            
            # Add symbol to Market Watch
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Failed to add {symbol} to Market Watch")
                return False
            
            logger.info(f"✅ {symbol} successfully configured")
            logger.info(f"📊 Spread: {symbol_info.spread} points")
            logger.info(f"💎 Contract size: {getattr(symbol_info, 'contract_size', 'N/A')}")
            logger.info(f"📏 Min lot: {symbol_info.volume_min}")
            logger.info(f"📏 Max lot: {symbol_info.volume_max}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up XAUUSD: {e}")
            return False
    
    async def get_roboforex_trading_conditions(self) -> Dict[str, Any]:
        """Get RoboForex specific trading conditions for XAUUSD_TDS"""
        try:
            # Try XAUUSD_TDS first, then fallback to XAUUSD
            symbol = "XAUUSD_TDS"
            xauusd_info = mt5.symbol_info(symbol)
            
            if xauusd_info is None:
                logger.info("🔄 XAUUSD_TDS not found, using standard XAUUSD")
                symbol = "XAUUSD"
                xauusd_info = mt5.symbol_info(symbol)
                
            account_info = mt5.account_info()
            
            if not xauusd_info or not account_info:
                return {}
            
            conditions = {
                "broker": "RoboForex",
                "server": account_info.server,
                "leverage": account_info.leverage,
                "currency": account_info.currency,
                "symbol_used": symbol,  # Track which symbol is being used
                "xauusd": {
                    "symbol_name": symbol,
                    "spread": xauusd_info.spread,
                    "contract_size": getattr(xauusd_info, 'contract_size', 100),
                    "min_lot": xauusd_info.volume_min,
                    "max_lot": xauusd_info.volume_max,
                    "lot_step": xauusd_info.volume_step,
                    "margin_required": getattr(xauusd_info, 'margin_initial', 0),
                    "swap_long": getattr(xauusd_info, 'swap_long', 0),
                    "swap_short": getattr(xauusd_info, 'swap_short', 0),
                    "tick_value": getattr(xauusd_info, 'tick_value', 0),
                    "tick_size": getattr(xauusd_info, 'tick_size', 0.01),
                    "digits": xauusd_info.digits,
                    "point": xauusd_info.point
                },
                "trading_hours": {
                    "current_time": datetime.now().isoformat(),
                    "server_time": datetime.fromtimestamp(xauusd_info.time).isoformat() if xauusd_info.time else None
                },
                "account_type": "Demo" if "demo" in account_info.server.lower() else "Live"
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"❌ Error getting trading conditions: {e}")
            return {}
    
    async def validate_ftmo_compliance(self) -> Dict[str, Any]:
        """Validate FTMO compliance settings for RoboForex"""
        try:
            account = mt5.account_info()
            if not account:
                return {"valid": False, "reason": "No account info"}
            
            compliance_checks = {
                "leverage_check": account.leverage <= 100,  # FTMO max leverage
                "currency_check": account.currency == "USD",  # FTMO accounts in USD
                "netting_check": True,  # RoboForex supports netting
                "hedging_prohibited": not account.fifo_close,  # Check hedging settings
                "account_balance": account.balance,
                "server_type": "demo" in account.server.lower()
            }
            
            # Overall compliance
            compliance_checks["ftmo_compliant"] = all([
                compliance_checks["leverage_check"],
                compliance_checks["currency_check"],
                compliance_checks["netting_check"]
            ])
            
            return compliance_checks
            
        except Exception as e:
            logger.error(f"❌ Error validating FTMO compliance: {e}")
            return {"valid": False, "reason": str(e)}
    
    async def test_connection_quality(self) -> Dict[str, Any]:
        """Test connection quality and latency for XAUUSD_TDS"""
        try:
            # Determine which symbol to use
            symbol = "XAUUSD_TDS"
            if mt5.symbol_info(symbol) is None:
                symbol = "XAUUSD"
                
            start_time = time.time()
            
            # Test symbol info retrieval
            symbol_info = mt5.symbol_info(symbol)
            info_latency = (time.time() - start_time) * 1000
            
            # Test tick retrieval
            start_time = time.time()
            tick = mt5.symbol_info_tick(symbol)
            tick_latency = (time.time() - start_time) * 1000
            
            # Test account info
            start_time = time.time()
            account = mt5.account_info()
            account_latency = (time.time() - start_time) * 1000
            
            quality_report = {
                "connection_status": "connected" if self.is_connected else "disconnected",
                "latency": {
                    "symbol_info": f"{info_latency:.2f}ms",
                    "tick_data": f"{tick_latency:.2f}ms",
                    "account_info": f"{account_latency:.2f}ms",
                    "average": f"{(info_latency + tick_latency + account_latency) / 3:.2f}ms"
                },
                "data_quality": {
                    "symbol_info_available": symbol_info is not None,
                    "tick_data_available": tick is not None,
                    "account_data_available": account is not None,
                    "last_tick_time": datetime.fromtimestamp(tick.time).isoformat() if tick else None
                },
                "server_info": {
                    "server": account.server if account else None,
                    "company": account.company if account else None
                }
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"❌ Error testing connection quality: {e}")
            return {"error": str(e)}
    
    def disconnect(self) -> bool:
        """Disconnect from MetaTrader 5"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                self.account_info = None
                logger.info("🔌 Disconnected from RoboForex MT5")
                return True
            return True
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "connected": self.is_connected,
            "broker": "RoboForex",
            "account": self.account_info.login if self.account_info else None,
            "server": self.account_info.server if self.account_info else None,
            "balance": self.account_info.balance if self.account_info else None,
            "currency": self.account_info.currency if self.account_info else None
        }

# Example usage and testing
async def main():
    """Test RoboForex connector"""
    
    print("🤖 RoboForex MT5 Connector Test")
    print("=" * 50)
    
    # Initialize connector
    connector = RoboForexMT5Connector()
    
    # Note: You need to provide real credentials for testing
    # This is just an example - DO NOT commit real credentials
    demo_login = 12345678  # Replace with your demo account
    demo_password = "YourPassword"  # Replace with your password
    demo_server = "RoboForex-Demo"
    
    print("⚠️ This is a test script. Please provide your RoboForex demo credentials.")
    print("📝 Edit the credentials in the script before running.")
    
    # Test connection (uncomment and add real credentials to test)
    # if await connector.connect(demo_login, demo_password, demo_server):
    #     print("✅ Connection test passed")
        
    #     # Setup XAUUSD
    #     await connector.setup_xauusd_symbol()
        
    #     # Get trading conditions
    #     conditions = await connector.get_roboforex_trading_conditions()
    #     print(f"📊 Trading conditions: {conditions}")
        
    #     # Test connection quality
    #     quality = await connector.test_connection_quality()
    #     print(f"📡 Connection quality: {quality}")
        
    #     # Validate FTMO compliance
    #     compliance = await connector.validate_ftmo_compliance()
    #     print(f"✅ FTMO compliance: {compliance}")
        
    #     # Disconnect
    #     connector.disconnect()
    # else:
    #     print("❌ Connection test failed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())