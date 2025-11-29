"""
MetaTrader 5 MCP Server - Broker Integration Module

This module manages broker-specific connections and configurations,
especially for autonomous agent integration with RoboForex.
"""

import json
import logging
import os
import sys
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import MetaTrader5 as mt5

# Add parent directory to path to import RoboForex connector
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

try:
    from roboforex_mt5_connector import RoboForexMT5Connector
except ImportError:
    # Fallback for when RoboForex connector is not available
    RoboForexMT5Connector = None

logger = logging.getLogger("mt5-mcp-broker-integration")

class BrokerManager:
    """Manages broker-specific connections and configurations"""
    
    def __init__(self):
        """Initialize the broker manager"""
        self.broker_type = os.environ.get("MT5_BROKER", "default")
        self.config_path = os.environ.get("MT5_CONFIG_PATH")
        self.autonomous_mode = os.environ.get("MT5_AUTONOMOUS_MODE", "false").lower() == "true"
        self.connector = None
        self.is_connected = False
        self.last_connection_test = None
        
        logger.info(f"🏢 Broker Manager initialized for: {self.broker_type}")
        if self.autonomous_mode:
            logger.info("🤖 Autonomous agent mode enabled")
    
    async def initialize_broker(self) -> bool:
        """
        Initialize broker-specific connector
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self.broker_type == "roboforex":
                return await self._initialize_roboforex()
            else:
                return await self._initialize_default()
        except Exception as e:
            logger.error(f"❌ Broker initialization failed: {e}")
            if self.autonomous_mode:
                # In autonomous mode, provide detailed error info
                await self._log_autonomous_error("BROKER_INIT_FAILED", str(e))
            return False
    
    async def _initialize_roboforex(self) -> bool:
        """Initialize RoboForex connector"""
        try:
            if RoboForexMT5Connector is None:
                logger.error("❌ RoboForex connector not available")
                return False
            
            logger.info("🔧 Initializing RoboForex connector...")
            self.connector = RoboForexMT5Connector(config_path=self.config_path)
            
            # Test basic functionality
            if await self._test_roboforex_connection():
                logger.info("✅ RoboForex connector initialized successfully")
                return True
            else:
                logger.error("❌ RoboForex connector test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ RoboForex initialization error: {e}")
            return False
    
    async def _initialize_default(self) -> bool:
        """Initialize default MT5 connection"""
        try:
            logger.info("🔧 Initializing default MT5 connection...")
            
            # Basic MT5 initialization
            if not mt5.initialize():
                logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            logger.info("✅ Default MT5 connector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Default MT5 initialization error: {e}")
            return False
    
    async def _test_roboforex_connection(self) -> bool:
        """Test RoboForex connection without credentials"""
        try:
            # Load configuration to validate structure
            config = self.connector._load_config()
            
            # Validate required configuration sections
            required_sections = ["broker_config", "connection_settings"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"❌ Missing configuration section: {section}")
                    return False
            
            # Test MT5 initialization
            if not mt5.initialize():
                logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Shutdown after test
            mt5.shutdown()
            
            logger.info("✅ RoboForex connection test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ RoboForex connection test failed: {e}")
            return False
    
    async def connect_with_credentials(self, login: int, password: str, server: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Connect to broker with credentials
        
        Args:
            login: Account login
            password: Account password  
            server: Server name (optional)
            
        Returns:
            Tuple[bool, Dict]: (success, connection_info)
        """
        try:
            if self.broker_type == "roboforex" and self.connector:
                success = await self.connector.connect(login, password, server)
                if success:
                    # Setup XAUUSD symbol
                    await self.connector.setup_xauusd_symbol()
                    
                    # Get connection info
                    connection_info = await self._get_roboforex_info()
                    
                    self.is_connected = True
                    self.last_connection_test = datetime.now()
                    
                    if self.autonomous_mode:
                        await self._log_autonomous_success("BROKER_CONNECTED", connection_info)
                    
                    return True, connection_info
                else:
                    error_info = {"error": "RoboForex connection failed"}
                    if self.autonomous_mode:
                        await self._log_autonomous_error("BROKER_CONNECT_FAILED", "RoboForex connection failed")
                    return False, error_info
            
            else:
                # Default connection
                if not mt5.initialize():
                    error_info = {"error": f"MT5 initialization failed: {mt5.last_error()}"}
                    return False, error_info
                
                if mt5.login(login, password, server or ""):
                    account_info = mt5.account_info()
                    connection_info = {
                        "broker": "default",
                        "server": account_info.server if account_info else "unknown",
                        "login": login,
                        "connected": True
                    }
                    
                    self.is_connected = True
                    self.last_connection_test = datetime.now()
                    
                    return True, connection_info
                else:
                    error_info = {"error": f"MT5 login failed: {mt5.last_error()}"}
                    return False, error_info
                    
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            if self.autonomous_mode:
                await self._log_autonomous_error("BROKER_CONNECT_ERROR", error_msg)
            return False, {"error": error_msg}
    
    async def _get_roboforex_info(self) -> Dict[str, Any]:
        """Get RoboForex connection information"""
        try:
            conditions = await self.connector.get_roboforex_trading_conditions()
            return {
                "broker": "RoboForex",
                "server": conditions.get("server", "unknown"),
                "symbol_used": conditions.get("symbol_used", "XAUUSD_TDS"),
                "leverage": conditions.get("leverage", "unknown"),
                "currency": conditions.get("currency", "USD"),
                "connected": True,
                "trading_conditions": conditions
            }
        except Exception as e:
            logger.error(f"❌ Error getting RoboForex info: {e}")
            return {"broker": "RoboForex", "connected": True, "error": str(e)}
    
    async def validate_connection(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate current connection status
        
        Returns:
            Tuple[bool, Dict]: (is_valid, status_info)
        """
        try:
            if not self.is_connected:
                return False, {"status": "not_connected", "message": "No active connection"}
            
            # Test connection health
            if self.broker_type == "roboforex" and self.connector:
                # Test with a simple operation
                try:
                    if not mt5.terminal_info():
                        return False, {"status": "terminal_error", "message": "MT5 terminal not responding"}
                    
                    return True, {
                        "status": "connected", 
                        "broker": "RoboForex",
                        "last_test": self.last_connection_test.isoformat() if self.last_connection_test else None
                    }
                except Exception as e:
                    return False, {"status": "connection_error", "message": str(e)}
            
            else:
                # Default validation
                try:
                    terminal_info = mt5.terminal_info()
                    if terminal_info:
                        return True, {
                            "status": "connected",
                            "broker": "default",
                            "terminal": terminal_info._asdict() if hasattr(terminal_info, '_asdict') else str(terminal_info)
                        }
                    else:
                        return False, {"status": "terminal_error", "message": "MT5 terminal not responding"}
                except Exception as e:
                    return False, {"status": "validation_error", "message": str(e)}
                    
        except Exception as e:
            logger.error(f"❌ Connection validation error: {e}")
            return False, {"status": "validation_failed", "message": str(e)}
    
    async def get_broker_capabilities(self) -> Dict[str, Any]:
        """
        Get broker-specific capabilities and information
        
        Returns:
            Dict: Broker capabilities and features
        """
        capabilities = {
            "broker_type": self.broker_type,
            "autonomous_mode": self.autonomous_mode,
            "connected": self.is_connected,
            "features": []
        }
        
        if self.broker_type == "roboforex":
            capabilities["features"].extend([
                "XAUUSD_TDS_support",
                "multi_timeframe_analysis", 
                "ftmo_compliance",
                "roboforex_conditions",
                "symbol_fallback"
            ])
            
            if self.connector:
                try:
                    config = self.connector._load_config()
                    capabilities["config_sections"] = list(config.keys())
                except:
                    pass
        
        capabilities["features"].extend([
            "market_data",
            "order_management", 
            "position_tracking",
            "history_analysis"
        ])
        
        return capabilities
    
    async def _log_autonomous_error(self, error_code: str, message: str):
        """Log errors for autonomous agent debugging"""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error_code": error_code,
            "message": message,
            "broker_type": self.broker_type,
            "autonomous_mode": self.autonomous_mode
        }
        logger.error(f"🤖 AUTONOMOUS_ERROR: {json.dumps(error_log)}")
    
    async def _log_autonomous_success(self, event_code: str, data: Dict[str, Any]):
        """Log successful events for autonomous agent monitoring"""
        success_log = {
            "timestamp": datetime.now().isoformat(),
            "event_code": event_code,
            "data": data,
            "broker_type": self.broker_type,
            "autonomous_mode": self.autonomous_mode
        }
        logger.info(f"🤖 AUTONOMOUS_SUCCESS: {json.dumps(success_log)}")
    
    def disconnect(self):
        """Disconnect from broker"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                logger.info("🔌 Disconnected from broker")
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")

# Global broker manager instance
broker_manager = BrokerManager()

# Convenience functions for MCP tools
async def get_broker_manager() -> BrokerManager:
    """Get the global broker manager instance"""
    return broker_manager

async def ensure_broker_initialized() -> bool:
    """Ensure broker is initialized"""
    if not broker_manager.connector and broker_manager.broker_type != "default":
        return await broker_manager.initialize_broker()
    return True

async def get_connection_status() -> Dict[str, Any]:
    """Get current connection status"""
    is_valid, status = await broker_manager.validate_connection()
    return {
        "valid": is_valid,
        "broker_type": broker_manager.broker_type,
        "autonomous_mode": broker_manager.autonomous_mode,
        **status
    }