"""
MetaTrader 5 MCP Server - Core server functionality.

This module contains the main server instance and core functionality
with enhanced broker integration for autonomous agents.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
import asyncio

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from fastmcp import FastMCP, Image
from pydantic import BaseModel, Field

# Import broker integration
from .broker_integration import get_broker_manager, ensure_broker_initialized, get_connection_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mt5-mcp-server")

# Create the MCP server with enhanced autonomous agent support
mcp = FastMCP(
    "MetaTrader 5 MCP Server",
    description="A Model Context Protocol server for MetaTrader 5 trading platform with autonomous agent integration",
    dependencies=["MetaTrader5", "pandas", "numpy", "fastmcp", "pydantic"],
)

# Initialize broker manager on startup
@mcp.on_startup
async def initialize_server():
    """Initialize the MCP server and broker integration"""
    logger.info("🚀 Initializing MetaTrader 5 MCP Server...")
    
    # Initialize broker manager
    broker_manager = await get_broker_manager()
    success = await broker_manager.initialize_broker()
    
    if success:
        logger.info(f"✅ Broker ({broker_manager.broker_type}) initialized successfully")
        
        # Log autonomous mode status
        if broker_manager.autonomous_mode:
            logger.info("🤖 Server running in autonomous agent mode")
            capabilities = await broker_manager.get_broker_capabilities()
            logger.info(f"🔧 Available capabilities: {capabilities['features']}")
    else:
        logger.warning("⚠️ Broker initialization failed, some functions may not work")
    
    logger.info("🌟 MetaTrader 5 MCP Server ready for connections")

@mcp.tool()
async def get_server_status() -> Dict[str, Any]:
    """
    Get comprehensive server status including broker integration.
    
    Returns:
        Dict containing server status, broker information, and capabilities
    """
    try:
        broker_manager = await get_broker_manager()
        connection_status = await get_connection_status()
        capabilities = await broker_manager.get_broker_capabilities()
        
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "broker_integration": {
                "broker_type": broker_manager.broker_type,
                "autonomous_mode": broker_manager.autonomous_mode,
                "connection_status": connection_status,
                "capabilities": capabilities
            },
            "mt5_terminal": {
                "initialized": bool(mt5.terminal_info()),
                "version": str(mt5.version()) if mt5.version() else None
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting server status: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@mcp.tool()
async def connect_to_broker(login: int, password: str, server: str = None) -> Dict[str, Any]:
    """
    Connect to MetaTrader 5 broker with automatic broker detection.
    
    Args:
        login: Account login number
        password: Account password 
        server: Server name (optional, will use configured server)
        
    Returns:
        Connection result with broker information
    """
    try:
        # Ensure broker is initialized
        await ensure_broker_initialized()
        
        broker_manager = await get_broker_manager()
        success, connection_info = await broker_manager.connect_with_credentials(login, password, server)
        
        if success:
            logger.info(f"✅ Connected to {connection_info.get('broker', 'broker')}")
            return {
                "success": True,
                "message": "Connected successfully",
                "connection_info": connection_info
            }
        else:
            logger.error(f"❌ Connection failed: {connection_info.get('error', 'Unknown error')}")
            return {
                "success": False,
                "message": "Connection failed", 
                "error": connection_info.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        return {
            "success": False,
            "message": "Connection error",
            "error": str(e)
        }

@mcp.tool()
async def validate_broker_connection() -> Dict[str, Any]:
    """
    Validate current broker connection status.
    
    Returns:
        Connection validation result
    """
    try:
        broker_manager = await get_broker_manager()
        is_valid, status_info = await broker_manager.validate_connection()
        
        return {
            "valid": is_valid,
            "broker_type": broker_manager.broker_type,
            "autonomous_mode": broker_manager.autonomous_mode,
            "status_info": status_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return {
            "valid": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add error handling for autonomous agents
@mcp.error_handler
async def handle_autonomous_errors(error: Exception, tool_name: str) -> Dict[str, Any]:
    """
    Handle errors with enhanced reporting for autonomous agents.
    """
    broker_manager = await get_broker_manager()
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "tool_name": tool_name,
        "timestamp": datetime.now().isoformat(),
        "broker_type": broker_manager.broker_type,
        "autonomous_mode": broker_manager.autonomous_mode
    }
    
    if broker_manager.autonomous_mode:
        # Enhanced error logging for autonomous agents
        logger.error(f"🤖 AUTONOMOUS_TOOL_ERROR: {error_info}")
        
        # Try to provide recovery suggestions
        recovery_suggestions = []
        
        if "connection" in str(error).lower():
            recovery_suggestions.extend([
                "Check MT5 terminal is running",
                "Verify account credentials", 
                "Validate server name",
                "Test network connectivity"
            ])
        
        if "symbol" in str(error).lower():
            recovery_suggestions.extend([
                "Verify symbol exists on broker",
                "Check symbol is added to Market Watch",
                "Try symbol fallback (XAUUSD_TDS -> XAUUSD)"
            ])
        
        error_info["recovery_suggestions"] = recovery_suggestions
    
    return error_info
class SymbolInfo(BaseModel):
    """Information about a trading symbol"""
    name: str
    description: Optional[str] = None
    path: Optional[str] = None
    session_deals: Optional[int] = None
    session_buy_orders: Optional[int] = None
    session_sell_orders: Optional[int] = None
    volume: Optional[float] = None
    volumehigh: Optional[float] = None
    volumelow: Optional[float] = None
    time: Optional[int] = None
    digits: Optional[int] = None
    spread: Optional[int] = None
    spread_float: Optional[bool] = None
    trade_calc_mode: Optional[int] = None
    trade_mode: Optional[int] = None
    start_time: Optional[int] = None
    expiration_time: Optional[int] = None
    trade_stops_level: Optional[int] = None
    trade_freeze_level: Optional[int] = None
    trade_exemode: Optional[int] = None
    swap_mode: Optional[int] = None
    swap_rollover3days: Optional[int] = None
    margin_hedged_use_leg: Optional[bool] = None
    expiration_mode: Optional[int] = None
    filling_mode: Optional[int] = None
    order_mode: Optional[int] = None
    order_gtc_mode: Optional[int] = None
    option_mode: Optional[int] = None
    option_right: Optional[int] = None
    bid: Optional[float] = None
    bidhigh: Optional[float] = None
    bidlow: Optional[float] = None
    ask: Optional[float] = None
    askhigh: Optional[float] = None
    asklow: Optional[float] = None
    last: Optional[float] = None
    lasthigh: Optional[float] = None
    lastlow: Optional[float] = None
    point: Optional[float] = None
    tick_value: Optional[float] = None
    tick_value_profit: Optional[float] = None
    tick_value_loss: Optional[float] = None
    tick_size: Optional[float] = None
    contract_size: Optional[float] = None
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None
    volume_step: Optional[float] = None
    swap_long: Optional[float] = None
    swap_short: Optional[float] = None
    margin_initial: Optional[float] = None
    margin_maintenance: Optional[float] = None

class AccountInfo(BaseModel):
    """Trading account information"""
    login: int
    trade_mode: int
    leverage: int
    limit_orders: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int
    fifo_close: bool
    balance: float
    credit: float
    profit: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    margin_so_call: float
    margin_so_so: float
    margin_initial: float
    margin_maintenance: float
    assets: float
    liabilities: float
    commission_blocked: float
    name: str
    server: str
    currency: str
    company: str

class OrderRequest(BaseModel):
    """Order request parameters"""
    action: int
    symbol: str
    volume: float
    type: int
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    deviation: Optional[int] = None
    magic: Optional[int] = None
    comment: Optional[str] = None
    type_time: Optional[int] = None
    type_filling: Optional[int] = None

class OrderResult(BaseModel):
    """Order execution result"""
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int
    retcode_external: int
    request: Dict[str, Any]

class Position(BaseModel):
    """Trading position information"""
    ticket: int
    time: int
    time_msc: int
    time_update: int
    time_update_msc: int
    type: int
    magic: int
    identifier: int
    reason: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    swap: float
    profit: float
    symbol: str
    comment: str
    external_id: str

class HistoryOrder(BaseModel):
    """Historical order information"""
    ticket: int
    time_setup: int
    time_setup_msc: int
    time_expiration: int
    type: int
    type_time: int
    type_filling: int
    state: int
    magic: int
    position_id: int
    position_by_id: int
    reason: int
    volume_initial: float
    volume_current: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    price_stoplimit: float
    symbol: str
    comment: str
    external_id: str

class Deal(BaseModel):
    """Deal information"""
    ticket: int
    order: int
    time: int
    time_msc: int
    type: int
    entry: int
    magic: int
    position_id: int
    reason: int
    volume: float
    price: float
    commission: float
    swap: float
    profit: float
    fee: float
    symbol: str
    comment: str
    external_id: str

# Initialize MetaTrader 5 connection
@mcp.tool()
def initialize() -> bool:
    """
    Initialize the MetaTrader 5 terminal.
    
    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return False
    
    logger.info("MT5 initialized successfully")
    return True

# Shutdown MetaTrader 5 connection
@mcp.tool()
def shutdown() -> bool:
    """
    Shut down the connection to the MetaTrader 5 terminal.
    
    Returns:
        bool: True if shutdown was successful.
    """
    mt5.shutdown()
    logger.info("MT5 connection shut down")
    return True

# Login to MetaTrader 5 account
@mcp.tool()
def login(login: int, password: str, server: str) -> bool:
    """
    Log in to the MetaTrader 5 trading account.
    
    Args:
        login: Trading account number
        password: Trading account password
        server: Trading server name
        
    Returns:
        bool: True if login was successful, False otherwise.
    """
    if not mt5.login(login=login, password=password, server=server):
        logger.error(f"MT5 login failed, error code: {mt5.last_error()}")
        return False
    
    logger.info(f"MT5 login successful to account #{login} on server {server}")
    return True

# Get account information
@mcp.tool()
def get_account_info() -> AccountInfo:
    """
    Get information about the current trading account.
    
    Returns:
        AccountInfo: Information about the trading account.
    """
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info, error code: {mt5.last_error()}")
        raise ValueError("Failed to get account info")
    
    # Convert named tuple to dictionary
    account_dict = account_info._asdict()
    return AccountInfo(**account_dict)

# Get terminal information
@mcp.tool()
def get_terminal_info() -> Dict[str, Any]:
    """
    Get information about the MetaTrader 5 terminal.
    
    Returns:
        Dict[str, Any]: Information about the terminal.
    """
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        logger.error(f"Failed to get terminal info, error code: {mt5.last_error()}")
        raise ValueError("Failed to get terminal info")
    
    # Convert named tuple to dictionary
    return terminal_info._asdict()

# Get version information
@mcp.tool()
def get_version() -> Dict[str, Any]:
    """
    Get the MetaTrader 5 version.
    
    Returns:
        Dict[str, Any]: Version information.
    """
    version = mt5.version()
    if version is None:
        logger.error(f"Failed to get version, error code: {mt5.last_error()}")
        raise ValueError("Failed to get version")
    
    return {
        "version": version[0],
        "build": version[1],
        "date": version[2]
    }
