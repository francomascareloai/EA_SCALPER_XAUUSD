"""
MetaTrader 5 MCP Server - A Model Context Protocol server for MetaTrader 5.

This server provides tools and resources for AI assistants to interact with MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from fastmcp import FastMCP, Image
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mt5-mcp-server")

# Create the MCP server
mcp = FastMCP(
    "MetaTrader 5 MCP Server",
    description="A Model Context Protocol server for MetaTrader 5 trading platform",
    dependencies=["MetaTrader5", "pandas", "numpy", "fastmcp", "pydantic"],
)

# Models for request/response data
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
