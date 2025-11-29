"""
MetaTrader 5 MCP Server - Market Data Functions

This module contains tools and resources for accessing market data from MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from fastmcp import FastMCP, Image
from pydantic import BaseModel, Field

# Import the main server instance
from mcp_metatrader5_server.server import mcp, SymbolInfo

logger = logging.getLogger("mt5-mcp-server.market_data")

# Get symbols
@mcp.tool()
def get_symbols() -> List[str]:
    """
    Get all available symbols (financial instruments) from the MetaTrader 5 terminal.
    
    Returns:
        List[str]: List of symbol names.
    """
    symbols = mt5.symbols_get()
    if symbols is None:
        logger.error(f"Failed to get symbols, error code: {mt5.last_error()}")
        raise ValueError("Failed to get symbols")
    
    return [symbol.name for symbol in symbols]

# Get symbols by group
@mcp.tool()
def get_symbols_by_group(group: str) -> List[str]:
    """
    Get symbols that match a specific group or pattern.
    
    Args:
        group: Filter for arranging a group of symbols (e.g., "*", "EUR*", etc.)
        
    Returns:
        List[str]: List of symbol names that match the group.
    """
    symbols = mt5.symbols_get(group=group)
    if symbols is None:
        logger.error(f"Failed to get symbols for group {group}, error code: {mt5.last_error()}")
        return []
    
    return [symbol.name for symbol in symbols]

# Get symbol information
@mcp.tool()
def get_symbol_info(symbol: str) -> SymbolInfo:
    """
    Get information about a specific symbol.
    
    Args:
        symbol: Symbol name
        
    Returns:
        SymbolInfo: Information about the symbol.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get info for symbol {symbol}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to get info for symbol {symbol}")
    
    # Convert named tuple to dictionary
    symbol_dict = symbol_info._asdict()
    return SymbolInfo(**symbol_dict)

# Get symbol tick information
@mcp.tool()
def get_symbol_info_tick(symbol: str) -> Dict[str, Any]:
    """
    Get the latest tick data for a symbol.
    
    Args:
        symbol: Symbol name
        
    Returns:
        Dict[str, Any]: Latest tick data for the symbol.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Failed to get tick for symbol {symbol}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to get tick for symbol {symbol}")
    
    # Convert named tuple to dictionary
    return tick._asdict()

# Select symbol in Market Watch
@mcp.tool()
def symbol_select(symbol: str, visible: bool = True) -> bool:
    """
    Select a symbol in the Market Watch window or remove a symbol from it.
    
    Args:
        symbol: Symbol name
        visible: Symbol visibility flag
            - True: Make the symbol visible in Market Watch
            - False: Hide the symbol from Market Watch
        
    Returns:
        bool: True if the symbol is selected successfully, False otherwise.
    """
    result = mt5.symbol_select(symbol, visible)
    if not result:
        logger.error(f"Failed to select symbol {symbol}, error code: {mt5.last_error()}")
    
    return result

# Copy rates from position
@mcp.tool()
def copy_rates_from_pos(
    symbol: str, 
    timeframe: int, 
    start_pos: int, 
    count: int
) -> List[Dict[str, Any]]:
    """
    Get bars from a specified symbol and timeframe starting from the specified position.
    
    Args:
        symbol: Symbol name
        timeframe: Timeframe as specified in TIMEFRAME_* constants:
            - 1: TIMEFRAME_M1 (1 minute)
            - 5: TIMEFRAME_M5 (5 minutes)
            - 15: TIMEFRAME_M15 (15 minutes)
            - 30: TIMEFRAME_M30 (30 minutes)
            - 60: TIMEFRAME_H1 (1 hour)
            - 240: TIMEFRAME_H4 (4 hours)
            - 1440: TIMEFRAME_D1 (1 day)
            - 10080: TIMEFRAME_W1 (1 week)
            - 43200: TIMEFRAME_MN1 (1 month)
        start_pos: Initial position for bar retrieval
        count: Number of bars to retrieve
        
    Returns:
        List[Dict[str, Any]]: List of bars with time, open, high, low, close, tick_volume, spread, and real_volume.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
    if rates is None:
        logger.error(f"Failed to copy rates for {symbol}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy rates for {symbol}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(rates)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df.to_dict('records')

# Copy rates from date
@mcp.tool()
def copy_rates_from_date(
    symbol: str, 
    timeframe: int, 
    date_from: datetime, 
    count: int
) -> List[Dict[str, Any]]:
    """
    Get bars from a specified symbol and timeframe starting from the specified date.
    
    Args:
        symbol: Symbol name
        timeframe: Timeframe (use TIMEFRAME_* constants)
        date_from: Start date for bar retrieval
        count: Number of bars to retrieve
        
    Returns:
        List[Dict[str, Any]]: List of bars with time, open, high, low, close, tick_volume, spread, and real_volume.
    """
    # Convert datetime to timestamp
    date_from_timestamp = int(date_from.timestamp())
    
    rates = mt5.copy_rates_from_date(symbol, timeframe, date_from_timestamp, count)
    if rates is None:
        logger.error(f"Failed to copy rates for {symbol} from date {date_from}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy rates for {symbol} from date {date_from}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(rates)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df.to_dict('records')

# Copy rates range
@mcp.tool()
def copy_rates_range(
    symbol: str, 
    timeframe: int, 
    date_from: datetime, 
    date_to: datetime
) -> List[Dict[str, Any]]:
    """
    Get bars from a specified symbol and timeframe within the specified date range.
    
    Args:
        symbol: Symbol name
        timeframe: Timeframe (use TIMEFRAME_* constants)
        date_from: Start date for bar retrieval
        date_to: End date for bar retrieval
        
    Returns:
        List[Dict[str, Any]]: List of bars with time, open, high, low, close, tick_volume, spread, and real_volume.
    """
    # Convert datetime to timestamp
    date_from_timestamp = int(date_from.timestamp())
    date_to_timestamp = int(date_to.timestamp())
    
    rates = mt5.copy_rates_range(symbol, timeframe, date_from_timestamp, date_to_timestamp)
    if rates is None:
        logger.error(f"Failed to copy rates for {symbol} in range {date_from} to {date_to}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy rates for {symbol} in range {date_from} to {date_to}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(rates)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df.to_dict('records')

# Copy ticks from position
@mcp.tool()
def copy_ticks_from_pos(
    symbol: str, 
    start_pos: int, 
    count: int, 
    flags: int = mt5.COPY_TICKS_ALL
) -> List[Dict[str, Any]]:
    """
    Get ticks from a specified symbol starting from the specified position.
    
    Args:
        symbol: Symbol name
        start_pos: Initial position for tick retrieval
        count: Number of ticks to retrieve
        flags: Type of requested ticks:
            - mt5.COPY_TICKS_ALL: All ticks (default)
            - mt5.COPY_TICKS_INFO: Ticks containing bid and/or ask price changes
            - mt5.COPY_TICKS_TRADE: Ticks containing last price and volume changes
        
    Returns:
        List[Dict[str, Any]]: List of ticks.
    """
    ticks = mt5.copy_ticks_from(symbol, start_pos, count, flags)
    if ticks is None:
        logger.error(f"Failed to copy ticks for {symbol}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy ticks for {symbol}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(ticks)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    if 'time_msc' in df.columns:
        df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
    
    return df.to_dict('records')

# Copy ticks from date
@mcp.tool()
def copy_ticks_from_date(
    symbol: str, 
    date_from: datetime, 
    count: int, 
    flags: int = mt5.COPY_TICKS_ALL
) -> List[Dict[str, Any]]:
    """
    Get ticks from a specified symbol starting from the specified date.
    
    Args:
        symbol: Symbol name
        date_from: Start date for tick retrieval
        count: Number of ticks to retrieve
        flags: Type of requested ticks
        
    Returns:
        List[Dict[str, Any]]: List of ticks.
    """
    # Convert datetime to timestamp in milliseconds
    date_from_timestamp = int(date_from.timestamp() * 1000)
    
    ticks = mt5.copy_ticks_from(symbol, date_from_timestamp, count, flags)
    if ticks is None:
        logger.error(f"Failed to copy ticks for {symbol} from date {date_from}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy ticks for {symbol} from date {date_from}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(ticks)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    if 'time_msc' in df.columns:
        df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
    
    return df.to_dict('records')

# Copy ticks range
@mcp.tool()
def copy_ticks_range(
    symbol: str, 
    date_from: datetime, 
    date_to: datetime, 
    flags: int = mt5.COPY_TICKS_ALL
) -> List[Dict[str, Any]]:
    """
    Get ticks from a specified symbol within the specified date range.
    
    Args:
        symbol: Symbol name
        date_from: Start date for tick retrieval
        date_to: End date for tick retrieval
        flags: Type of requested ticks
        
    Returns:
        List[Dict[str, Any]]: List of ticks.
    """
    # Convert datetime to timestamp in milliseconds
    date_from_timestamp = int(date_from.timestamp() * 1000)
    date_to_timestamp = int(date_to.timestamp() * 1000)
    
    ticks = mt5.copy_ticks_range(symbol, date_from_timestamp, date_to_timestamp, flags)
    if ticks is None:
        logger.error(f"Failed to copy ticks for {symbol} in range {date_from} to {date_to}, error code: {mt5.last_error()}")
        raise ValueError(f"Failed to copy ticks for {symbol} in range {date_from} to {date_to}")
    
    # Convert numpy array to list of dictionaries
    df = pd.DataFrame(ticks)
    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    if 'time_msc' in df.columns:
        df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
    
    return df.to_dict('records')

# Get last error
@mcp.tool()
def get_last_error() -> Dict[str, Any]:
    """
    Get the last error code and description.
    
    Returns:
        Dict[str, Any]: Last error code and description.
    """
    error_code = mt5.last_error()
    
    error_descriptions = {
        mt5.RES_S_OK: "OK",
        mt5.RES_E_FAIL: "Generic fail",
        mt5.RES_E_INVALID_PARAMS: "Invalid parameters",
        mt5.RES_E_NO_MEMORY: "No memory",
        mt5.RES_E_NOT_FOUND: "Not found",
        mt5.RES_E_INVALID_VERSION: "Invalid version",
        mt5.RES_E_AUTH_FAILED: "Authorization failed",
        mt5.RES_E_UNSUPPORTED: "Unsupported method",
        mt5.RES_E_AUTO_TRADING_DISABLED: "Auto-trading disabled",
        mt5.RES_E_INTERNAL_FAIL: "Internal failure",
        mt5.RES_E_DONE: "Request completed",
        mt5.RES_E_CANCELED: "Request canceled",
    }
    
    error_description = error_descriptions.get(error_code, "Unknown error")
    
    return {
        "code": error_code,
        "description": error_description
    }

# Resource for timeframe constants
@mcp.resource("mt5://timeframes")
def get_timeframes() -> str:
    """
    Get information about available timeframes in MetaTrader 5.
    
    Returns:
        str: Information about available timeframes.
    """
    timeframes = {
        "TIMEFRAME_M1": 1,
        "TIMEFRAME_M2": 2,
        "TIMEFRAME_M3": 3,
        "TIMEFRAME_M4": 4,
        "TIMEFRAME_M5": 5,
        "TIMEFRAME_M6": 6,
        "TIMEFRAME_M10": 10,
        "TIMEFRAME_M12": 12,
        "TIMEFRAME_M15": 15,
        "TIMEFRAME_M20": 20,
        "TIMEFRAME_M30": 30,
        "TIMEFRAME_H1": 60,
        "TIMEFRAME_H2": 120,
        "TIMEFRAME_H3": 180,
        "TIMEFRAME_H4": 240,
        "TIMEFRAME_H6": 360,
        "TIMEFRAME_H8": 480,
        "TIMEFRAME_H12": 720,
        "TIMEFRAME_D1": 1440,
        "TIMEFRAME_W1": 10080,
        "TIMEFRAME_MN1": 43200
    }
    
    result = "Available timeframes in MetaTrader 5:\n\n"
    for name, value in timeframes.items():
        result += f"{name}: {value}\n"
    
    return result

# Resource for tick flag constants
@mcp.resource("mt5://tick_flags")
def get_tick_flags() -> str:
    """
    Get information about tick flags in MetaTrader 5.
    
    Returns:
        str: Information about tick flags.
    """
    tick_flags = {
        "COPY_TICKS_ALL": mt5.COPY_TICKS_ALL,
        "COPY_TICKS_INFO": mt5.COPY_TICKS_INFO,
        "COPY_TICKS_TRADE": mt5.COPY_TICKS_TRADE
    }
    
    result = "Available tick flags in MetaTrader 5:\n\n"
    for name, value in tick_flags.items():
        result += f"{name}: {value}\n"
    
    return result
