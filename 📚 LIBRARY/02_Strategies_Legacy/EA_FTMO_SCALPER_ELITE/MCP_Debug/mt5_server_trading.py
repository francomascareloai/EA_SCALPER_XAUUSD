"""
MetaTrader 5 MCP Server - Trading Functions

This module contains tools and resources for trading operations in MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import the main server instance
from mt5_server import mcp, OrderRequest, OrderResult, Position, HistoryOrder, Deal

logger = logging.getLogger("mt5-mcp-server.trading")

# Send order
@mcp.tool()
def order_send(request: OrderRequest) -> OrderResult:
    """
    Send an order to the trade server.
    
    Args:
        request: Order parameters
        
    Returns:
        OrderResult: Order execution result.
    """
    # Convert request to dictionary
    request_dict = request.model_dump()
    
    # Send order
    result = mt5.order_send(request_dict)
    if result is None:
        logger.error(f"Failed to send order, error code: {mt5.last_error()}")
        raise ValueError("Failed to send order")
    
    # Convert named tuple to dictionary
    result_dict = result._asdict()
    
    # Convert request named tuple to dictionary if needed
    if hasattr(result_dict['request'], '_asdict'):
        result_dict['request'] = result_dict['request']._asdict()
    
    return OrderResult(**result_dict)

# Check order
@mcp.tool()
def order_check(request: OrderRequest) -> Dict[str, Any]:
    """
    Check if an order can be placed with the specified parameters.
    
    Args:
        request: Order parameters
        
    Returns:
        Dict[str, Any]: Order check result.
    """
    # Convert request to dictionary
    request_dict = request.model_dump()
    
    # Check order
    result = mt5.order_check(request_dict)
    if result is None:
        logger.error(f"Failed to check order, error code: {mt5.last_error()}")
        raise ValueError("Failed to check order")
    
    # Convert named tuple to dictionary
    result_dict = result._asdict()
    
    # Convert request named tuple to dictionary if needed
    if hasattr(result_dict['request'], '_asdict'):
        result_dict['request'] = result_dict['request']._asdict()
    
    return result_dict

# Get positions
@mcp.tool()
def positions_get(symbol: Optional[str] = None, group: Optional[str] = None) -> List[Position]:
    """
    Get open positions.
    
    Args:
        symbol: Symbol name. If specified, only positions for this symbol will be returned.
        group: Filter for arranging a group of positions (e.g., "*", "USD*", etc.)
        
    Returns:
        List[Position]: List of open positions.
    """
    if symbol is not None:
        positions = mt5.positions_get(symbol=symbol)
    elif group is not None:
        positions = mt5.positions_get(group=group)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        logger.error(f"Failed to get positions, error code: {mt5.last_error()}")
        return []
    
    result = []
    for position in positions:
        # Convert named tuple to dictionary
        position_dict = position._asdict()
        result.append(Position(**position_dict))
    
    return result

# Get position by ticket
@mcp.tool()
def positions_get_by_ticket(ticket: int) -> Optional[Position]:
    """
    Get an open position by its ticket.
    
    Args:
        ticket: Position ticket
        
    Returns:
        Optional[Position]: Position information or None if not found.
    """
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        logger.error(f"Failed to get position with ticket {ticket}, error code: {mt5.last_error()}")
        return None
    
    # Convert named tuple to dictionary
    position_dict = position[0]._asdict()
    return Position(**position_dict)

# Get orders
@mcp.tool()
def orders_get(symbol: Optional[str] = None, group: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get active orders.
    
    Args:
        symbol: Symbol name. If specified, only orders for this symbol will be returned.
        group: Filter for arranging a group of orders (e.g., "*", "USD*", etc.)
        
    Returns:
        List[Dict[str, Any]]: List of active orders.
    """
    if symbol is not None:
        orders = mt5.orders_get(symbol=symbol)
    elif group is not None:
        orders = mt5.orders_get(group=group)
    else:
        orders = mt5.orders_get()
    
    if orders is None:
        logger.error(f"Failed to get orders, error code: {mt5.last_error()}")
        return []
    
    result = []
    for order in orders:
        # Convert named tuple to dictionary
        order_dict = order._asdict()
        result.append(order_dict)
    
    return result

# Get order by ticket
@mcp.tool()
def orders_get_by_ticket(ticket: int) -> Optional[Dict[str, Any]]:
    """
    Get an active order by its ticket.
    
    Args:
        ticket: Order ticket
        
    Returns:
        Optional[Dict[str, Any]]: Order information or None if not found.
    """
    order = mt5.orders_get(ticket=ticket)
    if order is None or len(order) == 0:
        logger.error(f"Failed to get order with ticket {ticket}, error code: {mt5.last_error()}")
        return None
    
    # Convert named tuple to dictionary
    return order[0]._asdict()

# Get history orders
@mcp.tool()
def history_orders_get(
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    ticket: Optional[int] = None,
    position: Optional[int] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[HistoryOrder]:
    """
    Get orders from history within the specified date range.
    
    Args:
        symbol: Symbol name
        group: Filter for arranging a group of orders
        ticket: Order ticket
        position: Position ticket
        from_date: Start date for order retrieval
        to_date: End date for order retrieval
        
    Returns:
        List[HistoryOrder]: List of historical orders.
    """
    # Convert datetime to timestamp
    from_timestamp = int(from_date.timestamp()) if from_date else None
    to_timestamp = int(to_date.timestamp()) if to_date else None
    
    # Prepare request
    request = {}
    if symbol is not None:
        request["symbol"] = symbol
    if group is not None:
        request["group"] = group
    if ticket is not None:
        request["ticket"] = ticket
    if position is not None:
        request["position"] = position
    if from_timestamp is not None:
        request["from"] = from_timestamp
    if to_timestamp is not None:
        request["to"] = to_timestamp
    
    # Get history orders
    if request:
        orders = mt5.history_orders_get(**request)
    else:
        orders = mt5.history_orders_get()
    
    if orders is None:
        logger.error(f"Failed to get history orders, error code: {mt5.last_error()}")
        return []
    
    result = []
    for order in orders:
        # Convert named tuple to dictionary
        order_dict = order._asdict()
        result.append(HistoryOrder(**order_dict))
    
    return result

# Get history deals
@mcp.tool()
def history_deals_get(
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    ticket: Optional[int] = None,
    position: Optional[int] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[Deal]:
    """
    Get deals from history within the specified date range.
    
    Args:
        symbol: Symbol name
        group: Filter for arranging a group of deals
        ticket: Deal ticket
        position: Position ticket
        from_date: Start date for deal retrieval
        to_date: End date for deal retrieval
        
    Returns:
        List[Deal]: List of historical deals.
    """
    # Convert datetime to timestamp
    from_timestamp = int(from_date.timestamp()) if from_date else None
    to_timestamp = int(to_date.timestamp()) if to_date else None
    
    # Prepare request
    request = {}
    if symbol is not None:
        request["symbol"] = symbol
    if group is not None:
        request["group"] = group
    if ticket is not None:
        request["ticket"] = ticket
    if position is not None:
        request["position"] = position
    if from_timestamp is not None:
        request["from"] = from_timestamp
    if to_timestamp is not None:
        request["to"] = to_timestamp
    
    # Get history deals
    if request:
        deals = mt5.history_deals_get(**request)
    else:
        deals = mt5.history_deals_get()
    
    if deals is None:
        logger.error(f"Failed to get history deals, error code: {mt5.last_error()}")
        return []
    
    result = []
    for deal in deals:
        # Convert named tuple to dictionary
        deal_dict = deal._asdict()
        result.append(Deal(**deal_dict))
    
    return result

# Resource for order types
@mcp.resource("mt5://order_types")
def get_order_types() -> str:
    """
    Get information about order types in MetaTrader 5.
    
    Returns:
        str: Information about order types.
    """
    order_types = {
        "ORDER_TYPE_BUY": mt5.ORDER_TYPE_BUY,
        "ORDER_TYPE_SELL": mt5.ORDER_TYPE_SELL,
        "ORDER_TYPE_BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
        "ORDER_TYPE_SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
        "ORDER_TYPE_BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
        "ORDER_TYPE_SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
        "ORDER_TYPE_BUY_STOP_LIMIT": mt5.ORDER_TYPE_BUY_STOP_LIMIT,
        "ORDER_TYPE_SELL_STOP_LIMIT": mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        "ORDER_TYPE_CLOSE_BY": mt5.ORDER_TYPE_CLOSE_BY
    }
    
    result = "Available order types in MetaTrader 5:\n\n"
    for name, value in order_types.items():
        result += f"{name}: {value}\n"
    
    return result

# Resource for order filling types
@mcp.resource("mt5://order_filling_types")
def get_order_filling_types() -> str:
    """
    Get information about order filling types in MetaTrader 5.
    
    Returns:
        str: Information about order filling types.
    """
    filling_types = {
        "ORDER_FILLING_FOK": mt5.ORDER_FILLING_FOK,
        "ORDER_FILLING_IOC": mt5.ORDER_FILLING_IOC,
        "ORDER_FILLING_RETURN": mt5.ORDER_FILLING_RETURN
    }
    
    result = "Available order filling types in MetaTrader 5:\n\n"
    for name, value in filling_types.items():
        result += f"{name}: {value}\n"
    
    return result

# Resource for order time types
@mcp.resource("mt5://order_time_types")
def get_order_time_types() -> str:
    """
    Get information about order time types in MetaTrader 5.
    
    Returns:
        str: Information about order time types.
    """
    time_types = {
        "ORDER_TIME_GTC": mt5.ORDER_TIME_GTC,
        "ORDER_TIME_DAY": mt5.ORDER_TIME_DAY,
        "ORDER_TIME_SPECIFIED": mt5.ORDER_TIME_SPECIFIED,
        "ORDER_TIME_SPECIFIED_DAY": mt5.ORDER_TIME_SPECIFIED_DAY
    }
    
    result = "Available order time types in MetaTrader 5:\n\n"
    for name, value in time_types.items():
        result += f"{name}: {value}\n"
    
    return result

# Resource for trade request actions
@mcp.resource("mt5://trade_actions")
def get_trade_actions() -> str:
    """
    Get information about trade request actions in MetaTrader 5.
    
    Returns:
        str: Information about trade request actions.
    """
    actions = {
        "TRADE_ACTION_DEAL": mt5.TRADE_ACTION_DEAL,
        "TRADE_ACTION_PENDING": mt5.TRADE_ACTION_PENDING,
        "TRADE_ACTION_SLTP": mt5.TRADE_ACTION_SLTP,
        "TRADE_ACTION_MODIFY": mt5.TRADE_ACTION_MODIFY,
        "TRADE_ACTION_REMOVE": mt5.TRADE_ACTION_REMOVE,
        "TRADE_ACTION_CLOSE_BY": mt5.TRADE_ACTION_CLOSE_BY
    }
    
    result = "Available trade request actions in MetaTrader 5:\n\n"
    for name, value in actions.items():
        result += f"{name}: {value}\n"
    
    return result
