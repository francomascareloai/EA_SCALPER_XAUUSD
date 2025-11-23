"""
Pydantic Models for MQL5 <-> Python Communication
Following PRD v2.0 JSON Schema Specification
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class OrderType(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


# === REQUEST MODELS ===

class TechnicalAnalysisRequest(BaseModel):
    """Request for Technical Analysis Score"""
    schema_version: str = Field(default="1.0", description="Schema version")
    req_id: str = Field(..., description="Unique request ID")
    timestamp: float = Field(..., description="Unix timestamp")
    timeout_ms: int = Field(default=200, description="Max processing time in ms")
    
    # Market Data
    symbol: str = Field(default="XAUUSD", description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (M5, M15, H1, etc)")
    current_price: float = Field(..., description="Current market price")
    
    # Optional Context
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class FundamentalAnalysisRequest(BaseModel):
    """Request for Fundamental Analysis Score"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    timeout_ms: int = Field(default=5000)
    
    symbol: str = Field(default="XAUUSD")
    context: Optional[Dict[str, Any]] = None


class SentimentAnalysisRequest(BaseModel):
    """Request for Sentiment Analysis Score"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    timeout_ms: int = Field(default=3000)
    
    symbol: str = Field(default="XAUUSD")
    context: Optional[Dict[str, Any]] = None


class ReasoningRequest(BaseModel):
    """Request for LLM Reasoning String"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    timeout_ms: int = Field(default=10000)
    
    # Trade Context
    symbol: str = Field(default="XAUUSD")
    direction: OrderType = Field(...)
    entry_price: float = Field(...)
    sl_price: float = Field(...)
    tp_price: float = Field(...)
    
    # Scores
    tech_score: int = Field(..., ge=0, le=100)
    fund_score: int = Field(..., ge=0, le=100)
    sent_score: int = Field(..., ge=0, le=100)
    total_score: int = Field(..., ge=0, le=100)
    
    # Context
    context: Optional[Dict[str, Any]] = None


# === RESPONSE MODELS ===

class TechnicalAnalysisResponse(BaseModel):
    """Response for Technical Analysis"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    
    # Core Fields
    tech_subscore: int = Field(..., ge=0, le=100, description="Technical score 0-100")
    signal_type: SignalType = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    
    # Optional Fields
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    
    # Metadata
    processing_time_ms: float = Field(...)
    error: Optional[str] = None


class FundamentalAnalysisResponse(BaseModel):
    """Response for Fundamental Analysis"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    
    fund_subscore: int = Field(..., ge=0, le=100)
    bias: SignalType = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # News Impact
    news_impact: Optional[str] = None
    key_events: Optional[list[str]] = None
    
    processing_time_ms: float = Field(...)
    error: Optional[str] = None


class SentimentAnalysisResponse(BaseModel):
    """Response for Sentiment Analysis"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    
    sent_subscore: int = Field(..., ge=0, le=100)
    market_sentiment: SignalType = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Sentiment Metrics
    bullish_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    bearish_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    processing_time_ms: float = Field(...)
    error: Optional[str] = None


class ReasoningResponse(BaseModel):
    """Response for LLM Reasoning"""
    schema_version: str = Field(default="1.0")
    req_id: str = Field(...)
    timestamp: float = Field(...)
    
    reasoning_string: str = Field(..., description="Natural language explanation")
    trade_class: str = Field(..., description="A+, A, B, C, D")
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Optional Warnings
    warnings: Optional[list[str]] = None
    
    processing_time_ms: float = Field(...)
    error: Optional[str] = None


# === HEALTH CHECK ===

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(default="healthy")
    timestamp: float = Field(...)
    version: str = Field(default="2.0.0")
    uptime_seconds: Optional[float] = None
