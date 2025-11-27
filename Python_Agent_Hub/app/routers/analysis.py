"""
Analysis Router
Handles all analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from app.models import (
    TechnicalAnalysisRequest,
    TechnicalAnalysisResponse,
    FundamentalAnalysisRequest,
    FundamentalAnalysisResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    RegimeDetectionRequest,
    RegimeDetectionResponse,
    MarketRegimeType,
    TradingActionType
)
from app.services.technical_agent import TechnicalAgent
from app.services.regime_detector import (
    RegimeDetector, 
    KalmanTrendFilter,
    get_regime_detector,
    get_kalman_filter
)
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Initialize Agents
technical_agent = TechnicalAgent()


@router.post("/technical", response_model=TechnicalAnalysisResponse)
async def technical_analysis(request: TechnicalAnalysisRequest):
    """
    Technical Analysis Endpoint
    Fast Lane: < 200ms target
    """
    logger.info(f"Technical analysis request: {request.req_id}")
    
    try:
        response = await technical_agent.analyze(request)
        
        # Check timeout
        if response.processing_time_ms > request.timeout_ms:
            logger.warning(f"Technical analysis exceeded timeout: {response.processing_time_ms}ms > {request.timeout_ms}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fundamental", response_model=FundamentalAnalysisResponse)
async def fundamental_analysis(request: FundamentalAnalysisRequest):
    """
    Fundamental Analysis Endpoint
    Slow Lane: Can take several seconds
    """
    logger.info(f"Fundamental analysis request: {request.req_id}")
    
    # Placeholder for Phase 2
    return FundamentalAnalysisResponse(
        req_id=request.req_id,
        timestamp=time.time(),
        fund_subscore=50,
        bias="neutral",
        confidence=0.5,
        processing_time_ms=100.0,
        error=None
    )


@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def sentiment_analysis(request: SentimentAnalysisRequest):
    """
    Sentiment Analysis Endpoint
    Slow Lane: Can take several seconds
    """
    logger.info(f"Sentiment analysis request: {request.req_id}")
    
    # Placeholder for Phase 2
    return SentimentAnalysisResponse(
        req_id=request.req_id,
        timestamp=time.time(),
        sent_subscore=50,
        market_sentiment="neutral",
        confidence=0.5,
        processing_time_ms=100.0,
        error=None
    )


@router.post("/regime", response_model=RegimeDetectionResponse)
async def regime_detection(request: RegimeDetectionRequest):
    """
    Regime Detection Endpoint
    
    Analyzes market regime using Hurst Exponent and Shannon Entropy.
    Returns trading action recommendation based on detected regime.
    
    Regimes:
    - PRIME_TRENDING: Strong trend, low noise -> Full size trading
    - NOISY_TRENDING: Trend with noise -> Half size trading
    - PRIME_REVERTING: Mean reversion, low noise -> Full size reverting
    - NOISY_REVERTING: Mean reversion with noise -> Half size reverting
    - RANDOM_WALK: No statistical edge -> NO TRADE
    """
    logger.info(f"Regime detection request: {request.req_id}")
    start_time = time.time()
    
    try:
        # Get regime detector (singleton)
        detector = get_regime_detector()
        kalman = get_kalman_filter()
        
        # Configure windows if different from default
        if request.hurst_window != 100:
            detector.hurst_window = request.hurst_window
        if request.entropy_window != 100:
            detector.entropy_window = request.entropy_window
        
        # Convert prices to numpy array
        prices = np.array(request.prices)
        
        # Detect regime
        analysis = detector.detect_regime(prices)
        
        # Get Kalman trend
        kalman_trend = kalman.get_trend_signal(prices)
        
        # Get score adjustment
        score_adjustment = detector.get_regime_score_adjustment(analysis.regime)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Map enum values
        regime_map = {
            "prime_trending": MarketRegimeType.PRIME_TRENDING,
            "noisy_trending": MarketRegimeType.NOISY_TRENDING,
            "prime_reverting": MarketRegimeType.PRIME_REVERTING,
            "noisy_reverting": MarketRegimeType.NOISY_REVERTING,
            "random_walk": MarketRegimeType.RANDOM_WALK,
            "unknown": MarketRegimeType.UNKNOWN
        }
        
        action_map = {
            "full_size": TradingActionType.FULL_SIZE,
            "half_size": TradingActionType.HALF_SIZE,
            "no_trade": TradingActionType.NO_TRADE,
            "unknown": TradingActionType.UNKNOWN
        }
        
        return RegimeDetectionResponse(
            req_id=request.req_id,
            timestamp=time.time(),
            regime=regime_map.get(analysis.regime.value, MarketRegimeType.UNKNOWN),
            action=action_map.get(analysis.action.value, TradingActionType.UNKNOWN),
            size_multiplier=analysis.size_multiplier,
            hurst_exponent=analysis.hurst_exponent,
            shannon_entropy=analysis.shannon_entropy,
            confidence=analysis.confidence,
            strategy_hint=analysis.strategy_hint,
            score_adjustment=score_adjustment,
            kalman_trend=kalman_trend,
            processing_time_ms=processing_time,
            error=None,
            details=analysis.details
        )
        
    except Exception as e:
        logger.error(f"Regime detection error: {e}")
        processing_time = (time.time() - start_time) * 1000
        
        return RegimeDetectionResponse(
            req_id=request.req_id,
            timestamp=time.time(),
            regime=MarketRegimeType.UNKNOWN,
            action=TradingActionType.NO_TRADE,
            size_multiplier=0.0,
            hurst_exponent=0.5,
            shannon_entropy=2.0,
            confidence=0.0,
            strategy_hint="Error during analysis",
            score_adjustment=-20,
            kalman_trend=None,
            processing_time_ms=processing_time,
            error=str(e),
            details=None
        )
