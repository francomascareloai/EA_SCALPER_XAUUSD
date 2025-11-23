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
    SentimentAnalysisResponse
)
from app.services.technical_agent import TechnicalAgent
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
