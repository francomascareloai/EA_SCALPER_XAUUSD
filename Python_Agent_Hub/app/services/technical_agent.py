"""
Technical Analysis Agent
Provides technical scoring based on market data
Integrates Regime Detection (Hurst + Entropy) for intelligent filtering
"""
import logging
import numpy as np
from typing import Optional, List
from app.models import TechnicalAnalysisRequest, TechnicalAnalysisResponse, SignalType
from app.services.regime_detector import (
    RegimeDetector, 
    KalmanTrendFilter,
    MarketRegime,
    RegimeAnalysis
)
import time

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """
    Technical Analysis Agent with Regime Detection Integration
    
    The Singularity Filter approach:
    1. Check regime FIRST (Hurst + Entropy)
    2. If RANDOM_WALK → reject trade
    3. Adjust score based on regime quality
    4. Use Kalman filter for trend confirmation
    """
    
    def __init__(self):
        self.name = "TechnicalAgent"
        self.regime_detector = RegimeDetector()
        self.kalman_filter = KalmanTrendFilter()
        self._last_regime: Optional[RegimeAnalysis] = None
        self._price_cache: List[float] = []
        logger.info(f"{self.name} initialized with Regime Detection")
    
    def update_price_cache(self, prices: List[float]):
        """Update internal price cache for regime detection"""
        self._price_cache = prices[-200:]  # Keep last 200 prices
    
    def get_regime_filter(self, prices: Optional[List[float]] = None) -> RegimeAnalysis:
        """
        Get current regime analysis
        
        Args:
            prices: Price array (uses cache if None)
        
        Returns:
            RegimeAnalysis with regime, action, and metrics
        """
        if prices is not None:
            self.update_price_cache(prices)
        
        if len(self._price_cache) < 20:
            logger.warning("Insufficient price data for regime detection")
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                action="no_trade",
                size_multiplier=0.0,
                hurst_exponent=0.5,
                shannon_entropy=2.0,
                confidence=0.0,
                strategy_hint="Insufficient data",
                details={}
            )
        
        prices_array = np.array(self._price_cache)
        self._last_regime = self.regime_detector.detect_regime(prices_array)
        return self._last_regime
    
    def should_trade(self, prices: Optional[List[float]] = None) -> bool:
        """
        Check if trading is allowed based on regime
        
        Returns:
            True if regime allows trading, False if RANDOM_WALK
        """
        regime = self.get_regime_filter(prices)
        return regime.regime != MarketRegime.RANDOM_WALK
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on regime
        
        Returns:
            1.0 for PRIME regimes, 0.5 for NOISY, 0.0 for RANDOM
        """
        if self._last_regime is None:
            return 0.5  # Conservative default
        return self._last_regime.size_multiplier
    
    async def analyze(self, request: TechnicalAnalysisRequest) -> TechnicalAnalysisResponse:
        """
        Analyze technical indicators and return score
        
        Enhanced with Regime Detection:
        - Checks regime filter first
        - Adjusts score based on regime quality
        - Uses Kalman filter for trend confirmation
        """
        start_time = time.time()
        
        try:
            # Base technical score (placeholder - implement real TA later)
            base_score = 75
            signal = SignalType.LONG
            confidence = 0.75
            
            # If we have price data in context, run regime detection
            if request.context and 'prices' in request.context:
                prices = request.context['prices']
                
                # Get regime analysis
                regime = self.get_regime_filter(prices)
                
                # Apply regime filter
                if regime.regime == MarketRegime.RANDOM_WALK:
                    # Random walk = no edge, heavily penalize score
                    base_score = max(0, base_score - 30)
                    confidence = confidence * 0.3
                    signal = SignalType.NEUTRAL
                    logger.info(f"Random walk detected - score penalized: {base_score}")
                    
                elif regime.regime in [MarketRegime.PRIME_TRENDING, MarketRegime.PRIME_REVERTING]:
                    # Prime regime = boost confidence
                    confidence = min(1.0, confidence * 1.2)
                    base_score = min(100, base_score + 10)
                    logger.info(f"Prime regime detected - score boosted: {base_score}")
                    
                elif regime.regime in [MarketRegime.NOISY_TRENDING, MarketRegime.NOISY_REVERTING]:
                    # Noisy regime = slight penalty
                    confidence = confidence * 0.9
                    logger.info(f"Noisy regime detected - confidence adjusted")
                
                # Use Kalman trend for signal confirmation
                kalman_trend = self.kalman_filter.get_trend_signal(np.array(prices))
                
                if kalman_trend == "bullish" and signal == SignalType.SHORT:
                    # Conflict - reduce confidence
                    confidence = confidence * 0.7
                elif kalman_trend == "bearish" and signal == SignalType.LONG:
                    # Conflict - reduce confidence
                    confidence = confidence * 0.7
                elif kalman_trend == signal.value.lower().replace("long", "bullish").replace("short", "bearish"):
                    # Agreement - boost confidence
                    confidence = min(1.0, confidence * 1.1)
            
            processing_time = (time.time() - start_time) * 1000
            
            return TechnicalAnalysisResponse(
                req_id=request.req_id,
                timestamp=time.time(),
                tech_subscore=int(base_score),
                signal_type=signal,
                confidence=confidence,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return TechnicalAnalysisResponse(
                req_id=request.req_id,
                timestamp=time.time(),
                tech_subscore=50,  # Neutral on error
                signal_type=SignalType.NEUTRAL,
                confidence=0.0,
                processing_time_ms=processing_time,
                error=str(e)
            )
