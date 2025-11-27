"""
Regime Detector Module
Implements Hurst Exponent, Shannon Entropy, and Kalman Filter
for market regime detection (Trending, Mean-Reverting, Random Walk)

Based on Singularity Architect specifications.
"""
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification"""
    PRIME_TRENDING = "prime_trending"       # H > 0.55, S < 1.5
    NOISY_TRENDING = "noisy_trending"       # H > 0.55, S >= 1.5
    PRIME_REVERTING = "prime_reverting"     # H < 0.45, S < 1.5
    NOISY_REVERTING = "noisy_reverting"     # H < 0.45, S >= 1.5
    RANDOM_WALK = "random_walk"             # 0.45 <= H <= 0.55
    UNKNOWN = "unknown"                      # Insufficient data


class TradingAction(str, Enum):
    """Recommended trading action based on regime"""
    FULL_SIZE = "full_size"                 # Trade with full position
    HALF_SIZE = "half_size"                 # Trade with 50% position
    NO_TRADE = "no_trade"                   # Do not trade
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result"""
    regime: MarketRegime
    action: TradingAction
    size_multiplier: float
    hurst_exponent: float
    shannon_entropy: float
    confidence: float
    strategy_hint: str
    details: Dict


class RegimeDetector:
    """
    Market Regime Detection using Hurst Exponent and Shannon Entropy
    
    The Singularity Filter combines both metrics:
    - Hurst > 0.55 + Low Entropy = PRIME_TRENDING (full size)
    - Hurst > 0.55 + High Entropy = NOISY_TRENDING (half size)
    - Hurst < 0.45 + Low Entropy = PRIME_REVERTING (full size)
    - Hurst < 0.45 + High Entropy = NOISY_REVERTING (half size)
    - Hurst ~ 0.5 = RANDOM_WALK (no trade)
    """
    
    def __init__(
        self,
        hurst_window: int = 100,
        entropy_window: int = 100,
        entropy_bins: int = 10,
        hurst_trending_threshold: float = 0.55,
        hurst_reverting_threshold: float = 0.45,
        entropy_low_threshold: float = 1.5,
        entropy_high_threshold: float = 2.5
    ):
        self.hurst_window = hurst_window
        self.entropy_window = entropy_window
        self.entropy_bins = entropy_bins
        self.hurst_trending = hurst_trending_threshold
        self.hurst_reverting = hurst_reverting_threshold
        self.entropy_low = entropy_low_threshold
        self.entropy_high = entropy_high_threshold
        
        logger.info(f"RegimeDetector initialized: hurst_window={hurst_window}, entropy_window={entropy_window}")
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Hurst Exponent using R/S Analysis
        
        H > 0.5: Trending (persistent)
        H < 0.5: Mean-reverting (anti-persistent)
        H = 0.5: Random walk
        
        Returns: (hurst_value, standard_error)
        """
        if len(prices) < 20:
            logger.warning("Insufficient data for Hurst calculation")
            return 0.5, 0.0
        
        try:
            # Use log prices for better numerical stability
            log_prices = np.log(prices)
            
            # Calculate returns
            returns = np.diff(log_prices)
            
            # R/S Analysis with multiple window sizes
            max_k = min(len(returns) // 2, 50)
            min_k = 10
            
            if max_k < min_k:
                return 0.5, 0.0
            
            rs_values = []
            window_sizes = []
            
            for n in range(min_k, max_k + 1):
                # Divide series into non-overlapping subseries
                num_subseries = len(returns) // n
                if num_subseries < 1:
                    continue
                
                rs_list = []
                for i in range(num_subseries):
                    subseries = returns[i * n:(i + 1) * n]
                    
                    # Mean-adjusted cumulative deviate
                    mean_r = np.mean(subseries)
                    cumdev = np.cumsum(subseries - mean_r)
                    
                    # Range
                    R = np.max(cumdev) - np.min(cumdev)
                    
                    # Standard deviation
                    S = np.std(subseries, ddof=1)
                    
                    if S > 0:
                        rs_list.append(R / S)
                
                if rs_list:
                    rs_values.append(np.mean(rs_list))
                    window_sizes.append(n)
            
            if len(rs_values) < 3:
                return 0.5, 0.0
            
            # Linear regression on log-log scale
            log_n = np.log(window_sizes)
            log_rs = np.log(rs_values)
            
            # Fit: log(R/S) = H * log(n) + c
            coeffs = np.polyfit(log_n, log_rs, 1)
            hurst = coeffs[0]
            
            # Calculate standard error
            y_pred = np.polyval(coeffs, log_n)
            residuals = log_rs - y_pred
            std_error = np.std(residuals) / np.sqrt(len(residuals))
            
            # Clamp to reasonable range
            hurst = np.clip(hurst, 0.0, 1.0)
            
            return float(hurst), float(std_error)
            
        except Exception as e:
            logger.error(f"Hurst calculation error: {e}")
            return 0.5, 0.0
    
    def calculate_shannon_entropy(self, returns: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Shannon Entropy of returns distribution
        
        Low entropy: Structured, predictable moves
        High entropy: Noisy, random moves
        
        Returns: (entropy_value, interpretation)
        """
        if len(returns) < 10:
            logger.warning("Insufficient data for entropy calculation")
            return 2.0, "unknown"
        
        try:
            # Remove NaN and infinite values
            returns = returns[np.isfinite(returns)]
            
            if len(returns) < 10:
                return 2.0, "unknown"
            
            # Create histogram (probability distribution)
            hist, _ = np.histogram(returns, bins=self.entropy_bins, density=True)
            
            # Normalize to get probabilities
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            
            # Remove zero probabilities (undefined log)
            hist = hist[hist > 0]
            
            # Shannon entropy: H = -sum(p * log(p))
            entropy = -np.sum(hist * np.log2(hist))
            
            # Interpretation
            if entropy < self.entropy_low:
                interpretation = "low_noise"
            elif entropy < self.entropy_high:
                interpretation = "medium_noise"
            else:
                interpretation = "high_noise"
            
            return float(entropy), interpretation
            
        except Exception as e:
            logger.error(f"Entropy calculation error: {e}")
            return 2.0, "unknown"
    
    def detect_regime(self, prices: np.ndarray) -> RegimeAnalysis:
        """
        Main regime detection using combined Hurst + Entropy filter
        
        Args:
            prices: Array of close prices (at least 100 values recommended)
        
        Returns:
            RegimeAnalysis with regime, action, and details
        """
        if len(prices) < 20:
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                action=TradingAction.NO_TRADE,
                size_multiplier=0.0,
                hurst_exponent=0.5,
                shannon_entropy=2.0,
                confidence=0.0,
                strategy_hint="Insufficient data",
                details={"error": "Need at least 20 price points"}
            )
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate Hurst Exponent
        hurst, hurst_se = self.calculate_hurst_exponent(prices[-self.hurst_window:])
        
        # Calculate Shannon Entropy
        entropy, entropy_interp = self.calculate_shannon_entropy(returns[-self.entropy_window:])
        
        # Determine regime using Singularity Filter
        regime, action, size_mult, strategy, confidence = self._classify_regime(
            hurst, entropy, hurst_se
        )
        
        return RegimeAnalysis(
            regime=regime,
            action=action,
            size_multiplier=size_mult,
            hurst_exponent=hurst,
            shannon_entropy=entropy,
            confidence=confidence,
            strategy_hint=strategy,
            details={
                "hurst_std_error": hurst_se,
                "entropy_interpretation": entropy_interp,
                "hurst_window": self.hurst_window,
                "entropy_window": self.entropy_window,
                "prices_analyzed": len(prices)
            }
        )
    
    def _classify_regime(
        self, 
        hurst: float, 
        entropy: float,
        hurst_se: float
    ) -> Tuple[MarketRegime, TradingAction, float, str, float]:
        """
        Classify market regime based on Hurst and Entropy
        
        Returns: (regime, action, size_multiplier, strategy_hint, confidence)
        """
        # Base confidence from Hurst standard error
        base_confidence = max(0.0, 1.0 - hurst_se * 5)
        
        # Random walk detection (no edge)
        if self.hurst_reverting <= hurst <= self.hurst_trending:
            return (
                MarketRegime.RANDOM_WALK,
                TradingAction.NO_TRADE,
                0.0,
                "Random walk detected - no statistical edge. Avoid trading.",
                base_confidence * 0.9
            )
        
        # Trending regime (H > 0.55)
        if hurst > self.hurst_trending:
            if entropy < self.entropy_low:
                return (
                    MarketRegime.PRIME_TRENDING,
                    TradingAction.FULL_SIZE,
                    1.0,
                    "Strong trending regime with low noise. Use momentum/breakout strategies with full size.",
                    base_confidence * 0.95
                )
            else:
                return (
                    MarketRegime.NOISY_TRENDING,
                    TradingAction.HALF_SIZE,
                    0.5,
                    "Trending but noisy. Use trend strategies with reduced size and wider stops.",
                    base_confidence * 0.7
                )
        
        # Mean-reverting regime (H < 0.45)
        if hurst < self.hurst_reverting:
            if entropy < self.entropy_low:
                return (
                    MarketRegime.PRIME_REVERTING,
                    TradingAction.FULL_SIZE,
                    1.0,
                    "Strong mean-reversion regime. Use contrarian/fade strategies with full size.",
                    base_confidence * 0.95
                )
            else:
                return (
                    MarketRegime.NOISY_REVERTING,
                    TradingAction.HALF_SIZE,
                    0.5,
                    "Mean-reverting but noisy. Use reversion strategies with reduced size.",
                    base_confidence * 0.7
                )
        
        # Fallback
        return (
            MarketRegime.UNKNOWN,
            TradingAction.NO_TRADE,
            0.0,
            "Unable to classify regime",
            0.0
        )
    
    def get_regime_score_adjustment(self, regime: MarketRegime) -> int:
        """
        Get score adjustment based on regime
        
        Returns: Score modifier (-30 to +10)
        """
        adjustments = {
            MarketRegime.PRIME_TRENDING: 10,
            MarketRegime.PRIME_REVERTING: 10,
            MarketRegime.NOISY_TRENDING: 0,
            MarketRegime.NOISY_REVERTING: 0,
            MarketRegime.RANDOM_WALK: -30,
            MarketRegime.UNKNOWN: -20
        }
        return adjustments.get(regime, 0)


class KalmanTrendFilter:
    """
    Kalman Filter for adaptive trend estimation
    Superior to Moving Averages - no fixed lag
    """
    
    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 1.0
    ):
        """
        Initialize Kalman Filter
        
        Args:
            process_variance: How much the true price changes (Q)
            measurement_variance: How noisy our measurements are (R)
        """
        self.Q = process_variance
        self.R = measurement_variance
        
        # State variables (will be initialized on first update)
        self.x = None  # Estimated price
        self.P = 1.0   # Estimation error covariance
        self.initialized = False
        
        logger.info(f"KalmanTrendFilter initialized: Q={process_variance}, R={measurement_variance}")
    
    def reset(self):
        """Reset filter state"""
        self.x = None
        self.P = 1.0
        self.initialized = False
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update filter with new price measurement
        
        Args:
            measurement: New price observation
        
        Returns:
            (estimated_price, velocity_estimate)
        """
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement, 0.0
        
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        # Estimate velocity (change from prediction)
        velocity = measurement - x_pred
        
        return float(self.x), float(velocity)
    
    def filter_series(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to entire price series
        
        Returns:
            (filtered_prices, velocities)
        """
        self.reset()
        
        filtered = np.zeros_like(prices, dtype=float)
        velocities = np.zeros_like(prices, dtype=float)
        
        for i, price in enumerate(prices):
            filtered[i], velocities[i] = self.update(price)
        
        return filtered, velocities
    
    def get_trend_signal(self, prices: np.ndarray, threshold: float = 0.1) -> str:
        """
        Get trend direction from Kalman-filtered velocity
        
        Args:
            prices: Recent price array
            threshold: Velocity threshold for trend detection
        
        Returns:
            "bullish", "bearish", or "neutral"
        """
        if len(prices) < 5:
            return "neutral"
        
        _, velocities = self.filter_series(prices)
        
        # Average recent velocity
        avg_velocity = np.mean(velocities[-5:])
        
        # Normalize by price level
        norm_velocity = avg_velocity / prices[-1] * 100
        
        if norm_velocity > threshold:
            return "bullish"
        elif norm_velocity < -threshold:
            return "bearish"
        else:
            return "neutral"


# Singleton instance for easy access
_regime_detector: Optional[RegimeDetector] = None
_kalman_filter: Optional[KalmanTrendFilter] = None


def get_regime_detector() -> RegimeDetector:
    """Get or create regime detector singleton"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector


def get_kalman_filter() -> KalmanTrendFilter:
    """Get or create Kalman filter singleton"""
    global _kalman_filter
    if _kalman_filter is None:
        _kalman_filter = KalmanTrendFilter()
    return _kalman_filter
