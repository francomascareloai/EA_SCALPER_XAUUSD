#!/usr/bin/env python3
"""
ea_logic_python.py - Python port of EA_SCALPER_XAUUSD trading logic.

BATCH 4: Implements the EA's decision logic in Python for shadow testing.

This mirrors the logic in:
- CConfluenceScorer.mqh (signal scoring)
- CRegimeDetector.mqh (regime detection)
- CSessionFilter.mqh (session filtering)
- FTMO_RiskManager.mqh (position sizing)

Usage:
    from scripts.backtest.strategies.ea_logic_python import EALogic
    
    ea = EALogic()
    signal = ea.evaluate(bar_data, regime, session)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class Regime(Enum):
    TRENDING = 1
    RANGING = 0
    REVERTING = -1
    RANDOM = -2


class Session(Enum):
    ASIA = 'ASIA'
    LONDON = 'LONDON'
    OVERLAP = 'OVERLAP'
    NY = 'NY'
    CLOSE = 'CLOSE'


class SignalType(Enum):
    BUY = 1
    SELL = -1
    NONE = 0


@dataclass
class Signal:
    """Trade signal with all scoring components."""
    signal_type: SignalType
    confidence: float  # 0-100
    confluence_score: float  # 0-100
    regime_score: float
    session_score: float
    mtf_alignment: float
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_reward: float


@dataclass
class BarData:
    """Single bar data structure."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class RegimeDetector:
    """
    Port of CRegimeDetector.mqh
    
    Uses Hurst exponent to classify market regime.
    """
    
    HURST_TRENDING = 0.55
    HURST_REVERTING = 0.45
    
    def __init__(self, window: int = 100):
        self.window = window
        self.prices = []
        self.current_regime = Regime.RANGING
        self.hurst = 0.5
        self.entropy = 0.5
    
    def update(self, price: float) -> Regime:
        """Update with new price and return current regime."""
        self.prices.append(price)
        
        if len(self.prices) > self.window * 2:
            self.prices = self.prices[-self.window * 2:]
        
        if len(self.prices) >= self.window:
            self.hurst = self._calculate_hurst()
            self.entropy = self._calculate_entropy()
            self.current_regime = self._classify_regime()
        
        return self.current_regime
    
    def _calculate_hurst(self) -> float:
        """Calculate Hurst exponent using R/S method."""
        prices = np.array(self.prices[-self.window:])
        returns = np.diff(np.log(prices + 1e-10))
        
        if len(returns) < 2:
            return 0.5
        
        mean_ret = np.mean(returns)
        deviations = returns - mean_ret
        cum_dev = np.cumsum(deviations)
        
        R = np.max(cum_dev) - np.min(cum_dev)
        S = np.std(returns, ddof=1)
        
        if S > 1e-10 and R > 1e-10:
            RS = R / S
            n = len(returns)
            H = np.log(RS) / np.log(n)
            return np.clip(H, 0, 1)
        
        return 0.5
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of returns."""
        prices = np.array(self.prices[-50:])
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 10:
            return 0.5
        
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        
        H = -np.sum(hist * np.log(hist))
        max_H = np.log(10)
        
        return H / max_H
    
    def _classify_regime(self) -> Regime:
        """Classify regime based on Hurst."""
        if self.hurst > self.HURST_TRENDING:
            return Regime.TRENDING
        elif self.hurst < self.HURST_REVERTING:
            return Regime.REVERTING
        else:
            # Check entropy for random walk
            if self.entropy > 0.9:
                return Regime.RANDOM
            return Regime.RANGING


class SessionFilter:
    """
    Port of CSessionFilter.mqh
    
    Classifies trading sessions based on hour (UTC).
    """
    
    SESSION_HOURS = {
        Session.ASIA: (0, 7),
        Session.LONDON: (7, 12),
        Session.OVERLAP: (12, 16),
        Session.NY: (16, 21),
        Session.CLOSE: (21, 24),
    }
    
    SESSION_SCORES = {
        Session.ASIA: 0.5,
        Session.LONDON: 0.8,
        Session.OVERLAP: 1.0,
        Session.NY: 0.7,
        Session.CLOSE: 0.3,
    }
    
    @classmethod
    def get_session(cls, hour: int) -> Session:
        """Get session from hour (UTC)."""
        for session, (start, end) in cls.SESSION_HOURS.items():
            if start <= hour < end:
                return session
        return Session.CLOSE
    
    @classmethod
    def get_session_score(cls, session: Session) -> float:
        """Get trading quality score for session."""
        return cls.SESSION_SCORES.get(session, 0.5)


class ConfluenceScorer:
    """
    Port of CConfluenceScorer.mqh
    
    Scores trade setups based on multiple factors.
    """
    
    def __init__(self):
        self.weights = {
            'regime': 0.25,
            'session': 0.15,
            'mtf': 0.20,
            'rsi': 0.15,
            'structure': 0.15,
            'ml': 0.10
        }
    
    def score(self, factors: Dict[str, float]) -> float:
        """
        Calculate confluence score.
        
        Args:
            factors: Dict with keys matching self.weights
        
        Returns:
            Score 0-100
        """
        total = 0
        weight_sum = 0
        
        for key, weight in self.weights.items():
            if key in factors:
                total += factors[key] * weight * 100
                weight_sum += weight
        
        if weight_sum > 0:
            return total / weight_sum
        return 0


class RiskManager:
    """
    Port of FTMO_RiskManager.mqh
    
    Calculates position sizing using Kelly criterion.
    """
    
    # FTMO limits
    DAILY_DD_LIMIT = 0.05  # 5%
    TOTAL_DD_LIMIT = 0.10  # 10%
    
    def __init__(self, account_balance: float = 100000):
        self.account_balance = account_balance
        self.daily_equity_high = account_balance
        self.total_equity_high = account_balance
        
        # Kelly tracking
        self.wins = 0
        self.losses = 0
        self.total_win = 0
        self.total_loss = 0
    
    def calculate_lot_size(self, sl_pips: float, regime_score: float = 1.0,
                           session_score: float = 1.0) -> float:
        """
        Calculate lot size with adaptive Kelly.
        
        Args:
            sl_pips: Stop loss in pips
            regime_score: Regime quality multiplier (0-1)
            session_score: Session quality multiplier (0-1)
        
        Returns:
            Lot size
        """
        # Base risk: 1% of account
        base_risk = 0.01 * self.account_balance
        
        # Kelly adjustment
        kelly = self._calculate_kelly()
        
        # Adaptive factors
        risk_amount = base_risk * kelly * regime_score * session_score
        
        # Convert to lots (XAUUSD: $10/pip for 1 lot)
        pip_value = 10.0
        lot_size = risk_amount / (sl_pips * pip_value)
        
        # Clamp to reasonable range
        return np.clip(lot_size, 0.01, 5.0)
    
    def _calculate_kelly(self) -> float:
        """Calculate Kelly fraction."""
        total = self.wins + self.losses
        
        if total < 30:
            return 0.25  # Conservative until enough data
        
        win_rate = self.wins / total
        
        if self.total_loss == 0:
            return 0.5
        
        avg_win = self.total_win / max(1, self.wins)
        avg_loss = abs(self.total_loss) / max(1, self.losses)
        
        if avg_loss == 0:
            return 0.5
        
        payoff = avg_win / avg_loss
        kelly = (win_rate * payoff - (1 - win_rate)) / payoff
        
        # Use half Kelly for safety
        return np.clip(kelly * 0.5, 0.1, 0.5)
    
    def update_trade_result(self, pnl: float):
        """Update Kelly tracking with trade result."""
        if pnl > 0:
            self.wins += 1
            self.total_win += pnl
        else:
            self.losses += 1
            self.total_loss += pnl


class EALogic:
    """
    Main EA trading logic.
    
    Combines all components to generate trade signals.
    """
    
    # Minimum confluence for trade
    MIN_CONFLUENCE = 65
    
    # Risk:Reward targets
    MIN_RR = 1.5
    TARGET_RR = 2.5
    
    def __init__(self, account_balance: float = 100000):
        self.regime_detector = RegimeDetector()
        self.confluence_scorer = ConfluenceScorer()
        self.risk_manager = RiskManager(account_balance)
        
        # State
        self.last_signal = SignalType.NONE
        self.bars_since_signal = 0
    
    def evaluate(self, bar: BarData, hour: int, 
                 rsi: float = 50, ml_prob: float = 0.5,
                 mtf_alignment: float = 0.5) -> Optional[Signal]:
        """
        Evaluate current bar for trade signal.
        
        Args:
            bar: Current bar data
            hour: Current hour (UTC)
            rsi: RSI value (0-100)
            ml_prob: ML model probability
            mtf_alignment: MTF alignment score (0-1)
        
        Returns:
            Signal if conditions met, None otherwise
        """
        # Update regime
        regime = self.regime_detector.update(bar.close)
        
        # Skip RANDOM regime
        if regime == Regime.RANDOM:
            return None
        
        # Get session
        session = SessionFilter.get_session(hour)
        session_score = SessionFilter.get_session_score(session)
        
        # Calculate regime score
        if regime == Regime.TRENDING:
            regime_score = 1.0
        elif regime == Regime.REVERTING:
            regime_score = 0.7
        else:
            regime_score = 0.5
        
        # Determine direction
        direction = self._get_direction(bar, rsi, ml_prob, regime)
        
        if direction == SignalType.NONE:
            return None
        
        # Calculate confluence
        factors = {
            'regime': regime_score,
            'session': session_score,
            'mtf': mtf_alignment,
            'rsi': self._rsi_score(rsi, direction),
            'structure': 0.5,  # Placeholder
            'ml': ml_prob if direction == SignalType.BUY else (1 - ml_prob)
        }
        
        confluence = self.confluence_scorer.score(factors)
        
        if confluence < self.MIN_CONFLUENCE:
            return None
        
        # Calculate entry, SL, TP
        entry = bar.close
        atr = self._estimate_atr(bar)
        
        if direction == SignalType.BUY:
            sl = entry - atr * 1.5
            tp = entry + atr * self.TARGET_RR * 1.5
        else:
            sl = entry + atr * 1.5
            tp = entry - atr * self.TARGET_RR * 1.5
        
        sl_pips = abs(entry - sl) / 0.01  # XAUUSD pip = $0.01
        
        # Calculate lot size
        lot_size = self.risk_manager.calculate_lot_size(
            sl_pips, regime_score, session_score
        )
        
        rr = abs(tp - entry) / abs(entry - sl)
        
        return Signal(
            signal_type=direction,
            confidence=confluence,
            confluence_score=confluence,
            regime_score=regime_score,
            session_score=session_score,
            mtf_alignment=mtf_alignment,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            lot_size=lot_size,
            risk_reward=rr
        )
    
    def _get_direction(self, bar: BarData, rsi: float, 
                       ml_prob: float, regime: Regime) -> SignalType:
        """Determine trade direction."""
        # Simple logic - in real EA this is more complex
        
        if regime == Regime.TRENDING:
            # Follow trend
            if ml_prob > 0.65:
                return SignalType.BUY
            elif ml_prob < 0.35:
                return SignalType.SELL
        
        elif regime == Regime.REVERTING:
            # Counter-trend on oversold/overbought
            if rsi < 30 and ml_prob > 0.5:
                return SignalType.BUY
            elif rsi > 70 and ml_prob < 0.5:
                return SignalType.SELL
        
        return SignalType.NONE
    
    def _rsi_score(self, rsi: float, direction: SignalType) -> float:
        """Score RSI for direction."""
        if direction == SignalType.BUY:
            if rsi < 30:
                return 1.0
            elif rsi < 50:
                return 0.7
            else:
                return 0.3
        else:
            if rsi > 70:
                return 1.0
            elif rsi > 50:
                return 0.7
            else:
                return 0.3
    
    def _estimate_atr(self, bar: BarData) -> float:
        """Estimate ATR from single bar."""
        return (bar.high - bar.low) * 2  # Rough estimate


if __name__ == '__main__':
    # Quick test
    ea = EALogic()
    
    bar = BarData(
        timestamp=0,
        open=2000.0,
        high=2005.0,
        low=1995.0,
        close=2003.0
    )
    
    signal = ea.evaluate(bar, hour=14, rsi=35, ml_prob=0.7, mtf_alignment=0.8)
    
    if signal:
        print(f"Signal: {signal.signal_type.name}")
        print(f"Confluence: {signal.confluence_score:.1f}")
        print(f"Entry: {signal.entry_price:.2f}")
        print(f"SL: {signal.sl_price:.2f}")
        print(f"TP: {signal.tp_price:.2f}")
        print(f"Lot: {signal.lot_size:.2f}")
    else:
        print("No signal")
