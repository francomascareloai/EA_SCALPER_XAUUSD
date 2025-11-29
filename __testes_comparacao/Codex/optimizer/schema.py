from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any


@dataclass
class ParamSpace:
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    stop_loss_min: float = 50.0
    stop_loss_max: float = 600.0
    take_profit_min: float = 50.0
    take_profit_max: float = 1200.0
    atr_multiplier_min: float = 0.5
    atr_multiplier_max: float = 5.0
    risk_factor_min: float = 0.1
    risk_factor_max: float = 3.0
    session_start_default: str = "08:00"
    session_end_default: str = "17:00"


@dataclass
class Metrics:
    net_profit: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    sharpe_ratio: float


@dataclass
class TrialParams:
    StopLoss: float
    TakeProfit: float
    ATR_Multiplier: float
    RiskFactor: float
    SessionStart: str
    SessionEnd: str
    Lots: float = 0.10
    MagicNumber: int = 8888

    def to_json(self) -> Dict[str, object]:
        return {
            "Lots": self.Lots,
            "StopLoss": round(float(self.StopLoss), 4),
            "TakeProfit": round(float(self.TakeProfit), 4),
            "RiskFactor": round(float(self.RiskFactor), 4),
            "ATR_Multiplier": round(float(self.ATR_Multiplier), 4),
            "SessionStart": self.SessionStart,
            "SessionEnd": self.SessionEnd,
            "MagicNumber": self.MagicNumber,
        }


@dataclass
class OptimizeResult:
    params: TrialParams
    metrics_pred: Metrics
    score: float
    backend: str
    n_trials: int
    note: Optional[str] = None
    trials: Optional[List[Dict[str, Any]]] = None
