from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class NormCalibrator:
    ranges: Dict[str, tuple]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, q_lo: float = 0.05, q_hi: float = 0.95) -> "NormCalibrator":
        metrics = [
            "profit_factor",
            "sharpe_ratio",
            "win_rate_pct",
            "net_profit",
            "max_drawdown_pct",
        ]
        ranges = {}
        for m in metrics:
            lo = float(df[m].quantile(q_lo))
            hi = float(df[m].quantile(q_hi))
            if np.isclose(hi, lo):
                # fallback spreads
                if m == "max_drawdown_pct":
                    hi = lo + 10.0
                else:
                    hi = lo + 1.0
            ranges[m] = (lo, hi)
        return cls(ranges)

    def norm(self, name: str, value: float) -> float:
        lo, hi = self.ranges.get(name, (0.0, 1.0))
        if hi <= lo:
            return 0.0
        x = (value - lo) / (hi - lo)
        return float(max(0.0, min(1.0, x)))

