from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


def normalize_series(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if math.isclose(mx - mn, 0.0):
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def composite_score(
    metrics: Dict[str, float],
    calibrator: Optional[object] = None,
    weights: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, float]] = None,
) -> float:
    """Combina métricas em um escore único.

    score = 0.35 * norm(profit_factor) + 0.30 * norm(sharpe_ratio)
            + 0.20 * norm(win_rate_pct) + 0.15 * norm(net_profit)
            - penalty(drawdown)
    """
    # Valores brutos
    pf = metrics.get("profit_factor", 0.0)
    sh = metrics.get("sharpe_ratio", 0.0)
    wr = metrics.get("win_rate_pct", 0.0)
    npf = metrics.get("net_profit", 0.0)
    dd = metrics.get("max_drawdown_pct", 100.0)

    # Pesos default
    w = weights or {"pf": 0.35, "sh": 0.30, "wr": 0.20, "np": 0.15, "dd_pen": 0.40}

    # Normalização
    if calibrator is not None and hasattr(calibrator, "norm"):
        n_pf = calibrator.norm("profit_factor", pf)
        n_sh = calibrator.norm("sharpe_ratio", sh)
        n_wr = calibrator.norm("win_rate_pct", wr)
        n_np = calibrator.norm("net_profit", npf)
        n_dd = calibrator.norm("max_drawdown_pct", dd)
        dd_penalty = n_dd
    else:
        def n(x: float, lo: float, hi: float) -> float:
            return max(0.0, min(1.0, (x - lo) / (hi - lo))) if hi > lo else 0.0
        n_pf = n(pf, 0.8, 3.0)
        n_sh = n(sh, -0.5, 3.0)
        n_wr = n(wr, 40.0, 85.0)
        n_np = n(npf, -1000.0, 5000.0)
        dd_penalty = n(dd, 5.0, 25.0)

    score = w["pf"] * n_pf + w["sh"] * n_sh + w["wr"] * n_wr + w["np"] * n_np - w["dd_pen"] * dd_penalty

    # Restrições duras (FTMO-like)
    if constraints:
        max_dd = constraints.get("max_drawdown_max_pct")
        min_pf = constraints.get("profit_factor_min")
        if max_dd is not None and dd > max_dd:
            score -= 0.6  # penalidade forte
        if min_pf is not None and pf < min_pf:
            score -= 0.4
    return float(score)


def time_range_to_str(start: str, end: str) -> str:
    return f"{start}-{end}"
