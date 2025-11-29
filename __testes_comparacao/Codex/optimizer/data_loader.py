from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import ParamSpace
from .utils import seed_everything


REQUIRED_COLUMNS = [
    "StopLoss",
    "TakeProfit",
    "ATR_Multiplier",
    "RiskFactor",
    "SessionStart",
    "SessionEnd",
    # targets / métricas
    "net_profit",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate_pct",
    "sharpe_ratio",
]


def _coerce_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {c}")
    # Coerções
    float_cols = [
        "StopLoss",
        "TakeProfit",
        "ATR_Multiplier",
        "RiskFactor",
        "net_profit",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "sharpe_ratio",
    ]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["SessionStart", "SessionEnd"]:
        df[c] = df[c].astype(str)
    df = df.dropna().reset_index(drop=True)
    return df


def load_from_dir(input_dir: Path) -> pd.DataFrame:
    files: List[Path] = []
    if input_dir.exists():
        files = list(input_dir.glob("*.csv")) + list(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV/JSON encontrado em {input_dir}")

    frames: List[pd.DataFrame] = []
    for f in files:
        if f.suffix.lower() == ".csv":
            frames.append(pd.read_csv(f))
        else:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            frames.append(pd.json_normalize(data))

    df = pd.concat(frames, axis=0, ignore_index=True)
    return _coerce_and_validate(df)


def generate_synthetic(space: ParamSpace, n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Gera dataset sintético de backtest plausível para XAUUSD M5."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    sl = rng.uniform(space.stop_loss_min, space.stop_loss_max, size=n)
    tp = rng.uniform(space.take_profit_min, space.take_profit_max, size=n)
    atr = rng.uniform(space.atr_multiplier_min, space.atr_multiplier_max, size=n)
    risk = rng.uniform(space.risk_factor_min, space.risk_factor_max, size=n)

    # Sessões simplificadas: duas janelas comuns
    sessions = rng.choice([("08:00", "12:00"), ("09:00", "17:00"), ("07:00", "10:30"), ("13:00", "18:00")], size=n)
    s_start = [s[0] for s in sessions]
    s_end = [s[1] for s in sessions]

    # Gerar métricas dependentes dos parâmetros com ruído
    # Heurística: tp > sl tende a melhorar PF e lucro; risco maior aumenta lucro esperado e drawdown; atr moderado melhora sharpe
    ratio = (tp / (sl + 1e-6)).clip(0, 20)
    base_pf = 0.6 + 0.9 * np.tanh((ratio - 1.0)) + 0.1 * rng.normal(size=n)
    base_pf = np.clip(base_pf, 0.5, 4.0)

    max_dd = (3.0 + 5.0 * risk + 8.0 * np.maximum(0, 1.2 - atr)) + 1.2 * rng.normal(size=n)
    max_dd = np.clip(max_dd, 1.0, 35.0)

    sharpe = 0.2 + 1.2 * np.tanh((ratio - 0.8)) + 0.3 * (2.0 - np.abs(atr - 1.8)) + 0.1 * rng.normal(size=n)
    sharpe = np.clip(sharpe, -0.5, 3.5)

    winrate = 45 + 25 * np.tanh(ratio - 0.7) - 5 * (risk - 1.0) + 2.0 * rng.normal(size=n)
    winrate = np.clip(winrate, 25.0, 90.0)

    net_profit = 200 * (ratio - 0.8) + 120 * (risk) + 50 * (atr) + 30 * rng.normal(size=n)
    net_profit = np.clip(net_profit, -1500.0, 8000.0)

    df = pd.DataFrame(
        {
            "StopLoss": sl,
            "TakeProfit": tp,
            "ATR_Multiplier": atr,
            "RiskFactor": risk,
            "SessionStart": s_start,
            "SessionEnd": s_end,
            "net_profit": net_profit,
            "max_drawdown_pct": max_dd,
            "profit_factor": base_pf,
            "win_rate_pct": winrate,
            "sharpe_ratio": sharpe,
        }
    )
    return _coerce_and_validate(df)

