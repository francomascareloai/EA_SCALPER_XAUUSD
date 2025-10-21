from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _try_import_sklearn():
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        return {
            "RandomForestRegressor": RandomForestRegressor,
            "Ridge": Ridge,
            "split": train_test_split,
            "r2": r2_score,
        }
    except Exception:
        return None


@dataclass
class FittedModels:
    backend: str
    model_net_profit: object
    model_sharpe: object
    model_drawdown: object
    model_pf: object
    model_wr: object

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        np_pred = _predict(self.model_net_profit, X)
        sh_pred = _predict(self.model_sharpe, X)
        dd_pred = _predict(self.model_drawdown, X)
        pf_pred = _predict(self.model_pf, X)
        wr_pred = _predict(self.model_wr, X)
        return {
            "net_profit": np_pred,
            "sharpe_ratio": sh_pred,
            "max_drawdown_pct": dd_pred,
            "profit_factor": pf_pred,
            "win_rate_pct": wr_pred,
        }


def _predict(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float)
    # Fallback linear
    w, b = model
    return X @ w + b


def _fit_linear_fallback(X: np.ndarray, y: np.ndarray):
    # Regressão linear ridge-like simples via pseudo-inversa
    reg = 1e-3
    XtX = X.T @ X + reg * np.eye(X.shape[1])
    w = np.linalg.pinv(XtX) @ X.T @ y
    b = float(np.mean(y) - np.mean(X, axis=0) @ w)
    return (w, b)


def train_models(df: pd.DataFrame) -> FittedModels:
    X = df[["StopLoss", "TakeProfit", "ATR_Multiplier", "RiskFactor"]].to_numpy(dtype=float)
    y_np = df["net_profit"].to_numpy(dtype=float)
    y_sh = df["sharpe_ratio"].to_numpy(dtype=float)
    y_dd = df["max_drawdown_pct"].to_numpy(dtype=float)
    y_pf = df["profit_factor"].to_numpy(dtype=float)
    y_wr = df["win_rate_pct"].to_numpy(dtype=float)

    sk = _try_import_sklearn()
    if sk:
        split = sk["split"]
        Xtr, Xte, ytr_np, yte_np = split(X, y_np, test_size=0.2, random_state=42)
        Xtr2, Xte2, ytr_sh, yte_sh = split(X, y_sh, test_size=0.2, random_state=42)
        Xtr3, Xte3, ytr_dd, yte_dd = split(X, y_dd, test_size=0.2, random_state=42)

        # Modelo robusto + linear
        rf = sk["RandomForestRegressor"](n_estimators=300, random_state=42)
        rf.fit(Xtr, ytr_np)
        ridge_sh = sk["Ridge"](alpha=1.0)
        ridge_sh.fit(Xtr2, ytr_sh)
        ridge_dd = sk["Ridge"](alpha=1.0)
        ridge_dd.fit(Xtr3, ytr_dd)
        ridge_pf = sk["Ridge"](alpha=0.5)
        ridge_pf.fit(X, y_pf)
        ridge_wr = sk["Ridge"](alpha=0.5)
        ridge_wr.fit(X, y_wr)

        backend = "sklearn"
        return FittedModels(backend, rf, ridge_sh, ridge_dd, ridge_pf, ridge_wr)

    # Fallback totalmente numpy
    m_np = _fit_linear_fallback(X, y_np)
    m_sh = _fit_linear_fallback(X, y_sh)
    m_dd = _fit_linear_fallback(X, y_dd)
    m_pf = _fit_linear_fallback(X, y_pf)
    m_wr = _fit_linear_fallback(X, y_wr)
    return FittedModels("numpy-linear", m_np, m_sh, m_dd, m_pf, m_wr)
