from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import OptimizeResult, ParamSpace, TrialParams, Metrics
from .utils import composite_score


def _try_import_optuna():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception:
        return None


def _predict_score(models, params_array: np.ndarray, calibrator=None, weights=None, constraints=None) -> Tuple[float, Dict[str, float]]:
    preds = models.predict(params_array)
    m = {
        "net_profit": float(preds["net_profit"][0]),
        "sharpe_ratio": float(preds["sharpe_ratio"][0]),
        "max_drawdown_pct": float(preds["max_drawdown_pct"][0]),
        "profit_factor": float(preds["profit_factor"][0]),
        "win_rate_pct": float(preds["win_rate_pct"][0]),
    }
    score = composite_score(m, calibrator=calibrator, weights=weights, constraints=constraints)
    return score, m


def optimize(space: ParamSpace, models, n_trials: int = 150, seed: int = 42, calibrator=None, weights=None, constraints: dict | None = None) -> OptimizeResult:
    optuna = _try_import_optuna()

    def suggest_to_array(sl, tp, atr, risk) -> np.ndarray:
        return np.array([[sl, tp, atr, risk]], dtype=float)

    if optuna:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            sl = trial.suggest_float("StopLoss", space.stop_loss_min, space.stop_loss_max)
            tp = trial.suggest_float("TakeProfit", space.take_profit_min, space.take_profit_max)
            atr = trial.suggest_float("ATR_Multiplier", space.atr_multiplier_min, space.atr_multiplier_max)
            risk = trial.suggest_float("RiskFactor", space.risk_factor_min, space.risk_factor_max)
            score, m = _predict_score(models, suggest_to_array(sl, tp, atr, risk), calibrator=calibrator, weights=weights, constraints=constraints)
            # Pequena preferência por tp >= 1.5*sl
            if tp < 1.5 * sl:
                score -= 0.05
            trial.set_user_attr("metrics", m)
            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_trial
        sl = float(best.params["StopLoss"]) ; tp = float(best.params["TakeProfit"]) ; atr = float(best.params["ATR_Multiplier"]) ; risk = float(best.params["RiskFactor"]) 
        _, metrics = _predict_score(models, suggest_to_array(sl, tp, atr, risk))
        params = TrialParams(
            StopLoss=sl,
            TakeProfit=tp,
            ATR_Multiplier=atr,
            RiskFactor=risk,
            SessionStart=space.session_start_default,
            SessionEnd=space.session_end_default,
        )
        trials = []
        for t in study.trials:
            p = t.params
            met = t.user_attrs.get("metrics", {})
            item = {"value": t.value}
            item.update(p)
            item.update(met)
            trials.append(item)
        return OptimizeResult(
            params=params,
            metrics_pred=Metrics(**metrics),
            score=float(best.value),
            backend="optuna",
            n_trials=n_trials,
            trials=trials,
        )

    # Fallback: Random Search
    rng = np.random.default_rng(seed)
    best_score = -1e9
    best_metrics = None
    best_params = None
    trials = []
    for _ in range(n_trials):
        sl = rng.uniform(space.stop_loss_min, space.stop_loss_max)
        tp = rng.uniform(space.take_profit_min, space.take_profit_max)
        atr = rng.uniform(space.atr_multiplier_min, space.atr_multiplier_max)
        risk = rng.uniform(space.risk_factor_min, space.risk_factor_max)
        score, m = _predict_score(models, np.array([[sl, tp, atr, risk]], dtype=float), calibrator=calibrator, weights=weights, constraints=constraints)
        score -= 0.05 if tp < 1.5 * sl else 0.0
        if score > best_score:
            best_score = score
            best_metrics = m
            best_params = (sl, tp, atr, risk)
        trials.append({
            "value": float(score),
            "StopLoss": float(sl),
            "TakeProfit": float(tp),
            "ATR_Multiplier": float(atr),
            "RiskFactor": float(risk),
            **m,
        })

    params = TrialParams(
        StopLoss=float(best_params[0]),
        TakeProfit=float(best_params[1]),
        ATR_Multiplier=float(best_params[2]),
        RiskFactor=float(best_params[3]),
        SessionStart=space.session_start_default,
        SessionEnd=space.session_end_default,
    )
    return OptimizeResult(
        params=params,
        metrics_pred=Metrics(**best_metrics),
        score=float(best_score),
        backend="random-search",
        n_trials=n_trials,
        note="Fallback sem Optuna",
        trials=trials,
    )
