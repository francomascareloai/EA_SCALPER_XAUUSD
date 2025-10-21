from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def summarize_performance(
    df: pd.DataFrame,
    best_pred: Dict[str, float],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: mediana do dataset
    baseline = {
        "net_profit": float(df["net_profit"].median()),
        "max_drawdown_pct": float(df["max_drawdown_pct"].median()),
        "profit_factor": float(df["profit_factor"].median()),
        "win_rate_pct": float(df["win_rate_pct"].median()),
        "sharpe_ratio": float(df["sharpe_ratio"].median()),
    }
    # Baseline 2: melhor PF da amostra
    idx_best_pf = int(df["profit_factor"].idxmax())
    best_row = df.loc[idx_best_pf]
    baseline_best = {
        "net_profit": float(best_row["net_profit"]),
        "max_drawdown_pct": float(best_row["max_drawdown_pct"]),
        "profit_factor": float(best_row["profit_factor"]),
        "win_rate_pct": float(best_row["win_rate_pct"]),
        "sharpe_ratio": float(best_row["sharpe_ratio"]),
    }

    rows = []
    for base_name, base_vals in [("median", baseline), ("best_pf", baseline_best)]:
        for k in ["net_profit", "max_drawdown_pct", "profit_factor", "win_rate_pct", "sharpe_ratio"]:
            b = base_vals[k]
            o = float(best_pred.get(k, np.nan))
            rows.append({"baseline": base_name, "metric": k, "baseline_value": b, "optimized": o, "delta": o - b})
    summary = pd.DataFrame(rows)
    csv_path = out_dir / "performance_summary.csv"
    summary.to_csv(csv_path, index=False)
    try:
        _plots(summary, out_dir)
    except Exception:
        pass
    return csv_path


def _plots(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    metrics = summary["metric"].tolist()
    b = summary["baseline"].to_numpy()
    o = summary["optimized"].to_numpy()
    x = np.arange(len(metrics))
    w = 0.35
    plt.bar(x - w / 2, b, w, label="baseline")
    plt.bar(x + w / 2, o, w, label="optimized")
    plt.xticks(x, metrics, rotation=15)
    plt.title("Performance — baseline vs optimized")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_metrics.png", dpi=140)
    plt.close()

    # Equidade simulada (apenas visual): baseline vs optimized
    np.random.seed(42)
    steps = 200
    base_drift = np.clip(summary.loc[summary.metric == "net_profit", "baseline"].values[0] / 5000.0, -0.5, 1.5)
    opt_drift = np.clip(summary.loc[summary.metric == "net_profit", "optimized"].values[0] / 5000.0, -0.5, 1.5)
    base_curve = np.cumsum(np.random.normal(base_drift, 1.0, size=steps))
    opt_curve = np.cumsum(np.random.normal(opt_drift, 1.0, size=steps))
    plt.figure(figsize=(8, 4))
    plt.plot(base_curve, label="baseline")
    plt.plot(opt_curve, label="optimized")
    plt.title("Equity (simulada p/ visualização)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_equity_demo.png", dpi=140)
    plt.close()
