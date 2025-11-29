from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from optimizer.data_loader import generate_synthetic, load_from_dir
from optimizer.models import train_models
from optimizer.optimizer_core import optimize
from optimizer.schema import ParamSpace
from optimizer.mql5_generator import generate_ea_file
from optimizer.report import summarize_performance
from optimizer.utils import ensure_dir, save_json, seed_everything
from optimizer.calibration import NormCalibrator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("EA Optimizer AI — Codex")
    ap.add_argument("--input-dir", type=str, default=None, help="Diretório com CSV/JSON de backtests")
    ap.add_argument("--output-dir", type=str, default="output", help="Diretório de saída")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--timeframe", type=str, default="M5")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--demo", action="store_true", help="Usar dados sintéticos")
    # Restrições e pesos
    ap.add_argument("--max-dd", type=float, default=8.0, help="Max drawdown permitido (%%)")
    ap.add_argument("--min-pf", type=float, default=1.6, help="Profit factor mínimo")
    ap.add_argument("--w-pf", type=float, default=0.35, help="Peso PF")
    ap.add_argument("--w-sh", type=float, default=0.30, help="Peso Sharpe")
    ap.add_argument("--w-wr", type=float, default=0.20, help="Peso WinRate")
    ap.add_argument("--w-np", type=float, default=0.15, help="Peso NetProfit")
    ap.add_argument("--w-dd", type=float, default=0.40, help="Peso penalidade DD")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(42)

    space = ParamSpace(symbol=args.symbol, timeframe=args.timeframe)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Etapa 1: Dados
    if args.demo or args.input_dir is None:
        print("[INFO] Modo DEMO: gerando dataset sintético...")
        df = generate_synthetic(space, n=600, seed=42)
    else:
        df = load_from_dir(Path(args.input_dir))
    print(f"[INFO] Dataset carregado: {len(df)} linhas")

    # Etapa 2: Modelos
    models = train_models(df)
    print(f"[INFO] Backend de modelos: {models.backend}")

    # Calibração dinâmica do score
    calibrator = NormCalibrator.from_dataframe(df)
    weights = {"pf": args.w_pf, "sh": args.w_sh, "wr": args.w_wr, "np": args.w_np, "dd_pen": args.w_dd}
    constraints = {"max_drawdown_max_pct": args.max_dd, "profit_factor_min": args.min_pf}

    # Etapa 3: Otimização
    result = optimize(space, models, n_trials=args.trials, seed=42, calibrator=calibrator, weights=weights, constraints=constraints)
    print(f"[INFO] Otimização backend={result.backend} score={result.score:.4f}")

    # Etapa 4: Exportação de parâmetros e EA
    params_path = out_dir / "optimized_params.json"
    save_json(result.params.to_json(), params_path)
    print(f"[INFO] Parâmetros salvos em: {params_path}")

    ea_path = out_dir / "EA_OPTIMIZER_XAUUSD.mq5"
    generate_ea_file(result.params, ea_path)
    print(f"[INFO] EA MQL5 gerado em: {ea_path}")

    # Etapa 5: Relatório
    # Predições do melhor
    best_pred = {
        "net_profit": result.metrics_pred.net_profit,
        "max_drawdown_pct": result.metrics_pred.max_drawdown_pct,
        "profit_factor": result.metrics_pred.profit_factor,
        "win_rate_pct": result.metrics_pred.win_rate_pct,
        "sharpe_ratio": result.metrics_pred.sharpe_ratio,
    }
    csv_path = summarize_performance(df, best_pred, out_dir)
    print(f"[INFO] Relatório salvo em: {csv_path}")

    # Export trials/metadata
    if result.trials:
        import pandas as pd
        pd.DataFrame(result.trials).to_csv(out_dir / "study_trials.csv", index=False)
    meta = {
        "model_backend": models.backend,
        "optimizer_backend": result.backend,
        "trials": result.n_trials,
        "constraints": constraints,
        "weights": weights,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
    }
    save_json(meta, out_dir / "metadata.json")

    print("[DONE] Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
