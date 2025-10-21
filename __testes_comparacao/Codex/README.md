# EA Optimizer AI — Codex

Este módulo implementa o desafio "EA Optimizer AI" contido em `__testes_comparacao/desafio_ea_optimizer_ai.md` com um pipeline end‑to‑end:

- Importação de dados de backtest (CSV/JSON) ou geração sintética (modo demo)
- Treino de modelos para prever métricas (lucro, drawdown, sharpe)
- Otimização automática de parâmetros (Optuna; fallback em Random Search)
- Exportação dos melhores parâmetros em `optimized_params.json`
- Geração automática do EA MQL5 `EA_OPTIMIZER_XAUUSD.mq5` (MQL5 moderno com `CTrade`)
- EA v1.2: Trailing stop por ATR e Daily Loss Guard (DD diário)
- Relatório comparativo + gráficos em `output/`

## Estrutura
- `cli.py` — ponto de entrada (CLI)
- `optimizer/` — código do pipeline (dados, modelos, otimização, exportação, relatórios)
- `templates/` — template base do EA
- `sample_data/` — gerador e exemplo de dados sintéticos
- `output/` — artefatos gerados (params, EA, gráficos, CSV)
- `ARQUITETURA.md` — diagrama textual e explicação de integrações

## Requisitos
Python 3.11+. Recomendado usar venv.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Caso `scikit-learn` ou `optuna` não sejam instaláveis no ambiente, o código entra em modo fallback (regressão linear numpy + random search), permitindo executar o fluxo demo.

## Uso (Demo)
Executa com dados sintéticos e exporta tudo em `output/`:

```
python cli.py --demo
```

Uso com dados reais (pasta contendo CSV/JSON de backtests, com colunas como: `StopLoss,TakeProfit,ATR_Multiplier,RiskFactor,SessionStart,SessionEnd,net_profit,max_drawdown_pct,profit_factor,win_rate_pct,sharpe_ratio`):

```
python cli.py --input-dir /caminho/para/backtests --symbol XAUUSD --timeframe M5
```

Artefatos esperados em `output/`:
- `optimized_params.json`
- `EA_OPTIMIZER_XAUUSD.mq5`
- `performance_summary.csv`
- `plot_metrics.png` e `plot_equity_demo.png`
- `study_trials.csv` (Optuna trials) e `metadata.json`

## Integração rápida com MetaTrader 5
- Copie `output/EA_OPTIMIZER_XAUUSD.mq5` para a pasta `MQL5/Experts/EA_SCALPER_XAUUSD/` do seu terminal.
- Compile no MetaEditor; rode no Strategy Tester com o símbolo e timeframe correspondentes (ex: XAUUSD M5).
- Opcionalmente, utilize o `optimized_params.json` para parametrizar outras variantes do seu EA.

Observação: a versão 1.1 do EA usa `#include <Trade/Trade.mqh>` e as APIs `CTrade::Buy/Sell`. Os inputs foram renomeados para `StopLossPoints` e `TakeProfitPoints` (pontos do símbolo). 

### Parâmetros adicionais do EA v1.2
- `UseTrailingStop` (bool, padrão true)
- `ATR_Period` (int), `TrailingATRMultiplier` (double), `TrailingMinPoints` (double)
- `EnableDailyLossGuard` (bool), `DailyLossMaxPct` (%) e `DayResetTime` (HH:MM)

### Restrições/pesos via CLI
Exemplo conservador FTMO:

```
python cli.py --demo --trials 300 --max-dd 8.0 --min-pf 1.6 \
  --w-pf 0.40 --w-sh 0.30 --w-wr 0.15 --w-np 0.10 --w-dd 0.45
```

## Licença
Somente para uso dentro deste repositório, conforme desafio. Alterações limitadas à pasta `__testes_comparacao/Codex`.
