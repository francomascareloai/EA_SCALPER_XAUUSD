# Arquitetura — EA Optimizer AI

Diagrama textual do fluxo ponta a ponta (MT5 ⇄ Python ⇄ EA gerado):

```
[MetaTrader 5 Strategy Tester]
      │  (executa EAs e exporta relatórios CSV/HTML/JSON)
      ▼
[backtest_results/  (CSV/JSON)]  ──►  [Python: Data Loader]
                                      - Normaliza colunas e valida esquema
                                      - Suporte a dados sintéticos (modo demo)
                                      ▼
                              [Python: Model Trainer]
                                      - Modelos p/ prever: net_profit, sharpe_ratio, max_drawdown_pct
                                      - sklearn RandomForest/Ridge (fallback: regressão numpy)
                                      ▼
                                [Python: Optimizer]
                                      - Optuna TPE (fallback: Random Search)
                                      - Objetivo: maximizar score = f(profit_factor, sharpe, winrate) - penalidade(drawdown)
                                      - Restrições FTMO (ex.: drawdown ≤ 8%)
                                      ▼
                           [optimized_params.json]
                                      │
                                      ├─► [Python: Report]
                                      │      - performance_summary.csv
                                      │      - gráficos (plot_metrics.png, plot_equity_demo.png)
                                      │
                                      └─► [Python: MQL5 Generator]
                                             - `EA_OPTIMIZER_XAUUSD.mq5`
                                             - Cabeçalho de inputs e lógica OnInit/OnTick
                                             - Logs informativos
```

Canais de comunicação:
- Input: arquivos CSV/JSON de backtest salvos pelo MT5 (ou ferramenta de parsing).
- Saída: `optimized_params.json` e `EA_OPTIMIZER_XAUUSD.mq5` para compilar no MetaEditor.
- Integração futura (opcional): FastAPI para expor endpoint `POST /optimize` e `GET /artifacts` (não necessário para o desafio, mas suportado pela arquitetura).

## Componentes
- Data Loader: robusto a ausência de dados reais (gera dataset sintético reprodutível para XAUUSD M5).
- Model Trainer: abstrai o backend (sklearn se disponível; senão, fallback numérico simples).
- Optimizer: TPE/Optuna com priores sensatas por ativo/timeframe; retorna conjunto único "melhor".
- Generator MQL5: template seguro, com inputs otimizados e exemplo de estratégia baseada em MA + ATR.
- Reports: comparação baseline × otimizado e gráficos de diagnóstico.

## Segurança e limites
- Tudo isolado em `__testes_comparacao/Codex/`.
- Não altera outros diretórios do repositório.
- Sem dependência obrigatória de MT5 no ambiente local para executar o fluxo demo.

