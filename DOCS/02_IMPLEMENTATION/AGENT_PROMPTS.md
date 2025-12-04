# PROMPTS DETALHADOS PARA OS 5 AGENTES

**Data**: 2025-12-01  
**Uso**: Copie e cole em sessões separadas do Factory  
**Objetivo**: Trabalho paralelo com máxima clareza

---

## 🔮 ORACLE - Validação e Backtest

### Prompt 1: Validação de Dados
```
Oracle, preciso que você valide os dados tick convertidos para Parquet.

CONTEXTO:
- Dados em: data/processed/ticks_2020.parquet até ticks_2025.parquet
- Total: 318 milhões de ticks, 5.5 GB
- Colunas: timestamp, bid, ask, volume, spread (cents), mid_price, timestamp_unix
- Script existente: scripts/oracle/validate_data.py

TAREFAS:
1. Rode validate_data.py em CADA arquivo Parquet (2020-2025)
2. Gere um relatório consolidado com:
   - Gaps detectados (ignore weekends)
   - Anomalias de spread (> $2.00 ou negativo)
   - Cobertura por sessão (Asian/London/NY)
   - Score de qualidade GENIUS (0-100)
3. Identifique se há regime diversity suficiente (trending/ranging/reverting)

OUTPUT:
- Salvar relatório em: DOCS/04_REPORTS/VALIDATION/DATA_QUALITY_GENIUS.md
- Formato: Markdown com tabelas

CRITÉRIOS DE APROVAÇÃO:
- Gaps críticos (>24h non-weekend): 0
- Spread anomalies: < 0.1%
- Cobertura de sessões: > 90% cada
- Score GENIUS: >= 80

Se algum critério falhar, liste exatamente o que está errado e sugira correção.
```

### Prompt 2: Backtest com Ablation Study (Análise de Filtros)
```
Oracle, preciso que você rode um backtest sistemático testando cada filtro.

CONTEXTO:
- Dados: data/processed/ticks_2024.parquet (usar só 2024 para speed)
- Script: scripts/backtest/tick_backtester.py
- Estratégia base: SMC (Order Blocks + FVG + Liquidity Sweeps)

METODOLOGIA ABLATION STUDY:
Rode 6 backtests separados:

1. BASELINE: Estratégia pura, SEM nenhum filtro
2. +REGIME: Baseline + filtro de regime (só trending/ranging)
3. +SESSION: Baseline + filtro de sessão (só London/NY overlap)
4. +MTF: Baseline + alinhamento multi-timeframe (M5 + H1)
5. +CONFLUENCE: Baseline + score mínimo de confluência (>= 70)
6. ALL_FILTERS: Todos os filtros combinados

MÉTRICAS A COLETAR (para cada teste):
- Total trades
- Win Rate (%)
- Profit Factor
- Sharpe Ratio
- Max Drawdown (%)
- Average trade duration
- Trades por sessão

OUTPUT:
1. Tabela comparativa: DOCS/04_REPORTS/BACKTESTS/ABLATION_STUDY.md
2. Ranking: Qual filtro adiciona mais valor (delta Sharpe vs baseline)
3. Recomendação: Quais filtros manter, quais descartar

FORMATO DA TABELA:
| Config | Trades | WR% | PF | Sharpe | MaxDD | Delta vs Baseline |
|--------|--------|-----|----|---------|----|------------------|
| BASELINE | ... | ... | ... | ... | ... | 0 |
| +REGIME | ... | ... | ... | ... | ... | +X% |
```

### Prompt 3: Walk-Forward Analysis
```
Oracle, rode Walk-Forward Analysis completo.

CONTEXTO:
- Script: scripts/oracle/walk_forward.py
- Dados: 2020-2024 (deixar 2025 como holdout final)
- Melhor configuração do Ablation Study

PARÂMETROS WFA:
- Training window: 12 meses
- Test window: 3 meses
- Rolling (não anchored)
- Purge gap: 1 semana (evitar lookahead)

MÉTRICAS:
- WFE (Walk-Forward Efficiency) global
- WFE por regime (trending >= 0.65, ranging >= 0.50)
- % de janelas OOS positivas
- Degradação IS → OOS

OUTPUT:
- DOCS/04_REPORTS/VALIDATION/WFA_REPORT.md
- Gráfico: Performance IS vs OOS por janela

CRITÉRIOS GO/NO-GO:
- WFE global >= 0.60: GO
- WFE global 0.50-0.59: CAUTIOUS
- WFE global < 0.50: NO-GO (estratégia overfitted)
```

---

## ⚒️ FORGE - Código e Implementação

### Prompt 1: Criar Segmentador de Dados
```
Forge, crie um script para segmentar tick data por regime e sessão.

ARQUIVO: scripts/backtest/segment_data.py

FUNCIONALIDADES:

1. SEGMENTAÇÃO POR REGIME:
   - Calcular Hurst exponent em janelas de 1 hora
   - Trending: Hurst > 0.55
   - Ranging: 0.45 <= Hurst <= 0.55
   - Reverting: Hurst < 0.45
   - Output: data/segments/regime_trending.parquet, etc.

2. SEGMENTAÇÃO POR SESSÃO:
   - Asian: 00:00-08:00 UTC
   - London: 08:00-16:00 UTC
   - NY: 13:00-21:00 UTC
   - Overlap London/NY: 13:00-16:00 UTC
   - Output: data/segments/session_*.parquet

3. ESTATÍSTICAS:
   - % de tempo em cada regime
   - Volatilidade média por sessão
   - Spread médio por sessão

INPUT:
- data/processed/ticks_*.parquet

OUTPUT:
- data/segments/*.parquet
- data/segments/SEGMENT_STATS.json

DEPENDÊNCIAS:
- pandas, numpy, pyarrow
- Usar Hurst R/S method (não DFA)

TESTES:
- Rodar em ticks_2024.parquet primeiro (mais rápido)
- Validar que soma dos segmentos = total original
```

### Prompt 2: Criar Script de Ablation Study
```
Forge, crie um script automatizado para Ablation Study de filtros.

ARQUIVO: scripts/backtest/ablation_study.py

OBJETIVO:
Testar sistematicamente cada filtro da estratégia e medir impacto.

ESTRUTURA:

class AblationStudy:
    def __init__(self, data_path, base_strategy):
        self.data = load_parquet(data_path)
        self.strategy = base_strategy
        self.filters = {
            'regime': RegimeFilter(),
            'session': SessionFilter(),
            'mtf': MTFFilter(),
            'confluence': ConfluenceFilter(),
            'news': NewsFilter()
        }
    
    def run_baseline(self) -> BacktestResult:
        """Roda estratégia sem filtros"""
        pass
    
    def run_with_filter(self, filter_name: str) -> BacktestResult:
        """Roda estratégia + um filtro específico"""
        pass
    
    def run_all_combinations(self) -> pd.DataFrame:
        """Roda todas as combinações possíveis (2^n)"""
        pass
    
    def rank_filters(self) -> pd.DataFrame:
        """Ranking de filtros por contribuição marginal"""
        # Shapley values ou simple delta analysis
        pass
    
    def generate_report(self, output_path: str):
        """Gera relatório markdown"""
        pass

MÉTRICAS POR TESTE:
- trades, win_rate, profit_factor, sharpe, max_dd, avg_trade_duration

CLI:
python scripts/backtest/ablation_study.py \
    --data data/processed/ticks_2024.parquet \
    --output DOCS/04_REPORTS/BACKTESTS/ABLATION_STUDY.md

BONUS:
- Calcular Shapley values para contribuição de cada filtro
- Heatmap de correlação entre filtros
```

### Prompt 3: Revisar tick_backtester.py
```
Forge, revise o tick_backtester.py existente para suportar filtros.

ARQUIVO: scripts/backtest/tick_backtester.py (já existe, ~1000 linhas)

VERIFICAR:
1. Já suporta filtros individuais on/off?
2. Coleta métricas necessárias para Ablation Study?
3. Tem modo de comparação A/B?

SE NÃO SUPORTA:
Adicionar:
- Flag --disable-filter FILTER_NAME
- Flag --enable-only FILTER_NAME
- Output JSON com métricas detalhadas

SE JÁ SUPORTA:
Documentar como usar para Ablation Study.

OUTPUT:
- Código atualizado (se necessário)
- DOCS/05_GUIDES/USAGE/TICK_BACKTESTER_GUIDE.md
```

---

## 🔍 ARGUS - Pesquisa

### Prompt 1: Research de Filtros para Scalping
```
Argus, pesquise papers e implementações sobre filtros para scalping em forex/gold.

FOCO:
1. Regime filters: Quais métodos funcionam melhor? (Hurst, HMM, Entropy)
2. Session filters: Qual sessão tem melhor edge para XAUUSD?
3. MTF alignment: Multi-timeframe realmente ajuda em scalping?
4. Confluence scoring: Quantos sinais precisam concordar?

FONTES:
- arXiv (quant-ph, q-fin)
- SSRN
- GitHub repos de trading
- QuantConnect, Quantopian archives

OUTPUT:
- DOCS/03_RESEARCH/FINDINGS/FILTER_RESEARCH.md
- Para cada filtro: evidência a favor, evidência contra, recomendação
- Citar pelo menos 3 papers/repos por filtro

FORMATO:
## Filtro X
### Evidência a Favor
- Paper 1: [citação] - resultado
### Evidência Contra
- Paper 2: [citação] - limitação
### Implementações
- Repo 1: [link] - como implementa
### Recomendação
[Manter/Descartar/Modificar] - justificativa
```

### Prompt 2: Research EVT para Tail Risk
```
Argus, pesquise Extreme Value Theory (EVT) aplicada a trading.

CONTEXTO:
Monte Carlo tradicional assume distribuição normal, mas mercados têm fat tails.
EVT modela melhor os eventos extremos (crashes, spikes).

PERGUNTAS:
1. Como implementar GPD (Generalized Pareto Distribution) para tail risk?
2. Qual threshold usar para definir "extreme"? (95th percentile? 99th?)
3. Existem implementações Python prontas?
4. Como integrar EVT no Monte Carlo existente?

OUTPUT:
- DOCS/03_RESEARCH/FINDINGS/EVT_TAIL_RISK.md
- Código de exemplo (se encontrar)
- Recomendação de implementação

PAPERS CHAVE (se encontrar):
- McNeil, Frey & Embrechts - Quantitative Risk Management
- Qualquer paper de EVT + VaR/CVaR
```

---

## 🛡️ SENTINEL - Risco

### Prompt 1: Calibrar Kelly por Regime
```
Sentinel, calcule Kelly optimal para cada regime de mercado.

CONTEXTO:
- Dados: Resultados do backtest por regime (trending/ranging/reverting)
- Conta: FTMO $100k
- Max risk por trade: 1%

CÁLCULOS NECESSÁRIOS:

1. KELLY CLÁSSICO:
   f* = (p * b - q) / b
   onde p = win rate, q = 1-p, b = avg_win/avg_loss

2. KELLY CONSERVADOR:
   f_conservative = f* / 2 (half-Kelly)

3. KELLY POR REGIME:
   - Trending: calcular com dados só de trending
   - Ranging: calcular com dados só de ranging
   - Reverting: calcular (provavelmente f* ≈ 0, não operar)

4. AJUSTES:
   - Sample size correction: se N < 100 trades, penalizar
   - Drawdown constraint: limitar para DD < 10%

OUTPUT:
- DOCS/04_REPORTS/DECISIONS/KELLY_CALIBRATION.md
- Tabela: Regime | Win Rate | Avg Win | Avg Loss | Kelly | Half-Kelly | Recomendado

FORMATO:
| Regime | Trades | WR% | AvgW | AvgL | Kelly* | Half-Kelly | Recomendado |
|--------|--------|-----|------|------|--------|------------|-------------|
| Trending | 500 | 55% | $150 | $100 | 0.10 | 0.05 | 0.04 (4%) |
| Ranging | 300 | 48% | $120 | $110 | 0.02 | 0.01 | 0.01 (1%) |
| Reverting | 100 | 42% | $100 | $130 | -0.05 | N/A | NO TRADE |
```

### Prompt 2: Circuit Breaker Thresholds
```
Sentinel, defina thresholds para circuit breakers FTMO.

CONTEXTO:
- FTMO $100k
- Daily DD limit: 5% ($5,000)
- Total DD limit: 10% ($10,000)
- Precisamos de buffers de segurança

CALCULAR:

1. WARNING LEVELS:
   - Daily DD warning: X% (quando alertar)
   - Total DD warning: Y% (quando alertar)

2. STOP LEVELS:
   - Daily DD stop: A% (parar de operar hoje)
   - Total DD stop: B% (parar de operar até reset)

3. RECOVERY MODE:
   - Quando entrar em recovery mode
   - Quanto reduzir position size em recovery

RECOMENDAÇÕES TÍPICAS:
- Warning: 50% do limite
- Stop: 80% do limite
- Recovery: Reduce size 50%

OUTPUT:
- DOCS/04_REPORTS/DECISIONS/CIRCUIT_BREAKER_THRESHOLDS.md
- Pseudo-código para implementação no EA
```

---

## 🔥 CRUCIBLE - Estratégia e Mercado

### Prompt 1: Análise de Sessões XAUUSD
```
Crucible, analise qual sessão tem melhor edge para scalping XAUUSD.

ANÁLISE:
1. Asian (00:00-08:00 UTC):
   - Volatilidade média
   - Spread médio
   - Direção predominante
   - Setup mais comum

2. London (08:00-16:00 UTC):
   - Idem

3. NY (13:00-21:00 UTC):
   - Idem

4. Overlap London/NY (13:00-16:00 UTC):
   - Idem

DADOS:
- Usar dados Parquet de 2024
- Calcular estatísticas por sessão

OUTPUT:
- DOCS/03_RESEARCH/FINDINGS/SESSION_ANALYSIS.md
- Recomendação: Qual sessão focar? Qual evitar?
- Horários específicos de melhor edge
```

### Prompt 2: Setup Atual do Mercado
```
Crucible, analise o setup atual de XAUUSD (tempo real).

VERIFICAR:
1. Preço atual vs níveis chave (suporte/resistência)
2. Regime atual (trending/ranging?) - usar Hurst ou visual
3. Correlação com DXY (dólar forte = ouro fraco?)
4. Próximos eventos de risco (FOMC, NFP, etc.)
5. Sentimento (COT positioning se disponível)

USAR MCPs:
- twelve-data: preço atual
- perplexity: notícias e eventos
- coingecko: se precisar de correlações crypto

OUTPUT:
- Análise curta (1 página)
- Recomendação: Bom momento para scalping? Esperar?
```

---

## COMO USAR ESTES PROMPTS

1. **Abra 5 sessões** do Factory CLI
2. **Cole um prompt** em cada sessão
3. **Monitore** os resultados
4. **Consolide** os outputs no final

### Ordem Recomendada:
```
PRIMEIRO (paralelo):
├── ORACLE: Validação de dados
├── FORGE: Criar segment_data.py
└── ARGUS: Research de filtros

DEPOIS (quando acima terminar):
├── ORACLE: Ablation Study
├── SENTINEL: Kelly calibration
└── CRUCIBLE: Session analysis

POR ÚLTIMO:
├── ORACLE: WFA + Monte Carlo
└── ORACLE: GO/NO-GO decision
```

---

*Prompts criados em 2025-12-01 - Copie e use!*
