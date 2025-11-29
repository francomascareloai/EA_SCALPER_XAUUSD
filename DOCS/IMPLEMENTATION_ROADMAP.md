# 🗺️ Implementation Roadmap
## EA_SCALPER_XAUUSD - Plano de Melhorias Completo

**Versão**: 1.0
**Data**: 2025-11-28
**Status**: Em Execução

---

## 📊 Resumo Executivo

| Fase | Status | Prioridade | Estimativa |
|------|--------|------------|------------|
| 1. Papers & Indexação | 🔲 Pendente | 🔴 Crítica | 3-5 dias |
| 2. Triple Barrier + CV | 🔲 Pendente | 🔴 Crítica | 5-7 dias |
| 3. Sentiment Integration | 🔲 Pendente | 🟠 Alta | 7-10 dias |
| 4. Validação Final | 🔲 Pendente | 🟠 Alta | 5-7 dias |

**Total Estimado**: 20-29 dias

---

## FASE 1: Papers & Indexação

### 1.1 Papers para Baixar (arXiv - Gratuitos)

- [ ] **Deep Learning Statistical Arbitrage** (2021)
  - Link: https://arxiv.org/abs/2106.04028
  - PDF: https://arxiv.org/pdf/2106.04028.pdf
  - Relevância: Transformer para stat arb, convolutional architecture
  - Ação: Baixar, converter, indexar

- [ ] **StockGPT: GenAI for Stock Prediction** (2024)
  - Link: https://arxiv.org/abs/2404.05101
  - PDF: https://arxiv.org/pdf/2404.05101.pdf
  - Relevância: 119% return, Sharpe 6.5, attention mechanism
  - Ação: Baixar, converter, indexar

- [ ] **FinRL: Deep RL Framework** (2021)
  - Link: https://arxiv.org/abs/2111.09395
  - PDF: https://arxiv.org/pdf/2111.09395.pdf
  - Relevância: PPO/DQN/A2C para trading
  - Ação: Baixar, converter, indexar

- [ ] **Evolution of RL in Quant Finance Survey** (2024)
  - Link: https://arxiv.org/abs/2408.10932
  - PDF: https://arxiv.org/pdf/2408.10932.pdf
  - Relevância: Survey de 167 papers, state-of-art
  - Ação: Baixar, converter, indexar

- [ ] **Machine Learning Enhanced Multi-Factor Trading** (2025)
  - Link: https://arxiv.org/abs/2507.07107
  - PDF: https://arxiv.org/pdf/2507.07107.pdf
  - Relevância: 500-1000 factors, 20% return, Sharpe 2.0
  - Ação: Baixar, converter, indexar

### 1.2 Papers SSRN (Gratuitos com registro)

- [ ] **Machine Learning in Portfolio Decisions** (2024)
  - Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4988124
  - Relevância: Comprehensive survey, practical applications
  - Ação: Registrar SSRN, baixar, indexar

- [ ] **Generating Alpha using NLP** (2022)
  - Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4231338
  - Relevância: FinBERT, sentiment analysis para alpha
  - Ação: Baixar, indexar

### 1.3 Papers Clássicos (Buscar PDFs)

- [ ] **Kyle (1985) - Continuous Auctions and Insider Trading**
  - Relevância: Fundamento de microstructure
  - Ação: Buscar PDF gratuito ou biblioteca

- [ ] **Fama-French (1993) - Common Risk Factors**
  - Relevância: Factor models base
  - Ação: Buscar PDF gratuito

- [ ] **Jegadeesh-Titman (1993) - Returns to Buying Winners**
  - Relevância: Momentum anomaly documentation
  - Ação: Buscar PDF gratuito

### 1.4 Repositórios GitHub para Clonar

- [ ] **FinRL Library**
  - Repo: https://github.com/AI4Finance-Foundation/FinRL
  - Ação: `git clone` para referência de código

- [ ] **ML for Trading (Jansen)**
  - Repo: https://github.com/stefan-jansen/machine-learning-for-trading
  - Ação: `git clone` para notebooks e exemplos

- [ ] **MlFinLab (Hudson Thames)**
  - Repo: https://github.com/hudson-and-thames/mlfinlab
  - Ação: Instalar `pip install mlfinlab` para Triple Barrier

### 1.5 Indexação no RAG

- [ ] Converter PDFs baixados para markdown (se necessário)
- [ ] Executar ingest para cada arquivo:
  ```python
  mql5-books___ingest_file(filePath="path/to/paper.pdf")
  ```
- [ ] Validar indexação com queries de teste
- [ ] Atualizar `ML_TRADING_KNOWLEDGE_BASE.md` com novos recursos

---

## FASE 2: Triple Barrier + Cross-Validation

### 2.1 Implementar Triple Barrier Labeling

- [ ] Criar `Python_Agent_Hub/ml_pipeline/triple_barrier.py`
  ```python
  # Funções a implementar:
  # - triple_barrier_labels()
  # - get_volatility_barriers()
  # - meta_labeling()
  ```

- [ ] Criar script de teste `test_triple_barrier.py`
  - Testar com dados históricos XAUUSD
  - Comparar distribuição de labels vs fixed horizon
  - Validar que barreiras são volatility-adjusted

- [ ] Integrar com pipeline de treinamento existente
  - Modificar `train_model.py` para usar triple barrier
  - Salvar labels gerados para análise

### 2.2 Implementar Purged K-Fold CV

- [ ] Criar `Python_Agent_Hub/ml_pipeline/purged_cv.py`
  ```python
  # Classes a implementar:
  # - PurgedKFold
  # - ComboPurgedKFold (combinatorial purged)
  ```

- [ ] Implementar walk-forward analysis
  ```python
  # - WalkForwardSplit
  # - Calcular WFE (Walk-Forward Efficiency)
  ```

- [ ] Criar visualizações de CV splits
  - Plot de train/test splits
  - Verificar que não há overlap

### 2.3 Re-treinar Direction Model

- [ ] Criar novo script `train_with_improvements.py`
  - Usar Triple Barrier labels
  - Usar Purged K-Fold CV
  - Salvar métricas de cada fold

- [ ] Comparar com modelo anterior
  - Accuracy por fold
  - Out-of-sample performance
  - Sharpe ratio simulado

- [ ] Exportar modelo melhorado para ONNX
  - `direction_model_v2.onnx`
  - Atualizar `scaler_params_v2.json`

### 2.4 Validação de Melhorias

- [ ] Métricas a calcular:
  - [ ] Accuracy (train vs test)
  - [ ] Precision/Recall para cada classe
  - [ ] F1 Score
  - [ ] AUC-ROC
  - [ ] Walk-Forward Efficiency (WFE > 0.6)

- [ ] Criar relatório de comparação:
  - Before: Fixed horizon + Standard CV
  - After: Triple Barrier + Purged CV

---

## FASE 3: Sentiment Integration

### 3.1 Setup FinBERT

- [ ] Instalar dependências:
  ```bash
  pip install transformers torch
  pip install finbert-embedding  # ou usar HuggingFace
  ```

- [ ] Criar `Python_Agent_Hub/ml_pipeline/sentiment.py`
  ```python
  # Funções:
  # - load_finbert_model()
  # - get_sentiment_score(text)
  # - batch_sentiment_analysis(texts)
  ```

- [ ] Testar com exemplos de news XAUUSD

### 3.2 Coletar News Feed

- [ ] Identificar fontes gratuitas:
  - [ ] Investing.com RSS
  - [ ] ForexFactory calendar
  - [ ] Reuters (se API disponível)
  - [ ] Twitter/X (se API disponível)

- [ ] Criar scraper/collector
  ```python
  # - NewsCollector class
  # - Filtrar por XAUUSD/Gold keywords
  # - Timestamp alignment com price data
  ```

- [ ] Criar dataset de treino com news + prices

### 3.3 Arquitetura Híbrida

- [ ] Implementar fusion model:
  ```python
  class HybridLSTMTransformer(nn.Module):
      def __init__(self):
          # LSTM for price features
          # Transformer (FinBERT) for sentiment
          # Fusion layer
  ```

- [ ] Treinar modelo híbrido
- [ ] Comparar com LSTM-only

### 3.4 Integração com EA

- [ ] Criar endpoint `/api/v1/sentiment` no Python Hub
- [ ] Modificar `COnnxBrain.mqh` para incluir sentiment
- [ ] Testar latência (deve ser < 400ms total)

---

## FASE 4: Validação Final

### 4.1 Walk-Forward Analysis Completo

- [ ] Configurar WFA:
  - [ ] 10+ janelas de treino/teste
  - [ ] Período: 2020-2024 (4 anos)
  - [ ] Test window: 3 meses cada

- [ ] Executar WFA para cada modelo:
  - [ ] Baseline (antes das melhorias)
  - [ ] Triple Barrier only
  - [ ] Triple Barrier + Purged CV
  - [ ] Híbrido com sentiment

- [ ] Calcular WFE (Walk-Forward Efficiency)
  - Target: WFE > 0.6

### 4.2 Monte Carlo Simulation

- [ ] Configurar simulação:
  - [ ] 5,000+ runs
  - [ ] Bootstrap de trades históricos
  - [ ] Calcular distribuição de returns

- [ ] Métricas a extrair:
  - [ ] Expected return (média)
  - [ ] Drawdown distribution (95th percentile)
  - [ ] Probability of ruin
  - [ ] Confidence intervals

### 4.3 Comparar com Achilles Benchmark

- [ ] Replicar setup Achilles:
  - Budget: $1,000
  - Risk: 0.3
  - Período: 1 mês

- [ ] Comparar resultados:
  | Métrica | Achilles | Nosso EA |
  |---------|----------|----------|
  | Return | 62.3% | ? |
  | Max DD | ? | ? |
  | Win Rate | ? | ? |

### 4.4 Deploy em Demo

- [ ] Configurar conta demo FTMO
- [ ] Deploy EA com modelo melhorado
- [ ] Monitorar por 2 semanas mínimo
- [ ] Coletar métricas reais:
  - [ ] Slippage real vs backtest
  - [ ] Latência de execução
  - [ ] FTMO compliance

---

## FASE 5: Documentação & Manutenção (Ongoing)

### 5.1 Atualizar Documentos

- [ ] Atualizar `CLAUDE.md` com novos módulos
- [ ] Atualizar `INDEX.md` com melhorias
- [ ] Documentar novos scripts Python
- [ ] Criar changelog detalhado

### 5.2 Criar Testes Automatizados

- [ ] Testes unitários para Triple Barrier
- [ ] Testes unitários para Purged CV
- [ ] Testes de integração para pipeline completo
- [ ] CI/CD com GitHub Actions (se aplicável)

### 5.3 Monitoramento Contínuo

- [ ] Setup de alertas para:
  - [ ] Drawdown > threshold
  - [ ] Model drift detection
  - [ ] Regime change (Hurst/Entropy)
  - [ ] News events importantes

---

## 📚 Recursos de Referência Rápida

### RAG Queries Essenciais

```bash
# Triple Barrier
mql5-books: "triple barrier labeling volatility profit loss"

# Purged CV
mql5-books: "cross validation time series purged embargo"

# LSTM Architecture
mql5-books: "LSTM hidden state forget gate neural network"

# FinBERT
mql5-books: "sentiment analysis BERT financial news"

# XAUUSD Specific
mql5-books: "gold XAUUSD prediction LSTM commodity"
```

### Comandos Úteis

```bash
# Baixar paper arXiv
curl -o paper.pdf https://arxiv.org/pdf/XXXX.XXXXX.pdf

# Indexar no RAG (via MCP)
# Use mql5-books___ingest_file

# Treinar modelo
python Python_Agent_Hub/ml_pipeline/train_with_improvements.py

# Exportar ONNX
python Python_Agent_Hub/export_onnx.py --model direction_v2
```

### Links Importantes

| Recurso | URL |
|---------|-----|
| arXiv Quant Finance | https://arxiv.org/list/q-fin/recent |
| SSRN Finance | https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalbrowse&journal_id=556869 |
| FinRL GitHub | https://github.com/AI4Finance-Foundation/FinRL |
| MlFinLab Docs | https://mlfinlab.readthedocs.io/ |
| HuggingFace FinBERT | https://huggingface.co/ProsusAI/finbert |

---

## 📈 Tracking de Progresso

### Semana 1: ___ / ___ / ___

| Dia | Tarefa | Status | Notas |
|-----|--------|--------|-------|
| D1 | Baixar papers arXiv | 🔲 | |
| D2 | Indexar papers no RAG | 🔲 | |
| D3 | Clonar repos GitHub | 🔲 | |
| D4 | Testar queries RAG | 🔲 | |
| D5 | Documentar findings | 🔲 | |

### Semana 2: ___ / ___ / ___

| Dia | Tarefa | Status | Notas |
|-----|--------|--------|-------|
| D1 | Implementar Triple Barrier | 🔲 | |
| D2 | Testar Triple Barrier | 🔲 | |
| D3 | Implementar Purged CV | 🔲 | |
| D4 | Testar Purged CV | 🔲 | |
| D5 | Integrar com pipeline | 🔲 | |
| D6 | Re-treinar modelo | 🔲 | |
| D7 | Comparar métricas | 🔲 | |

### Semana 3: ___ / ___ / ___

| Dia | Tarefa | Status | Notas |
|-----|--------|--------|-------|
| D1-2 | Setup FinBERT | 🔲 | |
| D3-4 | News collector | 🔲 | |
| D5-6 | Treinar híbrido | 🔲 | |
| D7 | Integrar com EA | 🔲 | |

### Semana 4: ___ / ___ / ___

| Dia | Tarefa | Status | Notas |
|-----|--------|--------|-------|
| D1-2 | Walk-Forward Analysis | 🔲 | |
| D3-4 | Monte Carlo | 🔲 | |
| D5 | Comparar Achilles | 🔲 | |
| D6-7 | Deploy demo | 🔲 | |

---

## 🎯 Critérios de Sucesso

### Fase 1
- [ ] ≥10 papers indexados no RAG
- [ ] Queries retornando resultados relevantes

### Fase 2
- [ ] Triple Barrier implementado e testado
- [ ] WFE > 0.6 no novo modelo
- [ ] Melhoria mensurável vs baseline

### Fase 3
- [ ] FinBERT funcionando com latência < 100ms
- [ ] Modelo híbrido treinado
- [ ] Integração com EA completa

### Fase 4
- [ ] WFA completo (10+ janelas)
- [ ] Monte Carlo (5000+ runs)
- [ ] 2 semanas em demo sem violação FTMO
- [ ] Performance ≥ Achilles benchmark

---

## 📞 Pontos de Decisão

### Checkpoint 1 (Fim Fase 1)
**Pergunta**: Papers suficientes? RAG funcionando?
**Se NÃO**: Buscar mais papers, ajustar queries

### Checkpoint 2 (Fim Fase 2)
**Pergunta**: WFE > 0.6? Melhoria vs baseline?
**Se NÃO**: Revisar implementação, ajustar hiperparâmetros

### Checkpoint 3 (Fim Fase 3)
**Pergunta**: Híbrido melhor que LSTM-only?
**Se NÃO**: Considerar manter LSTM-only, ajustar fusion

### Checkpoint 4 (Fim Fase 4)
**Pergunta**: Pronto para live?
**Se NÃO**: Mais tempo em demo, identificar problemas

---

*Última atualização: 2025-11-28*
*Próxima revisão: [preencher após início]*
