# PROMPT: Iniciar Implementação do MASTER_EXECUTION_PLAN v5.2

**Copie e cole este prompt para iniciar uma nova sessão de implementação.**

---

## PROMPT

```
Você é o implementador do EA_SCALPER_XAUUSD. Temos um plano completo de validação e preciso que você:

## 1. REVISE O PLANO

Leia e analise completamente:
- `DOCS/02_IMPLEMENTATION/MASTER_EXECUTION_PLAN_FINAL.md` (v5.2 - 4,249 linhas)

Entenda:
- A AUDITORIA v5.2 que classifica scripts em: 🆕 CRIAR / 🔄 ESTENDER / ✅ PRONTO
- Os 7 princípios GENIUS (Kelly, Convexity, Phase Transitions, Fractals, Info Theory, Ensemble, Tail Risk)
- A ordem de implementação por BATCHes
- Os scripts que JÁ EXISTEM em scripts/oracle/ e scripts/backtest/

## 2. VERIFIQUE A INFRAESTRUTURA

Confirme que estes scripts existem e analise suas capacidades:
- scripts/oracle/walk_forward.py (398 linhas)
- scripts/oracle/monte_carlo.py (486 linhas)
- scripts/oracle/go_nogo_validator.py (570 linhas)
- scripts/oracle/deflated_sharpe.py (271 linhas)
- scripts/oracle/validate_data.py (733 linhas)
- scripts/backtest/tick_backtester.py (1014 linhas)

## 3. INICIE BATCH 1 (CRÍTICO)

BATCH 1 bloqueia TUDO. Implemente em ordem:

### 3.1 CRIAR: convert_tick_data.py
- Localização: scripts/data/convert_tick_data.py
- Input: Python_Agent_Hub/ml_pipeline/data/XAUUSD_ftmo_all_desde_2003.csv (24.8 GB)
- Output: data/processed/ticks_YYYY.parquet (chunked por ano/mês)
- Features: Leitura em chunks (RAM < 8GB), detecção automática de formato, normalização

### 3.2 ESTENDER: validate_data.py
Adicionar validações GENIUS ao script existente:
- Regime transition analysis (contar transições, diversity check)
- MTF consistency (H1.high == max(M5.high))
- Volatility clustering (autocorrelação de |returns|)
- Session coverage analysis (ASIA/LONDON/OVERLAP/NY/CLOSE >= 5%)
- Quality Score GENIUS 0-100

## 4. VERIFIQUE ESTRUTURA DE DIRETÓRIOS

Estes diretórios JÁ EXISTEM (criados em 2025-12-01):
- scripts/data/       ✅
- scripts/ml/         ✅
- scripts/live/       ✅
- data/processed/     ✅
- data/segments/      ✅

## 5. PADRÕES A SEGUIR

- Consulte MQL5/Include/EA_SCALPER/INDEX.md para arquitetura do EA
- Use mesma lógica de Hurst/Entropy do CRegimeDetector.mqh
- Siga convenções existentes em scripts/oracle/
- Docstrings completas em cada função
- Type hints em Python
- Testes básicos após implementação

## IMPORTANTE

- NÃO reimplemente o que já existe - ESTENDA
- Compile qualquer código MQL5 que modificar
- Rode scripts para validar que funcionam
- Documente decisões no código

## CONTEXTO ADICIONAL

- Dados tick: 24.8 GB (2003-2025)
- Dados bar: M5/M15/H1/H4 (2020-2025)
- Target: FTMO $100k Challenge
- EA já tem: Kelly adaptive, Regime detection, MTF alignment, Shannon Entropy

Comece revisando o plano e me diga:
1. Confirmação de que entendeu a estrutura
2. Scripts existentes que verificou
3. Seu plano de ação para BATCH 1
4. Alguma dúvida ou clarificação necessária

Depois de confirmar, comece a implementar convert_tick_data.py.
```

---

## CHECKLIST PRÉ-SESSÃO

Antes de iniciar, confirme:
- [x] MASTER_EXECUTION_PLAN_FINAL.md está na v5.2
- [x] Scripts oracle/ existem (walk_forward, monte_carlo, validate_data, etc.)
- [x] Diretórios organizados (scripts/data, scripts/ml, scripts/live, data/processed, data/segments)
- [ ] Dados tick disponíveis em Python_Agent_Hub/ml_pipeline/data/
- [ ] Ambiente Python funcional

---

## NOTAS

- **Esforço estimado BATCH 1**: ~12-16 horas (26GB data processing)
- **Bloqueador**: convert_tick_data.py bloqueia TODAS as outras fases
- **Prioridade**: Máxima - sem isso não há backtest
- **Infraestrutura**: Diretórios e scripts Oracle já organizados

---

*Criado: 2025-12-01*
*Referência: MASTER_EXECUTION_PLAN_FINAL.md v5.2*
