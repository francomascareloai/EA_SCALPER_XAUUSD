# Checklists - CRUCIBLE v3.0

## Pre-Trade Checklist (15 Gates) - Integrado com EA

```
GATES CRITICOS (FAIL = 🛑 NO GO imediato):
□ 1.  [CRegimeDetector] Hurst fora de 0.45-0.55 (nao Random Walk)?
□ 2.  [CRegimeDetector] Entropy: < 2.5?
□ 11. [FTMO_RiskManager] Daily DD: < 4% (buffer FTMO)?
□ 12. [FTMO_RiskManager] Total DD: < 8% (buffer FTMO)?
□ 15. [CConfluenceScorer] Score confluencia: >= 70?

GATES NORMAIS (contam para score):
□ 3.  [CSessionFilter] Sessao: Nao e Asia (22:00-07:00 GMT)?
□ 4.  [Manual] Spread: < 30 pontos?
□ 5.  [CNewsFilter] News: Nenhum HIGH impact em 30min?
□ 6.  [CMTFManager] H1 Trend: Alinhado com direcao do trade?
□ 7.  [EliteOrderBlock/EliteFVG] M15 Zone: Em OB ou FVG valido?
□ 8.  [CMTFManager] M5 Confirm: Candle de confirmacao presente?
□ 9.  [CFootprintAnalyzer] Order Flow: Delta confirma direcao?
□ 10. [CLiquiditySweepDetector] Liquidity: Sweep recente ocorreu?
□ 13. [Manual] Posicoes: < 3 abertas?
□ 14. [Manual/CEntryOptimizer] R:R: Minimo 2:1?

RESULTADO: [X/15] gates passados
├── >= 13: GO (Tier A) - Executar com size normal
├── 11-12: CAUTION (Tier B) - Executar com size 75%
└── < 11:  NO GO (Tier C/D) - Nao executar
```

---

## FTMO Compliance Checklist

```
LIMITES ABSOLUTOS ($100k account):
□ Max Daily Loss: 5% ($5,000)
□ Buffer diario usado: 4% ($4,000) - trigger de alerta
□ Max Total Loss: 10% ($10,000)
□ Buffer total usado: 8% ($8,000) - trigger de alerta

TARGETS:
□ Profit Target Phase 1: 10%
□ Profit Target Phase 2: 5%
□ Min Trading Days: 4

OPERACIONAL:
□ Risk por trade: <= 1% (ideal 0.5%)
□ Max leverage: Dentro do permitido
□ Weekend positions: Ciente dos riscos
□ News trading: Permitido (com cuidado)
```

---

## Code Review Checklist (20 items)

### FTMO Compliance (5)
```
□ Daily DD calculado corretamente?
□ Total DD calculado corretamente?
□ Buffer (4%/8%) implementado?
□ Emergency stop funciona?
□ Max lot limitado?
```

### Risk Management (5)
```
□ Kelly/Fractional Kelly implementado?
□ SL SEMPRE definido antes de entry?
□ Slippage controlado/limitado?
□ Magic number unico por estrategia?
□ Trade comments para identificacao?
```

### Entry Logic (5)
```
□ Regime filter ativo?
□ Session filter ativo?
□ News filter ativo?
□ MTF alignment verificado?
□ Confluencia minima exigida?
```

### Execution (5)
```
□ Retry em caso de requote?
□ Error handling em OrderSend?
□ Spread check antes de entry?
□ Latencia < 50ms?
□ Logging suficiente para debug?
```

**RESULTADO: [X/20]**

---

## Setup Validation Gates (Detalhado)

### Gate 1: Regime Check
```
PASS: Hurst < 0.45 (reverting) OU Hurst > 0.55 (trending)
FAIL: Hurst entre 0.45-0.55 (random walk)
```

### Gate 2: Entropy Check
```
PASS: Entropy < 2.5
FAIL: Entropy >= 2.5 (muito ruido)
```

### Gate 3: Session Check
```
PASS: London (08:00-12:00), NY (13:00-17:00), Overlap (12:00-16:00)
FAIL: Asia (22:00-07:00), After Hours (20:00-22:00)
```

### Gate 4: Spread Check
```
PASS: Spread <= 30 pontos
CAUTION: 25-30 pontos
FAIL: > 30 pontos
```

### Gate 5: News Check
```
PASS: Sem HIGH impact em 30min
CAUTION: MEDIUM impact em 15min
FAIL: HIGH impact em 30min
```

### Gate 6: HTF Alignment
```
PASS: H1 trend alinhado com direcao
FAIL: H1 contra direcao do trade
```

### Gate 7: MTF Zone
```
PASS: Preco em Order Block ou FVG valido no M15
FAIL: Preco em "no man's land"
```

### Gate 8: LTF Confirmation
```
PASS: M5 mostra candle de confirmacao (engulf, pin bar)
FAIL: Sem confirmacao clara
```

### Gate 9: Order Flow
```
PASS: Delta positivo para long, negativo para short
BONUS: Stacked imbalance presente
FAIL: Delta contra direcao
```

### Gate 10: Liquidity
```
PASS: Liquidity sweep recente (BSL para short, SSL para long)
BONUS: Sweep + retorno a zona
FAIL: Sem sweep visivel
```

### Gate 11-12: DD Check
```
PASS: Daily DD < 4% E Total DD < 8%
CAUTION: Daily 3-4% OU Total 6-8%
FAIL: Daily >= 4% OU Total >= 8%
```

### Gate 13: Position Count
```
PASS: < 3 posicoes abertas
CAUTION: 2 posicoes abertas
FAIL: >= 3 posicoes
```

### Gate 14: Risk/Reward
```
PASS: R:R >= 2:1
CAUTION: R:R 1.5:1 a 2:1
FAIL: R:R < 1.5:1
```

### Gate 15: Confluence Score
```
PASS: Score >= 70/100
CAUTION: Score 55-69
FAIL: Score < 55
```

---

## Regime-Based Strategy Selection

```
ANTES DE CADA TRADE - VERIFICAR:

1. □ Regime atual calculado (Hurst + Entropy)
2. □ Strategy correspondente selecionada:
   
   PRIME_TRENDING → TREND_FOLLOW
   - Entry: Breakout + Pullback
   - Exit: Trailing Stop
   - Size: 100%
   
   NOISY_TRENDING → TREND_FILTER
   - Entry: Pullbacks only
   - Exit: Fixed TP 1.5-2R
   - Size: 75%
   
   MEAN_REVERTING → RANGE_BOUNCE
   - Entry: S/R + Reversal
   - Exit: Opposite S/R
   - Size: 50%
   
   RANDOM_WALK → NO_TRADE
   - Entry: NENHUMA
   - Size: 0%

3. □ Size multiplier aplicado
4. □ Nao em transicao recente (< 15min desde mudanca)
```
