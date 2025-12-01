# MTF Manager – Future Improvements (CMTFManager.mqh)

**Contexto**  
Módulo atual já atende ao PRD (H1 direção, M15 estrutura, M5 execução) com filtros de Hurst, força de tendência, confluência e multiplicadores 100/75/50/0%. As melhorias abaixo são refinamentos incrementais, não bugs.

---

## 1. Trend M15 como vetor independente

- Hoje o `m_mtf.trend` herda direção do H1.  
- Melhorar para:
  - Calcular trend próprio em M15 (ex.: mini-versão de `DetermineTrend` e `CalculateTrendStrength` para M15).  
  - Rebaixar o alinhamento quando H1 e M15 divergem (ex.: PERFECT → GOOD / WEAK).  
  - Opcional: bloquear sinais quando H1 e M15 estão explicitamente opostos.

Benefício: validação cruzada entre H1 e M15, evitando operar pullbacks profundos contra a estrutura M15.

---

## 2. BOS/CHoCH M15 mais rico

- Lógica atual: usa apenas último swing e uma condição simples de rompimento com a barra anterior.  
- Melhorar para:
  - Analisar uma cadeia de swings (ex.: últimos 3–5) para classificar BOS/CHoCH com mais contexto.  
  - Exigir rompimento mínimo em ATR (ex.: breakout > 0.5–1.0 ATR) para filtrar ruído.  
  - Marcar explicitamente “trend leg” vs “pullback leg” em M15 com base na sequência de swings.

Benefício: BOS/CHoCH mais estável e menos sensível a falsos rompimentos de 1–2 candles.

---

## 3. Session gating dentro do MTF Manager

- Campo `session_ok` existe em `SMTFConfluence`, mas não é utilizado.  
- Melhorar para:
  - Integrar um filtro de sessão (London / NY / Asia) diretamente no `GetConfluence()`.  
  - Preencher `session_ok` com base em janela horária configurável.  
  - Ponderar confiança: fora de sessão “core” (ex.: London/NY), reduzir confiança ou forçar alinhamento mínimo maior.

Benefício: amarrar o mesmo MTF manager à disciplina de sessões do PRD, evitando setups “perfeitos” em horários de liquidez baixa.

---

## 4. Tuning fino de thresholds de confluência

- Hoje `m_min_trend_strength` e `m_min_confluence` são usados em `CanTradeLong/Short`, mas com valores default genéricos (30 / 60).  
- Melhorar para:
  - Expor estes thresholds nos inputs do EA com presets por regime.  
  - Ajustar por regime detectado (ex.: exigir mais confluência em regime noisy, permitir menos em prime).  
  - Validar via backtest/WFA quais patamares de 60/70/80 maximizam SQN e reduzem DD.

Benefício: knobs que já existem passam a ser calibrados empiricamente, não “hard-coded”.

---

## 5. Telemetria e debug dedicado

- Adicionar hooks opcionais de log:
  - Dump de `SMTFConfluence` (alignment, confidence, position_size_mult) na hora do sinal.  
  - Flags de qual pilar faltou (HTF, MTF, LTF, sessão, regime) quando um sinal é rejeitado.  

Benefício: facilita post-mortem de entradas perdidas e falsos positivos, e ajuda a calibrar thresholds e pesos de confluência.

