# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 2: Design Detalhado do EA em MQL5

---

# ⚙️ SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5

## 3.1 Módulos e Responsabilidades

### COrderBlockModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Detectar Order Blocks bullish/bearish em múltiplos timeframes (M15, H1, H4) |
| **Inputs** | Série de candles, lookback period, threshold de validação |
| **Outputs** | `bool hasValidOB`, `double obLevel`, `ENUM_OB_TYPE obType`, `int obStrength (0-100)` |

### CFVGModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Identificar Fair Value Gaps não mitigados, calcular probabilidade de fill |
| **Inputs** | Série de candles, minimum gap size (em pontos) |
| **Outputs** | `bool hasFVG`, `double fvgHigh`, `double fvgLow`, `bool fvgBullish`, `int fvgAge` |

### CLiquidityModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Mapear pools de liquidez (equal highs/lows, swing points), detectar sweeps |
| **Inputs** | Série de candles, swing detection parameters |
| **Outputs** | `bool liquiditySwept`, `double liquidityLevel`, `ENUM_SWEEP_TYPE sweepType` |

### CMarketStructureModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Analisar estrutura de mercado (HH/HL para bullish, LH/LL para bearish), detectar BOS/CHoCH |
| **Inputs** | Série de candles, structure timeframe |
| **Outputs** | `ENUM_MARKET_STRUCTURE currentStructure`, `bool hasBOS`, `bool hasCHoCH`, `int structureStrength` |

### CVolatilityModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Calcular ATR, detectar condições de volatilidade anormal, ajustar SL/TP dinamicamente |
| **Inputs** | ATR period, volatility thresholds |
| **Outputs** | `double currentATR`, `ENUM_VOLATILITY_STATE volState`, `double suggestedSL`, `double suggestedTP` |

### CSignalScoringModule
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Combinar todos os sinais em scores, integrar dados do Python Hub, calcular FinalScore |
| **Inputs** | Outputs de todos os módulos técnicos, dados do Python Hub (Fund, Sent) |
| **Outputs** | `double techScore`, `double finalScore`, `string reasoningComponents` |

### CFTMORiskManager
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Controle absoluto de risco, cálculo de lot size, veto de trades, dynamic drawdown control |
| **Inputs** | Account balance, equity, daily P/L, risk parameters |
| **Outputs** | `bool canTrade`, `double allowedLot`, `double currentDDPercent`, `ENUM_RISK_STATE riskState` |

### CTradeExecutor
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Enviar ordens ao servidor, retry logic, verificar execução, gerenciar posições abertas |
| **Inputs** | Trade parameters (direction, lot, SL, TP, magic number) |
| **Outputs** | `bool executionSuccess`, `ulong ticket`, `double actualFillPrice`, `double slippage` |

### CLogger
| Aspecto | Descrição |
|---------|-----------|
| **Responsabilidades** | Logging estruturado, push notifications, geração de reasoning strings, export para análise |
| **Inputs** | Trade data, module outputs, reasoning components |
| **Outputs** | Arquivos de log, notificações push, CSV para análise |

---

## 3.2 Pseudocódigo do OnTick Ideal

```pseudocode
FUNCTION OnTick():
    
    // ═══════════════════════════════════════════════════════
    // FASE 1: VERIFICAÇÕES INICIAIS (< 1ms)
    // ═══════════════════════════════════════════════════════
    
    // 1.1 Reset diário se novo dia
    IF IsNewDay():
        ResetDailyTracking()
        RiskManager.ResetDaily()
    
    // 1.2 Atualizar Risk Manager com equity atual
    RiskManager.Update(AccountEquity)
    
    // 1.3 Early exit se trading bloqueado
    IF NOT RiskManager.IsTradingAllowed():
        RETURN
    
    // 1.4 Verificar spread
    IF CurrentSpread > MaxSpreadAllowed:
        RETURN
    
    // ═══════════════════════════════════════════════════════
    // FASE 2: GERENCIAMENTO DE POSIÇÃO ABERTA (< 5ms)
    // ═══════════════════════════════════════════════════════
    
    IF HasOpenPosition():
        ManageOpenPosition()  // Trailing stop, break-even
        RETURN  // Não abrir novas posições
    
    // ═══════════════════════════════════════════════════════
    // FASE 3: RATE LIMITING
    // ═══════════════════════════════════════════════════════
    
    TickCounter++
    IF TickCounter MOD 5 != 0:
        RETURN  // Processar a cada 5 ticks
    
    IF TradesToday >= MaxTradesPerDay:
        RETURN
    
    // ═══════════════════════════════════════════════════════
    // FASE 4: ANÁLISE TÉCNICA (< 20ms total)
    // ═══════════════════════════════════════════════════════
    
    START_TIMER()
    
    obData     = OrderBlockModule.Analyze()
    fvgData    = FVGModule.Analyze()
    liqData    = LiquidityModule.Analyze()
    structData = MarketStructureModule.Analyze()
    volData    = VolatilityModule.Analyze()
    
    analysisTime = STOP_TIMER()
    IF analysisTime > 20ms:
        Logger.Warn("Analysis exceeded 20ms")
    
    // ═══════════════════════════════════════════════════════
    // FASE 5: SCORING (< 5ms)
    // ═══════════════════════════════════════════════════════
    
    techScore = ScoringModule.ComputeTechScore(
        obData, fvgData, liqData, structData, volData
    )
    
    // Pre-filter: economizar processamento
    IF techScore < TECH_SCORE_PREFILTER (60):
        RETURN
    
    // Obter scores do Python (do cache, atualizado em OnTimer)
    pythonData = GetCachedPythonData()
    
    finalScore = ScoringModule.ComputeFinalScore(
        techScore, 
        pythonData.fundScore, 
        pythonData.sentScore
    )
    
    // ═══════════════════════════════════════════════════════
    // FASE 6: DECISÃO DE TRADE (< 2ms)
    // ═══════════════════════════════════════════════════════
    
    IF finalScore < EXECUTION_THRESHOLD (85):
        Logger.Debug("Signal rejected: " + finalScore)
        RETURN
    
    tradeDirection = DetermineTradeDirection(obData, structData, liqData)
    IF tradeDirection == NONE:
        RETURN  // Sem direção clara
    
    // Calcular SL/TP baseado em ATR
    slPoints = volData.currentATR * 1.5
    tpPoints = slPoints * RiskRewardRatio
    
    // ═══════════════════════════════════════════════════════
    // FASE 7: VALIDAÇÃO DE RISCO (CRÍTICO) (< 2ms)
    // ═══════════════════════════════════════════════════════
    
    riskCheck = RiskManager.CanOpenTrade(RiskPercent, slPoints)
    
    IF NOT riskCheck.approved:
        Logger.Info("VETOED: " + riskCheck.reason)
        SendPush("⚠️ Trade vetoed: " + riskCheck.reason)
        RETURN
    
    // ═══════════════════════════════════════════════════════
    // FASE 8: EXECUÇÃO (< 10ms)
    // ═══════════════════════════════════════════════════════
    
    lotSize = riskCheck.allowedLot
    
    result = TradeExecutor.OpenPosition(
        tradeDirection, lotSize, slPoints, tpPoints
    )
    
    IF result.success:
        TradesToday++
        reasoning = GenerateReasoningString(...)
        Logger.LogTrade(result.ticket, reasoning)
        SendPush(reasoning)

END FUNCTION
```

---

## 3.3 OnTimer para Chamadas Python (Assíncrono)

```pseudocode
FUNCTION OnTimer():  // Executado a cada 30 segundos
    
    IF NOT UsePythonHub:
        RETURN
    
    // Preparar request
    request = BuildPythonRequest(
        symbol = _Symbol,
        timeframe = CurrentTimeframe,
        price = CurrentPrice,
        atr = LastATR,
        structure = LastStructure
    )
    
    // Fazer chamada HTTP (com timeout de 5s)
    response = HTTPPost(PythonHubURL, request, timeout=5000)
    
    IF response.success:
        // Atualizar cache global
        PythonCache.techSubscore = response.tech_subscore
        PythonCache.fundScore = response.fund_score
        PythonCache.sentScore = response.sent_score
        PythonCache.fundBias = response.fund_bias
        PythonCache.sentBias = response.sent_bias
        PythonCache.llmReasoning = response.llm_reasoning
        PythonCache.lastUpdate = TimeCurrent()
        PythonCache.isValid = TRUE
    ELSE:
        // Marcar como stale após 5 minutos sem update
        IF TimeCurrent() - PythonCache.lastUpdate > 300:
            PythonCache.isValid = FALSE
            Logger.Warn("Python data stale - operating MQL5-only")

END FUNCTION
```
