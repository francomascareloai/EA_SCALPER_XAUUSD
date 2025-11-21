# Parâmetros dos EAs - EA_SCALPER_XAUUSD

## Overview

Este documento descreve todos os parâmetros configuráveis dos Expert Advisors (EAs) do projeto EA_SCALPER_XAUUSD, incluindo suas descrições detalhadas, valores recomendados, validações e exemplos de uso.

## Sumário

1. [EAs Principais](#eas-principais)
2. [EAs de Scalping](#eas-de-scalping)
3. [Parâmetros de Risco](#parâmetros-de-risco)
4. [Parâmetros de Machine Learning](#parâmetros-de-machine-learning)
5. [Parâmetros de Interface Visual](#parâmetros-de-interface-visual)
6. [Parâmetros de Performance](#parâmetros-de-performance)
7. [Configurações por Ativo](#configurações-por-ativo)
8. [Exemplos de Configuração](#exemplos-de-configuração)
9. [Validação de Parâmetros](#validação-de-parâmetros)
10. [Troubleshooting](#troubleshooting)

---

## EAs Principais

### XAUUSD ML Trading Bot

#### Descrição
EA avançado com Machine Learning para trading de XAUUSD (Gold/USD), incorporando múltiplas estratégias e gerenciamento de risco sofisticado.

#### Estrutura de Parâmetros

```mql5
//=== ML CONFIGURATION ===
input bool     EnableMLPrediction = true;                    // Enable ML Prediction
input double   MLConfidenceThreshold = 0.75;                 // ML Confidence Threshold (0.0-1.0)
input int      MLModelUpdateHours = 24;                      // ML Model Update Frequency (hours)

//=== RISK MANAGEMENT ===
input double   BaseRiskPercent = 0.01;                       // Base Risk per Trade (%)
input double   MaxDailyRisk = 0.02;                          // Maximum Daily Risk (%)
input double   MaxDrawdownPercent = 0.03;                    // Maximum Drawdown (%)
input bool     EnableFTMOCompliance = true;                  // Enable FTMO Compliance

//=== STRATEGY SELECTION ===
input bool     EnableSmartMoney = true;                      // Enable ICT Smart Money Strategy
input bool     EnableMLScalping = true;                      // Enable ML Adaptive Scalping
input bool     EnableVolatilityBreakout = true;              // Enable Volatility Breakout
input bool     EnableMultiTimeframe = true;                  // Enable Multi-Timeframe Analysis

//=== LATENCY OPTIMIZATION ===
input int      MaxLatencyMS = 120;                           // Maximum Acceptable Latency (ms)
input bool     EnablePreStops = true;                        // Enable Pre-Stop Validation
input bool     EnableOrderBuffering = true;                  // Enable Order Buffering
input double   SlippageTolerancePips = 2.0;                  // Slippage Tolerance (pips)

//=== VISUAL INTERFACE ===
input bool     EnableVisualInterface = true;                 // Enable Visual Dashboard
input bool     ShowRealTimeAnalysis = true;                  // Show Real-Time Analysis
input bool     ShowDecisionProcess = true;                   // Show Decision Process
input color    InterfaceColor = clrCyan;                     // Interface Color
```

#### Parâmetros Detalhados

| Parâmetro | Tipo | Padrão | Intervalo | Descrição |
|-----------|------|--------|-----------|-----------|
| `EnableMLPrediction` | bool | true | - | Ativa predições de Machine Learning |
| `MLConfidenceThreshold` | double | 0.75 | 0.0 - 1.0 | Limiar de confiança para decisões ML |
| `MLModelUpdateHours` | int | 24 | 1 - 168 | Frequência de atualização do modelo (horas) |
| `BaseRiskPercent` | double | 0.01 | 0.001 - 0.05 | Risco base por trade (%) |
| `MaxDailyRisk` | double | 0.02 | 0.005 - 0.10 | Risco máximo diário (%) |
| `MaxDrawdownPercent` | double | 0.03 | 0.01 - 0.15 | Drawdown máximo permitido (%) |
| `EnableFTMOCompliance` | bool | true | - | Ativa modo de conformidade FTMO |
| `MaxLatencyMS` | int | 120 | 50 - 500 | Latência máxima aceitável (ms) |
| `SlippageTolerancePips` | double | 2.0 | 0.5 - 10.0 | Tolerância de slippage (pips) |

### EA Scalping3 v1.0

#### Descrição
EA de scalping versátil com suporte para múltiplos ativos (Forex, Crypto, Gold, Índices).

#### Estrutura de Parâmetros

```mql5
//=== Trading Profiles ===
input SystemType SType = 0;                                   // Trading System applied

//=== Common Trading Inputs ===
input double RiskPercent = 3;                                 // Risk as % of Trading Capital
input ENUM_TIMEFRAMES TimeFrame = PERIOD_CURRENT;             // Time frame to run
input int    InpMagic = 298347;                               // EA identification no
input string TradeComment = "Scalping Robot";                 // Trade comment
input StartHour SHInput = 8;                                  // Start Hour
input EndHour EHInput = 21;                                   // End Hour

//=== Visual Settings ===
input color ChartColorTradingOff = clrPink;                   // Chart color when EA is Inactive
input color ChartColorTradingOn = clrBlack;                   // Chart color when EA is active
input bool  HideIndicators = true;                            // Hide Indicators on Chart?
```

#### Configurações por Ativo

**Forex Trading Inputs**
```mql5
input int    TpPoints = 200;                                  // Take Profit (10 points = 1 pip)
input int    SlPoints = 200;                                  // Stoploss Points (10 points = 1 pip)
input int    TslTriggerPointsInputs = 15;                     // Points in profit before Trailing SL
input int    TslPointsInputs = 10;                            // Trailing Stop Loss
```

**Crypto Related Inputs**
```mql5
input double TPasPct = 0.4;                                   // TP as % of Price
input double SLasPct = 0.4;                                   // SL as % of Price
input double TSLasPctofTP = 5;                                // Trail SL as % of TP
input double TSLTrgasPctofTP = 7;                             // Trigger of Trail SL % of TP
```

**Gold Related Inputs**
```mql5
input double TPasPctGold = 0.2;                               // TP as % of Price
input double SLasPctGold = 0.2;                               // SL as % of Price
input double TSLasPctofTPGold = 5;                            // Trail SL as % of TP
```

---

## EAs de Scalping

### Crosshair MTF Zones

#### Descrição
Indicador/EA que mostra zonas de suporte e resistência de múltiplos timeframes com crosshair interativo.

#### Parâmetros de Crosshair
```mql5
input color CrosshairColor = clrRed;                          // Crosshair color
input int   CrosshairWidth = 1;                               // Crosshair width
input ENUM_LINE_STYLE CrosshairStyle = STYLE_DOT;             // Crosshair line style
```

#### Parâmetros de Zonas
```mql5
input int   ZoneTransparency = 30;                            // Zone transparency (0-100)
input bool  ShowZoneHistory = false;                          // Show historical zones
input bool  EnableAlerts = true;                              // Enable zone alerts
```

#### Parâmetros de Notificação
```mql5
input bool  EnablePush = false;                               // Enable push notifications
input bool  EnableEmail = false;                              // Enable email notifications
input bool  EnableTelegram = false;                           // Enable Telegram notifications
input bool  EnableWhatsApp = false;                           // Enable WhatsApp notifications
```

#### Parâmetros por Timeframe
```mql5
// Monthly to M1 settings
input bool Show_MN1 = true;   input color Color_MN1 = clrMaroon;    // Monthly
input bool Show_W1  = true;   input color Color_W1  = clrTeal;      // Weekly
input bool Show_D1  = true;   input color Color_D1  = clrGreen;     // Daily
input bool Show_H4  = true;   input color Color_H4  = clrBlue;      // 4 Hour
input bool Show_H2  = true;   input color Color_H2  = clrIndigo;    // 2 Hour
input bool Show_H1  = true;   input color Color_H1  = clrOrange;    // 1 Hour
input bool Show_M30 = true;   input color Color_M30 = clrDarkViolet;// 30 Min
input bool Show_M15 = true;   input color Color_M15 = clrBrown;     // 15 Min
input bool Show_M5  = true;   input color Color_M5  = clrFireBrick; // 5 Min
input bool Show_M1  = true;   input color Color_M1  = clrDarkSlateGray; // 1 Min
```

---

## Parâmetros de Risco

### Tipos de Gestão de Risco

#### 1. Percentual Fixo
```mql5
input double FixedRiskPercent = 1.0;                          // Fixed risk percentage
input double MaxDailyLossPercent = 2.0;                       // Maximum daily loss
input double MaxWeeklyLossPercent = 5.0;                      // Maximum weekly loss
```

#### 2. Gestão Dinâmica
```mql5
input bool EnableDynamicRisk = true;                          // Enable dynamic risk
input double RiskMultiplier = 1.5;                            // Risk multiplier on wins
input double RiskReducer = 0.5;                               // Risk reducer on losses
input int ConsecutiveWinsForIncrease = 3;                     // Wins before risk increase
input int ConsecutiveLossesForDecrease = 2;                   // Losses before risk decrease
```

#### 3. Parâmetros FTMO
```mql5
input bool EnableFTMO = true;                                 // Enable FTMO compliance
input double MaxDailyLossFTMO = 2.0;                          // FTMO daily loss limit
input double MaxTotalLossFTMO = 10.0;                         // FTMO total loss limit
input int MinTradingDaysFTMO = 10;                            // Minimum trading days
input double ProfitTargetFTMO = 10.0;                         // FTMO profit target
```

### Gestão de Posições

#### Stop Loss e Take Profit
```mql5
input double FixedSLPips = 20.0;                              // Fixed Stop Loss (pips)
input double FixedTPPips = 30.0;                              // Fixed Take Profit (pips)
input bool EnableTrailingSL = true;                           // Enable trailing stop
input double TrailingDistance = 15.0;                         // Trailing distance (pips)
input double TrailingStep = 5.0;                              // Trailing step (pips)
```

#### Gestão de Lotes
```mql5
input double FixedLotSize = 0.01;                             // Fixed lot size
input bool EnableAutoLot = true;                              // Enable auto lot sizing
input double LotPerRisk = 0.1;                                // Lot per 1% risk
input double MaxLotSize = 1.0;                                // Maximum lot size
input double MinLotSize = 0.01;                               // Minimum lot size
```

---

## Parâmetros de Machine Learning

### Configurações de Modelos

#### Predição e Confiança
```mql5
input bool EnableMLModels = true;                             // Enable ML models
input double MinConfidenceScore = 0.75;                       // Minimum confidence score
input int MaxPredictionTime = 100;                            // Max prediction time (ms)
input string ModelPath = "ML/Models/";                        // ML models path
```

#### Treinamento e Atualização
```mql5
input bool EnableOnlineLearning = true;                       // Enable online learning
input int RetrainingInterval = 168;                           // Retraining interval (hours)
input int MinTrainingSamples = 1000;                          // Minimum training samples
input double LearningRate = 0.001;                            // Learning rate
```

#### Features e Indicadores
```mql5
input bool UsePriceAction = true;                             // Use price action features
input bool UseVolumeAnalysis = true;                          // Use volume analysis
input bool UseSentimentData = false;                          // Use sentiment data
input bool UseTechnicalIndicators = true;                     // Use technical indicators
input int FeatureWindowSize = 50;                             // Feature window size
```

---

## Parâmetros de Interface Visual

### Dashboard e Exibição

#### Interface Principal
```mql5
input bool ShowDashboard = true;                              // Show main dashboard
input color DashboardColor = clrCyan;                         // Dashboard color
input int DashboardX = 20;                                    // Dashboard X position
input int DashboardY = 20;                                    // Dashboard Y position
input bool EnableTooltips = true;                             // Enable tooltips
```

#### Análise em Tempo Real
```mql5
input bool ShowRealTimeSignals = true;                        // Show real-time signals
input bool ShowPredictionConfidence = true;                   // Show prediction confidence
input bool ShowRiskMetrics = true;                            // Show risk metrics
input bool ShowPerformanceStats = true;                       // Show performance statistics
```

#### Customização Visual
```mql5
input color BuySignalColor = clrLime;                         // Buy signal color
input color SellSignalColor = clrRed;                         // Sell signal color
input color NeutralSignalColor = clrYellow;                   // Neutral signal color
input int SignalSize = 3;                                     // Signal size
input bool ShowLabels = true;                                 // Show signal labels
```

---

## Parâmetros de Performance

### Otimização de Velocidade

#### Latência e Execução
```mql5
input int MaxExecutionTime = 50;                              // Max execution time (ms)
input bool EnablePreCalculation = true;                       // Enable pre-calculation
input bool UseAsyncOperations = true;                         // Use async operations
input int MaxConcurrentTrades = 3;                            // Maximum concurrent trades
```

#### Cache e Memória
```mql5
input bool EnableDataCache = true;                            // Enable data caching
input int CacheSize = 1000;                                   // Cache size (items)
input int MaxHistoryBars = 1000;                              // Maximum history bars
input bool OptimizeMemory = true;                             // Optimize memory usage
```

### Conectividade e Rede

#### Configurações de Conexão
```mql5
input int ConnectionTimeout = 5000;                           // Connection timeout (ms)
input int MaxRetries = 3;                                     // Maximum retry attempts
input int RetryDelay = 1000;                                  // Retry delay (ms)
input bool EnableConnectionCheck = true;                      // Enable connection monitoring
```

---

## Configurações por Ativo

### Forex (Pares de Moedas)

#### Configurações Padrão
```mql5
// Risk Management
input double ForexRiskPercent = 1.0;                          // Risk per trade
input double ForexMaxSpread = 3.0;                            // Maximum spread (pips)
input double ForexCommission = 7.0;                           // Commission per lot

// Timeframes
input ENUM_TIMEFRAMES ForexPrimaryTF = PERIOD_M15;            // Primary timeframe
input ENUM_TIMEFRAMES ForexSecondaryTF = PERIOD_H1;           // Secondary timeframe

// Trading Hours
input int ForexStartHour = 8;                                 // Start trading hour
input int ForexEndHour = 21;                                  // End trading hour
input bool TradeDuringNews = false;                           // Trade during news
```

### Criptomoedas

#### Configurações Crypto
```mql5
// Risk Management
input double CryptoRiskPercent = 2.0;                         // Risk per trade
input double CryptoVolatilityLimit = 5.0;                     // Volatility limit
input bool EnableCryptoHedging = true;                        // Enable crypto hedging

// Specific Pairs
input bool TradeBTCUSD = true;                                // Trade BTC/USD
input bool TradeETHUSD = true;                                // Trade ETH/USD
input double CryptoMinVolume = 0.001;                         // Minimum volume
```

### Ouro (XAUUSD)

#### Configurações Gold
```mql5
// Risk Management
input double GoldRiskPercent = 0.5;                           // Risk per trade
input double GoldMaxSpread = 50;                              // Maximum spread (points)
input bool EnableGoldHedging = false;                         // Enable gold hedging

// Volatility
input double GoldVolatilityThreshold = 20;                    // Volatility threshold
input bool AvoidHighVolatility = true;                        // Avoid high volatility periods

// Trading Sessions
input bool TradeAsianSession = true;                          // Trade Asian session
input bool TradeEuropeanSession = true;                       // Trade European session
input bool TradeAmericanSession = true;                       // Trade American session
```

---

## Exemplos de Configuração

### Configuração Conservadora (Iniciantes)

```mql5
// Risk Management
input double BaseRiskPercent = 0.005;                        // 0.5% por trade
input double MaxDailyRisk = 0.01;                             // 1% máximo diário
input double MaxDrawdownPercent = 0.02;                       // 2% drawdown máximo

// ML Configuration
input bool EnableMLPrediction = true;
input double MLConfidenceThreshold = 0.85;                    // Alta confiança
input int MLModelUpdateHours = 48;                            // Atualizações menos frequentes

// Strategy Selection
input bool EnableSmartMoney = false;                          // Desativar estratégias complexas
input bool EnableMLScalping = true;                           // Manter scalping ML
input bool EnableVolatilityBreakout = false;                  // Desativar breakout

// Latency Optimization
input int MaxLatencyMS = 200;                                 // Latência mais permissiva
input bool EnablePreStops = true;
input double SlippageTolerancePips = 3.0;                     // Mais tolerância
```

### Configuração Agressiva (Experientes)

```mql5
// Risk Management
input double BaseRiskPercent = 0.02;                          // 2% por trade
input double MaxDailyRisk = 0.05;                             // 5% máximo diário
input double MaxDrawdownPercent = 0.08;                       // 8% drawdown

// ML Configuration
input bool EnableMLPrediction = true;
input double MLConfidenceThreshold = 0.65;                    // Menor confiança, mais trades
input int MLModelUpdateHours = 12;                            // Atualizações frequentes

// Strategy Selection
input bool EnableSmartMoney = true;                           // Todas as estratégias
input bool EnableMLScalping = true;
input bool EnableVolatilityBreakout = true;
input bool EnableMultiTimeframe = true;

// Performance
input int MaxLatencyMS = 50;                                  // Latência restrita
input int MaxConcurrentTrades = 5;                            // Mais trades simultâneos
```

### Configuração FTMO

```mql5
// FTMO Compliance
input bool EnableFTMOCompliance = true;
input double MaxDailyLossFTMO = 2.0;                          // Limite diário FTMO
input double MaxTotalLossFTMO = 10.0;                         // Limite total FTMO
input int MinTradingDaysFTMO = 10;                            // Dias mínimos
input double ProfitTargetFTMO = 10.0;                         // Alvo de lucro

// Risk Management
input double BaseRiskPercent = 0.01;                          // 1% por trade
input double MaxDailyRisk = 0.02;                             // 2% máximo diário
input double MaxDrawdownPercent = 0.05;                       // 5% drawdown

// Trading Parameters
input int MaxConcurrentTrades = 2;                            // Limite de trades
input double MaxLotSize = 0.1;                                // Lote máximo
input bool EnableHedging = false;                             // Sem hedging
```

### Configuração Alta Frequência (HFT)

```mql5
// Latency Optimization
input int MaxLatencyMS = 10;                                  // Latência muito baixa
input bool EnablePreStops = true;
input bool EnableOrderBuffering = true;
input double SlippageTolerancePips = 0.5;                     // Slippage mínimo

// Performance
input int MaxExecutionTime = 20;                              // Execução rápida
input bool EnableDataCache = true;
input int CacheSize = 5000;                                   // Cache grande
input bool UseAsyncOperations = true;

// ML Configuration
input bool EnableMLPrediction = true;
input double MLConfidenceThreshold = 0.60;                    // Decisões rápidas
input int MLModelUpdateHours = 6;                             // Atualizações muito frequentes
```

---

## Validação de Parâmetros

### Funções de Validação

```mql5
// Função de validação principal
bool ValidateInputParameters() {
    // Validar risk parameters
    if (BaseRiskPercent <= 0 || BaseRiskPercent > 0.05) {
        Print("❌ BaseRiskPercent deve estar entre 0 e 5%");
        return false;
    }

    if (MaxDailyRisk <= BaseRiskPercent || MaxDailyRisk > 0.10) {
        Print("❌ MaxDailyRisk deve ser maior que BaseRiskPercent e menor que 10%");
        return false;
    }

    // Validar ML parameters
    if (EnableMLPrediction) {
        if (MLConfidenceThreshold <= 0 || MLConfidenceThreshold > 1) {
            Print("❌ MLConfidenceThreshold deve estar entre 0 e 1");
            return false;
        }

        if (MLModelUpdateHours < 1 || MLModelUpdateHours > 168) {
            Print("❌ MLModelUpdateHours deve estar entre 1 e 168 horas");
            return false;
        }
    }

    // Validar latency parameters
    if (MaxLatencyMS < 10 || MaxLatencyMS > 1000) {
        Print("❌ MaxLatencyMS deve estar entre 10 e 1000 ms");
        return false;
    }

    // Validar slippage
    if (SlippageTolerancePips < 0.1 || SlippageTolerancePips > 20) {
        Print("❌ SlippageTolerancePips deve estar entre 0.1 e 20 pips");
        return false;
    }

    Print("✅ Todos os parâmetros validados com sucesso");
    return true;
}

// Validação específica por ativo
bool ValidateAssetParameters(string symbol) {
    double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

    if (FixedLotSize < minLot || FixedLotSize > maxLot) {
        Print("❌ Lot size fora dos limites para ", symbol);
        Print("   Mínimo: ", minLot, " Máximo: ", maxLot);
        return false;
    }

    if (MathMod(FixedLotSize, lotStep) != 0) {
        Print("❌ Lot size deve ser múltiplo de ", lotStep);
        return false;
    }

    return true;
}
```

### Sistema de Verificação

```mql5
// Sistema de verificação em tempo real
void CheckParameterSanity() {
    static datetime lastCheck = 0;

    if (TimeCurrent() - lastCheck < 3600) return; // Verificar a cada hora

    lastCheck = TimeCurrent();

    // Verificar drawdown atual
    double currentDrawdown = CalculateCurrentDrawdown();
    if (currentDrawdown > MaxDrawdownPercent) {
        Print("⚠️ Drawdown atual (", currentDrawdown, "%) excede o limite");
        DisableTrading("Drawdown limit exceeded");
    }

    // Verificar latência
    int currentLatency = GetCurrentLatency();
    if (currentLatency > MaxLatencyMS) {
        Print("⚠️ Latência atual (", currentLatency, "ms) excede o limite");
        if (currentLatency > MaxLatencyMS * 2) {
            DisableTrading("High latency detected");
        }
    }

    // Verificar spread
    double currentSpread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                           SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
    if (currentSpread > MaxSpreadPips) {
        Print("⚠️ Spread atual (", currentSpread, " pips) muito alto");
        // Não executar trades até spread normalizar
    }
}
```

---

## Troubleshooting

### Problemas Comuns

#### 1. Erro de Validação de Parâmetros

**Sintoma:** EA não inicializa com erro de parâmetros inválidos

**Causas Possíveis:**
- Risk percent muito alto ou baixo
- ML confidence threshold fora do intervalo
- Lot size não compatível com o ativo

**Solução:**
```mql5
// Debug parameters
void DebugParameters() {
    Print("=== DEBUG PARAMETERS ===");
    Print("BaseRiskPercent: ", BaseRiskPercent);
    Print("MaxDailyRisk: ", MaxDailyRisk);
    Print("MLConfidenceThreshold: ", MLConfidenceThreshold);
    Print("FixedLotSize: ", FixedLotSize);

    // Verificar limites do ativo
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    Print("Symbol min lot: ", minLot);
    Print("Symbol max lot: ", maxLot);
    Print("========================");
}
```

#### 2. Performance Lent

**Sintoma:** EA demora para executar análises

**Causas Possíveis:**
- Cache desativado
- ML models muito complexos
- Histórico de dados excessivo

**Solução:**
```mql5
// Otimizar performance
input bool EnableDataCache = true;                            // Ativar cache
input int MaxHistoryBars = 500;                               // Reduzir histórico
input bool EnablePreCalculation = true;                       // Ativar pré-cálculo
input int MLModelUpdateHours = 48;                            // Reduzir frequência de atualização
```

#### 3. Trades Não Executando

**Sintoma:** EA analisa mas não executa trades

**Causas Possíveis:**
- Risk limits atingidos
- Latência alta
- Parâmetros ML muito restritivos

**Solução:**
```mql5
// Verificar why trades aren't executing
void DebugTradingStatus() {
    Print("=== TRADING STATUS ===");
    Print("Trading enabled: ", IsTradeAllowed());
    Print("Risk limits ok: ", CheckRiskLimits());
    Print("ML confidence: ", GetLastMLConfidence());
    Print("Current latency: ", GetCurrentLatency(), "ms");
    Print("Daily P&L: ", GetDailyPnL());
    Print("=====================");
}
```

### Ferramentas de Diagnóstico

#### 1. Monitor de Parâmetros em Tempo Real

```mql5
// Painel de diagnóstico
void CreateDiagnosticPanel() {
    int panelX = 300;
    int panelY = 20;

    CreateLabel("DiagnosticTitle", panelX, panelY, "DIAGNÓSTICO EA", clrRed);

    CreateLabel("RiskStatus", panelX, panelY + 20, "Risk: OK", clrGreen);
    CreateLabel("MLStatus", panelX, panelY + 40, "ML: OK", clrGreen);
    CreateLabel("LatencyStatus", panelX, panelY + 60, "Latency: OK", clrGreen);
    CreateLabel("TradingStatus", panelX, panelY + 80, "Trading: Active", clrGreen);
}

void UpdateDiagnosticPanel() {
    // Risk status
    double currentDD = CalculateCurrentDrawdown();
    string riskStatus = currentDD < MaxDrawdownPercent ? "Risk: OK" : "Risk: HIGH";
    color riskColor = currentDD < MaxDrawdownPercent ? clrGreen : clrRed;
    ObjectSetString(0, "RiskStatus", OBJPROP_TEXT, riskStatus);
    ObjectSetInteger(0, "RiskStatus", OBJPROP_COLOR, riskColor);

    // ML status
    double mlConf = GetLastMLConfidence();
    string mlStatus = mlConf >= MLConfidenceThreshold ? "ML: OK" : "ML: LOW";
    color mlColor = mlConf >= MLConfidenceThreshold ? clrGreen : clrYellow;
    ObjectSetString(0, "MLStatus", OBJPROP_TEXT, mlStatus);
    ObjectSetInteger(0, "MLStatus", OBJPROP_COLOR, mlColor);

    // Latency status
    int latency = GetCurrentLatency();
    string latencyStatus = latency <= MaxLatencyMS ? "Latency: OK" : "Latency: HIGH";
    color latencyColor = latency <= MaxLatencyMS ? clrGreen : clrRed;
    ObjectSetString(0, "LatencyStatus", OBJPROP_TEXT, latencyStatus);
    ObjectSetInteger(0, "LatencyStatus", OBJPROP_COLOR, latencyColor);
}
```

#### 2. Sistema de Alertas

```mql5
// Alertas configuráveis
void CheckAndAlert() {
    // Alerta de drawdown
    double currentDD = CalculateCurrentDrawdown();
    if (currentDD > MaxDrawdownPercent * 0.8) { // 80% do limite
        Alert("⚠️ AVISO: Drawdown接近 limite - ", currentDD, "%");
    }

    // Alerta de latência
    int latency = GetCurrentLatency();
    if (latency > MaxLatencyMS * 0.8) { // 80% do limite
        Alert("⚠️ AVISO: Latência alta detectada - ", latency, "ms");
    }

    // Alerta de ML confidence
    double mlConf = GetLastMLConfidence();
    if (mlConf < MLConfidenceThreshold * 0.9) { // 90% do limiar
        Alert("⚠️ AVISO: Baixa confiança ML - ", mlConf);
    }

    // Alerta de risco diário
    double dailyRisk = GetDailyRisk();
    if (dailyRisk > MaxDailyRisk * 0.8) { // 80% do limite
        Alert("⚠️ AVISO: Risco diário接近 limite - ", dailyRisk, "%");
    }
}
```

Este guia completo de parâmetros dos EAs cobre todos os aspectos necessários para configurar, validar e otimizar os Expert Advisors do projeto EA_SCALPER_XAUUSD, incluindo exemplos práticos para diferentes perfis de trader e ferramentas de troubleshooting.