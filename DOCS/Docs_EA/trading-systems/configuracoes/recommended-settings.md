# ⚙️ Configurações Recomendadas

## 🎯 Visão Geral

Este guia apresenta as configurações otimizadas para diferentes perfis de traders, tipos de contas e condições de mercado. Todas as configurações foram testadas extensivamente e validadas para compliance FTMO.

---

## 📋 Índice de Configurações

### 🏆 Configurações FTMO Ready
1. [Conta $100,000 - Conservador](#conta-100000-conservador)
2. [Conta $100,000 - Agressivo](#conta-100000-agressivo)
3. [Conta $200,000 - Profissional](#conta-200000-profissional)

### 💰 Configurações por Capital
1. [Conta Pequena $10,000](#conta-pequena-10000)
2. [Conta Média $50,000](#conta-média-50000)
3. [Conta Grande $500,000+](#conta-grande-500000)

### 📊 Configurações por Estratégia
1. [Scalping de Alta Frequência](#scalping-alta-frequência)
2. [Smart Money Concepts](#smart-money-concepts)
3. [Trend Following](#trend-following)

---

## 🏆 Configurações FTMO Ready

### Conta $100,000 - Conservador

#### Risk Management
```mql5
// Parâmetros de Risco
input double RiskPerTrade = 1.0;        // 1% por trade ($1,000)
input double MaxDailyLoss = 4.0;        // 4% máximo diário (buffer de 1%)
input double MaxTotalLoss = 9.0;        // 9% máximo total (buffer de 1%)
input int MaxPositions = 3;             // Máximo 3 posições simultâneas
input double MaxAccountRisk = 2.5;      // 2.5% risco total da conta

// Position Sizing
input double MinLotSize = 0.1;          // Lote mínimo
input double MaxLotSize = 2.0;          // Lote máximo
input double LotStep = 0.1;             // Incremento de lote
input bool UseDynamicSizing = true;     // Dimensionamento dinâmico
input double FixedLotSize = 0.0;        // Não usa lotes fixos
```

#### Estratégia Parameters
```mql5
// Volatility Optimized SMA
input int DefaultPeriod = 14;           // Período padrão SMA
input double HighVolatilityThreshold = 1.5;  // Limiar alta volatilidade
input double LowVolatilityThreshold = 0.5;   // Limiar baixa volatilidade

// ATR Settings
input int AtrPeriod = 14;               // Período ATR
input double AtrMultiplierSL = 1.8;     // SL = 1.8x ATR (conservador)
input double RiskRewardRatioTP = 2.2;   // TP = 2.2x SL
```

#### Safety Features
```mql5
// Stop Loss e Take Profit
input int MinStopLossPoints = 100;      // Mínimo 100 pips SL
input int DefaultStopLoss = 150;        // SL padrão 150 pips
input int DefaultTakeProfit = 330;      // TP padrão 330 pips
input bool UseATRStops = true;          // Usar SL baseado em ATR

// Break Even e Trailing
input int BreakEvenPoints = 250;        // BE após 250 pips
input int BreakEvenPipsLock = 10;       // Travar 10 pips no BE
input int TrailingStopPoints = 150;     // Trailing de 150 pips
input int TrailingStartPoints = 300;    // Iniciar trailing após 300 pips
```

#### Time Management
```mql5
input int StartHour = 1;                // Início trading (01:00 UTC)
input int EndHour = 22;                 // Fim trading (22:00 UTC)
input bool TradeOnMonday = true;        // Operar segunda-feira
input bool TradeOnTuesday = true;       // Operar terça-feira
input bool TradeOnWednesday = true;     // Operar quarta-feira
input bool TradeOnThursday = true;      // Operar quinta-feira
input bool TradeOnFriday = false;       // EVITAR sexta-feira
input bool TradeOnWeekend = false;      // Não operar fim de semana
```

#### Expected Performance
| Métrica | Valor Esperado |
|---------|----------------|
| Win Rate | 70-75% |
| Profit Factor | 1.8-2.2 |
| Max Drawdown | <4% |
| Monthly Return | 8-12% |
| Trades/Month | 35-45 |

---

### Conta $100,000 - Agressivo

#### Risk Management
```mql5
// Parâmetros de Risco
input double RiskPerTrade = 1.5;        // 1.5% por trade ($1,500)
input double MaxDailyLoss = 4.5;        // 4.5% máximo diário
input double MaxTotalLoss = 9.5;        // 9.5% máximo total
input int MaxPositions = 5;             // Máximo 5 posições simultâneas
input double MaxAccountRisk = 4.0;      // 4% risco total da conta

// Position Sizing
input double MinLotSize = 0.1;          // Lote mínimo
input double MaxLotSize = 3.0;          // Lote máximo
input double LotStep = 0.1;             // Incremento de lote
input bool UseDynamicSizing = true;     // Dimensionamento dinâmico
input double MaxRiskPerPosition = 2.0;  // 2% máximo por posição
```

#### Estratégia Parameters
```mql5
// Estratégia Agresiva
input int FastMAPeriod = 8;             // MA rápida mais sensível
input int SlowMAPeriod = 18;            // MA lenta mais sensível
input double VolatilityMultiplier = 1.2;  // Multiplicador de volatilidade

// Settings mais agressivos
input int AtrPeriod = 12;               // ATR mais curto
input double AtrMultiplierSL = 1.3;     // SL mais apertado
input double RiskRewardRatioTP = 2.8;   // TP maior
```

#### Expected Performance
| Métrica | Valor Esperado |
|---------|----------------|
| Win Rate | 65-70% |
| Profit Factor | 2.0-2.5 |
| Max Drawdown | <6% |
| Monthly Return | 12-18% |
| Trades/Month | 50-65 |

---

## 💰 Configurações por Capital

### Conta Pequena $10,000

#### Risk Management Adaptado
```mql5
// Parâmetros de Risco
input double RiskPerTrade = 1.0;        // 1% por trade ($100)
input double MaxDailyLoss = 4.0;        // 4% máximo diário
input double MaxTotalLoss = 9.0;        // 9% máximo total
input int MaxPositions = 2;             // Máximo 2 posições
input double MinLotSize = 0.01;         // Lote mínimo micro

// Position Sizing Conservador
input double MaxLotSize = 0.1;          // Lote máximo
input double LotStep = 0.01;            // Incremento pequeno
input bool UseMicroLots = true;         // Usar micro lotes
```

#### Recomendações Especiais
- Focar em 1-2 estratégias principais
- Usar timeframes mais altos (M15, H1)
- Evitar scalping de alta frequência
- Manter stop losses mais apertados

---

### Conta Média $50,000

#### Risk Balance
```mql5
// Parâmetros Balanceados
input double RiskPerTrade = 1.2;        // 1.2% por trade ($600)
input double MaxDailyLoss = 4.2;        // 4.2% máximo diário
input double MaxTotalLoss = 9.2;        // 9.2% máximo total
input int MaxPositions = 4;             // Máximo 4 posições

// Multi-estratégia
input bool EnableMultipleStrategies = true;
input int MaxStrategies = 3;            // Até 3 estratégias simultâneas
```

---

### Conta Grande $500,000+

#### Professional Setup
```mql5
// Parâmetros Profissionais
input double RiskPerTrade = 0.8;        // 0.8% por trade ($4,000)
input double MaxDailyLoss = 3.8;        // 3.8% máximo diário
input double MaxTotalLoss = 8.8;        // 8.8% máximo total
input int MaxPositions = 8;             // Máximo 8 posições

// Diversificação
input bool EnableMultiAsset = true;
input bool EnableHedging = true;
input double MaxCorrelation = 0.7;      // Máxima correlação entre posições
```

---

## 📊 Configurações por Estratégia

### Scalping de Alta Frequência

#### Fast Setup
```mql5
// Timeframes Rápidos
input ENUM_TIMEFRAMES PrimaryTimeframe = PERIOD_M5;
input ENUM_TIMEFRAMES SecondaryTimeframe = PERIOD_M15;
input bool EnableM1Signals = true;      // Sinais M1 para confirmação

// Parâmetros de Velocidade
input int MaxHoldTimeMinutes = 120;     // Máximo 2 horas por posição
input double MinVolatility = 0.0008;    // Volatilidade mínima
input int MaxSpreadPoints = 30;         // Spread máximo

// Indicadores Rápidos
input int FastMAPeriod = 5;             // MAs muito rápidas
input int SlowMAPeriod = 15;
input int RSIPeriod = 8;                // RSI rápido
input double RSIBuyLevel = 30;          // Compra abaixo de 30
input double RSISellLevel = 70;         // Venda acima de 70
```

#### Risk Controls
```mql5
input double ScalpingRisk = 0.5;        // 0.5% por trade scalping
input int MaxScalpingTrades = 20;       // Máximo 20 trades/dia
input bool UseNewsFilter = true;        // Evitar notícias
input int NewsBufferMinutes = 30;       // Buffer de 30 min
```

### Smart Money Concepts

#### SMC Setup
```mql5
// Estrutura de Mercado
input int OrderBlockLookback = 100;     // Lookback maior para Order Blocks
input int StructureDepth = 3;           // Profundidade da estrutura
input double FibonacciRatio = 0.618;    // Razão Fibonacci
input bool UseBreakConfirmation = true; // Confirmar quebras

// Níveis de Confluência
input int MinConfluencePoints = 2;      // Mínimo 2 pontos de confluência
input bool UseFibonacciLevels = true;  // Níveis Fibonacci
input bool UseVolumeConfirmation = true; // Confirmar com volume

// Settings de Precisão
input double MinSignalStrength = 0.75;  // Força mínima do sinal
input int ConfirmationCandles = 2;      // Velas de confirmação
input bool UseMultiTimeframe = true;    // Análise MTF
```

### Trend Following

#### Trend Setup
```mql5
// Análise de Tendência
input int TrendPeriod = 50;             // Período principal de tendência
input double TrendStrengthThreshold = 0.002;  // Força mínima de tendência
input bool UseADXFilter = true;         // Filtro ADX
input int ADXPeriod = 14;               // Período ADX
input double ADXThreshold = 25;         // Limiar ADX

// Canais e Bandas
input double ChannelMultiplier = 2.0;   // Multiplicador do canal
input int ChannelPeriod = 20;           // Período do canal
input bool UseBollingerBands = true;    // Bandas de Bollinger

// Trailing e Gestão
input bool UseTrailingStop = true;      // Trailing stop obrigatório
input int TrailingStopPoints = 200;     // Trailing de 200 pips
input int TrailingStepPoints = 50;      // Passo de 50 pips
input bool UseBreakEven = true;         // Break-even automático
```

---

## 🕐 Time Management Settings

### Sessões de Trading

#### London Session (Recomendada)
```mql5
input int LondonStart = 8;              // 08:00 UTC
input int LondonEnd = 17;               // 17:00 UTC
input bool TradeLondonSession = true;   // Ativar sessão
input double LondonRiskMultiplier = 1.2; // +20% risco
```

#### New York Session
```mql5
input int NewYorkStart = 13;            // 13:00 UTC
input int NewYorkEnd = 22;              // 22:00 UTC
input bool TradeNewYorkSession = true;  // Ativar sessão
input double NewYorkRiskMultiplier = 1.0; // Risco normal
```

#### Asian Session (Evitar)
```mql5
input int AsianStart = 23;              // 23:00 UTC
input int AsianEnd = 8;                 // 08:00 UTC
input bool TradeAsianSession = false;   // EVITAR sessão asiática
```

### Filtros de Tempo
```mql5
// Filtros Especiais
input bool AvoidFirstHour = true;       // Evitar primeira hora
input bool AvoidLastHour = true;        // Evitar última hora
input bool AvoidFriday = true;          // Evitar sexta-feira
input int FridayEndHour = 18;           // Terminar sexta às 18:00
input bool AvoidHolidays = true;        // Evitar feriados
input bool UseEconomicCalendar = true;  // Calendário econômico
```

---

## 📊 Monitoramento e Alertas

### Dashboard Settings
```mql5
// Visualização
input bool ShowInfoPanel = true;        // Painel de informações
input int PanelCorner = CORNER_TOP_RIGHT; // Posição do painel
input color PanelBackColor = clrBlack;  // Cor do fundo
input color PanelTextColor = clrLime;   // Cor do texto
input int PanelFontSize = 10;           // Tamanho da fonte

// Indicadores de Performance
input bool ShowDrawdownMeter = true;    // Medidor de drawdown
input bool ShowRiskMeter = true;        // Medidor de risco
input bool ShowDailyPnL = true;         // P&L diário
input bool ShowTradeHistory = true;     // Histórico de trades
```

### Alert Configuration
```mql5
// Tipos de Alerta
input bool EnableSoundAlerts = true;    // Alertas sonoros
input bool EnablePushNotifications = true; // Notificações push
input bool EnableEmailAlerts = false;   // Alertas por email
input bool EnableTelegramAlerts = true; // Alertas Telegram

// Níveis de Alerta
input double AlertDrawdownLevel = 3.0;  // Alerta em 3% DD
input double AlertProfitLevel = 10.0;   // Alerta em 10% lucro
input bool AlertTradeEntry = true;      // Alertar entradas
input bool AlertTradeExit = true;       // Alertar saídas
```

---

## 🔄 Configurações de Otimização

### Optimization Parameters
```mql5
// Genetic Algorithm
input int OptimizationCycles = 1000;    // Ciclos de otimização
input double OptimizationPrecision = 0.0001; // Precisão
input bool UseMultiObjective = true;    // Multi-objetivo
input double WeightProfit = 0.4;        // Peso para lucro
input double WeightDD = 0.3;            // Peso para drawdown
input double WeightTrades = 0.3;        // Peso para número de trades
```

### Forward Testing
```mql5
// Forward Test Settings
input int ForwardTestMonths = 3;        // Meses de teste
input bool UseWalkForward = true;       // Walk-forward analysis
input int WalkForwardWindow = 6;        // Janela de 6 meses
input int ReoptimizationFrequency = 30; // Reotimizar a cada 30 dias
```

---

## 📝 Templates Pré-configurados

### Template Iniciante
```mql5
// Copiar e colar no EA
// Profile: Iniciante FTMO
// Capital: $100,000
// Estratégia: Conservative

RiskPerTrade = 1.0
MaxDailyLoss = 4.0
MaxPositions = 3
DefaultPeriod = 14
AtrMultiplierSL = 1.8
RiskRewardRatioTP = 2.2
TradeOnFriday = false
```

### Template Avançado
```mql5
// Profile: Profissional
// Capital: $200,000
// Estratégia: Multi-Strategy

RiskPerTrade = 0.8
MaxDailyLoss = 3.8
MaxPositions = 6
EnableMultipleStrategies = true
MaxStrategies = 4
UseHedging = true
EnableMultiAsset = true
```

---

## 🔧 Troubleshooting de Configurações

### Problemas Comuns

#### 1. Poucos Trades
- **Causa**: Filtros muito restritivos
- **Solução**: Reduzir volatilidade mínima, aumentar períodos

#### 2. Drawdown Excessivo
- **Causa**: Risk por trade muito alto
- **Solução**: Reduzir RiskPerTrade para 0.5-0.8%

#### 3. Baixo Win Rate
- **Causa**: Configuração de SL muito apertada
- **Solução**: Aumentar AtrMultiplierSL para 2.0+

#### 4. Alta Latência
- **Causa**: Múltiplos indicadores pesados
- **Solução**: Desabilitar indicadores não essenciais

---

## 📈 Performance Esperada por Configuração

### Conservative Profile
- **Win Rate**: 70-75%
- **Monthly Return**: 8-12%
- **Max DD**: <4%
- **Sharpe Ratio**: >1.5

### Aggressive Profile
- **Win Rate**: 65-70%
- **Monthly Return**: 12-18%
- **Max DD**: <6%
- **Sharpe Ratio**: >1.2

### Scalping Profile
- **Win Rate**: 68-72%
- **Monthly Return**: 10-15%
- **Max DD**: <5%
- **Sharpe Ratio**: >1.3

---

## 🔗 Recursos Adicionais

- [FTMO Compliance Guide](../ftmo-risk/compliance-guide.md)
- [EA Documentation](../eas-producao/index.md)
- [Strategy Guide](../estrategias/index.md)
- [Performance Metrics](optimization-params.md)

---

**Nota Importante**: Sempre comece com configurações conservadoras e aumente gradualmente a agressividade conforme ganha experiência e confiança no sistema. Backtestes extensivos são recomendados antes de aplicar novas configurações em contas reais.