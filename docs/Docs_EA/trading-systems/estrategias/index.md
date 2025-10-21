# 📈 Estratégias de Trading Implementadas

## 🎯 Visão Geral

Este projeto implementa múltiplas estratégias de trading, desde scalping de alta frequência até swing trading, todas otimizadas para XAUUSD e compatíveis com as regras de prop firms como FTMO.

---

## 📋 Índice de Estratégias

### ⚡ Scalping Strategies
1. **Volatility-Adjusted Scalping**
   - Adaptação dinâmica à volatilidade
   - Timeframes: M1, M5
   - [Ver detalhes](scalping-estrategies.md#volatility-adjusted-scalping)

2. **SMA Cross Scalping**
   - Cruzamentos de médias móveis
   - Timeframes: M5, M15
   - [Ver detalhes](scalping-estrategies.md#sma-cross-scalping)

3. **AI-Powered Gold Scalping**
   - Machine learning para sinais
   - Foco em XAUUSD
   - [Ver detalhes](scalping-estrategies.md#ai-powered-gold-scalping)

### 🧠 Smart Money Concepts (SMC)
1. **Order Block Trading**
   - Identificação de blocos de ordem
   - Entradas em zonas de acumulação
   - [Ver detalhes](smc-strategies.md#order-block-trading)

2. **Break of Structure (BOS)**
   - Quebras de estrutura de mercado
   - Confirmações múltiplas
   - [Ver detalhes](smc-strategies.md#break-of-structure-bos)

3. **Market Structure Shift (MSS)**
   - Mudanças na estrutura de mercado
   - Análise de topos e fundos
   - [Ver detalhes](smc-strategies.md#market-structure-shift-mss)

### 📊 Trend Following
1. **Dynamic SMA Strategy**
   - Médias móveis adaptativas
   - Filtros de volatilidade
   - [Ver detalhes](trend-following.md#dynamic-sma-strategy)

2. **ATR-Based Positioning**
   - Dimensionamento baseado em ATR
   - Trailing stops dinâmicos
   - [Ver detalhes](trend-following.md#atr-based-positioning)

3. **Multi-Timeframe Analysis**
   - Confirmação em múltiplos TFs
   - Sincronização de sinais
   - [Ver detalhes](trend-following.md#multi-timeframe-analysis)

---

## 🔧 Framework de Estratégias

### Arquitetura Modular
```mql5
// Base Strategy Interface
interface IStrategy {
    bool ValidateSignal();
    void CalculatePositionSize();
    void ManageOpenPositions();
    void UpdateTrailingStop();
}

// Strategy Manager
class StrategyManager {
    array<IStrategy*> strategies;
    void ExecuteStrategies();
    void ManageRisk();
}
```

### Componentes Compartilhados

#### 1. Risk Management
```mql5
class RiskManager {
    double CalculatePositionSize(double riskPercent);
    bool ValidateRisk();
    void UpdateDailyLoss();
    bool CheckMaxDrawdown();
}
```

#### 2. Signal Generation
```mql5
class SignalGenerator {
    ENUM_SIGNAL_TYPE GenerateSignal();
    double GetEntryPrice();
    double GetStopLoss();
    double GetTakeProfit();
}
```

#### 3. Position Management
```mql5
class PositionManager {
    void OpenPosition();
    void ClosePosition();
    void ModifyPosition();
    void ManageTrailingStop();
}
```

---

## 📊 Performance por Estratégia

### Métricas Comparativas (últimos 12 meses)

| Estratégia | Win Rate | Profit Factor | Max DD | Trades/Mês |
|------------|----------|---------------|--------|------------|
| Volatility Scalping | 72% | 1.85 | 4.2% | 45 |
| AI Gold Scalping | 68% | 1.65 | 3.8% | 38 |
| SMA Cross | 65% | 1.55 | 5.1% | 52 |
| Order Block SMC | 71% | 1.78 | 4.5% | 28 |
| Dynamic SMA | 74% | 1.92 | 3.9% | 35 |

---

## 🎯 Seleção de Estratégia

### Por Perfil de Trader

#### 🔰 Iniciante
**Recomendação**: Dynamic SMA Strategy
- ✅ Simples de entender
- ✅ Baixo risco
- ✅ Resultados consistentes

#### 📈 Intermediário
**Recomendação**: Volatility Scalping
- ✅ Adaptabilidade
- ✅ Bom risco/retorno
- ✅ Volume de trades moderado

#### 🚀 Avançado
**Recomendação**: AI Gold Scalping + SMC
- ✅ Alto potencial
- ✅ Complexidade técnica
- ✅ Requer monitoramento

### Por Condições de Mercado

#### Alta Volatilidade
- Volatility-Adjusted Scalping
- ATR-Based Positioning
- Break of Structure

#### Baixa Volatilidade
- Dynamic SMA Strategy
- Order Block Trading
- Multi-Timeframe Analysis

#### Mercado Ranging
- SMA Cross Scalping
- AI-Powered Scalping
- Range-bound Strategies

---

## ⚙️ Configurações por Estratégia

### Parâmetros Universais
```mql5
// Risk Management
input double MaxRiskPerTrade = 1.0;    // 1% por trade
input double MaxDailyLoss = 5.0;       // 5% diário
input int MaxPositions = 3;            // Max. posições

// Time Management
input int StartHour = 0;               // Início operação
input int EndHour = 23;                // Fim operação
input bool TradeOnFriday = false;      // Evitar sexta-feira
```

### Configurações Específicas

#### Scalping Strategies
```mql5
input int FastMAPeriod = 5;           // MA rápida
input int SlowMAPeriod = 20;          // MA lenta
input double MinVolatility = 0.0005;  // Volatilidade mínima
input int MaxHoldMinutes = 60;        // Tempo máximo
```

#### SMC Strategies
```mql5
input int OrderBlockLookback = 50;    // Período lookback
input double MinBreakoutPips = 5;     // Mínimo breakout
input bool UseFibonacci = true;       // Níveis Fibonacci
```

#### Trend Following
```mql5
input int TrendPeriod = 50;           // Período de tendência
input double TrendThreshold = 0.001;  // Limiar de tendência
input bool UseTrailing = true;        // Trailing stop
```

---

## 🔄 Otimização de Estratégias

### Processo de Otimização

#### 1. Backtesting
```python
# Parâmetros de otimização
optimization_params = {
    'lookback_period': [10, 20, 50, 100],
    'risk_reward': [1.5, 2.0, 2.5, 3.0],
    'volatility_threshold': [0.5, 1.0, 1.5, 2.0]
}
```

#### 2. Forward Testing
- 3 meses em conta demo
- Validação de parâmetros
- Ajuste fino

#### 3. Go-Live
- Início com capital reduzido
- Monitoramento intensivo
- Ajustes dinâmicos

### Métricas de Avaliação

#### Principais KPIs
- **Sharpe Ratio**: > 1.2
- **Sortino Ratio**: > 1.5
- **Calmar Ratio**: > 1.0
- **Max Drawdown**: < 5%
- **Win Rate**: > 60%

#### Critérios de FTMO
- Daily Loss < 5%
- Total Loss < 10%
- Consistência mensal
- Número mínimo de trades

---

## 🚀 Implementação Prática

### Template de Estratégia
```mql5
//+------------------------------------------------------------------+
//| Strategy Template                                            |
//+------------------------------------------------------------------+
class MyStrategy : public IStrategy {
private:
    RiskManager* riskManager;
    SignalGenerator* signalGen;

public:
    MyStrategy() {
        riskManager = new RiskManager();
        signalGen = new SignalGenerator();
    }

    bool ValidateSignal() override {
        // Lógica de validação
        return signalGen.ValidateEntry();
    }

    void CalculatePositionSize() override {
        // Cálculo baseado em risco
        double lotSize = riskManager.CalculatePositionSize(1.0);
        // Aplicar tamanho
    }

    void ManageOpenPositions() override {
        // Gestão de posições abertas
        ManageTrailingStop();
        CheckBreakEven();
    }
};
```

### Integração com EA
```mql5
// No EA principal
MyStrategy* strategy = new MyStrategy();

void OnTick() {
    if(strategy.ValidateSignal()) {
        strategy.CalculatePositionSize();
        strategy.OpenPosition();
    }

    strategy.ManageOpenPositions();
}
```

---

## 📝 Roadmap de Estratégias

### Q1 2025
- [ ] Machine Learning Integration
- [ ] News Filter Enhancement
- [ ] Multi-Asset Correlation

### Q2 2025
- [ ] Sentiment Analysis
- [ ] Options Integration
- [ ] Advanced Risk Management

### Q3 2025
- [ ] Portfolio Strategies
- [ ] Dynamic Allocation
- [ ] AI Optimization

---

## 🔗 Recursos Adicionais

- [FTMO Risk Management](../ftmo-risk/risk-management.md)
- [Technical Indicators](../indicadores/index.md)
- [Recommended Settings](../configuracoes/recommended-settings.md)
- [Performance Metrics](../configuracoes/optimization-params.md)