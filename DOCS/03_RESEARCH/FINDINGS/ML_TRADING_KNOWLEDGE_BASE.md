# 📚 ML Trading Knowledge Base
## Guia Completo para Trading Quantitativo de Elite - EA_SCALPER_XAUUSD

**Versão**: 1.0
**Data**: 2025-11-28
**Baseado em**: Deep Research com RAG (68,426 chunks) + Web Search + Perplexity

---

## Índice

1. [Market Microstructure](#1-market-microstructure)
2. [Regime Detection (Hurst/Entropy)](#2-regime-detection)
3. [Machine Learning para Trading](#3-machine-learning-para-trading)
4. [LSTM vs Transformer](#4-lstm-vs-transformer)
5. [Triple Barrier Labeling](#5-triple-barrier-labeling)
6. [Smart Money Concepts (SMC)](#6-smart-money-concepts)
7. [ONNX Integration em MQL5](#7-onnx-integration)
8. [Risk Management & Kelly Criterion](#8-risk-management)
9. [Gold/XAUUSD Específico](#9-xauusd-específico)
10. [Papers Essenciais](#10-papers-essenciais)
11. [RAG Queries Prontas](#11-rag-queries)
12. [Código de Referência](#12-código-referência)

---

## 1. Market Microstructure

### 1.1 O Que REALMENTE Move os Preços

**Fonte RAG**: `market_microstructure_intro.pdf` (1,685 chunks)

#### Kyle Model (1985) - Fundamento

O modelo Kyle explica como informação é incorporada nos preços:

```
Informed Trader → Order Flow → Price Impact → Market Maker Adjustment
```

**Conceitos-chave do RAG**:
- **Adverse Selection**: Market makers perdem para informed traders
- **Price Impact**: λ (lambda) mede quanto preço move por unidade de order flow
- **Information Asymmetry**: Spread reflete risco de trading contra informados

#### Aplicação no EA

```cpp
// Se detectar adverse selection (spread alto + volume baixo), reduzir size
if(spread > 2.0 * average_spread && volume < 0.5 * average_volume) {
    position_multiplier *= 0.5;  // Reduce exposure
}
```

### 1.2 Order Flow e Liquidity

**Fonte RAG**: `market_microstructure_intro.pdf`

```
Liquidity Pools:
- BSL (Buy-Side Liquidity) = Stop losses de shorts acima de highs
- SSL (Sell-Side Liquidity) = Stop losses de longs abaixo de lows

Smart Money busca essa liquidez ANTES de mover o preço na direção real.
```

### 1.3 Livros Recomendados (Não no RAG - ADQUIRIR)

| Livro | Autor | Por Que |
|-------|-------|---------|
| **Trading and Exchanges** | Larry Harris | Bíblia da microstructure |
| **Trades, Quotes and Prices** | Bouchaud et al. | Quantitative microstructure |
| **Algorithmic and High-Frequency Trading** | Cartea et al. | Optimal execution math |

---

## 2. Regime Detection

### 2.1 Hurst Exponent (H)

**Fonte RAG**: `Algorithmic_Trading_Methods.pdf` (3,775 chunks)
**Fonte Web**: Perplexity research

#### Interpretação

| H Value | Regime | Significado | Estratégia |
|---------|--------|-------------|------------|
| H > 0.55 | TRENDING | Preço persiste na direção | Momentum, breakout |
| H < 0.45 | MEAN-REVERTING | Preço reverte à média | Contrarian, fade |
| H ≈ 0.50 | RANDOM WALK | Sem padrão previsível | **NÃO OPERAR** |

#### Código Python (Validado)

```python
import numpy as np

def get_hurst_exponent(ts, max_lag=20):
    """
    Calcula Hurst exponent usando método de variância.
    
    Args:
        ts: Array de preços ou retornos
        max_lag: Máximo lag para análise
    
    Returns:
        H: Hurst exponent (0 a 1)
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    
    # Regressão log-log
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

# Uso
prices = np.array([...])  # Seus preços
H = get_hurst_exponent(prices)

if H > 0.55:
    regime = "TRENDING"
elif H < 0.45:
    regime = "MEAN_REVERTING"
else:
    regime = "RANDOM_WALK"  # Não operar!
```

### 2.2 Shannon Entropy (S)

**Fonte RAG**: `Algorithmic_Trading_Methods.pdf`

#### Interpretação

| S Value | Noise Level | Ação |
|---------|-------------|------|
| S < 1.5 | LOW | Full position size |
| 1.5 ≤ S < 2.5 | MEDIUM | 50% position size |
| S ≥ 2.5 | HIGH | Não operar |

#### Código Python

```python
import numpy as np

def calculate_entropy(returns, bins=10):
    """
    Calcula Shannon Entropy dos retornos.
    
    Args:
        returns: Array de retornos
        bins: Número de bins para histograma
    
    Returns:
        S: Shannon entropy
    """
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log2(hist))

# Uso
returns = np.diff(prices) / prices[:-1]
S = calculate_entropy(returns)
```

### 2.3 Matriz de Decisão Combinada

```
┌─────────────────┬───────────────┬───────────────┐
│                 │  Entropy < 1.5 │ Entropy >= 1.5│
├─────────────────┼───────────────┼───────────────┤
│  Hurst > 0.55   │ PRIME_TREND   │ NOISY_TREND   │
│  (Trending)     │ Size: 100%    │ Size: 50%     │
├─────────────────┼───────────────┼───────────────┤
│  Hurst < 0.45   │ PRIME_REVERT  │ NOISY_REVERT  │
│  (Reverting)    │ Size: 100%    │ Size: 50%     │
├─────────────────┼───────────────┼───────────────┤
│  Hurst ~ 0.50   │ RANDOM_WALK   │ RANDOM_WALK   │
│  (Random)       │ Size: 0%      │ Size: 0%      │
└─────────────────┴───────────────┴───────────────┘
```

---

## 3. Machine Learning para Trading

### 3.1 Arquiteturas Principais

**Fonte RAG**: Papers de ML (2,500+ chunks combinados)

#### LSTM (Long Short-Term Memory)

**Quando usar**: Price sequences, regime detection, direction prediction

```python
import torch
import torch.nn as nn

class DirectionLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # [P(bear), P(bull)]
        )
    
    def forward(self, x):
        # x: (batch, seq_len=100, features=15)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return torch.softmax(self.fc(last_hidden), dim=1)
```

#### Transformer para Time Series

**Quando usar**: Long-range dependencies, sentiment analysis

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Linear(15, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.softmax(self.fc(x[:, -1, :]), dim=1)
```

### 3.2 Paper Achilles - XAUUSD Específico

**Fonte RAG**: `achilles_xauusd_neural_network.pdf` (82 chunks)

**RESULTADO COMPROVADO**:
- Budget: $1,000
- Profit: $1,623.52 em 1 mês (62.3% return!)
- Modelo: LSTM com apenas 9,544 parâmetros
- Target: XAUUSD especificamente

**Arquitetura Achilles**:
```
Input → LSTM (small) → Dense → Output (price prediction)
+ FinBERT sentiment → Combined decision
```

**Lições do Achilles para nosso EA**:
1. Modelo pequeno funciona (9.5k params)
2. Previsão minute-per-minute
3. Sentimento de news é crucial
4. Risk = 0.3 funcionou

---

## 4. LSTM vs Transformer

### 4.1 Comparativo (Perplexity Research 2024)

| Aspecto | LSTM | Transformer |
|---------|------|-------------|
| Price Differences | ✅ Superior | ⚠️ Marginal |
| Absolute Prices | ⚠️ Ok | ⚠️ Inconsistent |
| Long Sequences | ⚠️ Vanishing gradient | ✅ Attention helps |
| Sentiment/News | ⚠️ Ok | ✅ Superior |
| Training Speed | ⚠️ Slower | ✅ Parallelizable |
| Parameters | ⚠️ More | ✅ Fewer |

### 4.2 Recomendação: Arquitetura HÍBRIDA

```
┌─────────────────────────────────────────────────────┐
│                 HYBRID ARCHITECTURE                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Price Data ──► LSTM ──────────┐                   │
│                                 ├──► Fusion ──► Output
│  News/Sentiment ──► Transformer┘                   │
│                                                     │
│  Benefits:                                          │
│  - LSTM captures price dynamics                    │
│  - Transformer extracts semantic signals           │
│  - Best of both worlds                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. Triple Barrier Labeling

### 5.1 Por Que É CRÍTICO

**Fonte**: López de Prado - Advances in Financial Machine Learning

**Problema com Fixed Horizon**:
```
Fixed Horizon: "O preço subiu ou desceu após N barras?"
→ Ignora volatilidade, ignora path, ignora risco

Triple Barrier: "Qual barreira foi tocada primeiro?"
→ Reflete realidade do trading com SL e TP
```

### 5.2 As Três Barreiras

```
        ┌───────────────────────────────┐ Upper Barrier (Take Profit)
        │                               │
        │     ╱╲      ╱╲               │
Entry ──┼────╱──╲────╱──╲──────────────┼── Vertical Barrier (Time)
        │   ╱    ╲  ╱    ╲             │
        │  ╱      ╲╱      ╲            │
        └───────────────────────────────┘ Lower Barrier (Stop Loss)
        
Labels:
+1 = Upper barrier hit first (profit)
-1 = Lower barrier hit first (loss)
 0 = Vertical barrier hit (timeout)
```

### 5.3 Implementação Python

```python
import numpy as np
import pandas as pd

def triple_barrier_labels(
    prices: pd.Series,
    volatility: pd.Series,  # ATR ou std
    pt_multiplier: float = 2.0,  # Take profit = 2x volatility
    sl_multiplier: float = 1.0,  # Stop loss = 1x volatility
    max_holding_period: int = 20  # Barras máximas
) -> pd.Series:
    """
    Implementa Triple Barrier Labeling de López de Prado.
    
    Args:
        prices: Série de preços de fechamento
        volatility: ATR ou rolling std (mesma escala que prices)
        pt_multiplier: Multiplicador para take profit
        sl_multiplier: Multiplicador para stop loss
        max_holding_period: Máximo de barras para manter posição
    
    Returns:
        labels: Série com +1, -1, ou 0
    """
    labels = pd.Series(index=prices.index, dtype=float)
    
    for i in range(len(prices) - max_holding_period):
        entry_price = prices.iloc[i]
        entry_vol = volatility.iloc[i]
        
        # Definir barreiras (volatility-adjusted!)
        upper_barrier = entry_price + pt_multiplier * entry_vol
        lower_barrier = entry_price - sl_multiplier * entry_vol
        
        # Procurar qual barreira é tocada primeiro
        for j in range(1, max_holding_period + 1):
            future_price = prices.iloc[i + j]
            
            # Upper barrier hit (profit)
            if future_price >= upper_barrier:
                labels.iloc[i] = 1
                break
            
            # Lower barrier hit (loss)
            if future_price <= lower_barrier:
                labels.iloc[i] = -1
                break
            
            # Vertical barrier (timeout)
            if j == max_holding_period:
                # Label baseado em return final
                final_return = (future_price - entry_price) / entry_price
                if final_return > 0:
                    labels.iloc[i] = 1
                elif final_return < 0:
                    labels.iloc[i] = -1
                else:
                    labels.iloc[i] = 0
    
    return labels

# Uso
df['atr'] = calculate_atr(df, period=14)
df['label'] = triple_barrier_labels(
    df['close'], 
    df['atr'],
    pt_multiplier=2.0,
    sl_multiplier=1.0,
    max_holding_period=20
)
```

### 5.4 Meta-Labeling

**Conceito**: Não prever direção, mas SIM/NÃO em sinal existente

```python
def meta_labeling(primary_signal, triple_barrier_label):
    """
    Primary signal: +1 (buy) ou -1 (sell) de outro modelo
    Triple barrier label: resultado real
    
    Meta-label: 1 se signal foi correto, 0 se errado
    """
    return (primary_signal == triple_barrier_label).astype(int)
```

---

## 6. Smart Money Concepts (SMC)

### 6.1 Order Blocks

**Fonte RAG**: `Smart Money Concept.pdf` (55 chunks)

```
BULLISH ORDER BLOCK:          BEARISH ORDER BLOCK:
                              
     Rally ↑                       Drop ↓
       │                            │
  ┌────┴────┐                  ┌────┴────┐
  │  LAST   │ ← Entry Zone     │  LAST   │
  │  DOWN   │                  │   UP    │
  │ CANDLE  │                  │ CANDLE  │
  └─────────┘                  └─────────┘
       │                            │
  Acumulação                   Distribuição
  antes rally                  antes drop
```

**Quality Score Factors**:
| Fator | Pontos |
|-------|--------|
| Displacement forte (>2x ATR) | +20 |
| Volume acima da média | +15 |
| BOS após formar | +20 |
| Fresh (primeiro toque) | +15 |
| Confluência com FVG | +10 |
| Liquidez swept antes | +20 |

### 6.2 Fair Value Gap (FVG)

```
BULLISH FVG:                    
                               
   Candle 3 ──►  ┌───┐         
                 │   │         
                 │   │         
   GAP ────────► │░░░│ ← Fill Zone (50% ideal)
                 │░░░│         
                 └───┘         
   Candle 1 ──►  ┌───┐         
                 │   │         
                 └───┘         
                               
   FVG = Low[3] - High[1] > 0  
```

### 6.3 Liquidity Concepts

```
LIQUIDITY SWEEP:

    ═══ Equal Highs (BSL) ═══
           │ │ │
           │ │ │← Stops acumulados
    ───────┼─┼─┼───────
           │ │ │
    RANGE  │ │ │
           │ │ │
    ───────┼─┼─┼───────
           │ │ │← Stops acumulados
    ═══ Equal Lows (SSL) ═══

Sweep = Price quebra, pega stops, REVERTE
→ Sinal de entrada na direção oposta
```

---

## 7. ONNX Integration

### 7.1 Workflow Completo

**Fonte RAG**: `mql5.pdf` (22,762 chunks) + `neuronetworksbook.pdf` (4,540 chunks)

```
Python (Treino)              ONNX (Bridge)            MQL5 (Produção)
─────────────────           ─────────────            ─────────────────
1. Treinar modelo    →      model.onnx       →      OnnxCreate()
2. Salvar scaler     →      scaler.json      →      Normalize()
3. Exportar ONNX     →                       →      OnnxRun()
4. Validar           →                       →      Trade decision
```

### 7.2 Código MQL5

```cpp
class COnnxBrain {
private:
    long m_model_handle;
    float m_input[];
    float m_output[];
    double m_means[];
    double m_stds[];
    
public:
    bool Initialize() {
        // Carregar modelo
        m_model_handle = OnnxCreate("Models\\direction_model.onnx", ONNX_DEFAULT);
        if(m_model_handle == INVALID_HANDLE) {
            Print("ONNX: Failed to load model");
            return false;
        }
        
        // Definir shapes
        ulong input_shape[] = {1, 100, 15};  // batch, seq, features
        ulong output_shape[] = {1, 2};        // batch, classes
        
        OnnxSetInputShape(m_model_handle, 0, input_shape);
        OnnxSetOutputShape(m_model_handle, 0, output_shape);
        
        // Alocar buffers
        ArrayResize(m_input, 1500);   // 100 * 15
        ArrayResize(m_output, 2);
        
        // Carregar parâmetros de normalização
        LoadScalerParams();
        
        return true;
    }
    
    double GetBullishProbability() {
        // Coletar features
        double features[];
        CollectFeatures(features);
        
        // Normalizar (CRÍTICO - deve bater com treino!)
        for(int i = 0; i < ArraySize(features); i++) {
            int feat_idx = i % 15;
            m_input[i] = (float)((features[i] - m_means[feat_idx]) / m_stds[feat_idx]);
        }
        
        // Inferência
        if(!OnnxRun(m_model_handle, ONNX_NO_CONVERSION, m_input, m_output)) {
            Print("ONNX: Inference failed");
            return 0.5;
        }
        
        return (double)m_output[1];  // P(bullish)
    }
};
```

### 7.3 As 15 Features do Modelo

| # | Feature | Cálculo | Normalização |
|---|---------|---------|--------------|
| 1 | Returns | (close - prev) / prev | StandardScaler |
| 2 | Log Returns | log(close / prev) | StandardScaler |
| 3 | Range % | (high - low) / close | StandardScaler |
| 4 | RSI M5 | RSI(14) on M5 | ÷ 100 |
| 5 | RSI M15 | RSI(14) on M15 | ÷ 100 |
| 6 | RSI H1 | RSI(14) on H1 | ÷ 100 |
| 7 | ATR Norm | ATR(14) / close | StandardScaler |
| 8 | MA Distance | (close - MA20) / MA20 | StandardScaler |
| 9 | BB Position | (close - mid) / width | Já -1 a 1 |
| 10 | Hurst | Rolling Hurst(100) | Já 0 a 1 |
| 11 | Entropy | Rolling Entropy(100) | ÷ 4 |
| 12 | Session | 0=Asia, 1=London, 2=NY | Categorical |
| 13 | Hour Sin | sin(2π × hour / 24) | Já -1 a 1 |
| 14 | Hour Cos | cos(2π × hour / 24) | Já -1 a 1 |
| 15 | OB Distance | Dist to nearest OB / ATR | StandardScaler |

---

## 8. Risk Management

### 8.1 Kelly Criterion

**Fonte RAG**: `ml_for_trading_jansen_overview.md`

```
Kelly Formula:
f* = (bp - q) / b

Onde:
- f* = fração ótima do capital
- b = odds (R:R ratio)
- p = probabilidade de win
- q = 1 - p (probabilidade de loss)

Exemplo:
- Win rate: 65%
- R:R: 2:1
- f* = (2 × 0.65 - 0.35) / 2 = 0.475 = 47.5%

REGRA: Usar Fractional Kelly (25-50% do full Kelly)
→ f_fractional = 0.5 × 0.475 = 23.75%
```

### 8.2 FTMO Compliance

**Fonte RAG**: Seu INDEX.md

| Regra | Limite FTMO | Buffer | Trigger |
|-------|-------------|--------|---------|
| Daily Loss Max | 5% | 4% | SOFT STOP |
| Total Loss Max | 10% | 8% | EMERGENCY |
| Profit P1 | 10% | - | Meta |
| Profit P2 | 5% | - | Meta |

### 8.3 Position Sizing Formula

```cpp
double CalculateLots(double sl_points) {
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = equity * risk_percent / 100.0;
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    
    double lots = risk_amount / (sl_points * tick_value);
    
    // Apply multipliers
    lots *= regime_multiplier;  // 0.5 ou 1.0 (Hurst/Entropy)
    lots *= mtf_multiplier;     // 0.5, 0.75, ou 1.0 (MTF alignment)
    
    // Clamp
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    
    return MathMax(min_lot, MathMin(lots, max_lot));
}
```

---

## 9. XAUUSD Específico

### 9.1 Características do Ouro

**Fonte RAG**: `achilles_xauusd_neural_network.pdf` + `gold_forex_ml_overview.md`

| Aspecto | Comportamento |
|---------|---------------|
| Safe Haven | Sobe em crises, cai em risk-on |
| USD Correlation | Negativa (mas enfraquecendo desde 2020) |
| Volatility | Alta, especialmente em news |
| Best Sessions | London + NY overlap |
| Spread típico | 20-50 points |

### 9.2 Sessões para Trading

```
ASIA (00:00-07:00 GMT):
- Baixo volume
- Spreads altos
- EVITAR

LONDON (07:00-15:00 GMT):
- Alto volume
- Tendências claras
- OPERAR

NY (13:00-21:00 GMT):
- Continuação de London
- Pode reverter tarde
- OPERAR até 17:00

OVERLAP (13:00-16:00 GMT):
- MELHOR período
- Máxima liquidez
- PRIORIZAR
```

### 9.3 Correlações Importantes

| Correlação | Relação | Trading Implication |
|------------|---------|---------------------|
| Gold vs DXY | Negativa (-0.3 a -0.7) | DXY forte → Gold fraco |
| Gold vs VIX | Positiva (safe haven) | VIX alto → Gold forte |
| Gold vs Real Yields | Negativa (opportunity cost) | Yields altos → Gold fraco |
| Gold vs Oil | Fraca/variável | Não usar como sinal |

---

## 10. Papers Essenciais

### 10.1 Papers JÁ INDEXADOS no RAG

| Paper | Chunks | Foco |
|-------|--------|------|
| achilles_xauusd_neural_network.pdf | 82 | LSTM para XAUUSD |
| autoformer_decomposition.pdf | 200 | Transformer decomposition |
| informer_long_sequence.pdf | 188 | Efficient attention |
| fedformer_frequency.pdf | 223 | Frequency-domain |
| patchtst_time_series.pdf | 193 | Patch-based Transformer |
| finrl_deep_rl_trading.pdf | 95 | RL para trading |
| news_aware_reinforcement_trading.pdf | 102 | RL + sentiment |
| market_microstructure_intro.pdf | 1,685 | Kyle model, order flow |

### 10.2 Papers para BAIXAR

| Paper | Link | Prioridade |
|-------|------|------------|
| Deep Learning Statistical Arbitrage | arxiv.org/abs/2106.04028 | 🔴 |
| StockGPT | arxiv.org/abs/2404.05101 | 🔴 |
| FinRL Framework | arxiv.org/abs/2111.09395 | 🟠 |
| RL in Quant Finance Survey | arxiv.org/abs/2408.10932 | 🟠 |
| ML in Portfolio SSRN | ssrn.com/4988124 | 🟠 |

### 10.3 Clássicos (Fundamentos)

| Paper | Ano | Contribuição |
|-------|-----|--------------|
| Kyle - Informed Trading | 1985 | Microstructure base |
| Fama-French 3-Factor | 1993 | Asset pricing |
| Jegadeesh-Titman Momentum | 1993 | Momentum anomaly |
| Hochreiter - LSTM | 1997 | Arquitetura LSTM |
| Thorp - Kelly Criterion | 1968 | Position sizing |

---

## 11. RAG Queries Prontas

### 11.1 Queries por Módulo

```python
# CRegimeDetector
query_rag("Hurst exponent calculation R/S analysis trending mean reverting", "books", 10)
query_rag("Shannon entropy information theory market noise", "books", 10)

# COnnxBrain
query_rag("OnnxCreate OnnxRun model inference shape input output", "docs", 15)
query_rag("ONNX model MQL5 normalization example", "docs", 10)

# EliteOrderBlock
query_rag("order block bullish bearish institutional displacement", "books", 10)

# EliteFVG
query_rag("fair value gap imbalance fill zone price", "books", 10)

# CLiquiditySweepDetector
query_rag("liquidity sweep stop hunt equal highs lows", "books", 10)

# Direction Model
query_rag("LSTM neural network time series prediction hidden state", "books", 15)

# Risk Management
query_rag("Kelly criterion position sizing drawdown risk", "books", 10)

# XAUUSD Specific
query_rag("gold XAUUSD price prediction LSTM neural network", "books", 10)
```

### 11.2 Queries de Troubleshooting

```python
# Se modelo não converge
query_rag("neural network training convergence batch normalization", "books", 10)

# Se overfitting
query_rag("overfitting regularization dropout cross validation", "books", 10)

# Se performance ruim em live
query_rag("backtest overfitting walk forward validation", "books", 10)
```

---

## 12. Código de Referência

### 12.1 Feature Engineering Completo

```python
import pandas as pd
import numpy as np
import talib as ta

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria todas as 15 features para o Direction Model.
    
    Args:
        df: DataFrame com OHLCV data
    
    Returns:
        features: DataFrame com 15 features normalizadas
    """
    f = pd.DataFrame(index=df.index)
    
    # Price features (1-3)
    f['returns'] = df['close'].pct_change()
    f['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    f['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # RSI multi-timeframe (4-6)
    f['rsi_m5'] = ta.RSI(df['close'], timeperiod=14) / 100
    # RSI M15 e H1 precisam de resampling ou dados separados
    f['rsi_m15'] = f['rsi_m5']  # Placeholder
    f['rsi_h1'] = f['rsi_m5']   # Placeholder
    
    # Volatility (7)
    f['atr_norm'] = ta.ATR(df['high'], df['low'], df['close'], 14) / df['close']
    
    # Trend (8)
    ma20 = ta.SMA(df['close'], 20)
    f['ma_distance'] = (df['close'] - ma20) / ma20
    
    # Bollinger (9)
    upper, mid, lower = ta.BBANDS(df['close'], 20)
    f['bb_position'] = (df['close'] - mid) / (upper - lower)
    
    # Regime (10-11)
    f['hurst'] = df['close'].rolling(100).apply(get_hurst_exponent)
    f['entropy'] = df['close'].pct_change().rolling(100).apply(calculate_entropy) / 4
    
    # Temporal (12-14)
    hour = df.index.hour
    f['session'] = hour.map(lambda h: 0 if h < 7 else (1 if h < 15 else 2))
    f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Structure (15)
    # OB distance precisa do detector de OB
    f['ob_distance'] = 0.0  # Placeholder
    
    return f.dropna()
```

### 12.2 Purged K-Fold Cross-Validation

```python
from sklearn.model_selection import BaseCrossValidator
import numpy as np

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold para time series - evita information leakage.
    """
    def __init__(self, n_splits=5, purge_gap=10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size
            
            # Train indices (before test, with gap)
            train_end = max(0, test_start - self.purge_gap)
            train_indices = np.arange(0, train_end)
            
            # Test indices
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Uso
cv = PurgedKFold(n_splits=5, purge_gap=20)
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

---

## Changelog

| Versão | Data | Mudanças |
|--------|------|----------|
| 1.0 | 2025-11-28 | Versão inicial baseada em deep research |

---

*"Trade with the institutions, not against them."*
*"Always consult RAG before coding."*
