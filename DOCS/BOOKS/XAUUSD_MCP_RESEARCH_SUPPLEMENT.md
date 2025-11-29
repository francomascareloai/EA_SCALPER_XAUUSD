# XAUUSD MCP Research Supplement
# Informações Extras Descobertas via Perplexity, GitHub e Brave Search

Este documento complementa o XAUUSD_DEEP_FUNDAMENTALS_GUIDE.md com descobertas adicionais dos MCPs.

---

## 1. MELHORES MODELOS ML PARA GOLD (Pesquisa 2024-2025)

### Performance Comparativa

| Modelo | Métrica | Performance | Fonte |
|--------|---------|-------------|-------|
| **Gaussian Process Regression** | RRMSE | 0.8706% | TandFOnline 2025 |
| **XGBoost** | R² | 0.9797 | GeeksForGeeks |
| **Random Forest** | R² | 0.9787 | GeeksForGeeks |
| **Lasso + Polynomial** | R² | 0.9687 | GeeksForGeeks |
| **ARIMA** | Accuracy | 68.4% (5-min) | SciTePress 2025 |

### INSIGHT CRÍTICO: USO (Crude Oil) é o Driver #1

```
DESCOBERTA SURPREENDENTE:

O preço do petróleo (USO) é o fator MAIS importante para prever gold - 
MAIS QUE 2X mais importante que qualquer outro fator!

Por quê?
- Oil correlaciona com inflação
- Oil correlaciona com força do dólar
- Oil reflete condições macroeconômicas globais
- Oil é proxy para risk sentiment

IMPLICAÇÃO: Adicionar USO como feature no modelo!
```

### Código de Feature Importance

```python
# Feature importance do XGBoost para Gold
FEATURE_IMPORTANCE = {
    'USO': 0.42,      # Crude Oil - DOMINANTE
    'DXY': 0.18,      # Dollar Index
    'Real_Yields': 0.15,
    'VIX': 0.08,
    'SPY': 0.07,
    'SLV': 0.05,      # Silver
    'Other': 0.05
}

# USO tem mais que 2x a importância de qualquer outro fator!
```

---

## 2. PREVISÕES AI PARA GOLD 2025

### Comparativo de Modelos AI

| Modelo | Q1 2025 | Q2 2025 | Q3 2025 | Q4 2025 |
|--------|---------|---------|---------|---------|
| **ChatGPT-4 Turbo** | $2,826-3,038 | $3,084-3,315 | $3,366-3,617 | $3,673-3,947 |
| **Claude AI** | $2,580-2,750 | $2,650-2,850 | $2,700-2,950 | $2,750-3,000 |
| **Meta AI** | $1,830-1,970 | $1,900-2,050 | $1,950-2,120 | $2,000-2,180 |

### Interpretação

```
VARIÂNCIA ENORME entre modelos:
- ChatGPT-4: Muito otimista (preços recordes)
- Claude: Moderado (continuação de trend)
- Meta: Conservador (correção)

LIÇÃO: Não confiar em previsões de preço absoluto
MELHOR: Usar ML para DIREÇÃO e PROBABILIDADE, não preço exato
```

---

## 3. DETALHES DA QUEBRA DE CORRELAÇÃO (2022-2024)

### Dados Quantitativos Específicos

```python
CORRELATION_BREAKDOWN = {
    'historical': {
        'gold_real_yields': -0.82,  # Erb & Harvey research
        'period': '2004-2021',
        'real_duration': 18,  # 100bps = 18% gold move
    },
    'post_2022': {
        'gold_real_yields': 'QUEBROU',
        'cause': 'Central bank buying',
        'etf_outflow': '800 tonnes',  # 2022-2024
        'central_bank_buying': '1,100+ tonnes/year',
        'main_buyers': ['China', 'Poland', 'Turkey', 'India']
    },
    'implication': 'Real yields ainda importam, mas não explicam tudo'
}

# PIMCO Research:
# 100bps aumento em real yields = 18% queda em gold (historical)
# Real duration do gold = 18 anos
```

### Os 4 Drivers da Quebra

1. **Geopolítica** - Guerra Ucrânia desde Mar/2022
2. **Central Bank Demand** - Compras recordes
3. **De-dollarization** - Diversificação de reservas
4. **Fiscal Concerns** - Hedge contra riscos fiscais

---

## 4. APIs GRATUITAS PARA GOLD DATA

### APIs Disponíveis

| API | Tipo | Free Tier | Notas |
|-----|------|-----------|-------|
| **TwelveData** | Price + OHLCV | Sim | 5-min intervals |
| **TraderMade** | Live + Hist | Sim | Multi-metal |
| **API Ninjas** | CME Futures | Sim | USD only |
| **MetalpriceAPI** | Spot + 150 currencies | Sim | Bid/Ask |
| **Metals-API** | Historical | Sim | Currency convert |
| **GoldAPI.io** | Real-time | Sim | JSON REST |
| **FRED** | Macro data | Sim | Unlimited |

### Código de Integração

```python
import requests
import os

class GoldDataFetcher:
    def __init__(self):
        self.apis = {
            'twelvedata': {
                'url': 'https://api.twelvedata.com/time_series',
                'params': {'symbol': 'XAU/USD', 'interval': '5min'}
            },
            'tradermade': {
                'url': 'https://marketdata.tradermade.com/api/v1/live',
                'params': {'currency': 'XAUUSD'}
            },
            'api_ninjas': {
                'url': 'https://api.api-ninjas.com/v1/goldprice',
                'headers': {'X-Api-Key': os.getenv('API_NINJAS_KEY')}
            }
        }
    
    def fetch_twelvedata(self, api_key):
        params = self.apis['twelvedata']['params'].copy()
        params['apikey'] = api_key
        response = requests.get(self.apis['twelvedata']['url'], params=params)
        return response.json()
    
    def fetch_tradermade(self, api_key):
        params = self.apis['tradermade']['params'].copy()
        params['api_key'] = api_key
        response = requests.get(self.apis['tradermade']['url'], params=params)
        return response.json()
    
    def fetch_api_ninjas(self, api_key):
        headers = {'X-Api-Key': api_key}
        response = requests.get(
            self.apis['api_ninjas']['url'], 
            headers=headers
        )
        return response.json()

# Uso:
fetcher = GoldDataFetcher()
data = fetcher.fetch_twelvedata(os.getenv('TWELVEDATA_KEY'))
```

---

## 5. ETL PIPELINE COMPLETO PARA GOLD

```python
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

class GoldETLPipeline:
    def __init__(self, db_path='gold_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def extract(self, api_response):
        """Extract data from API response"""
        if 'values' in api_response:
            return pd.DataFrame(api_response['values'])
        return pd.DataFrame([api_response])
    
    def transform(self, df):
        """Transform and calculate indicators"""
        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate indicators
        if 'close' in df.columns:
            df['SMA_10'] = df['close'].rolling(10).mean()
            df['SMA_20'] = df['close'].rolling(20).mean()
            df['SMA_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(20).mean()
            df['BB_std'] = df['close'].rolling(20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Add timestamp
        df['processed_at'] = datetime.now()
        
        return df
    
    def load(self, df, table_name='gold_prices'):
        """Load to SQLite"""
        df.to_sql(table_name, self.conn, if_exists='append', index=False)
        print(f"Loaded {len(df)} rows to {table_name}")
    
    def run_pipeline(self, api_response):
        """Run full ETL pipeline"""
        df = self.extract(api_response)
        df = self.transform(df)
        self.load(df)
        return df
    
    def query(self, sql):
        """Query the database"""
        return pd.read_sql(sql, self.conn)
    
    def close(self):
        self.conn.close()
```

---

## 6. REPOSITÓRIOS GITHUB RELEVANTES

### Encontrados via GitHub Search

| Repo | Descrição | Link |
|------|-----------|------|
| **NICEGOLD-ProjectP** | Sistema de trading gold completo | nicetpad2/NICEGOLD-ProjectP |
| **DekTradingSignal** | AI trading + sentiment | JonusNattapong/DekTradingSignal |
| **AxiomEdge-PRO** | ML Trading Framework | sam-minns/AxiomEdge-PRO |
| **Gold-analysis-prediction** | Análise e predição | trajceskijovan/Gold-analysis-and-prediction |

### Bibliotecas Python para MT5

```python
# MAS - Trading library for MetaTrader 5
# pip install mas

from mas import MT5, Strategy, Backtest

# Suporta:
# - Gold (XAUUSD)
# - Forex
# - Índices
# - Ações
# - Crypto

# Features:
# - Backtesting
# - AI Strategy Generation
# - Automated deployment
```

---

## 7. ESTRATÉGIAS QUANTITATIVAS ADICIONAIS

### Breakout Momentum (Backtested)

```python
# Resultados de backtest em Gold ETFs:
BREAKOUT_STRATEGY = {
    'entry': 'Price breaks key S/R with volume confirmation',
    'hold_period': 20,  # days
    'results': {
        'avg_gain': 0.86,  # % over 20 days
        'with_volume_filter': 1.25,  # % with volume confirmation
        'win_rate': 0.62
    }
}
```

### RSI Multi-Timeframe (90%+ Accuracy)

```python
RSI_STRATEGY = {
    'approach': 'Multi-timeframe RSI with dynamic thresholds',
    'timeframes': ['M15', 'H1', 'H4'],
    'backtested_accuracy': 0.90,  # 90%+ em backtests robustos
    'key_insight': 'Adaptar thresholds ao contexto de trend'
}
```

### Mean Reversion com Bollinger

```python
MEAN_REVERSION = {
    'entry': 'Price touches/compresses Bollinger Bands',
    'adjust_to': 'Volatility regime',
    'risk_management': {
        'stop_loss': '2%',
        'take_profit': '3%',
        'position_sizing': 'Kelly fraction'
    }
}
```

---

## 8. EDGE COMPUTING E LATÊNCIA

### Otimização de Latência

```
DESCOBERTA:
Edge computing reduz latência de inferência em 80%!

IMPLEMENTAÇÃO:
1. Deploy modelo localmente (não cloud)
2. Usar ONNX para inferência rápida
3. Adaptive learning intraday

BENEFÍCIO:
- Regime shifts capturados mais rápido
- Execução mais rápida
- Menos slippage
```

---

## 9. LIMITAÇÕES IDENTIFICADAS

### O que APIs de Gold NÃO fornecem:

1. **Dados Macro Integrados** - Precisa de FRED separado
2. **COT em Tempo Real** - Apenas semanal (CFTC)
3. **Order Flow** - Precisa de broker específico
4. **News Sentiment** - Precisa de API separada

### Solução: Integrar Múltiplas Fontes

```python
DATA_SOURCES = {
    'price': ['TwelveData', 'TraderMade'],
    'macro': ['FRED', 'World Bank'],
    'sentiment': ['NewsAPI', 'FinBERT'],
    'positioning': ['CFTC COT (weekly)'],
    'flows': ['World Gold Council']
}
```

---

## 10. RESUMO: O QUE OS MCPs ADICIONARAM

| Descoberta | Fonte | Impacto |
|------------|-------|---------|
| **USO é driver #1** | Perplexity | Adicionar Oil como feature! |
| **GPR: 0.87% error** | TandFOnline | Considerar Gaussian Process |
| **XGBoost: R² 0.98** | GeeksForGeeks | Modelo baseline forte |
| **Real duration = 18 anos** | PIMCO | Quantifica sensibilidade |
| **800t ETF outflow** | Saxo | Explica divergência |
| **APIs gratuitas** | Perplexity | Lista completa de fontes |
| **Edge computing 80%** | LuxAlgo | Otimização crítica |
| **RSI MTF 90%** | LuxAlgo | Estratégia validada |

---

*Documento complementar ao XAUUSD_DEEP_FUNDAMENTALS_GUIDE.md*
*Gerado via Perplexity, GitHub e Brave Search MCPs*
