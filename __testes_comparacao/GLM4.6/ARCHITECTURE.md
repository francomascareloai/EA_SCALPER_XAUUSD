# 🏗️ **EA Optimizer AI - Arquitetura Completa**

## 🎯 **Visão Geral**
O **EA Optimizer AI** é um sistema inteligente de otimização automática que analisa resultados de backtests e otimiza parâmetros críticos de Expert Advisors (EAs) utilizando Machine Learning.

## 📊 **Arquitetura do Sistema**

```
┌─────────────────────────────────────────────────────────────────┐
│                    EA OPTIMIZER AI SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   DADOS     │    │   OTIMizador  │    │   GERADOR EA    │   │
│  │   INPUT     │───▶│    PYTHON     │───▶│     MQL5        │   │
│  │             │    │              │    │                 │   │
│  │ • CSV/JSON  │    │ • Optuna     │    │ • Template      │   │
│  │ • Backtest  │    │ • Scikit-learn│   │ • Auto-compile  │   │
│  │ • Histórico │    │ • Pandas     │    │ • Deploy-ready  │   │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│         │                   │                     │           │
│         ▼                   ▼                     ▼           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ VALIDAÇÃO   │    │  RELATÓRIOS   │    │  META TRADER 5  │   │
│  │   RESULTADOS│    │   VISUAIS     │    │   EXECUÇÃO      │   │
│  │             │    │              │    │                 │   │
│  │ • Cross-val │    │ • Gráficos   │    │ • Trading real  │   │
│  │ • Backtest  │    │ • Métricas   │    │ • Monitoramento │   │
│  │ • Simulação │    │ • Export CSV │    │ • Logs          │   │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 **Componentes Principais**

### 1. **Módulo de Ingestão de Dados** (`data_loader.py`)
- Lê arquivos CSV/JSON de backtests
- Valida e limpa dados históricos
- Extrai métricas chave: profit, drawdown, winrate, sharpe ratio

### 2. **Motor de Otimização** (`optimizer.py`)
- **Optuna**: Hyperparameter optimization
- **Scikit-learn**: Modelos de regressão e classificação
- **Pandas/Numpy**: Processamento de dados
- Algoritmos: TPE, Random Search, Bayesian Optimization

### 3. **Gerador de EA MQL5** (`mql5_generator.py`)
- Template engine para EA otimizado
- Injeção automática de parâmetros otimizados
- Validação sintática do código MQL5

### 4. **Sistema de Visualização** (`visualizer.py`)
- Gráficos comparativos (antes/depois)
- Análise de performance
- Exportação de relatórios

### 5. **Validador de Resultados** (`validator.py`)
- Backtesting automatizado
- Validação cruzada
- Métricas de risco

## 🔄 **Fluxo de Dados**

1. **Input**: Arquivos de backtest (CSV/JSON)
2. **Processamento**: Limpeza e feature engineering
3. **Otimização**: Busca automática de melhores parâmetros
4. **Validação**: Cross-validation e backtesting
5. **Geração**: EA MQL5 otimizado
6. **Visualização**: Relatórios e gráficos de performance
7. **Deploy**: EA pronto para MetaTrader 5

## 📁 **Estrutura de Diretórios**

```
GLM4.6/
├── src/
│   ├── data_loader.py          # Ingestão de dados
│   ├── optimizer.py            # Motor de otimização ML
│   ├── mql5_generator.py       # Gerador de EA MQL5
│   ├── visualizer.py           # Visualizações
│   ├── validator.py            # Validação de resultados
│   └── main.py                 # Orquestrador principal
├── data/
│   ├── input/                  # Dados de backtest
│   │   ├── sample_backtest.csv
│   │   └── historical_data.json
│   └── processed/              # Dados processados
├── templates/
│   └── ea_template.mq5         # Template MQL5
├── output/
│   ├── optimized_params.json   # Parâmetros otimizados
│   ├── EA_OPTIMIZER_XAUUSD.mq5 # EA final
│   ├── performance_report.html # Relatório completo
│   └── charts/                 # Gráficos de performance
├── requirements.txt            # Dependências Python
└── README.md                   # Documentação
```

## 🎯 **Parâmetros Otimizáveis**

### Risk Management
- **Stop Loss**: Dinâmico baseado em ATR
- **Take Profit**: Proporção risco:retorno
- **Risk Factor**: Percentual de risco por trade
- **Max Drawdown**: Limite máximo de perda

### Technical Indicators
- **ATR Multiplier**: Ajuste de volatilidade
- **MA Periods**: Períodos das médias móveis
- **RSI Thresholds**: Níveis de sobrevrevenda/vendido
- **Bollinger Bands**: Desvio padrão

### Trading Sessions
- **Asian Session**: Horários de negociação
- **European Session**: Janelas de ativação
- **US Session**: Períodos de alta volatilidade

### Position Sizing
- **Lot Size**: Dimensionamento dinâmico
- **Max Positions**: Limite de operações simultâneas
- **Pyramiding**: Adição de posições

## 🚀 **Tecnologias Utilizadas**

- **Python 3.11+**: Linguagem principal
- **Optuna**: Otimização de hiperparâmetros
- **Scikit-learn**: Machine Learning
- **Pandas**: Manipulação de dados
- **Plotly**: Visualizações interativas
- **Jinja2**: Templates MQL5
- **FastAPI**: API REST (opcional)

## 🔐 **Critérios de Otimização**

### Primary Metrics
- **Profit Factor**: Razão entre lucros e perdas
- **Sharpe Ratio**: Retorno ajustado ao risco
- **Sortino Ratio**: Variação do Sharpe com downside risk
- **Maximum Drawdown**: Perda máxima tolerável

### Secondary Metrics
- **Win Rate**: Percentual de trades vencedores
- **Average Trade**: Ticket médio de operações
- **Recovery Factor**: Fator de recuperação
- **Calmar Ratio**: Retorno ajustado ao drawdown máximo

## 🎮 **Interface de Uso**

```python
from src.main import EAOptimizer

# Inicializar otimizador
optimizer = EAOptimizer(
    data_path="data/input/sample_backtest.csv",
    symbol="XAUUSD",
    timeframe="M5"
)

# Executar otimização
results = optimizer.optimize(
    n_trials=100,
    timeout=3600  # 1 hora
)

# Gerar EA otimizado
optimizer.generate_ea(
    output_path="output/EA_OPTIMIZER_XAUUSD.mq5"
)

# Visualizar resultados
optimizer.plot_results()
```

## 📈 **Métricas de Sucesso**

- **Otimização de 50%+** no Profit Factor
- **Redução de 30%+** no Maximum Drawdown
- **Aumento de 25%+** no Sharpe Ratio
- **Geração automática** de EA MQL5 funcional
- **Compatibilidade total** com MetaTrader 5

---

**Status**: ✅ Arquitetura definida e pronta para implementação