# Variáveis Globais e Constantes - EA_SCALPER_XAUUSD

## Overview

Este documento documenta todas as variáveis globais e constantes utilizadas no projeto EA_SCALPER_XAUUSD, organizadas por categoria, linguagem e funcionalidade.

## Sumário

1. [Constantes MQL5/MQL4](#constantes-mql5mql4)
2. [Constantes Python](#constantes-python)
3. [Constantes de Sistema](#constantes-de-sistema)
4. [Constantes de Trading](#constantes-de-trading)
5. [Constantes de Machine Learning](#constantes-de-machine-learning)
6. [Constantes de Interface](#constantes-de-interface)
7. [Constantes de Performance](#constantes-de-performance)
8. [Convenções de Nomenclatura](#convenções-de-nomenclatura)
9. [Gerenciamento de Constantes](#gerenciamento-de-constantes)
10. [Exemplos de Uso](#exemplos-de-uso)

---

## Constantes MQL5/MQL4

### 1. Identificação e Versão

```mql5
// EA Identification
#define EA_NAME            "EA_SCALPER_XAUUSD"
#define EA_VERSION         "2.0"
#define EA_AUTHOR          "Elite Trading"
#define EA_COPYRIGHT       "Copyright 2024, Elite Trading"
#define EA_LINK            "https://github.com/elite-trading/EA_SCALPER_XAUUSD"

// Magic Numbers
#define MAGIC_NUMBER_BASE  20241201
#define MAGIC_NUMBER_XAUUSD (MAGIC_NUMBER_BASE + 100)
#define MAGIC_NUMBER_FOREX  (MAGIC_NUMBER_BASE + 200)
#define MAGIC_NUMBER_CRYPTO (MAGIC_NUMBER_BASE + 300)
```

### 2. Trading Constants

```mql5
// Trading Parameters
#define MIN_LOT_SIZE       0.01
#define MAX_LOT_SIZE       10.0
#define LOT_STEP           0.01
#define POINT_VALUE        0.01

// Risk Management
#define MAX_RISK_PERCENT   5.0        // Maximum risk per trade (%)
#define MAX_DAILY_LOSS     10.0       // Maximum daily loss (%)
#define MAX_DRAWDOWN       15.0       // Maximum drawdown (%)
#define MIN_RR_RATIO       1.5        // Minimum risk/reward ratio

// Stop Loss & Take Profit
#define MIN_SL_POINTS     10         // Minimum stop loss (points)
#define MAX_SL_POINTS     500        // Maximum stop loss (points)
#define MIN_TP_POINTS     15         // Minimum take profit (points)
#define MAX_TP_POINTS     1000       // Maximum take profit (points)

// Spread Limits
#define MAX_SPREAD_XAUUSD 50         // Maximum spread for XAUUSD
#define MAX_SPREAD_FOREX  5          // Maximum spread for Forex pairs
#define MAX_SPREAD_CRYPTO 20         // Maximum spread for Crypto
```

### 3. Time Constants

```mql5
// Timeframes
#define TIMEFRAME_M1      PERIOD_M1
#define TIMEFRAME_M5      PERIOD_M5
#define TIMEFRAME_M15     PERIOD_M15
#define TIMEFRAME_M30     PERIOD_M30
#define TIMEFRAME_H1      PERIOD_H1
#define TIMEFRAME_H4      PERIOD_H4
#define TIMEFRAME_D1      PERIOD_D1
#define TIMEFRAME_W1      PERIOD_W1

// Trading Sessions
#define ASIAN_START       22         // Asian session start (UTC)
#define ASIAN_END         7          // Asian session end (UTC)
#define EUROPEAN_START    7          // European session start (UTC)
#define EUROPEAN_END      16         // European session end (UTC)
#define AMERICAN_START    13         // American session start (UTC)
#define AMERICAN_END      22         // American session end (UTC)

// Time Constants
#define SECONDS_PER_MINUTE 60
#define SECONDS_PER_HOUR   3600
#define SECONDS_PER_DAY    86400
#define SECONDS_PER_WEEK   604800
```

### 4. Technical Indicators

```mql5
// RSI Constants
#define RSI_PERIOD        14
#define RSI_OVERSOLD      30.0
#define RSI_OVERBOUGHT    70.0
#define RSI_MIDDLE        50.0

// Moving Averages
#define MA_FAST_PERIOD    9
#define MA_SLOW_PERIOD    21
#define MA_SIGNAL_PERIOD  50

// Bollinger Bands
#define BB_PERIOD         20
#define BB_DEVIATION      2.0

// MACD
#define MACD_FAST_EMA     12
#define MACD_SLOW_EMA     26
#define MACD_SIGNAL_SMA   9

// ATR
#define ATR_PERIOD        14
#define ATR_MULTIPLIER    2.0
```

### 5. ICT Smart Money Concepts

```mql5
// Order Block Constants
#define MIN_OB_SIZE       10.0       // Minimum order block size (points)
#define MAX_OB_AGE_BARS   50         // Maximum order block age (bars)
#define OB_VALIDATION_PIPS 5         // Order block validation range (pips)

// FVG (Fair Value Gap) Constants
#define MIN_FVG_SIZE      5.0        // Minimum FVG size (points)
#define MAX_FVG_AGE_BARS  30         // Maximum FVG age (bars)
#define FVG_FILL_PERCENT  70.0       // FVG considered filled at 70%

// Liquidity Constants
#define LIQUIDITY_SWING_PIPS 15      // Liquidity swing identification (pips)
#define LIQUIDITY_WICK_PERCENT 0.3   // Wick percentage for liquidity identification
```

### 6. Error Codes

```mql5
// Error Codes
#define ERR_SUCCESS                0
#define ERR_NO_TRADE_PERMISSION   1
#define ERR_INSUFFICIENT_FUNDS    2
#define ERR_MARKET_CLOSED         3
#define ERR_INVALID_PRICE         4
#define ERR_INVALID_STOPS         5
#define ERR_TRADE_DISABLED        6
#define ERR_INVALID_VOLUME        7
#define ERR_SERVER_BUSY          100
#define ERR_NETWORK_TIMEOUT      101
```

---

## Constantes Python

### 1. Configuração do Sistema

```python
# System Configuration
SYSTEM_NAME = "EA_SCALPER_XAUUSD"
SYSTEM_VERSION = "2.0.0"
SYSTEM_AUTHOR = "Elite Trading"

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
```

### 2. API Configuration

```python
# OpenRouter API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT = 30
OPENROUTER_MAX_RETRIES = 3

# LiteLLM Configuration
LITELLM_CACHE_TTL = 3600
LITELLM_REQUEST_TIMEOUT = 60
LITELLM_MAX_TOKENS = 4096

# Model Configuration
DEFAULT_MODEL = "openrouter/anthropic/claude-3-5-sonnet"
BACKUP_MODEL = "openrouter/openai/gpt-4o"
ML_MODEL_PATH = "models/xauusd_ml_model.pkl"
```

### 3. Database and Cache

```python
# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_MAX_CONNECTIONS = 10

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ea_trading.db")
DATABASE_POOL_SIZE = 5
DATABASE_MAX_OVERFLOW = 10
```

### 4. Trading Constants

```python
# Trading Parameters
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "M5"
DEFAULT_LOT_SIZE = 0.01
DEFAULT_RISK_PERCENT = 1.0

# Risk Management
MAX_POSITIONS = 3
MAX_DAILY_TRADES = 10
MAX_CONSECUTIVE_LOSSES = 3
MIN_WIN_RATE = 0.6

# Spread and Slippage
MAX_SPREAD_POINTS = 50
MAX_SLIPPAGE_POINTS = 5
SPREAD_THRESHOLD = 20
```

### 5. Machine Learning

```python
# ML Configuration
ML_CONFIDENCE_THRESHOLD = 0.75
ML_FEATURE_WINDOW = 50
ML_PREDICTION_HORIZON = 5
ML_MODEL_UPDATE_HOURS = 24

# Feature Engineering
TECHNICAL_INDICATORS = ["RSI", "MACD", "BB", "ATR", "EMA", "SMA"]
MARKET_FEATURES = ["spread", "volume", "volatility", "session_time"]
PATTERN_FEATURES = ["support_resistance", "trend_strength", "momentum"]

# Model Types
ENSEMBLE_MODELS = ["xgboost", "random_forest", "lstm"]
CLASSIFICATION_MODELS = ["logistic_regression", "svm", "neural_network"]
REGRESSION_MODELS = ["linear_regression", "gradient_boosting", "neural_network"]
```

### 6. Notification Constants

```python
# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_PARSE_MODE = "HTML"
TELEGRAM_TIMEOUT = 10

# Alert Types
ALERT_TYPES = {
    "trade_entry": "🚀",
    "trade_exit": "📊",
    "risk_alert": "⚠️",
    "system_error": "❌",
    "daily_summary": "📈"
}

# Message Templates
TRADE_ENTRY_TEMPLATE = """
🚀 <b>Trade Entry - {symbol}</b>
📊 <b>Direction:</b> {direction}
💰 <b>Price:</b> ${price:.2f}
📈 <b>SL:</b> ${sl:.2f}
📉 <b>TP:</b> ${tp:.2f}
🎯 <b>Confidence:</b> {confidence:.1%}
"""
```

---

## Constantes de Sistema

### 1. Performance

```python
# Performance Constants
MAX_EXECUTION_TIME_MS = 100
MAX_LATENCY_MS = 50
MAX_MEMORY_USAGE_MB = 512
MAX_CPU_USAGE_PERCENT = 80

# Cache Configuration
CACHE_SIZE_MB = 100
CACHE_TTL_SECONDS = 3600
MAX_CACHE_ITEMS = 10000
```

### 2. Logging

```python
# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Log Levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}
```

### 3. Security

```python
# Security Constants
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT_MINUTES = 30
TOKEN_EXPIRY_HOURS = 24
ENCRYPTION_KEY_SIZE = 32

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_BURST_SIZE = 10
RATE_LIMIT_WINDOW_SECONDS = 60
```

---

## Constantes de Trading

### 1. Market Sessions

```python
# Market Sessions
MARKET_SESSIONS = {
    "sydney": {"open": "21:00", "close": "06:00", "timezone": "AEDT"},
    "tokyo": {"open": "23:00", "close": "08:00", "timezone": "JST"},
    "london": {"open": "07:00", "close": "16:00", "timezone": "GMT"},
    "new_york": {"open": "12:00", "close": "21:00", "timezone": "EST"}
}

# Session Overlaps
SESSION_OVERLAPS = {
    "london_tokyo": {"start": "07:00", "end": "08:00"},
    "london_new_york": {"start": "12:00", "end": "16:00"},
    "sydney_tokyo": {"start": "23:00", "end": "06:00"}
}
```

### 2. Risk Management

```python
# Risk Management Constants
MAX_RISK_PER_TRADE = 0.02      # 2% risk per trade
MAX_DAILY_RISK = 0.05          # 5% maximum daily risk
MAX_WEEKLY_RISK = 0.10         # 10% maximum weekly risk
MAX_MONTHLY_RISK = 0.20        # 20% maximum monthly risk

# Position Sizing
MIN_RISK_REWARD_RATIO = 1.5
DEFAULT_RISK_REWARD_RATIO = 2.0
MAX_RISK_REWARD_RATIO = 5.0

# Drawdown Limits
WARNING_DRAWDOWN = 0.05        # 5% drawdown warning
CRITICAL_DRAWDOWN = 0.10       # 10% drawdown critical
MAX_DRAWDOWN = 0.20            # 20% maximum drawdown
```

### 3. Strategy Parameters

```python
# Strategy Weights
STRATEGY_WEIGHTS = {
    "ml_prediction": 0.30,
    "smart_money": 0.25,
    "technical_analysis": 0.20,
    "volume_analysis": 0.15,
    "market_structure": 0.10
}

# Entry Conditions
MIN_CONFLUENCE_SCORE = 70.0     # Minimum confluence score for entry
MIN_CONFIDENCE_LEVEL = 0.75     # Minimum confidence level
MAX_SPREAD_THRESHOLD = 50       # Maximum spread for trading
```

---

## Constantes de Machine Learning

### 1. Model Configuration

```python
# Model Paths and Files
MODEL_BASE_PATH = "models/"
SCALER_PATH = f"{MODEL_BASE_PATH}scaler.pkl"
ENCODER_PATH = f"{MODEL_BASE_PATH}encoder.pkl"
FEATURE_IMPORTANCE_PATH = f"{MODEL_BASE_PATH}feature_importance.json"

# Model Parameters
N_ESTIMATORS = 100
MAX_DEPTH = 10
LEARNING_RATE = 0.01
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_weighted"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
```

### 2. Feature Engineering

```python
# Feature Windows
SHORT_WINDOW = 5
MEDIUM_WINDOW = 20
LONG_WINDOW = 50
EXTRA_LONG_WINDOW = 200

# Technical Indicators Periods
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
WILLIAMS_R_PERIOD = 14
CCI_PERIOD = 20
ADX_PERIOD = 14
```

### 3. Prediction Thresholds

```python
# Confidence Thresholds
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.60
MIN_CONFIDENCE = 0.50

# Signal Strength
STRONG_BUY_THRESHOLD = 0.80
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35
STRONG_SELL_THRESHOLD = 0.20
```

---

## Constantes de Interface

### 1. Colors and Themes

```mql5
// Color Constants
COLOR_BUY_SIGNAL = clrLime
COLOR_SELL_SIGNAL = clrRed
COLOR_NEUTRAL = clrYellow
COLOR_BACKGROUND = clrBlack
COLOR_TEXT = clrWhite
COLOR_DASHBOARD = clrCyan

// Chart Colors
COLOR_BULLISH = clrGreen
COLOR_BEARISH = clrRed
COLOR_NEUTRAL_BAR = clrGray

// Alert Colors
COLOR_WARNING = clrOrange
COLOR_ERROR = clrRed
COLOR_SUCCESS = clrGreen
COLOR_INFO = clrBlue
```

### 2. Display Settings

```mql5
// Display Constants
DASHBOARD_X = 20
DASHBOARD_Y = 20
DASHBOARD_WIDTH = 300
DASHBOARD_HEIGHT = 200
FONT_SIZE = 10
FONT_NAME = "Arial"

// Signal Display
SIGNAL_SIZE = 3
LABEL_OFFSET = 20
ARROW_SIZE = 2
LINE_WIDTH = 2
```

---

## Constantes de Performance

### 1. Optimization

```python
# Performance Optimization
MAX_CONCURRENT_TASKS = 5
TASK_TIMEOUT_SECONDS = 60
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1

# Memory Management
MEMORY_LIMIT_MB = 1024
GARBAGE_COLLECTION_INTERVAL = 300
CACHE_CLEANUP_INTERVAL = 600
```

### 2. Monitoring

```python
# Monitoring Constants
HEALTH_CHECK_INTERVAL = 60
METRICS_COLLECTION_INTERVAL = 30
PERFORMANCE_LOG_INTERVAL = 300
ALERT_THRESHOLD_ERRORS = 5
ALERT_THRESHOLD_LATENCY = 1000
```

---

## Convenções de Nomenclatura

### 1. MQL5/MQL4

| Tipo | Convenção | Exemplo |
|------|-----------|---------|
| Constantes | `NOME_EM_MAISCULAS` | `MAX_RISK_PERCENT` |
| Variáveis Globais | `g_` prefix | `g_tradingEnabled` |
| Variáveis Locais | `camelCase` | `currentPrice` |
| Funções | `PascalCase` | `CalculateRisk()` |
| Classes | `C` prefix | `CTradeManager` |
| Estruturas | `S` prefix | `STradeInfo` |

### 2. Python

| Tipo | Convenção | Exemplo |
|------|-----------|---------|
| Constantes | `MAIUSCULAS_COM_UNDERSCORE` | `MAX_RISK_PERCENT` |
| Variáveis | `snake_case` | `current_price` |
| Funções | `snake_case` | `calculate_risk()` |
| Classes | `PascalCase` | `TradeManager` |
| Módulos | `snake_case` | `trading_utils.py` |
| Privado | `_` prefix | `_internal_method()` |

### 3. JSON/YAML/TOML

| Tipo | Convenção | Exemplo |
|------|-----------|---------|
| Chaves | `snake_case` | `max_risk_percent` |
| Valores | `kebab-case` | `max-risk-percent` |
| Seções | `PascalCase` | `RiskManagement` |

---

## Gerenciamento de Constantes

### 1. Centralização

```python
# constants.py - Arquivo central de constantes
class TradingConstants:
    # Risk Management
    MAX_RISK_PERCENT = 2.0
    MAX_DAILY_LOSS = 10.0
    MIN_RR_RATIO = 1.5

    # Technical Indicators
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30.0
    RSI_OVERBOUGHT = 70.0

class MLConstants:
    CONFIDENCE_THRESHOLD = 0.75
    FEATURE_WINDOW = 50
    MODEL_UPDATE_HOURS = 24

# Uso em outros módulos
from constants import TradingConstants, MLConstants

risk_percent = TradingConstants.MAX_RISK_PERCENT
confidence = MLConstants.CONFIDENCE_THRESHOLD
```

### 2. Configuração Dinâmica

```python
# dynamic_config.py
class DynamicConstants:
    def __init__(self):
        self.load_from_config()

    def load_from_config(self):
        """Carrega constantes do arquivo de configuração"""
        config = self.load_config_file()

        self.MAX_RISK_PERCENT = config.get('max_risk_percent', 2.0)
        self.CONFIDENCE_THRESHOLD = config.get('confidence_threshold', 0.75)

    def update_constant(self, name: str, value):
        """Atualiza constante em runtime"""
        setattr(self, name, value)
        self.save_to_config()

    def validate_constant(self, name: str, value):
        """Valida valor da constante"""
        validation_rules = {
            'MAX_RISK_PERCENT': lambda v: 0 < v <= 10,
            'CONFIDENCE_THRESHOLD': lambda v: 0 <= v <= 1,
        }

        if name in validation_rules:
            return validation_rules[name](value)
        return True
```

### 3. Versionamento

```python
# versioned_constants.py
class VersionedConstants:
    VERSION = "2.0.0"

    # Constants v1.0
    V1_0 = {
        'MAX_RISK_PERCENT': 1.0,
        'RSI_PERIOD': 14,
    }

    # Constants v2.0
    V2_0 = {
        'MAX_RISK_PERCENT': 2.0,
        'RSI_PERIOD': 14,
        'CONFIDENCE_THRESHOLD': 0.75,
    }

    @classmethod
    def get_constants(cls, version: str = None):
        """Retorna constantes para versão específica"""
        if version is None:
            version = cls.VERSION

        if version.startswith("1."):
            return cls.V1_0
        elif version.startswith("2."):
            return cls.V2_0
        else:
            raise ValueError(f"Unsupported version: {version}")
```

---

## Exemplos de Uso

### 1. MQL5 - Constantes de Trading

```mql5
//+------------------------------------------------------------------+
//| Exemplo de uso de constantes em MQL5                           |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>

CTrade trade;

// Função de cálculo de risco
double CalculateLotSize(double riskPercent, double stopLossPoints) {
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (riskPercent / 100.0);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = riskAmount / (stopLossPoints * tickValue);

    // Aplicar limites de constantes
    lotSize = MathMax(lotSize, MIN_LOT_SIZE);
    lotSize = MathMin(lotSize, MAX_LOT_SIZE);

    // Ajustar para passo de lote
    lotSize = MathFloor(lotSize / LOT_STEP) * LOT_STEP;

    return lotSize;
}

// Função de validação de spread
bool IsSpreadAcceptable() {
    double spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                    SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;

    if (_Symbol == "XAUUSD") {
        return spread <= MAX_SPREAD_XAUUSD;
    } else {
        return spread <= MAX_SPREAD_FOREX;
    }
}
```

### 2. Python - Constantes de ML

```python
# ml_predictor.py
import numpy as np
from constants import MLConstants, TradingConstants

class MLPredictor:
    def __init__(self):
        self.confidence_threshold = MLConstants.CONFIDENCE_THRESHOLD
        self.feature_window = MLConstants.FEATURE_WINDOW
        self.max_risk = TradingConstants.MAX_RISK_PERCENT

    def predict(self, features):
        """Faz predição com base nas features"""
        prediction = self.model.predict_proba(features)
        confidence = np.max(prediction)

        # Usar constantes para validação
        if confidence < self.confidence_threshold:
            return None, confidence

        signal = np.argmax(prediction)
        return signal, confidence

    def calculate_position_size(self, signal, confidence):
        """Calcula tamanho da posição baseado no sinal e confiança"""
        base_risk = self.max_risk

        # Ajustar risco baseado na confiança
        if confidence > 0.9:
            risk_multiplier = 1.2
        elif confidence > 0.8:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.8

        adjusted_risk = base_risk * risk_multiplier
        return adjusted_risk
```

### 3. Configuração Multi-ambiente

```python
# config_factory.py
import os
from constants import TradingConstants, MLConstants

class ConfigFactory:
    @staticmethod
    def create_config(environment=None):
        """Cria configuração baseada no ambiente"""
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')

        if environment == 'production':
            return ProductionConfig()
        elif environment == 'staging':
            return StagingConfig()
        else:
            return DevelopmentConfig()

class BaseConfig:
    MAX_RISK_PERCENT = TradingConstants.MAX_RISK_PERCENT
    CONFIDENCE_THRESHOLD = MLConstants.CONFIDENCE_THRESHOLD

class ProductionConfig(BaseConfig):
    MAX_RISK_PERCENT = 0.5  # Mais conservador
    CONFIDENCE_THRESHOLD = 0.85  # Mais exigente
    DEBUG_MODE = False

class DevelopmentConfig(BaseConfig):
    MAX_RISK_PERCENT = 2.0  # Mais agressivo para testes
    CONFIDENCE_THRESHOLD = 0.60  # Mais permissivo
    DEBUG_MODE = True

# Uso
config = ConfigFactory.create_config()
risk = config.MAX_RISK_PERCENT
confidence = config.CONFIDENCE_THRESHOLD
```

Este guia completo de variáveis globais e constantes cobre todos os aspectos necessários para gerenciar valores constantes no projeto EA_SCALPER_XAUUSD, incluindo convenções de nomenclatura, gerenciamento dinâmico e exemplos práticos de uso.