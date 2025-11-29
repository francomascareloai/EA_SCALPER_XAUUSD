# Guia de Configuração de Arquivos - EA_SCALPER_XAUUSD

## Overview

Este documento descreve todos os arquivos de configuração do projeto EA_SCALPER_XAUUSD nos formatos YAML, JSON e TOML, incluindo suas estruturas, parâmetros e exemplos de uso.

## Sumário

1. [Arquivos JSON](#arquivos-json)
2. [Arquivos YAML](#arquivos-yaml)
3. [Arquivos TOML](#arquivos-toml)
4. [Estrutura de Diretórios](#estrutura-de-diretórios)
5. [Templates de Configuração](#templates-de-configuração)
6. [Validação de Sintaxe](#validação-de-sintaxe)
7. [Gerenciamento de Versões](#gerenciamento-de-versões)
8. [Exemplos Práticos](#exemplos-práticos)
9. [Troubleshooting](#troubleshooting)

---

## Arquivos JSON

### 1. config_sistema.json

#### Descrição
Configuração principal do sistema de processamento multi-agente e validação FTMO.

#### Estrutura
```json
{
  "version": "6.0",
  "batch_size": 100,
  "max_threads": 4,
  "timeout_per_file": 60,
  "enable_real_processing": true,
  "enable_ftmo_validation": true,
  "enable_auto_backup": true
}
```

#### Parâmetros Detalhados

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `version` | string | "6.0" | Versão da configuração do sistema |
| `batch_size` | integer | 100 | Número de arquivos processados por lote |
| `max_threads` | integer | 4 | Número máximo de threads simultâneas |
| `timeout_per_file` | integer | 60 | Timeout por arquivo em segundos |
| `enable_real_processing` | boolean | true | Ativa processamento real |
| `enable_ftmo_validation` | boolean | true | Ativa validação FTMO |
| `enable_auto_backup` | boolean | true | Ativa backup automático |

#### Validação
```json
{
  "required": ["version", "batch_size", "max_threads"],
  "constraints": {
    "batch_size": {"min": 1, "max": 1000},
    "max_threads": {"min": 1, "max": 32},
    "timeout_per_file": {"min": 10, "max": 300}
  }
}
```

### 2. config_multi_agente.json

#### Descrição
Configuração do sistema multi-agente Qwen com especializações diferentes.

#### Estrutura Completa
```json
{
    "versao": "1.0",
    "sistema": "Multi-Agente Qwen 3",
    "orquestrador": "Trae AI (Claude 4 Sonnet)",
    "data_criacao": "2025-08-13 10:56:45",
    "capacidades": {
        "terminais_simultaneos": 5,
        "processamento_paralelo": true,
        "validacao_cruzada": true,
        "especializacao_profunda": true
    },
    "comandos_inicializacao": [
        "qwen chat --model qwen3-coder-plus --system-file prompts/classificador_system.txt",
        "qwen chat --model qwen3-coder-plus --system-file prompts/analisador_system.txt",
        "qwen chat --model qwen3-coder-plus --system-file prompts/gerador_system.txt",
        "qwen chat --model qwen3-coder-plus --system-file prompts/validador_system.txt",
        "qwen chat --model qwen3-coder-plus --system-file prompts/documentador_system.txt"
    ],
    "agentes": [
        {
            "Terminal": 1,
            "Especialidade": "Analise e classificacao de codigos MQL4/MQL5/Pine",
            "Nome": "Classificador_Trading",
            "Prompt": "prompts/classificador_system.txt",
            "Modelo": "qwen3-coder-plus"
        },
        {
            "Terminal": 2,
            "Especialidade": "Extracao completa de metadados",
            "Nome": "Analisador_Metadados",
            "Prompt": "prompts/analisador_system.txt",
            "Modelo": "qwen3-coder-plus"
        },
        {
            "Terminal": 3,
            "Especialidade": "Extracao de snippets reutilizaveis",
            "Nome": "Gerador_Snippets",
            "Prompt": "prompts/gerador_system.txt",
            "Modelo": "qwen3-coder-plus"
        },
        {
            "Terminal": 4,
            "Especialidade": "Analise de conformidade FTMO",
            "Nome": "Validador_FTMO",
            "Prompt": "prompts/validador_system.txt",
            "Modelo": "qwen3-coder-plus"
        },
        {
            "Terminal": 5,
            "Especialidade": "Geracao de documentacao e indices",
            "Nome": "Documentador_Trading",
            "Prompt": "prompts/documentador_system.txt",
            "Modelo": "qwen3-coder-plus"
        }
    ]
}
```

#### Detalhes dos Agentes

| Agente | Terminal | Especialidade | Modelo |
|--------|----------|---------------|---------|
| Classificador_Trading | 1 | Análise e classificação de códigos | qwen3-coder-plus |
| Analisador_Metadados | 2 | Extração de metadados | qwen3-coder-plus |
| Gerador_Snippets | 3 | Extração de snippets | qwen3-coder-plus |
| Validador_FTMO | 4 | Validação FTMO | qwen3-coder-plus |
| Documentador_Trading | 5 | Documentação | qwen3-coder-plus |

### 3. mcp.json

#### Descrição
Configuração dos servidores MCP (Model Context Protocol) para integrações externas.

#### Estrutura
```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "-e",
        "GITHUB_TOOLSETS",
        "-e",
        "GITHUB_READ_ONLY",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE",
        "GITHUB_TOOLSETS": "",
        "GITHUB_READ_ONLY": ""
      }
    },
    "playwright": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp@latest",
        "--browser=",
        "--headless=",
        "--viewport-size="
      ]
    }
  }
}
```

### 4. Arquivos de Metadados (.meta.json)

#### Descrição
Arquivos de metadados para EAs e indicadores do sistema.

#### Estrutura Padrão
```json
{
  "file_info": {
    "name": "EA_XAUUSD_Scalper_v1.0",
    "path": "/EAs/Scalping/EA_XAUUSD_Scalper_v1.0.mq5",
    "type": "EA",
    "language": "MQL5",
    "size_bytes": 15420
  },
  "trading_info": {
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "strategy": "Scalping",
    "risk_level": "Medium"
  },
  "performance_info": {
    "backtested": true,
    "profit_factor": 1.85,
    "max_drawdown": 12.5,
    "win_rate": 68.3
  },
  "metadata": {
    "created_date": "2024-01-15",
    "last_modified": "2024-10-18",
    "version": "1.0.0",
    "author": "Elite Trading",
    "description": "Advanced XAUUSD scalping EA with ML integration"
  }
}
```

---

## Arquivos YAML

### 1. config.yaml (BMAD Method)

#### Descrição
Configuração do módulo BMAD™ Core para agentes TEA.

#### Estrutura
```yaml
# Powered by BMAD™ Core
name: bmm
short-title: BMad Method Module
author: Brian (BMad) Madison

# TEA Agent Configuration
tea_use_mcp_enhancements: true # Enable Playwright MCP capabilities (healing, exploratory, verification)
```

### 2. LiteLLM Proxy Configuration

#### Descrição
Configuração do servidor proxy LiteLLM para balanceamento de carga e caching.

#### Estrutura
```yaml
model_list:
  - model_name: "claude-3-5-sonnet"
    litellm_params:
      model: "openrouter/anthropic/claude-3-5-sonnet"
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: "https://openrouter.ai/api/v1"
      max_tokens: 4096
      temperature: 0.1

  - model_name: "gpt-4o"
    litellm_params:
      model: "openrouter/openai/gpt-4o"
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: "https://openrouter.ai/api/v1"
      max_tokens: 4096
      temperature: 0.2

  - model_name: "claude-3-haiku"
    litellm_params:
      model: "openrouter/anthropic/claude-3-haiku"
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: "https://openrouter.ai/api/v1"
      max_tokens: 4096
      temperature: 0.1

litellm_settings:
  drop_params: true
  set_verbose: true
  success_callback: ["langfuse"]

router_settings:
  retry_after: 5
  allowed_fails: 3
  usage_budgets:
    prompt_budget: 1000000
    completion_budget: 1000000

cache_settings:
  type: "redis"
  redis_url: os.environ/REDIS_URL
  ttl: 3600

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

### 3. Trading Configuration

#### Descrição
Configuração de parâmetros de trading para diferentes ativos e estratégias.

#### Estrutura Completa
```yaml
# Trading Configuration for EA_SCALPER_XAUUSD
project:
  name: "EA_SCALPER_XAUUSD"
  version: "2.0"
  description: "Advanced XAUUSD Scalping System with ML Integration"

assets:
  xauusd:
    symbol: "XAUUSD"
    description: "Gold vs US Dollar"
    point: 0.01
    digits: 2
    spread_avg: 20
    volatility_threshold: 15

    risk_management:
      max_risk_percent: 0.01
      max_daily_loss: 0.02
      max_positions: 2
      min_distance_points: 50

    timeframes:
      primary: "M5"
      secondary: ["M15", "H1", "H4"]
      scalping: "M1"
      swing: "H4"

    sessions:
      asian:
        start: "22:00"
        end: "07:00"
        enabled: true
        max_spread: 30
      european:
        start: "07:00"
        end: "16:00"
        enabled: true
        max_spread: 25
      american:
        start: "13:00"
        end: "22:00"
        enabled: true
        max_spread: 20

strategies:
  ml_scalping:
    enabled: true
    confidence_threshold: 0.75
    update_frequency_hours: 24
    features:
      - price_action
      - volume_analysis
      - technical_indicators
      - sentiment_data

  smart_money:
    enabled: true
    ict_concepts:
      - market_structure_shift
      - fvg_concept
      - liquidity_zones
      - order_blocks

  volatility_breakout:
    enabled: true
    min_volatility: 15
    max_spread_points: 30
    breakout_confirmation: true

ml_configuration:
  models:
    prediction:
      type: "ensemble"
      models:
        - "xgboost_classifier"
        - "lstm_predictor"
        - "transformer_model"
      confidence_threshold: 0.75

  training:
    data_source: "historical_ticks"
    lookback_period: 1000
    validation_split: 0.2
    retrain_frequency: "weekly"

  features:
    technical:
      - RSI
      - MACD
      - Bollinger_Bands
      - ATR
      - Volume
    market:
      - spread
      - volatility
      - session_time
      - day_of_week

risk_management:
  position_sizing:
    method: "fixed_percentage"
    risk_per_trade: 0.01
    max_lot_size: 1.0
    min_lot_size: 0.01

  stop_loss:
    method: "atr_based"
    atr_multiplier: 2.0
    min_sl_points: 50
    max_sl_points: 200

  take_profit:
    method: "risk_reward_ratio"
    rr_ratio: 1.5
    min_tp_points: 75
    max_tp_points: 300

  trailing_stop:
    enabled: true
    activation_profit: 50
    trail_distance: 30
    trail_step: 10

notifications:
  telegram:
    enabled: true
    bot_token: os.environ/TELEGRAM_BOT_TOKEN
    chat_id: os.environ/TELEGRAM_CHAT_ID
    alerts:
      - trade_entry
      - trade_exit
      - risk_alerts
      - system_errors

  discord:
    enabled: false
    webhook_url: os.environ/DISCORD_WEBHOOK_URL
    alerts:
      - daily_summary
      - performance_metrics

performance:
  optimization:
    latency_threshold_ms: 100
    enable_caching: true
    cache_size_mb: 100
    async_operations: true

  monitoring:
    log_level: "INFO"
    metrics_collection: true
    health_check_interval: 60

  backup:
    enabled: true
    frequency: "daily"
    retention_days: 30
    compression: true
```

---

## Arquivos TOML

### 1. pyproject.toml (MCP Code Checker)

#### Descrição
Configuração do projeto Python MCP Code Checker com dependências e ferramentas.

#### Estrutura Completa
```toml
[project]
name = "mcp-code-checker"
version = "0.1.0"
authors = [
    {name = "Your Name"},
]
description = "An MCP server for running code checks (pylint and pytest)"
readme = "README.md"
requires-python = ">=3.13"
license = {text = "MIT"}
keywords = ["mcp", "server", "code-checker", "pylint", "pytest", "claude", "ai", "assistant"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "pathspec>=0.12.1",
    "mcp>=1.3.0",
    "mcp[server]>=1.3.0",
    "mcp[cli]>=1.3.0",
    "pylint>=3.3.3",
    "pytest>=8.3.5",
    "pytest-json-report>=1.5.0",
    "pytest-asyncio>=0.25.3",
    "mypy>=1.9.0",
    "structlog>=24.5.0",
    "python-json-logger>=3.2.1",
]

[project.optional-dependencies]
dev = [
    "black>=24.10.0",
    "isort>=5.13.2",
    ]

[project.urls]
"Homepage" = "https://github.com/yourusername/mcp-code-checker"
"Bug Tracker" = "https://github.com/yourusername/mcp-code-checker/issues"
"Documentation" = "https://github.com/yourusername/mcp-code-checker#readme"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_default_fixture_loop_scope = "function"
pythonpath = ["."]

[tool.black]
line-length = 88
target-version = ["py313"]

[tool.isort]
profile = "black"
line_length = 88
float_to_top = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
strict_optional = true
mypy_path = "stubs"
disable_error_code = ["unused-ignore"]

[[tool.mypy.overrides]]
module = ["pytest.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["mcp.server.fastmcp"]
ignore_missing_imports = true

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### 2. codex_mcp_config.toml

#### Descrição
Configuração completa dos servidores MCP para integração com Codex CLI.

#### Estrutura Detalhada
```toml
# Converted from .trae/mcp.json for Codex CLI
# Place this content under your ~/.codex/config.toml
# Schema per https://github.com/openai/codex/blob/main/docs/config.md#mcp_servers

[mcp_servers.context7]
command = "npx"
args = ["-y", "@upstash/context7-mcp"]
env = {}

[mcp_servers.sequential_thinking]
command = "npx"
args = ["-y", "@ahxxm/server-sequential-thinking"]
env = {}

[mcp_servers.puppeteer]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-puppeteer"]
env = {}

[mcp_servers.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "" }

[mcp_servers.everything]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-everything"]
env = {}

# Custom MCP Servers for EA_SCALPER_XAUUSD
[mcp_servers.mcp_code_checker]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\mcp-code-checker\\src\\main.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.file_analyzer]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_file_analyzer.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.ftmo_validator]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_ftmo_validator.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.metadata_generator]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_metadata_generator.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.code_classifier]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_code_classifier.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.batch_processor]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_batch_processor.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.task_manager]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_task_manager.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.trading_classifier]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\trading_classifier_mcp.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.api_integration]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\api_integration_mcp.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.code_analysis]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\code_analysis_mcp.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.project_scaffolding]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\project_scaffolding_mcp.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}

[mcp_servers.youtube_transcript]
command = "python"
args = ["c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\MCP_Integration\\servers\\mcp_youtube_transcript.py"]
env = {
    PYTHONPATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD",
    VIRTUAL_ENV = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv",
    PATH = "c:\\Users\\Admin\\Documents\\EA_SCALPER_XAUUSD\\.venv\\Scripts;C:\\Python313;C:\\Python313\\Scripts"
}
```

---

## Estrutura de Diretórios

### Organização de Arquivos de Configuração

```
EA_SCALPER_XAUUSD/
├── docs/configuration/           # Documentação de configuração
├── configs/                      # Configurações centralizadas
│   ├── trading/                 # Configurações de trading
│   │   ├── xauusd.yaml         # Config XAUUSD específica
│   │   ├── risk_management.yaml # Gestão de risco
│   │   └── strategies.yaml      # Config de estratégias
│   ├── mcp/                     # Configurações MCP
│   │   ├── codex_mcp_config.toml
│   │   └── mcp_servers.json
│   ├── ml/                      # Configurações ML
│   │   ├── models.yaml
│   │   └── training_params.json
│   └── deployment/              # Config de deployment
│       ├── production.json
│       ├── staging.yaml
│       └── development.toml
├── 📋 DOCUMENTACAO_FINAL/        # Configurações finais
│   └── CONFIGURACOES/
│       └── CONFIG_FINAL/
│           ├── config_sistema.json
│           └── config_multi_agente.json
└── 🤖 AI_AGENTS/                # Config MCP agents
    └── MCP_Code_Checker/
        └── pyproject.toml
```

### Convenções de Nomenclatura

| Tipo de Arquivo | Extensão | Convenção | Exemplo |
|-----------------|----------|-----------|---------|
| Config Sistema | .json | `config_[nome].json` | `config_sistema.json` |
| Config Trading | .yaml | `[symbol]_[tipo].yaml` | `xauusd_trading.yaml` |
| Config MCP | .toml | `[sistema]_mcp_config.toml` | `codex_mcp_config.toml` |
| Metadados | .json | `[arquivo].meta.json` | `ea_scalper.meta.json` |
| Python Project | .toml | `pyproject.toml` | `pyproject.toml` |
| LiteLLM | .yaml | `litellm_[env].yaml` | `litellm_production.yaml` |

---

## Templates de Configuração

### Template JSON para Sistema

```json
{
  "version": "1.0.0",
  "metadata": {
    "created_date": "YYYY-MM-DD",
    "author": "Author Name",
    "description": "Configuration description"
  },
  "system": {
    "name": "system_name",
    "environment": "development|staging|production",
    "debug": false
  },
  "features": {
    "feature_1": {
      "enabled": true,
      "config": {}
    }
  },
  "limits": {
    "max_items": 100,
    "timeout_seconds": 60
  }
}
```

### Template YAML para Trading

```yaml
# Trading Configuration Template
project:
  name: "Project Name"
  version: "1.0.0"
  environment: "development"

assets:
  symbol_name:
    symbol: "SYMBOL"
    description: "Description"
    point: 0.01
    digits: 2

    risk_management:
      max_risk_percent: 0.01
      max_positions: 1

    sessions:
      session_name:
        start: "HH:MM"
        end: "HH:MM"
        enabled: true

strategies:
  strategy_name:
    enabled: true
    parameters:
      param_1: value_1
      param_2: value_2

notifications:
  telegram:
    enabled: false
    alerts: []
```

### Template TOML para Python

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Project description"
authors = [{name = "Author Name"}]
requires-python = ">=3.8"
dependencies = [
    "package1>=1.0.0",
    "package2>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "black",
    "pytest",
]

[tool.tool_name]
setting_1 = "value_1"
setting_2 = 42

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

---

## Validação de Sintaxe

### JSON Validation

```python
# validate_json.py
import json
import sys
from typing import Dict, Any

def validate_json_file(file_path: str) -> bool:
    """Valida sintaxe de arquivo JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ {file_path}: JSON válido")

        # Validar estrutura específica
        if file_path.endswith('config_sistema.json'):
            return validate_config_sistema(data)
        elif file_path.endswith('config_multi_agente.json'):
            return validate_config_multi_agente(data)

        return True

    except json.JSONDecodeError as e:
        print(f"❌ {file_path}: Erro JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Erro - {e}")
        return False

def validate_config_sistema(data: Dict[str, Any]) -> bool:
    """Valida config_sistema.json"""
    required_fields = ['version', 'batch_size', 'max_threads']

    for field in required_fields:
        if field not in data:
            print(f"❌ Campo obrigatório ausente: {field}")
            return False

    # Validar tipos
    if not isinstance(data['batch_size'], int) or data['batch_size'] < 1:
        print("❌ batch_size deve ser inteiro >= 1")
        return False

    if not isinstance(data['max_threads'], int) or data['max_threads'] < 1:
        print("❌ max_threads deve ser inteiro >= 1")
        return False

    print("✅ config_sistema.json validado")
    return True

def validate_config_multi_agente(data: Dict[str, Any]) -> bool:
    """Valida config_multi_agente.json"""
    required_sections = ['versao', 'sistema', 'agentes']

    for section in required_sections:
        if section not in data:
            print(f"❌ Seção obrigatória ausente: {section}")
            return False

    # Validar agentes
    if 'agentes' in data and isinstance(data['agentes'], list):
        for i, agente in enumerate(data['agentes']):
            required_agent_fields = ['Terminal', 'Nome', 'Modelo']
            for field in required_agent_fields:
                if field not in agente:
                    print(f"❌ Agente {i}: campo obrigatório ausente: {field}")
                    return False

    print("✅ config_multi_agente.json validado")
    return True

if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        '📋 DOCUMENTACAO_FINAL/CONFIGURACOES/CONFIG_FINAL/config_sistema.json',
        '📋 DOCUMENTACAO_FINAL/CONFIGURACOES/CONFIG_FINAL/config_multi_agente.json'
    ]

    all_valid = True
    for file_path in files:
        if not validate_json_file(file_path):
            all_valid = False

    sys.exit(0 if all_valid else 1)
```

### YAML Validation

```python
# validate_yaml.py
import yaml
import sys
from typing import Dict, Any

def validate_yaml_file(file_path: str) -> bool:
    """Valida sintaxe de arquivo YAML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        print(f"✅ {file_path}: YAML válido")
        return True

    except yaml.YAMLError as e:
        print(f"❌ {file_path}: Erro YAML - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Erro - {e}")
        return False

if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        'configs/trading/xauusd.yaml',
        'configs/litellm/production.yaml'
    ]

    all_valid = True
    for file_path in files:
        if not validate_yaml_file(file_path):
            all_valid = False

    sys.exit(0 if all_valid else 1)
```

### TOML Validation

```python
# validate_toml.py
import toml
import sys
from typing import Dict, Any

def validate_toml_file(file_path: str) -> bool:
    """Valida sintaxe de arquivo TOML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)

        print(f"✅ {file_path}: TOML válido")
        return True

    except toml.TomlDecodeError as e:
        print(f"❌ {file_path}: Erro TOML - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Erro - {e}")
        return False

if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        'configs/codex_mcp_config.toml',
        '🤖 AI_AGENTS/MCP_Code_Checker/pyproject.toml'
    ]

    all_valid = True
    for file_path in files:
        if not validate_toml_file(file_path):
            all_valid = False

    sys.exit(0 if all_valid else 1)
```

---

## Gerenciamento de Versões

### Versionamento de Configurações

#### JSON Schema Versioning
```json
{
  "schema_version": "2.0",
  "config_version": "1.2.3",
  "compatibility": {
    "min_schema_version": "1.0",
    "max_schema_version": "2.0"
  },
  "migration_info": {
    "from": "1.0",
    "to": "2.0",
    "migration_required": true
  }
}
```

#### YAML Versioning
```yaml
# Configuration Versioning
version: "2.1.0"
schema_version: "1.0"

# Compatibility Matrix
compatibility:
  min_version: "2.0.0"
  max_version: "2.1.0"

# Migration Information
migration:
  auto_migrate: true
  backup_before_migrate: true
  migration_steps:
    - version: "2.0.0"
      description: "Add risk management section"
      required: true
    - version: "2.1.0"
      description: "Update ML configuration"
      required: false
```

#### TOML Versioning
```toml
[version]
config = "1.2.3"
schema = "2.0"

[compatibility]
min_version = "1.0.0"
max_version = "1.2.3"

[migration]
auto_migrate = true
backup_before = true
```

---

## Exemplos Práticos

### 1. Configuração Completa para Produção

#### config_sistema_production.json
```json
{
  "version": "6.0",
  "environment": "production",
  "batch_size": 50,
  "max_threads": 8,
  "timeout_per_file": 30,
  "enable_real_processing": true,
  "enable_ftmo_validation": true,
  "enable_auto_backup": true,
  "monitoring": {
    "enable_metrics": true,
    "log_level": "INFO",
    "alert_thresholds": {
      "error_rate": 0.05,
      "response_time": 1000,
      "memory_usage": 0.8
    }
  },
  "security": {
    "enable_audit_log": true,
    "rate_limiting": {
      "requests_per_minute": 100,
      "burst_size": 20
    }
  }
}
```

#### trading_production.yaml
```yaml
# Production Trading Configuration
project:
  name: "EA_SCALPER_XAUUSD"
  version: "2.1.0"
  environment: "production"

assets:
  xauusd:
    symbol: "XAUUSD"
    risk_management:
      max_risk_percent: 0.005  # 0.5% risk conservativo
      max_daily_loss: 0.01      # 1% máximo diário
      max_positions: 1          # Apenas uma posição por vez

    strategies:
      ml_scalping:
        enabled: true
        confidence_threshold: 0.85  # Alta confiança

      smart_money:
        enabled: false              # Desativado em produção

    sessions:
      asian:
        enabled: false             # Evitar sessão asiática
      european:
        enabled: true
        max_spread: 25
      american:
        enabled: true
        max_spread: 20

notifications:
  telegram:
    enabled: true
    alerts:
      - trade_entry
      - trade_exit
      - risk_alerts
      - system_errors
```

### 2. Configuração de Desenvolvimento

#### codex_mcp_dev.toml
```toml# Development MCP Configuration
[mcp_servers.mcp_code_checker]
command = "python"
args = ["src/main.py"]
env = {
    DEBUG = "true",
    LOG_LEVEL = "DEBUG",
    PYTHONPATH = ".",
    VIRTUAL_ENV = ".venv"
}

[mcp_servers.file_analyzer]
command = "python"
args = ["MCP_Integration/servers/mcp_file_analyzer.py"]
env = {
    DEBUG = "true",
    LOG_LEVEL = "DEBUG",
    PYTHONPATH = ".",
    VIRTUAL_ENV = ".venv"
}

# Development-specific servers
[mcp_servers.debug_tools]
command = "python"
args = ["tools/debug_server.py"]
env = {
    DEBUG_MODE = "true",
    VERBOSE_LOGGING = "true"
}
```

### 3. Configuração para Testes

#### config_test.json
```json
{
  "version": "6.0",
  "environment": "test",
  "batch_size": 10,
  "max_threads": 2,
  "timeout_per_file": 120,
  "enable_real_processing": false,
  "enable_ftmo_validation": true,
  "enable_auto_backup": false,
  "testing": {
    "mock_apis": true,
    "use_test_data": true,
    "simulate_failures": false,
    "performance_test": true
  }
}
```

---

## Troubleshooting

### Problemas Comuns

#### 1. Erro de Sintaxe JSON

**Sintoma:** `json.decoder.JSONDecodeError`

**Causa:** Arquivo JSON mal formatado

**Solução:**
```bash
# Validar JSON
python -m json.tool config_sistema.json

# Corrigir formatação
python -c "
import json
with open('config_sistema.json', 'r') as f:
    data = json.load(f)
with open('config_sistema_fixed.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

#### 2. Erro de Sintaxe YAML

**Sintoma:** `yaml.scanner.ScannerError`

**Causa:** Indentação incorreta ou caracteres inválidos

**Solução:**
```bash
# Validar YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Usar linter YAML
pip install yamllint
yamllint config.yaml
```

#### 3. Erro de Sintaxe TOML

**Sintoma:** `toml.decoder.TomlDecodeError`

**Causa:** Formatação TOML incorreta

**Solução:**
```bash
# Validar TOML
python -c "import toml; toml.load(open('config.toml'))"

# Verificar sintaxe online
# https://www.toml-lint.com/
```

### Ferramentas de Debug

#### Config Validator
```python
# config_validator.py
import json
import yaml
import toml
import sys
from pathlib import Path

class ConfigValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_all_configs(self, config_dir: str):
        """Valida todos os arquivos de configuração"""
        config_path = Path(config_dir)

        for file_path in config_path.rglob("*"):
            if file_path.is_file():
                self.validate_file(file_path)

    def validate_file(self, file_path: Path):
        """Valida arquivo individual"""
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.json':
                with open(file_path, 'r') as f:
                    json.load(f)
                print(f"✅ {file_path}: JSON válido")

            elif suffix in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {file_path}: YAML válido")

            elif suffix == '.toml':
                with open(file_path, 'r') as f:
                    toml.load(f)
                print(f"✅ {file_path}: TOML válido")

        except Exception as e:
            error_msg = f"❌ {file_path}: {str(e)}"
            self.errors.append(error_msg)
            print(error_msg)

    def print_summary(self):
        """Imprime resumo da validação"""
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  - {error}")

if __name__ == "__main__":
    validator = ConfigValidator()

    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = "."

    validator.validate_all_configs(config_dir)
    validator.print_summary()
```

Este guia completo de configuração de arquivos cobre todos os aspectos necessários para trabalhar com arquivos JSON, YAML e TOML no projeto EA_SCALPER_XAUUSD, incluindo validação, versionamento e troubleshooting.