# EA_SCALPER_XAUUSD - API Reference Guide

## Overview

O EA_SCALPER_XAUUSD é um sistema completo de trading algorítmico para XAUUSD (Gold/USD) que oferece múltiplas APIs para integração, automação e gestão de estratégias de trading. Este guia cobre todas as APIs disponíveis.

## Arquitetura da API

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend/UI   │────│   API Gateway    │────│   Backend       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   LiteLLM Proxy  │    │   MT5 MCP       │
                       └──────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────┐
                                            │   MetaTrader 5  │
                                            └─────────────────┘
```

## APIs Disponíveis

### 1. MetaTrader 5 MCP Server API
**Endpoint Base**: `http://localhost:8000`

API principal para comunicação com MetaTrader 5, permitindo operações de trading, obtenção de dados de mercado e gestão de contas.

#### Autenticação
```http
POST /auth/login
Content-Type: application/json

{
  "login": 12345678,
  "password": "your_password",
  "server": "RoboForex-Demo"
}
```

#### Endpoints Principais

##### Dados de Mercado
```http
GET /market/symbols
GET /market/symbols/{symbol}
GET /market/ticks/{symbol}
GET /market/bars/{symbol}/{timeframe}
GET /market/indicators/{symbol}/{timeframe}
```

##### Operações de Trading
```http
POST /trade/order
POST /trade/close
GET /trade/positions
GET /trade/history
GET /trade/orders
```

##### Gestão de Conta
```http
GET /account/info
GET /account/balance
GET /account/margin
GET /account/equity
```

##### Backtesting
```http
POST /backtest/start
GET /backtest/status/{id}
GET /backtest/results/{id}
```

### 2. LiteLLM Proxy API
**Endpoint Base**: `http://localhost:4000`

Proxy para múltiplos modelos de LLM (GPT-4, Claude, DeepSeek, etc.)

#### Configuração de Modelos
```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "openai/gpt-4"
      api_key: "your_openai_key"

  - model_name: "claude-3-opus"
    litellm_params:
      model: "anthropic/claude-3-opus-20240229"
      api_key: "your_anthropic_key"

  - model_name: "deepseek-r1-free"
    litellm_params:
      model: "openrouter/deepseek/deepseek-r1-free"
      api_key: "your_openrouter_key"
```

#### Endpoints
```http
POST /v1/chat/completions
POST /v1/embeddings
GET /v1/models
GET /health
```

### 3. AI Agent Management API
**Endpoint Base**: `http://localhost:8080`

API para gerenciamento de agentes autônomos de trading.

#### Agentes Disponíveis
- **MarketResearchAgent**: Análise de mercado multi-timeframe
- **StrategyDeveloperAgent**: Desenvolvimento de estratégias
- **BacktestAgent**: Execução de backtests automatizados
- **RiskManagementAgent**: Gestão de risco FTMO-compliant
- **MonitoringAgent**: Monitoramento em tempo real

#### Endpoints
```http
POST /agents/{agent_name}/execute
GET /agents/{agent_name}/status
GET /agents/{agent_name}/results
GET /agents/list
```

## Referência Completa de Endpoints

### MetaTrader 5 MCP API

#### Autenticação

##### POST /auth/login
Autentica usuário no MetaTrader 5.

**Request Body:**
```json
{
  "login": 12345678,
  "password": "senha_segura",
  "server": "RoboForex-Demo",
  "timeout": 10000
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123...",
  "account_info": {
    "login": 12345678,
    "server": "RoboForex-Demo",
    "balance": 10000.0,
    "equity": 10050.5,
    "margin": 150.0,
    "free_margin": 9900.5,
    "leverage": 100,
    "currency": "USD"
  }
}
```

#### Dados de Mercado

##### GET /market/symbols
Lista todos os símbolos disponíveis.

**Query Parameters:**
- `active_only` (boolean): Apenas símbolos ativos
- `filter` (string): Filtro de símbolo

**Response:**
```json
{
  "symbols": [
    {
      "name": "XAUUSD",
      "description": "Gold vs US Dollar",
      "digits": 2,
      "spread": 15,
      "volume_min": 0.01,
      "volume_max": 100.0,
      "volume_step": 0.01,
      "point": 0.01,
      "trade_contract_size": 100
    }
  ]
}
```

##### GET /market/bars/{symbol}/{timeframe}
Obtém barras de preços históricos.

**Path Parameters:**
- `symbol`: Símbolo do ativo (ex: XAUUSD)
- `timeframe`: Timeframe (M1, M5, M15, M30, H1, H4, D1)

**Query Parameters:**
- `count` (integer): Número de barras (default: 100)
- `start_time` (string): Data inicial (ISO 8601)
- `end_time` (string): Data final (ISO 8601)

**Response:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "H1",
  "bars": [
    {
      "time": "2024-01-01T00:00:00Z",
      "open": 2325.50,
      "high": 2327.25,
      "low": 2324.10,
      "close": 2326.85,
      "volume": 1250,
      "spread": 15
    }
  ]
}
```

##### GET /market/ticks/{symbol}
Obtém ticks em tempo real.

**Query Parameters:**
- `count` (integer): Número de ticks (default: 100)
- `offset` (integer): Offset do último tick

**Response:**
```json
{
  "symbol": "XAUUSD",
  "ticks": [
    {
      "time": "2024-01-01T12:00:00.123Z",
      "bid": 2325.85,
      "ask": 2326.00,
      "last": 2325.92,
      "volume": 0.5
    }
  ]
}
```

#### Operações de Trading

##### POST /trade/order
Envia ordem de compra/venda.

**Request Body:**
```json
{
  "symbol": "XAUUSD",
  "volume": 0.01,
  "order_type": "MARKET_BUY",
  "price": 0,
  "slippage": 10,
  "stop_loss": 2320.0,
  "take_profit": 2330.0,
  "magic_number": 12345,
  "comment": "EA Scalper"
}
```

**Response:**
```json
{
  "success": true,
  "order_ticket": 123456789,
  "execution_price": 2325.95,
  "message": "Order executed successfully"
}
```

##### GET /trade/positions
Lista posições abertas.

**Query Parameters:**
- `symbol` (string): Filtrar por símbolo
- `magic_number` (integer): Filtrar por magic number

**Response:**
```json
{
  "positions": [
    {
      "ticket": 123456789,
      "symbol": "XAUUSD",
      "type": "BUY",
      "volume": 0.01,
      "open_price": 2325.95,
      "current_price": 2327.12,
      "profit": 16.70,
      "swap": -0.25,
      "commission": -0.70,
      "stop_loss": 2320.0,
      "take_profit": 2330.0,
      "open_time": "2024-01-01T12:00:00Z",
      "magic_number": 12345
    }
  ]
}
```

##### POST /trade/close/{ticket}
Fecha posição específica.

**Path Parameters:**
- `ticket`: ID da posição

**Request Body (opcional):**
```json
{
  "volume": 0.01,
  "price": 0,
  "slippage": 10
}
```

#### Gestão de Conta

##### GET /account/info
Obtém informações detalhadas da conta.

**Response:**
```json
{
  "account_info": {
    "login": 12345678,
    "server": "RoboForex-Demo",
    "balance": 10500.75,
    "equity": 10523.20,
    "margin": 156.45,
    "free_margin": 10366.75,
    "margin_level": 6725.8,
    "leverage": 100,
    "currency": "USD",
    "company": "RoboForex",
    "name": "Trader Name",
    "stopout_mode": 0,
    "stopout_level": 20,
    "trade_allowed": true,
    "trade_expert": true
  }
}
```

#### Backtesting

##### POST /backtest/start
Inicia sessão de backtest.

**Request Body:**
```json
{
  "expert_file": "EA_XAUUSD_Scalper.ex5",
  "symbol": "XAUUSD",
  "period": "2024.01.01-2024.12.31",
  "deposit": 10000,
  "leverage": 100,
  "model": 4,
  "spread": 15,
  "optimization": false,
  "inputs": {
    "RiskPercent": 1.0,
    "MaxSpread": 20,
    "MagicNumber": 12345,
    "StopLossPips": 50,
    "TakeProfitPips": 100
  }
}
```

**Response:**
```json
{
  "success": true,
  "test_id": "bt_abc123...",
  "estimated_time": 180,
  "status": "queued"
}
```

##### GET /backtest/results/{test_id}
Obtém resultados do backtest.

**Response:**
```json
{
  "test_id": "bt_abc123...",
  "status": "completed",
  "results": {
    "total_trades": 1547,
    "profit_trades": 1018,
    "loss_trades": 529,
    "win_rate": 65.8,
    "profit_factor": 1.45,
    "recovery_factor": 2.1,
    "sharpe_ratio": 1.8,
    "max_drawdown": 8.2,
    "daily_drawdown": 4.1,
    "total_profit": 4250.75,
    "total_loss": -2925.50,
    "initial_deposit": 10000,
    "final_balance": 14250.75,
    "ftmo_compliance": {
      "daily_loss_check": true,
      "total_loss_check": true,
      "profit_target_check": true,
      "consistency_check": true,
      "overall_compliance": true
    }
  }
}
```

### LiteLLM Proxy API

#### POST /v1/chat/completions
Envia requisição de chat completion para LLM.

**Request Headers:**
```http
Content-Type: application/json
Authorization: Bearer any_string
```

**Request Body:**
```json
{
  "model": "deepseek-r1-free",
  "messages": [
    {
      "role": "system",
      "content": "Você é um especialista em trading de XAUUSD"
    },
    {
      "role": "user",
      "content": "Analise a estratégia de scalping para gold"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "deepseek-r1-free",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Para análise de scalping em XAUUSD..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

#### GET /v1/models
Lista modelos disponíveis.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1687882410,
      "owned_by": "openai"
    },
    {
      "id": "claude-3-opus",
      "object": "model",
      "created": 1687882411,
      "owned_by": "anthropic"
    },
    {
      "id": "deepseek-r1-free",
      "object": "model",
      "created": 1687882412,
      "owned_by": "openrouter"
    }
  ]
}
```

### AI Agent Management API

#### POST /agents/{agent_name}/execute
Executa tarefa específica do agente.

**Path Parameters:**
- `agent_name`: Nome do agente (market_research, strategy_developer, backtest, risk_management, monitoring)

**Request Body:**
```json
{
  "task_parameters": {
    "symbol": "XAUUSD",
    "timeframes": ["M5", "M15", "H1"],
    "analysis_period": "2024.01.01-2024.12.31",
    "risk_parameters": {
      "max_risk_percent": 1.0,
      "max_drawdown": 5.0
    }
  },
  "execution_mode": "async"
}
```

**Response:**
```json
{
  "success": true,
  "task_id": "task_xyz789...",
  "status": "running",
  "estimated_completion": 120,
  "message": "Market research started successfully"
}
```

#### GET /agents/{agent_name}/status/{task_id}
Verifica status de execução do agente.

**Response:**
```json
{
  "task_id": "task_xyz789...",
  "agent_name": "market_research",
  "status": "completed",
  "progress": 100,
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:02:15Z",
  "execution_time": 135,
  "results_available": true
}
```

#### GET /agents/{agent_name}/results/{task_id}
Obtém resultados da execução do agente.

**Response:**
```json
{
  "task_id": "task_xyz789...",
  "agent_name": "market_research",
  "results": {
    "market_analysis": {
      "trend_direction": "bullish",
      "support_levels": [2320, 2315, 2310],
      "resistance_levels": [2335, 2340, 2345],
      "volatility_analysis": {
        "current_volatility": 0.85,
        "historical_average": 0.75,
        "volatility_trend": "increasing"
      }
    },
    "timeframe_confluence": {
      "H4": "bullish",
      "H1": "bullish_pullback",
      "M15": "neutral",
      "M5": "oversold",
      "overall_signal": "buy_on_dip"
    },
    "recommended_strategy": {
      "entry_type": "limit_buy",
      "entry_price": 2322.50,
      "stop_loss": 2318.00,
      "take_profit": 2332.00,
      "risk_reward": 1:2,
      "confidence_level": 0.78
    }
  }
}
```

## Códigos de Erro

### Códigos Padrão HTTP
- `200 OK`: Requisição bem-sucedida
- `201 Created`: Recurso criado com sucesso
- `400 Bad Request`: Parâmetros inválidos
- `401 Unauthorized`: Autenticação necessária
- `403 Forbidden`: Acesso negado
- `404 Not Found`: Recurso não encontrado
- `429 Too Many Requests`: Limite de requisições excedido
- `500 Internal Server Error`: Erro interno do servidor

### Códigos de Erro Específicos

#### MetaTrader 5 MCP
- `MT5_1001`: Falha na inicialização do MT5
- `MT5_1002`: Credenciais inválidas
- `MT5_1003`: Símbolo não encontrado
- `MT5_1004`: Ordem rejeitada
- `MT5_1005`: Sem margem suficiente
- `MT5_1006`: Mercado fechado

#### LiteLLM Proxy
- `LLM_2001`: Chave de API inválida
- `LLM_2002`: Modelo não disponível
- `LLM_2003`: Limite de tokens excedido
- `LLM_2004`: Falha na resposta do modelo

#### AI Agent
- `AGENT_3001`: Agente não encontrado
- `AGENT_3002`: Parâmetros inválidos
- `AGENT_3003`: Falha na execução da tarefa
- `AGENT_3004`: Timeout na execução

## Rate Limiting

### Limites de Requisições
- **MetaTrader 5 MCP**: 100 requisições/por minuto
- **LiteLLM Proxy**: 60 requisições/por minuto
- **AI Agent API**: 30 requisições/por minuto

### Headers de Rate Limit
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067260
```

## Autenticação

### MetaTrader 5 MCP
Utiliza autenticação baseada em sessão via credenciais do MT5.

### LiteLLM Proxy
Autenticação via API key (pode usar qualquer string quando configurado).

### AI Agent API
Autenticação via JWT token gerado na primeira conexão.

## Webhooks

### Webhook de Trade Executed
```http
POST /webhooks/trade-executed
Content-Type: application/json

{
  "event_type": "trade_executed",
  "timestamp": "2024-01-01T12:00:00Z",
  "trade_details": {
    "ticket": 123456789,
    "symbol": "XAUUSD",
    "type": "BUY",
    "volume": 0.01,
    "price": 2325.95
  }
}
```

### Webhook de FTMO Alert
```http
POST /webhooks/ftmo-alert
Content-Type: application/json

{
  "event_type": "ftmo_alert",
  "timestamp": "2024-01-01T12:00:00Z",
  "alert_type": "daily_drawdown_warning",
  "current_drawdown": 4.2,
  "max_allowed": 5.0,
  "severity": "warning"
}
```

## SDKs e Bibliotecas

### Python SDK
```python
from ea_scalper_sdk import MT5Client, LLMClient, AgentClient

# Conectar ao MT5
mt5 = MT5Client(base_url="http://localhost:8000")
await mt5.connect(login=12345678, password="senha", server="RoboForex-Demo")

# Usar LLM
llm = LLMClient(base_url="http://localhost:4000")
response = await llm.chat("Analise XAUUSD")

# Gerenciar Agentes
agent = AgentClient(base_url="http://localhost:8080")
result = await agent.execute("market_research", {"symbol": "XAUUSD"})
```

### JavaScript SDK
```javascript
import { EAIClient } from 'ea-scalper-sdk';

const client = new EAIClient({
  mt5Url: 'http://localhost:8000',
  llmUrl: 'http://localhost:4000',
  agentUrl: 'http://localhost:8080'
});

// Executar análise
const analysis = await client.agents.execute('market_research', {
  symbol: 'XAUUSD'
});
```

## Exemplos de Uso

### Exemplo Básico - Market Data
```python
import requests

# Obter barras de XAUUSD
response = requests.get(
    "http://localhost:8000/market/bars/XAUUSD/H1",
    params={"count": 100}
)

if response.status_code == 200:
    data = response.json()
    bars = data["bars"]
    print(f"Obtidas {len(bars)} barras de XAUUSD H1")
```

### Exemplo Completo - Trade Automation
```python
import asyncio
from ea_scalper_sdk import MT5Client

async def automated_trading():
    mt5 = MT5Client("http://localhost:8000")

    # Conectar
    await mt5.connect(12345678, "senha", "RoboForex-Demo")

    # Analisar mercado
    bars = await mt5.get_bars("XAUUSD", "H1", 100)
    analysis = analyze_market(bars)

    if analysis["signal"] == "BUY":
        # Executar trade
        order = await mt5.place_order({
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_BUY",
            "stop_loss": analysis["sl"],
            "take_profit": analysis["tp"]
        })
        print(f"Ordem executada: {order['order_ticket']}")

    await mt5.disconnect()

asyncio.run(automated_trading())
```

## Próximos Passos

1. **Configurar Ambiente**: Instale dependências e configure servidores
2. **Testar Conexão**: Verifique conectividade com MT5 e APIs
3. **Implementar Estratégia**: Use os exemplos para criar sua estratégia
4. **Backtesting**: Teste estratégias antes de usar em produção
5. **Monitoramento**: Implemente monitoramento e alertas

## Suporte

- **Documentação**: Verifique os guias adicionais em `/docs/`
- **Exemplos**: Veja exemplos práticos em `/docs/examples/`
- **Issues**: Reporte problemas no GitHub repository
- **Comunidade**: Participe do Discord/Telegram para suporte

---

*Última atualização: Janeiro 2024*