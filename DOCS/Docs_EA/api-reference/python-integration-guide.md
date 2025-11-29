# Guia de Integração Python - EA_SCALPER_XAUUSD

## Overview

Este guia detalha como integrar suas aplicações Python com o sistema EA_SCALPER_XAUUSD, cobrindo desde a configuração inicial até implementações avançadas.

## Pré-requisitos

### Python Requirements
- Python 3.8+
- pip (gerenciador de pacotes Python)

### Bibliotecas Necessárias
```bash
pip install requests asyncio aiohttp websockets pandas numpy
pip install MetaTrader5 as mt5
pip install python-dotenv
pip install pydantic
pip install fastapi uvicorn
```

### Ambiente de Desenvolvimento
- MetaTrader 5 instalado
- Conta RoboForex (Demo ou Live)
- Docker (opcional, para ambiente isolado)

## Configuração Inicial

### 1. Instalação do SDK

#### Método 1: Via pip (Recomendado)
```bash
pip install ea-scalper-sdk
```

#### Método 2: Via GitHub (Desenvolvimento)
```bash
git clone https://github.com/your-org/ea-scalper-xauusd.git
cd ea-scalper-xauusd
pip install -e .
```

### 2. Configuração de Ambiente

Crie arquivo `.env` no diretório raiz do projeto:

```env
# MetaTrader 5 Configuration
MT5_LOGIN=12345678
MT5_PASSWORD=sua_senha_aqui
MT5_SERVER=RoboForex-Demo
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# API Configuration
MT5_MCP_URL=http://localhost:8000
LITELLM_URL=http://localhost:4000
AGENT_URL=http://localhost:8080

# LiteLLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Security
API_SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here
```

### 3. Validação de Configuração

```python
import os
from dotenv import load_dotenv

def validate_config():
    """Valida se todas as variáveis de ambiente necessárias estão configuradas"""
    load_dotenv()

    required_vars = [
        'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER',
        'OPENROUTER_API_KEY', 'API_SECRET_KEY'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Variáveis de ambiente obrigatórias não configuradas: {missing_vars}")

    print("✅ Configuração validada com sucesso!")

if __name__ == "__main__":
    validate_config()
```

## SDK Básico

### 1. Conexão com MetaTrader 5

```python
import asyncio
from ea_scalper_sdk import MT5Client
from ea_scalper_sdk.exceptions import MT5ConnectionError

async def connect_mt5():
    """Estabelece conexão com MetaTrader 5"""
    client = MT5Client()

    try:
        # Conectar usando variáveis de ambiente
        success = await client.connect_from_env()

        if success:
            print("✅ Conectado ao MetaTrader 5")

            # Obter informações da conta
            account_info = await client.get_account_info()
            print(f"📊 Conta: {account_info['login']}")
            print(f"💰 Saldo: {account_info['balance']}")
            print(f"🏢 Servidor: {account_info['server']}")

            return client
        else:
            print("❌ Falha na conexão")
            return None

    except MT5ConnectionError as e:
        print(f"❌ Erro de conexão: {e}")
        return None

# Uso
client = asyncio.run(connect_mt5())
```

### 2. Obtenção de Dados de Mercado

```python
async def get_market_data(client):
    """Obtém dados de mercado para XAUUSD"""

    # Obter símbolos disponíveis
    symbols = await client.get_symbols()
    print(f"📈 Símbolos disponíveis: {len(symbols)}")

    # Verificar se XAUUSD está disponível
    xauusd_info = await client.get_symbol_info("XAUUSD")
    if xauusd_info:
        print(f"💎 XAUUSD encontrado:")
        print(f"   - Spread: {xauusd_info['spread']} pontos")
        print(f"   - Lote Mín: {xauusd_info['volume_min']}")
        print(f"   - Lote Máx: {xauusd_info['volume_max']}")

    # Obter barras de preços
    bars = await client.get_bars("XAUUSD", "H1", 100)
    print(f"📊 Obtidas {len(bars)} barras de XAUUSD H1")

    # Obter ticks em tempo real
    ticks = await client.get_ticks("XAUUSD", 10)
    if ticks:
        last_tick = ticks[-1]
        print(f"🔄 Último tick: {last_tick['bid']}/{last_tick['ask']}")

    return bars, ticks

# Uso
if client:
    bars, ticks = asyncio.run(get_market_data(client))
```

### 3. Operações de Trading

```python
async def place_trade(client):
    """Executa ordem de compra/venda"""

    # Análise simples (substitua por sua estratégia)
    if len(bars) < 20:
        print("⚠️ Dados insuficientes para análise")
        return None

    # Calcular médias móveis simples
    closes = [bar['close'] for bar in bars[-20:]]
    ma_short = sum(closes[-10:]) / 10
    ma_long = sum(closes[-20:]) / 20

    current_price = bars[-1]['close']

    # Lógica de entrada simples
    if ma_short > ma_long and current_price > ma_short:
        # Sinal de compra
        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_BUY",
            "stop_loss": current_price - 50,  # 50 pips SL
            "take_profit": current_price + 100,  # 100 pips TP
            "magic_number": 12345,
            "comment": "EA Scalper Python"
        }

        try:
            result = await client.place_order(order_data)
            if result['success']:
                print(f"✅ Ordem de compra executada: Ticket {result['order_ticket']}")
                print(f"💰 Preço de execução: {result['execution_price']}")
                return result['order_ticket']
            else:
                print(f"❌ Falha na ordem: {result['message']}")
        except Exception as e:
            print(f"❌ Erro ao executar ordem: {e}")

    elif ma_short < ma_long and current_price < ma_short:
        # Sinal de venda
        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_SELL",
            "stop_loss": current_price + 50,
            "take_profit": current_price - 100,
            "magic_number": 12345,
            "comment": "EA Scalper Python"
        }

        try:
            result = await client.place_order(order_data)
            if result['success']:
                print(f"✅ Ordem de venda executada: Ticket {result['order_ticket']}")
                return result['order_ticket']
            else:
                print(f"❌ Falha na ordem: {result['message']}")
        except Exception as e:
            print(f"❌ Erro ao executar ordem: {e}")

    return None

# Uso
if client:
    ticket = asyncio.run(place_trade(client))
```

### 4. Gestão de Posições

```python
async def manage_positions(client):
    """Gerencia posições abertas"""

    # Obter posições abertas
    positions = await client.get_positions("XAUUSD")

    if not positions:
        print("📭 Nenhuma posição aberta")
        return

    print(f"📊 {len(positions)} posição(ões) aberta(s):")

    for position in positions:
        print(f"\n🎫 Ticket: {position['ticket']}")
        print(f"📈 Tipo: {position['type']}")
        print(f"💎 Volume: {position['volume']}")
        print(f"💰 Preço: {position['open_price']}")
        print(f"💲 Lucro: {position['profit']:.2f}")
        print(f"🛡️ SL: {position['stop_loss']}")
        print(f"🎯 TP: {position['take_profit']}")

        # Lógica de gestão de risco
        if position['profit'] < -100:  # Stop loss manual se perda > 100
            print(f"⚠️ Fechando posição por perda excessiva")
            close_result = await client.close_position(position['ticket'])
            if close_result['success']:
                print(f"✅ Posição {position['ticket']} fechada")

        elif position['profit'] > 50:  # Trailing stop
            new_sl = position['open_price'] + 20 if position['type'] == 'BUY' else position['open_price'] - 20
            modify_result = await client.modify_position(
                position['ticket'],
                stop_loss=new_sl
            )
            if modify_result['success']:
                print(f"📏 Stop loss ajustado para {new_sl}")

# Uso
if client:
    asyncio.run(manage_positions(client))
```

## Integração com LiteLLM

### 1. Configuração do Cliente LLM

```python
from ea_scalper_sdk import LLMClient
import asyncio

async def setup_llm():
    """Configura cliente LiteLLM"""

    client = LLMClient(base_url="http://localhost:4000")

    # Testar conexão
    models = await client.list_models()
    print(f"🤖 Modelos disponíveis: {[model['id'] for model in models['data']]}")

    return client

# Uso
llm_client = asyncio.run(setup_llm())
```

### 2. Análise de Mercado com IA

```python
async def ai_market_analysis(llm_client, market_data):
    """Usa IA para analisar dados de mercado"""

    # Preparar dados para análise
    price_data = {
        "current_price": market_data['current_price'],
        "trend": market_data['trend'],
        "support_levels": market_data['support'],
        "resistance_levels": market_data['resistance'],
        "indicators": market_data['indicators']
    }

    prompt = f"""
    Como especialista em trading de XAUUSD, analise os seguintes dados:

    Dados de Mercado:
    - Preço Atual: ${price_data['current_price']}
    - Tendência: {price_data['trend']}
    - Suportes: {price_data['support_levels']}
    - Resistências: {price_data['resistance_levels']}
    - Indicadores: {price_data['indicators']}

    Forneça:
    1. Análise técnica detalhada
    2. Sinal de trading (BUY/SELL/HOLD)
    3. Níveis de entrada, stop loss e take profit
    4. Nível de confiança (0-100%)
    5. Fatores de risco

    Responda em formato JSON.
    """

    try:
        response = await llm_client.chat_completion(
            model="deepseek-r1-free",
            messages=[
                {"role": "system", "content": "Você é um analista técnico especializado em trading de XAUUSD."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Extrair e parsear resposta JSON
        analysis_text = response['choices'][0]['message']['content']

        # Implementar parsing seguro do JSON
        import json
        try:
            analysis = json.loads(analysis_text)
            return analysis
        except json.JSONDecodeError:
            # Se não for JSON válido, extrair informações manualmente
            return {"raw_analysis": analysis_text}

    except Exception as e:
        print(f"❌ Erro na análise com IA: {e}")
        return None

# Uso
if llm_client and market_data:
    analysis = asyncio.run(ai_market_analysis(llm_client, market_data))
    if analysis:
        print("🤖 Análise de mercado gerada por IA:")
        print(f"📊 Sinal: {analysis.get('signal', 'N/A')}")
        print(f"🎯 Confiança: {analysis.get('confidence_level', 'N/A')}%")
```

### 3. Geração de Estratégias

```python
async def generate_trading_strategy(llm_client, historical_data):
    """Gera estratégia de trading baseada em dados históricos"""

    prompt = f"""
    Com base nos dados históricos de XAUUSD, crie uma estratégia de scalping:

    Período analisado: {historical_data['period']}
    Win rate histórico: {historical_data['win_rate']}%
    Fator de lucro: {historical_data['profit_factor']}
    Drawdown máximo: {historical_data['max_drawdown']}%

    Requisitos:
    1. Estratégia FTMO-compliant (máx 5% perda diária, 10% total)
    2. Risk/Reward mínimo de 1:1.5
    3. Máximo 2 trades simultâneos
    4. Horário de trading: 08:00-20:00 GMT

    Forneça:
    - Regras claras de entrada e saída
    - Parâmetros de gestão de risco
    - Indicadores recomendados
    - Configurações de otimização

    Formato: JSON com estrutura detalhada.
    """

    response = await llm_client.chat_completion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um quant especializado em desenvolvimento de EAs FTMO-compliant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )

    return response['choices'][0]['message']['content']
```

## Agentes Autônomos

### 1. Market Research Agent

```python
from ea_scalper_sdk import AgentClient

async def run_market_research():
    """Executa agente de pesquisa de mercado"""

    agent_client = AgentClient(base_url="http://localhost:8080")

    task_params = {
        "symbol": "XAUUSD",
        "timeframes": ["M5", "M15", "H1", "H4"],
        "analysis_period": "2024.01.01-2024.12.31",
        "indicators": ["RSI", "MACD", "EMA", "BB", "ATR"],
        "risk_parameters": {
            "max_risk_percent": 1.0,
            "max_drawdown": 5.0
        }
    }

    # Iniciar tarefa
    task_result = await agent_client.execute_task(
        agent_name="market_research",
        task_parameters=task_params
    )

    if task_result['success']:
        task_id = task_result['task_id']
        print(f"🔍 Pesquisa de mercado iniciada: {task_id}")

        # Monitorar progresso
        while True:
            status = await agent_client.get_task_status("market_research", task_id)
            print(f"📊 Progresso: {status['progress']}%")

            if status['status'] == 'completed':
                results = await agent_client.get_task_results("market_research", task_id)
                print("✅ Pesquisa concluída!")
                return results
            elif status['status'] == 'failed':
                print(f"❌ Pesquisa falhou: {status.get('error', 'Erro desconhecido')}")
                return None

            await asyncio.sleep(5)
    else:
        print(f"❌ Falha ao iniciar pesquisa: {task_result['message']}")
        return None

# Uso
research_results = asyncio.run(run_market_research())
```

### 2. Backtest Agent

```python
async def run_backtest(strategy_config):
    """Executa backtest automatizado"""

    agent_client = AgentClient(base_url="http://localhost:8080")

    backtest_params = {
        "expert_file": "EA_XAUUSD_Scalper_Generated.ex5",
        "symbol": "XAUUSD",
        "period": "2024.01.01-2024.12.31",
        "deposit": 10000,
        "leverage": 100,
        "optimization": True,
        "strategy_config": strategy_config
    }

    # Executar backtest
    task_result = await agent_client.execute_task(
        agent_name="backtest",
        task_parameters=backtest_params
    )

    if task_result['success']:
        task_id = task_result['task_id']

        # Aguardar conclusão (backtest pode demorar)
        while True:
            status = await agent_client.get_task_status("backtest", task_id)

            if status['status'] == 'completed':
                results = await agent_client.get_task_results("backtest", task_id)

                # Validar compliance FTMO
                if results['ftmo_compliance']['overall_compliance']:
                    print("✅ Backtest aprovado para FTMO!")
                else:
                    print("⚠️ Estratégia não é FTMO-compliant")

                return results
            elif status['status'] == 'failed':
                print(f"❌ Backtest falhou: {status.get('error')}")
                return None

            print(f"⏳ Backtest em andamento... {status.get('progress', 0)}%")
            await asyncio.sleep(30)  # Verificar a cada 30 segundos
```

## Sistema Completo de Trading

### 1. Main Trading Bot

```python
import asyncio
import logging
from datetime import datetime, time as dt_time
from ea_scalper_sdk import MT5Client, LLMClient, AgentClient

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TradingBot")

class TradingBot:
    """Bot de trading completo para XAUUSD"""

    def __init__(self):
        self.mt5_client = None
        self.llm_client = None
        self.agent_client = None
        self.is_running = False
        self.trading_hours = (dt_time(8, 0), dt_time(20, 0))  # GMT

    async def initialize(self):
        """Inicializa todos os clientes"""
        try:
            # Conectar MT5
            self.mt5_client = MT5Client()
            mt5_connected = await self.mt5_client.connect_from_env()

            if not mt5_connected:
                raise Exception("Falha na conexão MT5")

            # Conectar LiteLLM
            self.llm_client = LLMClient()

            # Conectar Agent Management
            self.agent_client = AgentClient()

            logger.info("✅ Bot inicializado com sucesso")
            return True

        except Exception as e:
            logger.error(f"❌ Falha na inicialização: {e}")
            return False

    async def check_trading_conditions(self):
        """Verifica se as condições de trading são favoráveis"""

        # Verificar horário de trading
        current_time = datetime.now().time()
        if not (self.trading_hours[0] <= current_time <= self.trading_hours[1]):
            return False, "Fora do horário de trading"

        # Verificar spread
        symbol_info = await self.mt5_client.get_symbol_info("XAUUSD")
        if symbol_info and symbol_info['spread'] > 30:
            return False, f"Spread muito alto: {symbol_info['spread']}"

        # Verificar volatilidade
        bars = await self.mt5_client.get_bars("XAUUSD", "M15", 20)
        if bars:
            # Calcular ATR simples
            ranges = [bar['high'] - bar['low'] for bar in bars]
            avg_range = sum(ranges) / len(ranges)

            if avg_range < 100:  # Menos de 10 pips de volatilidade
                return False, "Baixa volatilidade"

        return True, "Condições favoráveis"

    async def analyze_market(self):
        """Análise completa do mercado"""

        try:
            # Obter dados de múltiplos timeframes
            timeframes = ["M5", "M15", "H1", "H4"]
            market_data = {}

            for tf in timeframes:
                bars = await self.mt5_client.get_bars("XAUUSD", tf, 100)
                market_data[tf] = bars

            # Análise técnica básica
            h1_bars = market_data["H1"]
            if len(h1_bars) >= 20:
                closes = [bar['close'] for bar in h1_bars[-20:]]
                ma20 = sum(closes) / 20
                ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

                current_price = h1_bars[-1]['close']

                trend = "bullish" if current_price > ma20 else "bearish"
                strength = abs(current_price - ma20) / ma20 * 100
            else:
                trend = "unknown"
                strength = 0

            # Usar IA para análise avançada
            ai_analysis = await self.ai_enhanced_analysis(market_data)

            return {
                "trend": trend,
                "strength": strength,
                "current_price": current_price,
                "ai_analysis": ai_analysis,
                "time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Erro na análise de mercado: {e}")
            return None

    async def ai_enhanced_analysis(self, market_data):
        """Análise aprimorada com IA"""

        try:
            # Preparar resumo dos dados
            summary = {
                "h1_trend": self.calculate_trend(market_data["H1"]),
                "h4_trend": self.calculate_trend(market_data["H4"]),
                "volatility": self.calculate_volatility(market_data["M15"]),
                "key_levels": self.find_key_levels(market_data["H1"])
            }

            # Gerar prompt para IA
            prompt = f"""
            Analise o XAUUSD com os seguintes dados:

            Tendência H1: {summary['h1_trend']}
            Tendência H4: {summary['h4_trend']}
            Volatilidade: {summary['volatility']:.2f}
            Níveis chave: {summary['key_levels']}

            Forneça:
            1. Sinal de trading (BUY/SELL/HOLD)
            2. Nível de confiança (0-100)
            3. Preço de entrada sugerido
            4. Stop loss e take profit
            5. Risco/Reward ratio

            Resposta em JSON.
            """

            response = await self.llm_client.chat_completion(
                model="deepseek-r1-free",
                messages=[
                    {"role": "system", "content": "Você é um analista técnico de XAUUSD."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            import json
            try:
                return json.loads(response['choices'][0]['message']['content'])
            except:
                return {"raw": response['choices'][0]['message']['content']}

        except Exception as e:
            logger.error(f"❌ Erro na análise com IA: {e}")
            return None

    async def execute_trade(self, signal):
        """Executa ordem baseada no sinal"""

        try:
            if signal['signal'] == 'HOLD' or signal['confidence'] < 70:
                logger.info("⏸️ Sinal HOLD ou baixa confiança, aguardando...")
                return False

            # Calcular tamanho da posição (1% do capital)
            account_info = await self.mt5_client.get_account_info()
            risk_amount = account_info['balance'] * 0.01

            # Obter valor do pip para XAUUSD
            symbol_info = await self.mt5_client.get_symbol_info("XAUUSD")
            pip_value = symbol_info.get('trade_tick_value', 10) if symbol_info else 10

            # Calcular stop loss em pips
            current_price = signal['entry_price']
            sl_price = signal['stop_loss']
            sl_pips = abs(current_price - sl_price) * 100

            # Calcular tamanho da posição
            position_size = risk_amount / (sl_pips * pip_value)
            position_size = max(0.01, min(position_size, 1.0))  # Limitar entre 0.01 e 1.0

            # Arredondar para 2 casas decimais
            position_size = round(position_size, 2)

            # Preparar ordem
            order_type = "MARKET_BUY" if signal['signal'] == 'BUY' else "MARKET_SELL"

            order_data = {
                "symbol": "XAUUSD",
                "volume": position_size,
                "order_type": order_type,
                "stop_loss": signal['stop_loss'],
                "take_profit": signal['take_profit'],
                "magic_number": 12345,
                "comment": "AI Trading Bot"
            }

            # Executar ordem
            result = await self.mt5_client.place_order(order_data)

            if result['success']:
                logger.info(f"✅ Ordem executada: {signal['signal']} {position_size} lotes")
                logger.info(f"💰 Entry: {result['execution_price']}")
                logger.info(f"🛡️ SL: {signal['stop_loss']}")
                logger.info(f"🎯 TP: {signal['take_profit']}")
                return True
            else:
                logger.error(f"❌ Falha na ordem: {result['message']}")
                return False

        except Exception as e:
            logger.error(f"❌ Erro na execução da ordem: {e}")
            return False

    async def manage_risk(self):
        """Gestão de risco em tempo real"""

        try:
            positions = await self.mt5_client.get_positions("XAUUSD")

            if not positions:
                return

            total_risk = 0
            account_info = await self.mt5_client.get_account_info()
            max_risk = account_info['balance'] * 0.05  # 5% risco máximo

            for position in positions:
                risk = position['volume'] * abs(position['open_price'] - position['stop_loss']) * 100
                total_risk += risk

                # Fechar posição se perda > 2%
                if position['profit'] < -account_info['balance'] * 0.02:
                    logger.warning(f"⚠️ Fechando posição {position['ticket']} por stop loss manual")
                    await self.mt5_client.close_position(position['ticket'])

                # Trailing stop
                elif position['profit'] > 50:
                    new_sl = position['open_price'] + 20 if position['type'] == 'BUY' else position['open_price'] - 20
                    if new_sl > position['stop_loss'] if position['type'] == 'BUY' else new_sl < position['stop_loss']:
                        await self.mt5_client.modify_position(position['ticket'], stop_loss=new_sl)
                        logger.info(f"📏 Trailing stop ajustado para {new_sl}")

            # Verificar risco total
            if total_risk > max_risk:
                logger.warning(f"⚠️ Risco total ({total_risk:.2f}) excede o máximo ({max_risk:.2f})")

        except Exception as e:
            logger.error(f"❌ Erro na gestão de risco: {e}")

    async def run(self):
        """Loop principal do bot"""

        self.is_running = True
        logger.info("🚀 Bot de trading iniciado")

        while self.is_running:
            try:
                # Verificar condições de trading
                can_trade, reason = await self.check_trading_conditions()

                if can_trade:
                    # Análise de mercado
                    analysis = await self.analyze_market()

                    if analysis and analysis['ai_analysis']:
                        # Gerar sinal de trading
                        signal = analysis['ai_analysis']

                        logger.info(f"📊 Sinal gerado: {signal.get('signal', 'N/A')}")
                        logger.info(f"🎯 Confiança: {signal.get('confidence', 0)}%")

                        # Executar trade
                        await self.execute_trade(signal)
                else:
                    logger.info(f"⏸️ {reason}")

                # Gestão de risco
                await self.manage_risk()

                # Aguardar próximo ciclo
                await asyncio.sleep(60)  # Verificar a cada minuto

            except Exception as e:
                logger.error(f"❌ Erro no loop principal: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Para o bot"""
        self.is_running = False
        logger.info("🛑 Bot de trading parado")

    # Métodos utilitários
    def calculate_trend(self, bars):
        """Calcula tendência baseada em médias móveis"""
        if len(bars) < 20:
            return "unknown"

        closes = [bar['close'] for bar in bars[-20:]]
        ma20 = sum(closes) / 20

        if len(closes) >= 50:
            ma50 = sum(closes[-50:]) / 50
            current = closes[-1]

            if current > ma20 > ma50:
                return "strong_bullish"
            elif current > ma20 and ma20 < ma50:
                return "bullish"
            elif current < ma20 < ma50:
                return "strong_bearish"
            elif current < ma20 and ma20 > ma50:
                return "bearish"

        return "neutral" if closes[-1] > ma20 else "bearish"

    def calculate_volatility(self, bars):
        """Calcula volatilidade média"""
        if len(bars) < 10:
            return 0

        ranges = [bar['high'] - bar['low'] for bar in bars[-20:]]
        return sum(ranges) / len(ranges) * 100  # Converter para pips

    def find_key_levels(self, bars):
        """Encontra níveis de suporte e resistência"""
        if len(bars) < 50:
            return {"support": [], "resistance": []}

        highs = [bar['high'] for bar in bars]
        lows = [bar['low'] for bar in bars]

        # Encontrar topos e fundos significativos
        resistance_levels = sorted(set([round(h, 2) for h in highs if highs.count(h) >= 2]), reverse=True)[:3]
        support_levels = sorted(set([round(l, 2) for l in lows if lows.count(l) >= 2]))[:3]

        return {
            "support": support_levels,
            "resistance": resistance_levels
        }

# Função principal
async def main():
    """Função principal de execução"""

    bot = TradingBot()

    # Inicializar bot
    if await bot.initialize():
        try:
            # Executar bot
            await bot.run()
        except KeyboardInterrupt:
            logger.info("🛑 Interrupção pelo usuário")
            bot.stop()
        finally:
            # Limpeza
            if bot.mt5_client:
                await bot.mt5_client.disconnect()
    else:
        logger.error("❌ Falha na inicialização do bot")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Teste do Sistema

```python
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

class TestTradingBot(unittest.TestCase):
    """Testes unitários para o Trading Bot"""

    def setUp(self):
        self.bot = TradingBot()

    async def test_initialization(self):
        """Testa inicialização do bot"""
        with patch.object(self.bot.mt5_client, 'connect_from_env', return_value=True):
            result = await self.bot.initialize()
            self.assertTrue(result)

    def test_calculate_trend(self):
        """Testa cálculo de tendência"""
        # Criar dados de teste
        bars = [
            {"close": 2320 + i} for i in range(50)
        ]

        trend = self.bot.calculate_trend(bars)
        self.assertIn(trend, ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"])

    def test_calculate_volatility(self):
        """Testa cálculo de volatilidade"""
        bars = [
            {"high": 2320 + i, "low": 2318 + i} for i in range(20)
        ]

        volatility = self.bot.calculate_volatility(bars)
        self.assertGreater(volatility, 0)

    def test_find_key_levels(self):
        """Testa identificação de níveis chave"""
        bars = [
            {"high": 2325, "low": 2320},
            {"high": 2325, "low": 2320},
            {"high": 2330, "low": 2315},
            {"high": 2330, "low": 2315}
        ]

        levels = self.bot.find_key_levels(bars)
        self.assertIn("support", levels)
        self.assertIn("resistance", levels)

# Executar testes
if __name__ == "__main__":
    unittest.main()
```

## Deploy e Produção

### 1. Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    wine \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Instalar MetaTrader 5 (via Wine)
RUN wget -O mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" \
    && wine mt5setup.exe /quiet

# Expor portas
EXPOSE 8000 4000 8080

# Comando de inicialização
CMD ["python", "main.py"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    ports:
      - "8000:8000"
      - "4000:4000"
      - "8080:8080"
    environment:
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
      - MT5_SERVER=${MT5_SERVER}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  monitoring:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### 3. Script de Deploy

```bash
#!/bin/bash
# deploy.sh

echo "🚀 Deploy do EA_SCALPER_XAUUSD"

# Verificar variáveis de ambiente
if [ -z "$MT5_LOGIN" ] || [ -z "$MT5_PASSWORD" ]; then
    echo "❌ Variáveis de ambiente obrigatórias não configuradas"
    exit 1
fi

# Build e deploy
docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "✅ Deploy concluído"
echo "📊 Monitoramento: http://localhost:3000"
echo "🤖 MT5 MCP: http://localhost:8000"
echo "🧠 LiteLLM: http://localhost:4000"
echo "🤖 Agent API: http://localhost:8080"
```

## Monitoramento e Logging

### 1. Configuração Avançada de Logging

```python
import logging
import logging.handlers
from datetime import datetime

def setup_logging():
    """Configura logging avançado"""

    # Criar logger principal
    logger = logging.getLogger("EA_Scalper")
    logger.setLevel(logging.DEBUG)

    # Formatter personalizado
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handler para arquivo com rotação
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/trading_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para erros críticos
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
```

### 2. Sistema de Métricas

```python
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TradingMetrics:
    """Métricas de trading"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    def update(self, trade_result):
        """Atualiza métricas com resultado de trade"""
        self.total_trades += 1

        if trade_result['profit'] > 0:
            self.winning_trades += 1
            self.total_profit += trade_result['profit']
        else:
            self.losing_trades += 1
            self.total_loss += abs(trade_result['profit'])

        # Calcular win rate
        self.win_rate = (self.winning_trades / self.total_trades) * 100

        # Calcular profit factor
        if self.total_loss > 0:
            self.profit_factor = self.total_profit / self.total_loss

        # Atualizar drawdown
        if trade_result['profit'] < 0:
            self.current_drawdown += abs(trade_result['profit'])
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = max(0, self.current_drawdown - trade_result['profit'])

class MetricsCollector:
    """Coletor de métricas em tempo real"""

    def __init__(self):
        self.metrics = TradingMetrics()
        self.trade_history = deque(maxlen=1000)
        self.performance_data = defaultdict(list)
        self.start_time = time.time()

    def record_trade(self, trade_result):
        """Registra resultado de trade"""
        self.metrics.update(trade_result)
        self.trade_history.append(trade_result)

        # Registra timestamp
        self.performance_data['timestamps'].append(time.time() - self.start_time)
        self.performance_data['balance'].append(trade_result['balance'])
        self.performance_data['equity'].append(trade_result['equity'])

    def get_summary(self):
        """Retorna resumo das métricas"""
        uptime = time.time() - self.start_time
        trades_per_hour = (self.metrics.total_trades / uptime) * 3600 if uptime > 0 else 0

        return {
            "uptime_hours": uptime / 3600,
            "total_trades": self.metrics.total_trades,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "total_profit": self.metrics.total_profit - self.metrics.total_loss,
            "max_drawdown": self.metrics.max_drawdown,
            "trades_per_hour": trades_per_hour
        }
```

## Melhores Práticas

### 1. Gerenciamento de Erros

```python
import functools
import asyncio
from typing import Callable, Any

def retry_on_failure(max_retries=3, delay=1):
    """Decorator para retry de funções que falham"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

# Uso
@retry_on_failure(max_retries=3, delay=2)
async def place_order_with_retry(client, order_data):
    """Executa ordem com retry automático"""
    return await client.place_order(order_data)
```

### 2. Validação de Dados

```python
from pydantic import BaseModel, validator
from typing import Optional

class OrderRequest(BaseModel):
    """Modelo validado para requisição de ordem"""

    symbol: str
    volume: float
    order_type: str
    price: Optional[float] = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic_number: int
    comment: str = ""

    @validator('symbol')
    def validate_symbol(cls, v):
        if v != "XAUUSD":
            raise ValueError("Apenas XAUUSD é suportado")
        return v

    @validator('volume')
    def validate_volume(cls, v):
        if not 0.01 <= v <= 1.0:
            raise ValueError("Volume deve estar entre 0.01 e 1.0")
        return v

    @validator('order_type')
    def validate_order_type(cls, v):
        allowed = ["MARKET_BUY", "MARKET_SELL", "LIMIT_BUY", "LIMIT_SELL"]
        if v not in allowed:
            raise ValueError(f"Tipo de ordem deve ser um de: {allowed}")
        return v
```

### 3. Configuração por Ambiente

```python
from enum import Enum
from pydantic import BaseSettings

class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class Settings(BaseSettings):
    """Configurações baseadas no ambiente"""

    environment: Environment = Environment.DEVELOPMENT

    # MetaTrader 5
    mt5_login: int
    mt5_password: str
    mt5_server: str

    # APIs
    mt5_mcp_url: str = "http://localhost:8000"
    litellm_url: str = "http://localhost:4000"
    agent_url: str = "http://localhost:8080"

    # Segurança
    api_secret_key: str
    jwt_secret_key: str

    # Trading
    max_risk_percent: float = 1.0
    max_positions: int = 2
    trading_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"

    class Config:
        env_file = ".env"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

# Uso
settings = Settings()
logger = setup_logging()
logger.setLevel(getattr(logging, settings.log_level))
```

## Resumo

Este guia cobre:

✅ **Configuração completa** do ambiente Python
✅ **SDK básico** para interação com MT5, LiteLLM e Agentes
✅ **Sistema completo de trading** com IA
✅ **Deploy e produção** com Docker
✅ **Monitoramento e logging** avançado
✅ **Melhores práticas** de desenvolvimento

Com este guia, você pode:
- Integrar suas aplicações Python com o EA_SCALPER_XAUUSD
- Desenvolver estratégias de trading automatizadas
- Usar IA para análise e tomada de decisões
- Implementar gestão de risco robusta
- Deploy em ambiente de produção

Para exemplos práticos adicionais, consulte o diretório `/docs/examples/`.