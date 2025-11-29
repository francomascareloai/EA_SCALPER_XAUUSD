# Tutorial 01: Getting Started com EA_SCALPER_XAUUSD
================================================

## Overview

Este tutorial passo a passo vai guiá-lo através da configuração inicial e primeiro uso do sistema EA_SCALPER_XAUUSD. Ao final, você terá um ambiente funcional pronto para trading automatizado.

## Pré-requisitos

### Hardware Mínimo
- **Processador**: Intel i5 ou AMD Ryzen 5 (ou superior)
- **RAM**: 8GB (16GB recomendado)
- **Armazenamento**: 50GB livres
- **Internet**: Conexão estável

### Software Necessário
- **Windows 10/11** (obrigatório para MetaTrader 5)
- **Python 3.8+**
- **MetaTrader 5** instalado
- **Conta RoboForex** (Demo ou Live)

### Conta Broker
1. Abra conta **RoboForex** (Demo recomendado para início)
2. Configure a plataforma MT5 com RoboForex-Demo
3. Verifique se XAUUSD está disponível

## Passo 1: Instalação do Ambiente

### 1.1 Instalar Python

```bash
# Verificar se Python está instalado
python --version

# Se não estiver, instale a partir de:
# https://www.python.org/downloads/
# Durante instalação, MARQUE "Add Python to PATH"
```

### 1.2 Configurar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv ea_scalper_env

# Ativar ambiente (Windows)
ea_scalper_env\Scripts\activate

# Ativar ambiente (Linux/Mac)
source ea_scalper_env/bin/activate
```

### 1.3 Instalar Dependências

```bash
# Clonar repositório (se aplicável)
git clone https://github.com/your-org/ea-scalper-xauusd.git
cd ea-scalper-xauusd

# Instalar dependências
pip install -r requirements.txt

# Ou instalar individualmente
pip install requests aiohttp websockets pandas numpy
pip install MetaTrader5 python-dotenv pydantic
pip install fastapi uvicorn python-multipart
```

## Passo 2: Configurar MetaTrader 5

### 2.1 Instalar MetaTrader 5

1. Baixe MT5 do site RoboForex
2. Instale com configurações padrão
3. Inicie o MT5 e faça login com suas credenciais RoboForex

### 2.2 Configurar Terminal

1. **Adicionar XAUUSD**:
   - Clique com botão direito em "Market Watch"
   - Selecione "Symbols"
   - Encontre e adicione XAUUSD

2. **Configurar Timeframes**:
   - Abra gráfico XAUUSD
   - Adicione timeframes M1, M5, M15, H1, H4

3. **Verificar Conexão**:
   - Certifique-se que status é "Connected"
   - Verifique se há ticks chegando

## Passo 3: Configurar Projeto

### 3.1 Estrutura de Arquivos

Crie a seguinte estrutura:

```
ea-scalper-xauusd/
├── .env                    # Configurações sensíveis
├── config/
│   ├── mt5_config.json    # Config MT5
│   └── trading_config.json # Config trading
├── logs/                  # Logs do sistema
├── data/                  # Dados de mercado
├── docs/
│   ├── api-reference/
│   └── examples/
├── src/
│   └── trading_bot.py     # Seu bot principal
└── tests/                 # Testes unitários
```

### 3.2 Configurar Variáveis de Ambiente

Crie arquivo `.env` na raiz:

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
API_SECRET_KEY=sua_chave_secreta_aqui
JWT_SECRET_KEY=seu_jwt_secreto_aqui

# Trading Settings
RISK_PERCENT=1.0
MAX_POSITIONS=2
TRADING_ENABLED=true
LOG_LEVEL=INFO
```

### 3.3 Obter Chaves de API

#### OpenRouter (Recomendado)

1. Acesse https://openrouter.ai
2. Crie conta gratuita
3. Vá para Settings → API Keys
4. Copie sua chave
5. Adicione ao `.env`

#### Alternativas:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/

## Passo 4: Testar Conexão Básica

### 4.1 Executar Script de Teste

Crie arquivo `test_connection.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client

async def test_connection():
    """Testa conexão básica com MT5"""

    print("🚀 Testando Conexão EA_SCALPER_XAUUSD")
    print("=" * 50)

    # Carregar configuração
    load_dotenv()

    # Validar variáveis
    required = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing = [var for var in required if not os.getenv(var)]

    if missing:
        print(f"❌ Configure no .env: {missing}")
        return False

    try:
        # Conectar MT5
        client = MT5Client()

        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        print(f"🔌 Conectando ao servidor {server}...")
        success = await client.connect(login, password, server)

        if not success:
            print("❌ Falha na conexão")
            return False

        print("✅ Conectado com sucesso!")

        # Obter informações da conta
        account = await client.get_account_info()
        print(f"📊 Conta: {account['login']}")
        print(f"💰 Saldo: ${account['balance']:.2f}")
        print(f"🏢 Servidor: {account['server']}")

        # Verificar XAUUSD
        symbol = await client.get_symbol_info("XAUUSD")
        if symbol:
            print(f"✅ XAUUSD disponível - Spread: {symbol['spread']} pontos")
        else:
            print("❌ XAUUSD não encontrado")

        # Desconectar
        await client.disconnect()
        print("🔌 Desconectado")

        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    print("⚠️ Certifique-se de que o MT5 está aberto!")
    result = asyncio.run(test_connection())

    if result:
        print("\n🎉 Teste concluído com sucesso!")
        print("Seu ambiente está pronto para trading.")
    else:
        print("\n❌ Teste falhou!")
        print("Verifique a configuração e tente novamente.")
```

### 4.2 Executar Teste

```bash
# Ativar ambiente virtual
ea_scalper_env\Scripts\activate

# Executar teste
python test_connection.py
```

**Saída esperada:**
```
🚀 Testando Conexão EA_SCALPER_XAUUSD
==================================================
🔌 Conectando ao servidor RoboForex-Demo...
✅ Conectado com sucesso!
📊 Conta: 12345678
💰 Saldo: $10000.00
🏢 Servidor: RoboForex-Demo
✅ XAUUSD disponível - Spread: 15 pontos
🔌 Desconectado

🎉 Teste concluído com sucesso!
Seu ambiente está pronto para trading.
```

## Passo 5: Configurar LiteLLM Proxy

### 5.1 Criar Configuração LiteLLM

Crie arquivo `litellm_config.yaml`:

```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "openai/gpt-4"
      api_key: os.environ/OPENAI_API_KEY

  - model_name: "claude-3-opus"
    litellm_params:
      model: "anthropic/claude-3-opus-20240229"
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: "deepseek-r1-free"
    litellm_params:
      model: "openrouter/deepseek/deepseek-r1-free"
      api_key: os.environ/OPENROUTER_API_KEY

litellm_settings:
  drop_params: true  # Ignora parâmetros não suportados
  set_verbose: true
  success_callback: ["langfuse"]

general_settings:
  master_key: os.environ/API_SECRET_KEY
  database_url: "postgresql://user:password@localhost:5432/litellm"
```

### 5.2 Iniciar LiteLLM Proxy

```bash
# Instalar LiteLLM com proxy
pip install 'litellm[proxy]'

# Iniciar proxy
litellm --config litellm_config.yaml --port 4000 --host 0.0.0.0
```

### 5.3 Testar LiteLLM

Crie arquivo `test_litellm.py`:

```python
import requests
import json

def test_litellm():
    """Testa conexão com LiteLLM proxy"""

    url = "http://localhost:4000/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer any_string"
    }

    data = {
        "model": "deepseek-r1-free",
        "messages": [
            {
                "role": "user",
                "content": "Olá, qual o preço atual do ouro?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ LiteLLM proxy funcionando!")
            print(f"🤖 Resposta: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Erro: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Erro de conexão: {e}")
        return False

if __name__ == "__main__":
    test_litellm()
```

## Passo 6: Executar Primeiro Bot

### 6.1 Bot Simples de Demonstração

Crie arquivo `first_bot.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirstBot")

class FirstBot:
    """Primeiro bot de trading"""

    def __init__(self):
        self.mt5_client = None
        self.symbol = "XAUUSD"
        self.magic_number = 99999

    async def initialize(self):
        """Inicializa o bot"""
        try:
            load_dotenv()

            self.mt5_client = MT5Client()

            login = int(os.getenv('MT5_LOGIN'))
            password = os.getenv('MT5_PASSWORD')
            server = os.getenv('MT5_SERVER')

            success = await self.mt5_client.connect(login, password, server)

            if success:
                logger.info("✅ Bot inicializado com sucesso")
                return True
            else:
                logger.error("❌ Falha na inicialização")
                return False

        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False

    async def analyze_market(self):
        """Análise simples de mercado"""
        try:
            # Obter últimas 20 barras H1
            bars = await self.mt5_client.get_bars(self.symbol, "H1", 20)

            if not bars:
                return None

            # Calcular médias móveis simples
            closes = [bar['close'] for bar in bars]
            ma_short = sum(closes[-10:]) / 10
            ma_long = sum(closes[-20:]) / 20

            current_price = bars[-1]['close']

            # Determinar tendência
            if ma_short > ma_long:
                trend = "bullish"
                signal = "BUY" if current_price > ma_short else "WAIT"
            else:
                trend = "bearish"
                signal = "SELL" if current_price < ma_short else "WAIT"

            return {
                "trend": trend,
                "signal": signal,
                "price": current_price,
                "ma_short": ma_short,
                "ma_long": ma_long
            }

        except Exception as e:
            logger.error(f"❌ Erro na análise: {e}")
            return None

    async def place_demo_trade(self, signal):
        """Coloca ordem demonstrativa (simulada)"""

        if signal["signal"] == "WAIT":
            logger.info("⏸️ Nenhum sinal de trading")
            return

        logger.info(f"📊 Sinal: {signal['signal']}")
        logger.info(f"💰 Preço: ${signal['price']:.2f}")
        logger.info(f"📈 Tendência: {signal['trend']}")

        # SIMULAÇÃO - NÃO EXECUTA ORDEM REAL
        logger.info("📝 [SIMULAÇÃO] Ordem seria executada:")
        logger.info(f"   Tipo: {signal['signal']}")
        logger.info(f"   Volume: 0.01 lotes")
        logger.info(f"   SL: ${signal['price'] - 50 if signal['signal'] == 'BUY' else signal['price'] + 50:.2f}")
        logger.info(f"   TP: ${signal['price'] + 100 if signal['signal'] == 'BUY' else signal['price'] - 100:.2f}")

    async def run(self, cycles=5):
        """Executa o bot por alguns ciclos"""

        logger.info("🚀 Iniciando First Bot")

        for i in range(cycles):
            try:
                logger.info(f"\n--- Ciclo {i+1}/{cycles} ---")

                # Análise de mercado
                analysis = await self.analyze_market()

                if analysis:
                    await self.place_demo_trade(analysis)
                else:
                    logger.warning("⚠️ Falha na análise de mercado")

                # Aguardar próximo ciclo
                if i < cycles - 1:
                    logger.info("⏰ Aguardando 60 segundos...")
                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"❌ Erro no ciclo {i+1}: {e}")

        logger.info("✅ Bot concluído com sucesso!")

    async def cleanup(self):
        """Limpeza final"""
        if self.mt5_client:
            await self.mt5_client.disconnect()
            logger.info("🔌 Desconectado do MT5")

async def main():
    """Função principal"""
    print("🤖 First Bot - EA_SCALPER_XAUUSD")
    print("=" * 50)
    print("⚠️ MODO DEMONSTRAÇÃO - Nenhuma ordem real será executada")
    print()

    bot = FirstBot()

    # Inicializar
    if not await bot.initialize():
        print("❌ Falha na inicialização do bot")
        return

    try:
        # Executar bot por 5 ciclos
        await bot.run(cycles=5)
    finally:
        # Limpeza
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 Executar Primeiro Bot

```bash
# Garantir que MT5 está aberto
# Garantir que LiteLLM proxy está rodando (se for usar IA)

# Executar bot
python first_bot.py
```

**Saída esperada:**
```
🤖 First Bot - EA_SCALPER_XAUUSD
==================================================
⚠️ MODO DEMONSTRAÇÃO - Nenhuma ordem real será executada

✅ Bot inicializado com sucesso
🚀 Iniciando First Bot

--- Ciclo 1/5 ---
📊 Sinal: BUY
💰 Preço: $2325.45
📈 Tendência: bullish
📝 [SIMULAÇÃO] Ordem seria executada:
   Tipo: BUY
   Volume: 0.01 lotes
   SL: $2275.45
   TP: $2425.45
⏰ Aguardando 60 segundos...

✅ Bot concluído com sucesso!
```

## Passo 7: Verificação Final

### 7.1 Checklist de Configuração

Verifique se todos os itens estão configurados:

- [ ] **Python 3.8+** instalado
- [ ] **Ambiente virtual** criado e ativado
- [ ] **Dependências** instaladas
- [ ] **MetaTrader 5** instalado e funcionando
- [ ] **Conta RoboForex** configurada
- [ ] **XAUUSD** disponível no MT5
- [ ] **Arquivo .env** configurado
- [ ] **Chaves de API** obtidas
- [ ] **Conexão MT5** testada com sucesso
- [ ] **LiteLLM proxy** funcionando (opcional)
- [ ] **Primeiro bot** executando

### 7.2 Teste de Saúde do Sistema

Crie arquivo `health_check.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client, LLMClient

async def health_check():
    """Verificação completa do sistema"""

    print("🏥 HEALTH CHECK - EA_SCALPER_XAUUSD")
    print("=" * 50)

    load_dotenv()

    checks = []

    # 1. Variáveis de ambiente
    print("\n1️⃣ Verificando configuração...")
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER', 'OPENROUTER_API_KEY']
    env_ok = all(os.getenv(var) for var in required_vars)
    checks.append(("Variáveis de Ambiente", env_ok))
    print(f"   {'✅' if env_ok else '❌'} Variáveis de ambiente")

    # 2. Conexão MT5
    print("\n2️⃣ Testando conexão MT5...")
    try:
        client = MT5Client()
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        success = await client.connect(login, password, server)
        checks.append(("Conexão MT5", success))
        print(f"   {'✅' if success else '❌'} Conexão MT5")

        if success:
            # Verificar símbolo
            symbol = await client.get_symbol_info("XAUUSD")
            symbol_ok = symbol is not None
            checks.append(("XAUUSD Disponível", symbol_ok))
            print(f"   {'✅' if symbol_ok else '❌'} XAUUSD disponível")

            await client.disconnect()
    except Exception as e:
        checks.append(("Conexão MT5", False))
        print(f"   ❌ Erro: {e}")

    # 3. LiteLLM (opcional)
    print("\n3️⃣ Testando LiteLLM...")
    try:
        llm = LLMClient()
        models = await llm.list_models()
        llm_ok = 'data' in models and len(models['data']) > 0
        checks.append(("LiteLLM Proxy", llm_ok))
        print(f"   {'✅' if llm_ok else '❌'} LiteLLM proxy")

        if llm_ok:
            print(f"   📊 {len(models['data'])} modelos disponíveis")
    except Exception as e:
        checks.append(("LiteLLM Proxy", False))
        print(f"   ❌ Erro: {e}")

    # Resumo
    print("\n📋 RESUMO:")
    print("-" * 30)

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for name, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{name}: {status}")

    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")

    if passed == total:
        print("🎉 SISTEMA 100% FUNCIONAL!")
        print("✅ Você está pronto para trading automatizado!")
    elif passed >= total * 0.8:
        print("⚠️ SISTEMA QUASE FUNCIONAL")
        print("💡 Resolva os itens restantes para operação completa")
    else:
        print("❌ SISTEMA PRECISA DE CONFIGURAÇÃO")
        print("🔧 Revise a configuração antes de prosseguir")

    return passed == total

if __name__ == "__main__":
    print("Executando verificação completa do sistema...\n")
    result = asyncio.run(health_check())
```

Execute a verificação:
```bash
python health_check.py
```

## Passo 8: Próximos Passos

### 8.1 Personalizar Configuração

Edite o arquivo `config/trading_config.json`:

```json
{
  "trading": {
    "enabled": true,
    "risk_percent": 1.0,
    "max_positions": 2,
    "trading_hours": {
      "start": "08:00",
      "end": "20:00",
      "timezone": "GMT"
    }
  },
  "risk_management": {
    "max_daily_loss": 5.0,
    "max_total_loss": 10.0,
    "min_risk_reward": 1.5
  },
  "symbols": {
    "primary": "XAUUSD",
    "fallback": ["XAUUSD_TDS"]
  }
}
```

### 8.2 Explorar Exemplos

Navegue pelos exemplos disponíveis:
- `docs/examples/01-basic-mt5-connection.py` - Conexão básica
- `docs/examples/02-simple-trading-bot.py` - Bot simples
- `docs/examples/03-ai-enhanced-trading.py` - Bot com IA
- `docs/examples/04-backtesting-system.py` - Sistema de backtest

### 8.3 Estudar Documentação

Leia a documentação completa:
- `docs/api-reference/complete-api-reference.md` - Referência da API
- `docs/api-reference/python-integration-guide.md` - Guia Python
- `docs/examples/` - Exemplos práticos

## Troubleshooting Comum

### Problema: Conexão MT5 Falha
**Solução:**
1. Verifique se MT5 está aberto
2. Confirme credenciais no .env
3. Verifique se está no servidor correto
4. Tente reiniciar o MT5

### Problema: XAUUSD Não Encontrado
**Solução:**
1. Adicione XAUUSD ao Market Watch
2. Tente XAUUSD_TDS (sufixo RoboForex)
3. Verifique se conta suporta o símbolo

### Problema: LiteLLM Não Responde
**Solução:**
1. Verifique se proxy está rodando na porta 4000
2. Confirme chave de API OpenRouter
3. Teste com modelo gratuito primeiro

### Problema: Permissão Negada
**Solução:**
1. Execute como administrador
2. Verifique firewall/antivírus
3. Confirme permissões do MT5

## Recursos Adicionais

### Comunidade e Suporte
- **GitHub Issues**: Reporte problemas e sugestões
- **Discord**: Chat em tempo real com outros usuários
- **Documentação**: Guia completo de referência

### Ferramentas Úteis
- **MT5 Terminal**: Para monitoramento manual
- **VS Code**: Para desenvolvimento Python
- **Postman**: Para testar APIs
- **Git**: Para controle de versão

### Aprendizado
- **Trading Technical Analysis**: Conceitos básicos
- **Risk Management**: Gestão de risco essencial
- **FTMO Rules**: Regras específicas FTMO

---

## 🎉 Parabéns!

Você completou com sucesso a configuração inicial do EA_SCALPER_XAUUSD!

Seu sistema está pronto para:
- ✅ Conectar ao MetaTrader 5
- ✅ Analisar dados de XAUUSD
- ✅ Executar estratégias de trading
- ✅ Gerenciar risco automaticamente
- ✅ Usar IA para tomada de decisões

**Próximo recomendado:** Explore os exemplos em `docs/examples/` e comece a desenvolver suas próprias estratégias!