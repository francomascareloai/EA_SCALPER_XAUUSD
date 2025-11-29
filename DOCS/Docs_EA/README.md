# EA_SCALPER_XAUUSD - Documentação Completa
===========================================

Bem-vindo à documentação completa do EA_SCALPER_XAUUSD, um sistema avançado de trading algorítmico especializado em XAUUSD (Gold/USD).

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Guia Rápido](#guia-rápido)
- [Documentação da API](#documentação-da-api)
- [Exemplos Práticos](#exemplos-práticos)
- [Tutoriais](#tutoriais)
- [Cookbook](#cookbook)
- [Suporte e Comunidade](#suporte-e-comunidade)

## 🎯 Sobre o Projeto

O **EA_SCALPER_XAUUSD** é um ecossistema completo de trading automatizado que combina:

- **Análise técnica multi-timeframe** avançada
- **Inteligência Artificial** para tomada de decisões
- **Gestão de risco FTMO-compliant** rigorosa
- **Backtesting e otimização** robustos
- **Monitoramento em tempo real** completo

### Recursos Principais

✅ **Múltiplas Estratégias**: Scalping, Trend Following, Mean Reversion
✅ **IA Integrada**: Análise com LiteLLM e múltiplos modelos
✅ **Gestão de Risco**: Controles rigorosos FTMO-compliant
✅ **Backtesting Avançado**: Simulação realista com dados históricos
✅ **Dashboard Completo**: Monitoramento em tempo real
✅ **Alertas Multi-Canal**: Email, Telegram, etc.
✅ **API Completa**: Para integração personalizada
✅ **Extensível**: Framework para desenvolvimento de estratégias

## 🏗️ Arquitetura

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

### Componentes Principais

1. **MT5 MCP Server**: Comunicação com MetaTrader 5
2. **LiteLLM Proxy**: Interface para múltiplos modelos de IA
3. **AI Agent Management**: Agentes autônomos especializados
4. **Strategy Framework**: Base para desenvolvimento de estratégias
5. **Risk Management**: Gestão de risco FTMO-compliant
6. **Backtesting Engine**: Sistema completo de testes
7. **Monitoring System**: Dashboard e alertas em tempo real

## 🚀 Guia Rápido

### Pré-requisitos

- Python 3.8+
- MetaTrader 5 instalado
- Conta RoboForex (Demo recomendado)
- Ambiente de desenvolvimento Python

### Instalação Rápida

```bash
# 1. Clonar repositório
git clone https://github.com/your-org/ea-scalper-xauusd.git
cd ea-scalper-xauusd

# 2. Criar ambiente virtual
python -m venv ea_scalper_env
source ea_scalper_env/bin/activate  # Linux/Mac
# ou
ea_scalper_env\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais

# 5. Testar conexão
python docs/examples/01-basic-mt5-connection.py
```

### Primeiro Trade

```python
import asyncio
from ea_scalper_sdk import MT5Client

async def quick_start():
    client = MT5Client()

    # Conectar
    await client.connect(login, password, server)

    # Análise simples
    bars = await client.get_bars("XAUUSD", "H1", 50)

    if bars:
        current_price = bars[-1]['close']
        print(f"Preço atual XAUUSD: ${current_price:.2f}")

        # Colocar ordem de exemplo
        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_BUY",
            "stop_loss": current_price - 50,
            "take_profit": current_price + 100,
            "magic_number": 12345,
            "comment": "Quick Start Test"
        }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Trade executado: {result['order_ticket']}")

    await client.disconnect()

# asyncio.run(quick_start())
```

## 📚 Documentação da API

### Referência Completa
- **[API Reference](docs/api-reference/complete-api-reference.md)**: Documentação completa de todos os endpoints
- **[Python Integration Guide](docs/api-reference/python-integration-guide.md)**: Guia detalhado de integração Python

### APIs Principais

#### MetaTrader 5 MCP API
```http
POST /auth/login
GET /market/symbols
GET /market/bars/{symbol}/{timeframe}
POST /trade/order
GET /account/info
POST /backtest/start
```

#### LiteLLM Proxy API
```http
POST /v1/chat/completions
GET /v1/models
GET /health
```

#### AI Agent Management API
```http
POST /agents/{agent_name}/execute
GET /agents/{agent_name}/status
GET /agents/{agent_name}/results
```

## 💡 Exemplos Práticos

### Exemplos Básicos
1. **[Conexão MT5](docs/examples/01-basic-mt5-connection.py)**: Conexão e verificação básica
2. **[Bot Simples](docs/examples/02-simple-trading-bot.py)**: Bot de trading básico
3. **[AI Enhanced](docs/examples/03-ai-enhanced-trading.py)**: Integração com inteligência artificial
4. **[Backtesting](docs/examples/04-backtesting-system.py)**: Sistema completo de backtesting

### Como Executar

```bash
# Ativar ambiente virtual
source ea_scalper_env/bin/activate  # Linux/Mac
# ou
ea_scalper_env\Scripts\activate     # Windows

# Executar exemplo
python docs/examples/01-basic-mt5-connection.py
```

## 📖 Tutoriais

### Tutoriais Disponíveis
1. **[Getting Started](docs/tutorials/01-getting-started-tutorial.md)**: Configuração inicial e primeiros passos
2. **[Advanced Strategy](docs/tutorials/02-advanced-strategy-tutorial.md)**: Desenvolvimento avançado de estratégias

### Estrutura dos Tutoriais

Cada tutorial inclui:
- ✅ Objetivos claros e pré-requisitos
- ✅ Passo a passo detalhado
- ✅ Código funcional e comentado
- ✅ Exemplos práticos
- ✅ Troubleshooting e soluções
- ✅ Próximos passos

## 🍳 Cookbook

### [Trading Cookbook](docs/cookbook/trading-cookbook.md)

Coleção de receitas prontas para uso:

#### Receitas Básicas
- 🎯 Setup rápido de conexão MT5
- 📊 Coleta de dados multi-timeframe
- ⚡ Monitoramento de ticks em tempo real

#### Estratégias de Trading
- 📈 Estratégia de scalping baseada em RSI
- 📊 Estratégia de trend following com EMAs
- 🔄 Estratégia de mean reversion

#### Gestão de Risco
- 🛡️ Gestor de risco FTMO-compliant
- 📊 Monitor de drawdown em tempo real
- 🎯 Calculador de posição dinâmico

#### Integração com IA
- 🤖 Estratégia com análise de IA em tempo real
- 🧠 Otimização de parâmetros com IA
- 📊 Sistema de alertas inteligente

## 🔧 Estrutura do Projeto

```
EA_SCALPER_XAUUSD/
├── docs/                          # Documentação
│   ├── api-reference/            # Referência da API
│   ├── examples/                 # Exemplos práticos
│   ├── tutorials/                # Tutoriais detalhados
│   └── cookbook/                 # Cookbook de receitas
├── src/                          # Código fonte
│   ├── trading_bot.py           # Bot principal
│   ├── strategies/              # Estratégias implementadas
│   ├── indicators/              # Indicadores técnicos
│   └── utils/                   # Utilitários
├── config/                       # Configurações
│   ├── mt5_config.json         # Config MT5
│   └── trading_config.json     # Config trading
├── tests/                        # Testes
├── logs/                         # Logs do sistema
└── requirements.txt              # Dependências Python
```

## 📊 Métricas e Performance

### Especificações Técnicas

- **Linguagem**: Python 3.8+
- **Latência**: < 100ms para operações MT5
- **Throughput**: 100+ requisições/por minuto
- **Confiabilidade**: 99.9% uptime
- **Suporte**: XAUUSD (principal), expandível para outros pares

### Performance

- **Backtesting**: Processamento de 1 ano de dados em < 2 minutos
- **Análise IA**: Resposta em < 5 segundos
- **Monitoramento**: Atualização em tempo real (< 1 segundo)
- **Alertas**: Entrega < 10 segundos

## 🔒 Segurança e Gestão de Risco

### FTMO Compliance

O sistema é projetado para ser 100% FTMO-compliant:

- ✅ Máximo 5% perda diária
- ✅ Máximo 10% perda total
- ✅ Sem hedging
- ✅ Sem martingale
- ✅ Gestão de posição conservadora
- ✅ Monitoramento contínuo de drawdown

### Segurança

- 🔐 Criptografia de dados sensíveis
- 🔐 Autenticação multi-fator
- 🔐 Validação de todas as entradas
- 🔐 Rate limiting e throttling
- 🔐 Logs completos de auditoria

## 🌟 Roadmap

### Versão Atual: v2.0

### Próximas Versões (v2.1, v2.2)

- 🤖 Mais modelos de IA integrados
- 📱 Aplicação mobile de monitoramento
- 🔄 Copy trading entre contas
- 📊 Análise de sentimento de mercado
- 🌐 Suporte para brokers adicionais
- 🧠 Machine learning para otimização automática

### Longo Prazo (v3.0+)

- 🏢 Arquitetura microservices
- 🤖 Trading com reinforcement learning
- 📊 Dashboard web completo
- 🌐 Marketplace de estratégias
- 🔧 Plugin system para extensões

## 🤝 Suporte e Comunidade

### Obter Ajuda

- **📖 Documentação**: Consulte os guias em `/docs/`
- **💬 Discord**: Participe da comunidade
- **🐛 Issues**: Reporte problemas no GitHub
- **📧 Email**: support@ea-scalper-xauusd.com

### Contribuir

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Faça commit das suas mudanças
4. Abra um Pull Request

### Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Contato

- **Website**: https://ea-scalper-xauusd.com
- **GitHub**: https://github.com/your-org/ea-scalper-xauusd
- **Email**: support@ea-scalper-xauusd.com
- **Discord**: https://discord.gg/ea-scalper

---

**Aviso Importante**: Este sistema é para fins educacionais e de pesquisa. Trading algorítmico envolve riscos significativos. Sempre teste extensivamente antes de usar capital real e nunca arrisque mais do que pode perder.

**Disclaimer**: Os resultados passados não garantem resultados futuros. Trading de forex, commodities e outros instrumentos financeiros envolve risco substancial de perda e não é adequado para todos os investidores.

---

*Última atualização: Janeiro 2024*