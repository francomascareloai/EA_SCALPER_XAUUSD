# 🚀 Quick Start Guide - EA_SCALPER_XAUUSD

**Guia Rápido para Iniciantes (15 minutos)**

---

## 🎯 Visão Geral Rápida

O EA_SCALPER_XAUUSD é um sistema automatizado de trading para XAUUSD (Ouro) que inclui:
- **Especialistas Advisors (EAs)** para MetaTrader 5
- **Sistema de IA** para análise de mercado
- **Proxy server** para OpenRouter API
- **Scripts de automação** e gerenciamento

---

## ⚡ Instalação Rápida (5 minutos)

### Passo 1: Pré-requisitos Essenciais

**Você precisa ter instalado:**
- ✅ Python 3.11+
- ✅ Git
- ✅ MetaTrader 5

**Não tem?** [Veja o guia completo](01-instalacao-completa.md)

### Passo 2: Clonar o Projeto

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/EA_SCALPER_XAUUSD.git
cd EA_SCALPER_XAUUSD
```

### Passo 3: Ambiente Python

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Instalar dependências
pip install httpx python-dotenv mcp pylint pytest
```

### Passo 4: Configurar API Key

```bash
# Copiar arquivo de configuração
copy .env.example .env  # Windows
# ou
cp .env.example .env    # Linux/macOS

# Editar o arquivo .env e adicionar sua API Key da OpenRouter
# OPENROUTER_API_KEY=sk-or-v1-sua_chave_aqui
```

### Passo 5: Testar Instalação

```bash
# Testar sistema
python scripts/python/quick_test.py
```

Se aparecer "✅ Sistema funcionando", você está pronto!

---

## 🎮 Primeiros Passos (5 minutos)

### 1. Iniciar o Proxy Server

```bash
# Terminal 1
python scripts/python/simple_trading_proxy.py
```

Você deve ver:
```
🚀 SIMPLE TRADING PROXY INICIANDO...
📡 Host: 0.0.0.0:4000
🌐 IP Local: 192.168.1.100:4000
✅ Proxy pronto para uso!
```

### 2. Configurar MetaTrader 5

1. **Abra o MetaTrader 5**
2. **Habilite AutoTrading** (botão verde)
3. **Abra gráfico XAUUSD M5**
4. **Copie os EAs** para pasta MetaTrader/MQL5/Experts/

### 3. Ativar o EA

1. **Arraste o EA** `EA_FTMO_SCALPER_ELITE` para o gráfico
2. **Configure parâmetros básicos**:
   ```
   LotSize = 0.01
   StopLoss = 200
   TakeProfit = 400
   Enable Trading = true
   ```
3. **Clique em OK**

### 4. Verificar Funcionamento

No terminal do EA você deve ver:
```
EA_FTMO_SCALPER_ELITE initialized successfully
Connected to broker server
AutoTrading enabled
Ready for trading
```

---

## 🔧 Configuração Básica (3 minutos)

### Parâmetros Recomendados para Iniciantes

```mql5
// Risk Management
LotSize = 0.01          // Risco baixo
StopLoss = 200          // 20 pips
TakeProfit = 400        // 40 pips
MaxDrawdown = 10.0      // 10% máximo

// Trading Schedule
StartHour = 8           // Início 8:00
EndHour = 22            // Fim 22:00
MondayTrading = true    // Segunda sim
FridayTrading = false   // Sexta não (recomendado)

// Indicators
UseMAFilter = true      // Filtro Média Móvel
UseRSIFilter = true     // Filtro RSI
MA_Period = 20          // Média de 20 períodos
```

### Ajustes Importantes

1. **Magic Number**: Use um número único (ex: 12345)
2. **Timeframe**: Recomendado M5 para scalping
3. **Conta**: Comece com conta DEMO!

---

## 📊 Monitoramento Essencial (2 minutos)

### Verificações Diárias

**Manhã (antes de 8:00):**
```bash
# 1. Ativar ambiente
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows

# 2. Iniciar proxy
python scripts/python/simple_trading_proxy.py &

# 3. Verificar API
curl http://localhost:4000/health
```

**Durante o dia:**
- ✅ AutoTrading está ativo (botão verde)
- ✅ EA está rodando (ícone sorriso no gráfico)
- ✅ Sem erros no log do EA

**Fim do dia:**
- ✅ Verificar resultado do dia
- ✅ Salvar relatório se disponível

### Logs Importantes

```bash
# Verificar log do sistema
tail -f logs/system.log

# Verificar log do proxy
tail -f logs/proxy.log

# Verificar erros
grep ERROR logs/*.log
```

---

## 🚨 Configurações de Segurança Obrigatórias

### Risk Management

**NUNCA use em conta real sem:**
1. **Teste extensivo em DEMO** (mínimo 30 dias)
2. **Drawdown limitado** (máximo 10%)
3. **Tamanho de lote pequeno** (comece com 0.01)
4. **Stop loss sempre configurado**
5. **Monitoramento constante**

### Backups Automáticos

```bash
# Criar script de backup simples
echo "#!/bin/bash
cp .env backup/env_$(date +%Y%m%d)
cp logs/*.log backup/logs_$(date +%Y%m%d)/ 2>/dev/null || true
" > backup_daily.sh
chmod +x backup_daily.sh
```

---

## 🎯 Primeiro Trade - O Que Esperar

### Sinais do EA

O EA analisa:
- **Médias Móveis** (tendência)
- **RSI** (momento)
- **Volume** (confirmação)
- **Suporte/Resistência** (níveis)

### Tipos de Operações

**BUY Signal:**
- Preço acima da média móvel
- RSI entre 30-70 (não sobrecomprado)
- Volume crescente

**SELL Signal:**
- Preço abaixo da média móvel
- RSI entre 30-70 (não sobrevendido)
- Volume crescente

### Resultado Esperado (Mês 1)

- **Win Rate**: 45-55% (normal para scalping)
- **Profit Factor**: 1.2-1.5
- **Máximo Drawdown**: <10%
- **Trades por dia**: 5-15

---

## 🛠️ Comandos Essenciais

### Diagnóstico Rápido

```bash
# Verificar instalação
python scripts/python/quick_test.py

# Testar API
curl http://localhost:4000/health

# Verificar EAs
ls 📚\ LIBRARY\02_Strategies_Legacy\EA_FTMO_SCALPER_ELITE\MQL5_Source\*.mq5
```

### Problemas Comuns

**EA não aparece:**
```bash
# Copiar EAs manualmente
copy "📚\ LIBRARY\02_Strategies_Legacy\EA_FTMO_SCALPER_ELITE\MQL5_Source\*.mq5" "%APPDATA%\MetaQuotes\Terminal\*\MQL5\Experts\"
```

**Proxy não funciona:**
```bash
# Mudar porta
python scripts/python/simple_trading_proxy.py --port=4001
```

**Python não encontrado:**
```bash
# Usar python3
python3 scripts/python/quick_test.py
```

---

## 📈 Próximos Passos (Após 1 Semana)

### Otimização Básica

1. **Ajustar parâmetros** com base nos resultados
2. **Testar diferentes timeframes** (M1, M15)
3. **Adicionar filtros adicionais** se necessário

### Análise Avançada

1. **Instalar Claude Code** com MCP servers
2. **Usar sistema multi-agente** para otimização
3. **Implementar backtest automático**

### Documentação

Leia os guias completos:
- [Guia de Instalação Completa](01-instalacao-completa.md)
- [Guia de Configuração Inicial](02-configuracao-inicial.md)
- [Guia de Uso Diário](03-uso-diario.md)
- [Troubleshooting](04-troubleshooting.md)

---

## ⚠️ Avisos Importantes

### Riscos do Trading

- **Trading envolve risco de perda**
- **Performance passada não garante resultados futuros**
- **Comece sempre com conta DEMO**
- **Nunca arrisque mais do que pode perder**

### Segurança

- **Mantenha suas API keys seguras**
- **Não compartilhe senhas**
- **Use autenticação de dois fatores quando possível**
- **Faça backups regulares**

### Regulamentação

- **Verifique regulamentação local**
- **Cumpra regras de sua corretora**
- **Esteja ciente de implicações fiscais**

---

## 📞 Suporte Rápido

### Autoajuda

1. **Verifique logs**: `tail logs/*.log`
2. **Execute diagnóstico**: `python scripts/python/quick_test.py`
3. **Consulte troubleshooting**: [Guia completo](04-troubleshooting.md)

### Comunidade

- **GitHub Issues**: Reportar bugs
- **Discord/Telegram**: (links no README)
- **Documentação**: `📋 DOCUMENTACAO_FINAL/`

---

## ✅ Checklist de Início

**Antes de Começar:**
- [ ] Python 3.11+ instalado
- [ ] MetaTrader 5 funcionando
- [ ] Conta DEMO configurada
- [ ] API Key OpenRouter obtida

**Instalação:**
- [ ] Projeto clonado
- [ ] Ambiente virtual criado
- [ ] Dependências instaladas
- [ ] .env configurado
- [ ] Instalação testada

**Primeiro Trade:**
- [ ] Proxy server iniciado
- [ ] EA compilado e ativo
- [ ] Parâmetros configurados
- [ ] Monitoramento iniciado

**Segurança:**
- [ ] Risk management configurado
- [ ] Backup automatizado
- [ ] Documentação lida

---

## 🎉 Parabéns!

Você concluiu a configuração básica do EA_SCALPER_XAUUSD!

**Lembre-se:**
- Comece devagar e com cuidado
- Monitore constantemente
- Aprenda com os resultados
- Nunca pare de estudar

**Bons trades!** 📈💰

---

*Este guia é para iniciantes. Para configurações avançadas, consulte os outros guias de instalação.*