# Guia de Configuração Inicial - EA_SCALPER_XAUUSD

## 📋 Índice
1. [Visão Geral da Configuração](#visão-geral)
2. [Configuração do Ambiente](#config-ambiente)
3. [Configuração das Chaves de API](#config-api)
4. [Configuração do MetaTrader](#config-metatrader)
5. [Configuração dos Servidores MCP](#config-mcp)
6. [Configuração do Proxy Server](#config-proxy)
7. [Configuração dos EAs](#config-eas)
8. [Validação da Configuração](#validacao)
9. [Backup das Configurações](#backup)

---

## 🎯 Visão Geral da Configuração

Este guia orienta você através da configuração inicial de todos os componentes do sistema EA_SCALPER_XAUUSD:

- **Variáveis de ambiente** e credenciais
- **MetaTrader 5** para execução dos EAs
- **Servidores MCP** para integração com Claude Code
- **Proxy server** para OpenRouter
- **Especialistas Advisors** e seus parâmetros
- **Scripts de automatização**

---

## 🔧 Configuração do Ambiente

### Passo 1: Ativar Ambiente Virtual

#### Windows
```cmd
# Via CMD
venv\Scripts\activate

# Via PowerShell
.\venv\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
source venv/bin/activate
```

### Passo 2: Verificar Dependências

```bash
# Verifique se todas as dependências estão instaladas
pip list | grep -E "(httpx|python-dotenv|mcp|pylint|pytest)"

# Se faltar alguma dependência, instale:
pip install httpx python-dotenv mcp pylint pytest pytest-json-report
pip install structlog pathspec pytest-asyncio mypy
```

### Passo 3: Configurar Variáveis de Ambiente

#### Copiar Arquivo .env

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo
nano .env  # Linux/macOS
# ou
notepad .env  # Windows
```

#### Configurar Variáveis Básicas

```env
# ================================
# CONFIGURAÇÃO OPENROUTER
# ================================
OPENROUTER_API_KEY=sk-or-v1-sua_chave_api_aqui
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD Trading System"
OPENROUTER_SITE_URL="https://github.com/seu-usuario/EA_SCALPER_XAUUSD"

# ================================
# CONFIGURAÇÃO DE MODELOS
# ================================
DEFAULT_MODEL="openrouter/anthropic/claude-3-5-sonnet"
BACKUP_MODEL="openrouter/openai/gpt-4o"
FAST_MODEL="openrouter/meta-llama/llama-3.1-8b-instruct:free"

# ================================
# CONFIGURAÇÃO DE CACHE
# ================================
PROMPT_CACHE_TTL=3600
RESPONSE_CACHE_TTL=1800
ENABLE_CACHE=true

# ================================
# CONFIGURAÇÃO DO PROXY
# ================================
PROXY_HOST=0.0.0.0
PROXY_PORT=4000
PROXY_RATE_LIMIT=2.0

# ================================
# CONFIGURAÇÃO METATRADER
# ================================
MT5_LOGIN=seu_login
MT5_PASSWORD=sua_senha
MT5_SERVER=seu_servidor_broker

# ================================
# CONFIGURAÇÃO DE LOGGING
# ================================
LOG_LEVEL=INFO
LOG_FILE=logs/system.log
ENABLE_DEBUG=false
```

### Passo 4: Criar Estrutura de Diretórios

```bash
# Crie diretórios necessários
mkdir -p logs
mkdir -p data/backups
mkdir -p data/reports
mkdir -p temp
mkdir -p workspace/sessions
```

---

## 🔑 Configuração das Chaves de API

### OpenRouter API (Obrigatório)

1. **Criar conta OpenRouter**:
   - Acesse: https://openrouter.ai/
   - Crie sua conta gratuita

2. **Obter API Key**:
   - Vá para: https://openrouter.ai/keys
   - Copie sua API key

3. **Configurar no .env**:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-sua_chave_real_aqui
   ```

### GitHub Token (Opcional)

1. **Criar Personal Access Token**:
   - Acesse: https://github.com/settings/tokens
   - Crie um token com scopes: `repo`, `read:org`

2. **Configurar no MCP**:
   - Edite `.roo/mcp.json`:
   ```json
   {
     "mcpServers": {
       "github": {
         "env": {
           "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_seu_token_aqui"
         }
       }
     }
   }
   ```

### Broker API (Opcional)

Se seu broker oferece API REST:

```env
BROKER_API_KEY=sua_chave_broker
BROKER_API_URL=https://api.seubroker.com
BROKER_ACCOUNT=sua_conta
```

---

## 💻 Configuração do MetaTrader

### Passo 1: Configurar Conta Demo

1. **Abra o MetaTrader 5**
2. **Vá em**: Arquivo → Entrar na Conta de Negociação
3. **Selecione**: Abrir conta demo
4. **Preencha os dados**:
   - Nome: Seu nome
   - Email: Seu email
   - Telefone: Seu telefone
   - País: Brasil
   - Moeda: USD

### Passo 2: Configurar Permissões

1. **Vá em**: Ferramentas → Opções
2. **Aba Expert Advisors**:
   ```
   ✓ Permitir trading automatizado
   ✓ Permitir importação de DLL
   ✓ Permitir importação de experts externos
   ✓ Permitir solicitações WebRequest para URLs listadas
   ```

3. **Adicionar URLs permitidas**:
   ```
   https://openrouter.ai
   https://api.openai.com
   https://api.anthropic.com
   ```

### Passo 3: Configurar Gráficos

1. **Abra gráfico XAUUSD** (Gold vs USD)
2. **Timeframe recomendado**: M5 (5 minutos)
3. **Indicadores básicos** (opcional):
   - Moving Average (20 períodos)
   - RSI (14 períodos)
   - Volume

### Passo 4: Copiar EAs para MetaTrader

```bash
# Windows (exemplo)
copy "📚 LIBRARY\02_Strategies_Legacy\EA_FTMO_SCALPER_ELITE\MQL5_Source\*.mq5" "%APPDATA%\MetaQuotes\Terminal\*\MQL5\Experts\"

# Linux (via Wine)
cp "📚 LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MQL5_Source/*.mq5" "~/.wine/drive_c/users/$USER/AppData/Roaming/MetaQuotes/Terminal/*/MQL5/Experts/"
```

---

## 🤖 Configuração dos Servidores MCP

### Passo 1: Instalar MCP Code Checker

```bash
# Navegue até o diretório do MCP Code Checker
cd "🤖 AI_AGENTS/MCP_Code_Checker"

# Instale o servidor MCP
pip install -e .

# Instale dependências de desenvolvimento (opcional)
pip install -e ".[dev]"
```

### Passo 2: Configurar MCP para Claude Code

1. **Abra as configurações do Claude Code**
2. **Vá em**: Settings → MCP Servers
3. **Adicione os servidores**:

#### MCP Code Checker
```json
{
  "name": "code-checker",
  "command": "python",
  "args": ["-m", "mcp_code_checker", "--host", "localhost", "--port", "8001"],
  "cwd": "/caminho/completo/EA_SCALPER_XAUUSD/🤖 AI_AGENTS/MCP_Code_Checker"
}
```

#### MCP GitHub (se configurado)
```json
{
  "name": "github",
  "command": "docker",
  "args": [
    "run", "-i", "--rm",
    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
    "-e", "GITHUB_TOOLSETS",
    "-e", "GITHUB_READ_ONLY",
    "ghcr.io/github/github-mcp-server"
  ],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_seu_token_aqui",
    "GITHUB_TOOLSETS": "",
    "GITHUB_READ_ONLY": "true"
  }
}
```

### Passo 3: Configurar MCP MetaTrader (Opcional)

Se você tem o servidor MCP para MetaTrader:

```bash
# Navegue até o servidor MT5
cd "📚 LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MCP_Debug"

# Instale dependências
pip install -r requirements.txt

# Inicie o servidor
python main.py
```

Configure no Claude Code:
```json
{
  "name": "metatrader5",
  "command": "python",
  "args": ["-m", "mcp_metatrader5_server"],
  "cwd": "/caminho/completo/EA_SCALPER_XAUUSD/📚 LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MCP_Debug"
}
```

---

## 🌐 Configuração do Proxy Server

### Passo 1: Configurar Proxy Básico

1. **Edite o arquivo de configuração**:
   ```bash
   nano scripts/python/simple_trading_proxy.py
   ```

2. **Configure os modelos disponíveis**:
   ```python
   self.models = {
       "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
       "gpt-4o": "openai/gpt-4o",
       "deepseek-r1": "deepseek/deepseek-r1-0528:free",
       "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
       "default": "deepseek/deepseek-r1-0528:free"
   }
   ```

### Passo 2: Configurar Rate Limiting

```python
# No arquivo simple_trading_proxy.py
RATE_LIMIT_REQUESTS = 10  # requisições por minuto
RATE_LIMIT_WINDOW = 60    # segundos
CACHE_SIZE_LIMIT = 1000    # itens no cache
```

### Passo 3: Configurar Porta e Host

Edite as configurações no final do arquivo:
```python
def run_proxy(host='0.0.0.0', port=4000):
    # Configure conforme sua necessidade
```

### Passo 4: Testar o Proxy

```bash
# Inicie o proxy
python scripts/python/simple_trading_proxy.py

# Em outro terminal, teste
curl http://localhost:4000/health
curl http://localhost:4000/v1/models
```

---

## 📈 Configuração dos EAs

### Passo 1: Compilar EAs

1. **Abra o MetaEditor** (F4 no MetaTrader)
2. **Abra cada arquivo .mq5**:
   - `📚 LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MQL5_Source/EA_FTMO_SCALPER_ELITE.mq5`
3. **Compile** (F7)
4. **Verifique se não há erros**

### Passo 2: Configurar Parâmetros do EA

Ao arrastar o EA para o gráfico, configure:

#### Parâmetros de Risk Management
```
LotSize = 0.01
StopLoss = 200
TakeProfit = 400
MaxDrawdown = 10.0
```

#### Parâmetros de Sinal
```
UseMAFilter = true
UseRSIFilter = true
MA_Period = 20
RSI_Period = 14
RSI_Overbought = 70
RSI_Oversold = 30
```

#### Parâmetros de Tempo
```
StartHour = 8
EndHour = 22
MondayTrading = true
FridayTrading = false
```

### Passo 3: Configurar Magic Number

Cada EA deve ter um Magic Number único:
```
MagicNumber = 12345
```

### Passo 4: Configurar Notificações (Opcional)

```
EnablePushNotifications = true
EnableEmailNotifications = false
EmailServer = smtp.gmail.com
EmailLogin = seu_email@gmail.com
EmailPassword = sua_senha_app
```

---

## ✅ Validação da Configuração

### Teste 1: Verificar Conexão com APIs

```bash
# Teste OpenRouter
python -c "
import httpx
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}

with httpx.Client() as client:
    response = client.get('https://openrouter.ai/api/v1/models', headers=headers)
    print('Status:', response.status_code)
    print('Models available:', len(response.json().get('data', [])))
"
```

### Teste 2: Verificar Proxy Server

```bash
# Inicie o proxy em background
python scripts/python/simple_trading_proxy.py &

# Teste requisição
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Teste 3: Verificar MCP Servers

```bash
# Teste MCP Code Checker
python -m pytest "🤖 AI_AGENTS/MCP_Code_Checker/tests/" -v

# Teste conexão com servidores
python -c "
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    server_params = StdioServerParameters(
        command='python',
        args=['-m', 'mcp_code_checker']
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print('Tools available:', len(tools.tools))

asyncio.run(test_mcp())
"
```

### Teste 4: Verificar MetaTrader

1. **Execute um EA em modo visualização**
2. **Verifique no log do Expert**:
   - Deve mostrar "EA initialized successfully"
   - Deve mostrar conexão com o servidor
   - Deve mostrar permissões de trading

### Teste 5: Verificar Scripts Python

```bash
# Teste classificador
python 🔧\ WORKSPACE/Development/Core/classificador_qualidade_maxima.py

# Teste backup
python 🔧\ WORKSPACE/Development/Scripts/git_auto_backup.py --test

# Teste proxy LiteLLM
python scripts/python/litellm_server.py --test
```

---

## 💾 Backup das Configurações

### Criar Script de Backup

```bash
# Crie script de backup
cat > backup_config.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="data/backups/config_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup de arquivos de configuração
cp .env "$BACKUP_DIR/"
cp .roo/mcp.json "$BACKUP_DIR/"
cp scripts/python/config_*.json "$BACKUP_DIR/" 2>/dev/null || true

# Backup de logs importantes
cp logs/system.log "$BACKUP_DIR/" 2>/dev/null || true

# Criar arquivo de metadados
cat > "$BACKUP_DIR/backup_info.txt" << META
Backup criado em: $(date)
Sistema: $(uname -a)
Python: $(python --version)
META

echo "Backup salvo em: $BACKUP_DIR"
EOF

chmod +x backup_config.sh
./backup_config.sh
```

### Backup das Configurações do MetaTrader

1. **Exportar perfil**:
   - MetaTrader → Arquivo → Salvar Perfil
   - Nome: "EA_SCALPER_XAUUSD_Config"

2. **Exportar templates**:
   - Copie arquivos de `templates/`

3. **Salvar configurações dos EAs**:
   - Crie screenshots das configurações
   - Documente parâmetros utilizados

---

## 📝 Checklist de Configuração

### Configuração Básica
- [ ] Ambiente virtual ativado
- [ ] Dependências Python instaladas
- [ ] Arquivo .env configurado
- [ ] Chaves de API configuradas
- [ ] Estrutura de diretórios criada

### Configuração MetaTrader
- [ ] Conta demo configurada
- [ ] Permissões de trading habilitadas
- [ ] EAs compilados
- [ ] Gráficos configurados
- [ ] Parâmetros dos EAs ajustados

### Configuração MCP
- [ ] MCP Code Checker instalado
- [ ] Servidores configurados no Claude Code
- [ ] Conexão testada

### Configuração Proxy
- [ ] Proxy server configurado
- [ ] Modelos mapeados
- [ ] Rate limiting configurado
- [ ] Conexão testada

### Validação Final
- [ ] Testes de API passaram
- [ ] Proxy server funcional
- [ ] MCP servers online
- [ ] EAs carregando no MT5
- [ ] Scripts executando sem erros

---

## 🔗 Próximos Passos

Com a configuração inicial completa:

1. **Leia o Guia de Uso Diário**: `/docs/installation/03-uso-diario.md`
2. **Consulte o Troubleshooting**: `/docs/installation/05-troubleshooting.md`
3. **Acompanhe a documentação**: Verifique `📋 DOCUMENTACAO_FINAL/`

**Seu sistema está configurado e pronto para operação!** 🚀