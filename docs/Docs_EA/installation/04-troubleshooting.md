# Guia de Troubleshooting - EA_SCALPER_XAUUSD

## 📋 Índice
1. [Problemas Comuns de Instalação](#problemas-instalacao)
2. [Problemas com MetaTrader](#problemas-metatrader)
3. [Problemas com APIs e Conexões](#problemas-apis)
4. [Problemas com EAs](#problemas-eas)
5. [Problemas com Python e Scripts](#problemas-python)
6. [Problemas com MCP Servers](#problemas-mcp)
7. [Problemas de Performance](#problemas-performance)
8. [Recuperação de Sistema](#recuperacao-sistema)
9. [Ferramentas de Diagnóstico](#ferramentas-diagnostico)

---

## 🚨 Problemas Comuns de Instalação

### Problema: "python não reconhecido"

#### Sintomas
```
'python' não é reconhecido como comando interno
ou
Command 'python' not found
```

#### Causas
- Python não instalado
- Python não adicionado ao PATH
- Versões conflitantes (python vs python3)

#### Soluções

**Windows:**
```cmd
# Verificar se Python está instalado
where python
where python3

# Adicionar Python ao PATH manualmente
# 1. Abrir Propriedades do Sistema
# 2. Variáveis de Ambiente
# 3. Editar PATH
# 4. Adicionar: C:\Python311\ e C:\Python311\Scripts\

# Ou reinstalar com "Add to PATH" marcado
```

**Linux/macOS:**
```bash
# Verificar instalação
which python3
python3 --version

# Criar alias (temporário)
alias python=python3

# Criar link permanente
sudo ln -s /usr/bin/python3 /usr/bin/python
```

### Problema: Erro de permissão no Windows

#### Sintomas
```
Access denied
Execution Policy Error
```

#### Soluções

**PowerShell:**
```powershell
# Verificar política atual
Get-ExecutionPolicy

# Permitir execução de scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Para sessão atual apenas
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

**CMD como Administrador:**
```cmd
# Executar como Administrador
# Botão direito → Executar como administrador
```

### Problema: Virtual environment não funciona

#### Sintomas
```
'venv' is not recognized
Activation script not found
```

#### Soluções

**Recriar ambiente virtual:**
```bash
# Remover ambiente antigo
rm -rf venv

# Criar novo ambiente
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

**Verificar integridade:**
```bash
# Instalar dependências novamente
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 💻 Problemas com MetaTrader

### Problema: EA não aparece no MetaTrader

#### Sintomas
- EA não listado no Navigator
- EA aparece com ícone cinza
- Erro ao tentar adicionar EA ao gráfico

#### Diagnóstico
```bash
# Verificar se arquivo .mq5 existe
find . -name "*.mq5" -type f

# Verificar se arquivo .ex5 compilado existe
find . -name "*.ex5" -type f

# Verificar pasta correta do MetaTrader
# No MT5: File → Open Data Folder → MQL5 → Experts
```

#### Soluções

**1. Compilar EA:**
```bash
# Via MetaEditor
# 1. Abrir MetaTrader 5
# 2. Pressionar F4 (MetaEditor)
# 3. Abrir arquivo .mq5
# 4. Pressionar F7 (Compile)
# 5. Verificar "0 error(s), 0 warning(s)"
```

**2. Verificar pasta correta:**
```bash
# Windows
copy "caminho\do\EA\*.ex5" "%APPDATA%\MetaQuotes\Terminal\*\MQL5\Experts\"

# Linux (Wine)
cp "caminho/do/EA/*.ex5" "~/.wine/drive_c/users/$USER/AppData/Roaming/MetaQuotes/Terminal/*/MQL5/Experts/"
```

**3. Habilitar importação de DLL:**
- MetaTrader → Tools → Options → Expert Advisors
- Marcar "Allow DLL imports"

### Problema: EA não negocia

#### Sintomas
- EA carrega mas não abre posições
- Mensagem "Trading is disabled"
- Erro de permissão

#### Diagnóstico
```python
# Verificar status do trading no MetaTrader
import MetaTrader5 as mt5

if mt5.initialize():
    account = mt5.account_info()
    print(f"Trading enabled: {account.trade_mode_allowed}")
    print(f"Terminal trade allowed: {mt5.terminal_info().trade_allowed}")
    print(f"Server trade allowed: {account.server_trade_allowed}")
    mt5.shutdown()
```

#### Soluções

**1. Verificar permissões:**
```
MetaTrader → Tools → Options → Expert Advisors:
✓ Allow automated trading
✓ Allow DLL imports
```

**2. Verificar status do terminal:**
```
- Botão AutoTrading deve estar verde
- Verificar se conta permite trading
- Verificar horário de negociação
```

**3. Verificar configurações do EA:**
```
- Enable Trading = true
- Lot Size > 0
- Magic Number único
```

### Problema: Erro "Invalid Account"

#### Sintomas
```
Trade request failed
Invalid account
Invalid stops
```

#### Soluções

**1. Verificar símbolo:**
```python
import MetaTrader5 as mt5

if mt5.initialize():
    symbols = mt5.symbols_get()
    print("Símbolos disponíveis:")
    for s in symbols:
        if "GOLD" in s.name or "XAU" in s.name:
            print(f"  {s.name}: {s.trade_mode}")
    mt5.shutdown()
```

**2. Ajustar stops e lots:**
```
- Verificar mínimo de lote do símbolo
- Aumentar distância de stop loss
- Verificar horário de negociação
```

**3. Verificar tipo de conta:**
```
- Conta demo vs real
- Permissões da conta
- Limites da corretora
```

---

## 🔗 Problemas com APIs e Conexões

### Problema: OpenRouter API não funciona

#### Sintomas
```
401 Unauthorized
API Key invalid
Connection timeout
```

#### Diagnóstico
```bash
# Testar API Key diretamente
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

# Verificar variável de ambiente
echo $OPENROUTER_API_KEY
```

#### Soluções

**1. Verificar API Key:**
```bash
# Editar .env
nano .env

# Confirmar chave correta
OPENROUTER_API_KEY=sk-or-v1-sua_chave_correta_aqui
```

**2. Verificar quota:**
- Acessar https://openrouter.ai/usage
- Verificar se quota disponível
- Upgrade plano se necessário

**3. Testar com script simples:**
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

data = {
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "messages": [{"role": "user", "content": "Test"}]
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=data
)

print("Status:", response.status_code)
print("Response:", response.json())
```

### Problema: Proxy server não inicia

#### Sintomas
```
Port 4000 already in use
Connection refused
Permission denied
```

#### Diagnóstico
```bash
# Verificar se porta está ocupada
netstat -an | grep :4000
# ou
lsof -i :4000

# Verificar processo
ps aux | grep simple_trading_proxy
```

#### Soluções

**1. Mudar porta:**
```python
# Editar simple_trading_proxy.py
def run_proxy(host='0.0.0.0', port=4001):  # Mudar porta
```

**2. Matar processo antigo:**
```bash
# Matar processo na porta 4000
sudo kill -9 $(lsof -t -i:4000)

# Ou
pkill -f simple_trading_proxy
```

**3. Verificar firewall:**
```bash
# Linux
sudo ufw allow 4000

# Windows (admin)
netsh advfirewall firewall add rule name="Allow Port 4000" dir=in action=allow protocol=TCP localport=4000
```

### Problema: Conexão com GitHub falha

#### Sintomas
```
Authentication failed
Permission denied
SSH key error
```

#### Soluções

**1. Verificar token:**
```bash
# Testar token
curl -H "Authorization: token ghp_seu_token" \
     https://api.github.com/user

# Atualizar token no .roo/mcp.json
```

**2. Configurar SSH:**
```bash
# Gerar nova chave SSH
ssh-keygen -t ed25519 -C "seu_email@example.com"

# Adicionar ao GitHub
cat ~/.ssh/id_ed25519.pub
# Copiar e colar em GitHub → Settings → SSH keys
```

---

## 📈 Problemas com EAs

### Problema: EA apresenta erros de compilação

#### Sintomas
```
'function' is not defined
undeclared identifier
syntax error
```

#### Diagnóstico
1. Abrir MetaEditor
2. Compilar EA (F7)
3. Verificar aba "Errors"

#### Soluções Comuns

**1. Funções não definidas:**
```mql5
// Adicionar includes no topo
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
```

**2. Variáveis não declaradas:**
```mql5
// Declarar variáveis globais
input group "Risk Management"
input double LotSize = 0.01;
input int StopLoss = 200;
input int MagicNumber = 12345;
```

**3. Sintaxe incorreta:**
```mql5
// Corrigir sintaxe
if (signal == BUY) {
    // código correto
}
```

### Problema: EA entra e sai de posições rapidamente

#### Sintomas
- Múltiplas entradas/saídas em segundos
- Overtrading
- Spreads consumindo lucro

#### Causas
- Sinais muito frequentes
- Timeframe muito baixo
- Filtros insuficientes
- Latência de execução

#### Soluções

**1. Adicionar cooldown:**
```mql5
// Adicionar variável global
datetime LastTradeTime = 0;

// Antes de abrir posição
if (TimeCurrent() - LastTradeTime < 300) { // 5 minutos
    return;
}
LastTradeTime = TimeCurrent();
```

**2. Melhorar filtros:**
```mql5
// Adicionar filtros de confirmação
bool MA_Filter = iMA(NULL, 0, 20, 0, MODE_EMA, PRICE_CLOSE);
bool RSI_Filter = (iRSI(NULL, 0, 14, PRICE_CLOSE, 0) > 30 &&
                  iRSI(NULL, 0, 14, PRICE_CLOSE, 0) < 70);

if (!MA_Filter || !RSI_Filter) {
    return;
}
```

**3. Aumentar timeframe:**
- Mudar de M1 para M5
- Adicionar confirmação em timeframe superior

### Problema: Drawdown excessivo

#### Sintomas
- Perdas consecutivas
- Drawdown > 10%
- Saldo diminuindo rapidamente

#### Diagnóstico
```python
# Monitorar drawdown em tempo real
import MetaTrader5 as mt5

if mt5.initialize():
    account = mt5.account_info()
    current_dd = abs(account.balance - account.equity) / account.balance * 100

    if current_dd > 10:
        print("ALERTA: Drawdown crítico!")
        # Implementar parada automática
```

#### Soluções

**1. Reduzir tamanho da posição:**
```
LotSize = 0.005 (metade do atual)
```

**2. Aumentar stop loss:**
```
StopLoss = 300 (aumentar de 200)
```

**3. Adicionar máxima de posições:**
```mql5
// Limitar posições abertas
int MaxPositions = 1;
if (PositionsTotal() >= MaxPositions) {
    return;
}
```

**4. Implementar parada automática:**
```mql5
// Parar se drawdown > 8%
double currentDrawdown = (AccountInfoDouble(ACCOUNT_BALANCE) -
                         AccountInfoDouble(ACCOUNT_EQUITY)) /
                        AccountInfoDouble(ACCOUNT_BALANCE) * 100;

if (currentDrawdown > 8.0) {
    ExpertRemove();
    Alert("EA parado por drawdown excessivo");
}
```

---

## 🐍 Problemas com Python e Scripts

### Problema: "ModuleNotFoundError"

#### Sintomas
```
ModuleNotFoundError: No module named 'mcp'
ModuleNotFoundError: No module named 'httpx'
```

#### Soluções

**1. Verificar ambiente virtual:**
```bash
# Verificar se ambiente está ativo
which python
echo $VIRTUAL_ENV

# Reativar ambiente
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

**2. Instalar módulos faltantes:**
```bash
pip install mcp httpx python-dotenv pylint pytest

# Ou instalar requirements.txt
pip install -r requirements.txt
```

**3. Verificar lista de pacotes:**
```bash
pip list | grep mcp
pip list | grep httpx
```

### Problema: Scripts Python não executam

#### Sintomas
```
Permission denied
Python script not found
Syntax error
```

#### Soluções

**1. Verificar permissões (Linux/macOS):**
```bash
chmod +x scripts/python/*.py
chmod +x *.py
```

**2. Verificar shebang:**
```bash
# Adicionar no topo dos scripts
#!/usr/bin/env python3
```

**3. Verificar sintaxe:**
```bash
python -m py_compile script.py
```

### Problema: Scripts de automação falham

#### Sintomas
- Scripts não encontram arquivos
- Erros de caminho (path)
- Falha em execuções agendadas

#### Diagnóstico
```python
# Adicionar logging detalhado
import logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Verificar caminhos
import os
print("Diretório atual:", os.getcwd())
print("Arquivos no diretório:", os.listdir('.'))
```

#### Soluções

**1. Usar caminhos absolutos:**
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
```

**2. Verificar existência de diretórios:**
```python
os.makedirs('logs', exist_ok=True)
os.makedirs('data/backups', exist_ok=True)
```

**3. Usar try-except para tratamento de erros:**
```python
try:
    # código que pode falhar
    result = some_function()
except Exception as e:
    logging.error(f"Erro ao executar função: {e}")
    # tratamento adicional
```

---

## 🤖 Problemas com MCP Servers

### Problema: MCP server não inicia

#### Sintomas
```
Connection refused
Port already in use
Module not found
```

#### Diagnóstico
```bash
# Verificar se porta está em uso
netstat -an | grep :8001

# Verificar logs do MCP
tail -f logs/mcp_server.log
```

#### Soluções

**1. Instalar dependências MCP:**
```bash
cd "🤖 AI_AGENTS/MCP_Code_Checker"
pip install -e .
```

**2. Verificar configuração Claude Code:**
```json
{
  "name": "code-checker",
  "command": "python",
  "args": ["-m", "mcp_code_checker"],
  "cwd": "/caminho/absoluto/EA_SCALPER_XAUUSD/🤖 AI_AGENTS/MCP_Code_Checker"
}
```

**3. Testar MCP manualmente:**
```bash
cd "🤖 AI_AGENTS/MCP_Code_Checker"
python -m mcp_code_checker --help
```

### Problema: MCP não responde no Claude Code

#### Sintomas
- Ferramentas MCP não aparecem
- Timeout ao usar MCP
- "Server not responding"

#### Soluções

**1. Verificar logs do Claude Code:**
- Abrir Settings → Logs
- Procurar erros de MCP

**2. Reiniciar Claude Code:**
- Fechar completamente
- Reabrir e reconectar

**3. Verificar firewall:**
```bash
# Permitir conexões locais
sudo ufw allow from 127.0.0.1 to any port 8001
```

---

## ⚡ Problemas de Performance

### Problema: Sistema lento

#### Sintomas
- Scripts demoram para executar
- MetaTrader com lag
- Alta utilização de CPU

#### Diagnóstico
```bash
# Verificar uso de CPU
top
# ou
htop

# Verificar uso de memória
free -h

# Verificar processos Python
ps aux | grep python
```

#### Soluções

**1. Otimizar scripts Python:**
```python
# Usar cache para requisições
import functools
import time

@functools.lru_cache(maxsize=128)
def cached_api_call(params):
    # código da API
    pass
```

**2. Limitar threads/processos:**
```python
import threading
MAX_THREADS = 4

threading.Semaphore(MAX_THREADS)
```

**3. Limpar logs e cache:**
```bash
# Remover logs antigos
find logs/ -name "*.log" -mtime +7 -delete

# Limpar cache Python
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Problema: Memory leaks

#### Sintomas
- Uso de memória crescente
- Sistema fica lento com tempo
- Scripts travam

#### Diagnóstico
```python
import tracemalloc

tracemalloc.start()

# Seu código aqui
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

#### Soluções

**1. Fechar conexões:**
```python
import MetaTrader5 as mt5

# Sempre fechar conexão
mt5.shutdown()

# Usar context manager
with mt5.connected():
    # código que usa mt5
    pass
```

**2. Liberar recursos:**
```python
# Limpar variáveis grandes
del large_variable
import gc
gc.collect()
```

---

## 🔄 Recuperação de Sistema

### Backup Automático de Emergência

```bash
#!/bin/bash
# backup_emergencia.sh

BACKUP_DIR="emergency_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup de configurações críticas
cp .env "$BACKUP_DIR/"
cp -r .roo/ "$BACKUP_DIR/"
cp -r logs/ "$BACKUP_DIR/"
cp -r 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MQL5_Source/ "$BACKUP_DIR/"

# Comprimir backup
tar -czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup de emergência criado: ${BACKUP_DIR}.tar.gz"
```

### Restauração do Sistema

```bash
#!/bin/bash
# restore_system.sh

if [ -z "$1" ]; then
    echo "Uso: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE=$1
RESTORE_DIR="restore_$(date +%Y%m%d_%H%M%S)"

# Criar diretório de restauração
mkdir -p "$RESTORE_DIR"

# Extrair backup
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Restaurar configurações
cp "$RESTORE_DIR"*/.env .
cp -r "$RESTORE_DIR"*/.roo/ .roo/

# Reinstalar dependências
source venv/bin/activate
pip install -r requirements.txt --force-reinstall

echo "Sistema restaurado do backup: $BACKUP_FILE"
```

### Reset Completo do Sistema

```bash
#!/bin/bash
# reset_completo.sh

echo "⚠️ ATENÇÃO: Isso irá resetar todo o sistema!"
read -p "Continuar? (s/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    # Parar todos os serviços
    pkill -f python
    pkill -f simple_trading_proxy

    # Remover ambiente virtual
    rm -rf venv/

    # Limpar logs
    rm -rf logs/*

    # Criar novo ambiente
    python3 -m venv venv
    source venv/bin/activate

    # Reinstalar dependências
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "Sistema resetado com sucesso!"
fi
```

---

## 🛠️ Ferramentas de Diagnóstico

### Script de Diagnóstico Completo

```python
#!/usr/bin/env python3
# diagnosticar_sistema.py

import os
import sys
import subprocess
import platform
import json
from datetime import datetime

def check_python():
    """Verificar instalação Python"""
    print("🐍 Verificando Python...")
    print(f"Versão: {sys.version}")
    print(f"Caminho: {sys.executable}")

    try:
        import virtualenv
        print("✅ virtualenv disponível")
    except ImportError:
        print("❌ virtualenv não encontrado")

def check_environment():
    """Verificar ambiente virtual"""
    print("\n🔧 Verificando ambiente virtual...")

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Ambiente virtual ativo")
        print(f"Prefixo: {sys.prefix}")
    else:
        print("❌ Ambiente virtual não está ativo")

def check_dependencies():
    """Verificar dependências principais"""
    print("\n📦 Verificando dependências...")

    required_packages = [
        'httpx', 'python-dotenv', 'mcp', 'pylint',
        'pytest', 'structlog', 'pathspec'
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} não encontrado")

def check_env_file():
    """Verificar arquivo .env"""
    print("\n🔑 Verificando .env...")

    if os.path.exists('.env'):
        print("✅ Arquivo .env encontrado")

        from dotenv import load_dotenv
        load_dotenv()

        required_vars = ['OPENROUTER_API_KEY']
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print(f"✅ {var}: {'*' * (len(value) - 4)}{value[-4:]}")
            else:
                print(f"❌ {var} não configurado")
    else:
        print("❌ Arquivo .env não encontrado")

def check_metatrader():
    """Verificar MetaTrader"""
    print("\n💱 Verificando MetaTrader...")

    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            print("✅ MetaTrader 5 conectado")
            account = mt5.account_info()
            print(f"Conta: {account.login}")
            print(f"Servidor: {account.server}")
            mt5.shutdown()
        else:
            print("❌ Não foi possível conectar ao MetaTrader 5")
    except ImportError:
        print("❌ MetaTrader5 não instalado")
    except Exception as e:
        print(f"❌ Erro ao verificar MetaTrader: {e}")

def check_directories():
    """Verificar estrutura de diretórios"""
    print("\n📁 Verificando estrutura de diretórios...")

    required_dirs = [
        'logs', 'data', 'temp', 'scripts', '📚', '🤖', '🔧'
    ]

    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}")
        else:
            print(f"❌ {dir_name} não encontrado")

def check_ports():
    """Verificar portas em uso"""
    print("\n🌐 Verificando portas...")

    ports = [4000, 8001]  # Proxy, MCP

    for port in ports:
        try:
            result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
            if f":{port}" in result.stdout:
                print(f"⚠️ Porta {port} em uso")
            else:
                print(f"✅ Porta {port} livre")
        except:
            print(f"❓ Não foi possível verificar porta {port}")

def main():
    """Função principal"""
    print("=" * 50)
    print("🔍 DIAGNÓSTICO DO SISTEMA EA_SCALPER_XAUUSD")
    print("=" * 50)
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Arquitetura: {platform.machine()}")
    print("=" * 50)

    check_python()
    check_environment()
    check_dependencies()
    check_env_file()
    check_metatrader()
    check_directories()
    check_ports()

    print("\n" + "=" * 50)
    print("🏁 Diagnóstico concluído!")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

### Script de Teste de Conectividade

```python
#!/usr/bin/env python3
# testar_conexoes.py

import requests
import time
import os
from dotenv import load_dotenv

def test_openrouter():
    """Testar conexão com OpenRouter"""
    print("🔗 Testando OpenRouter...")

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("❌ API Key não configurada")
        return False

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            models = response.json().get('data', [])
            print(f"✅ Conectado! {len(models)} modelos disponíveis")
            return True
        else:
            print(f"❌ Erro {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Erro de conexão: {e}")
        return False

def test_proxy():
    """Testar proxy local"""
    print("\n🌐 Testando proxy local...")

    try:
        response = requests.get(
            "http://localhost:4000/health",
            timeout=5
        )

        if response.status_code == 200:
            print("✅ Proxy respondendo")
            return True
        else:
            print(f"❌ Proxy retornou erro {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Proxy não está rodando")
        return False
    except Exception as e:
        print(f"❌ Erro ao testar proxy: {e}")
        return False

def main():
    print("🧪 TESTES DE CONECTIVIDADE")
    print("=" * 30)

    openrouter_ok = test_openrouter()
    proxy_ok = test_proxy()

    print("\n" + "=" * 30)
    if openrouter_ok and proxy_ok:
        print("✅ Todos os testes passaram!")
    else:
        print("❌ Alguns testes falharam - verifique configuração")
    print("=" * 30)

if __name__ == "__main__":
    main()
```

Este guia abrange os problemas mais comuns e suas soluções, fornecendo ferramentas para diagnóstico e recuperação do sistema quando necessário.