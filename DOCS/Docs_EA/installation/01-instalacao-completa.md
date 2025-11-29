# Guia Completo de Instalação - EA_SCALPER_XAUUSD

## 📋 Índice
1. [Visão Geral do Sistema](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação no Windows](#instalação-windows)
4. [Instalação no Linux](#instalação-linux)
5. [Instalação no macOS](#instalação-macos)
6. [Configuração do MetaTrader](#configuração-metatrader)
7. [Verificação da Instalação](#verificação-instalação)
8. [Próximos Passos](#proximos-passos)

---

## 🎯 Visão Geral do Sistema

O EA_SCALPER_XAUUSD é um sistema completo de trading automatizado que inclui:

- **Especialistas Advisors (EAs)** para MetaTrader 4/5
- **Sistema Multi-Agente** com IA para análise de mercado
- **Proxy Server** para integração com OpenRouter
- **Ferramentas de Classificação** e organização de código
- **MCP Servers** para integração com Claude Code
- **Scripts de Backup** e automação

---

## 📚 Pré-requisitos Detalhados

### Sistema Operacional
- **Windows 10/11** (Recomendado para MetaTrader)
- **Ubuntu 20.04+** ou **Debian 11+**
- **macOS 11+** (com algumas limitações)

### Software Essencial

#### Python (Obrigatório)
- **Versão**: 3.11+ (recomendado 3.13)
- **Por quê**: Scripts de automação, servidores MCP, proxies

```bash
# Verificar instalação
python --version
# ou
python3 --version
```

#### Node.js (Opcional)
- **Versão**: 18+ (LTS recomendado)
- **Por quê**: Algumas ferramentas de frontend e scripts

```bash
# Verificar instalação
node --version
npm --version
```

#### Git (Obrigatório)
- **Versão**: 2.30+
- **Por quê**: Controle de versão do projeto

```bash
# Verificar instalação
git --version
```

#### MetaTrader (Obrigatório para Trading)
- **MetaTrader 5** (recomendado) ou **MetaTrader 4**
- **Por quê**: Execução dos EAs de trading

### Hardware Mínimo

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| Processador | 2 núcleos | 4+ núcleos |
| Memória RAM | 4 GB | 8+ GB |
| Armazenamento | 10 GB livres | 20+ GB livres |
| Rede | 1 Mbps | 10+ Mbps |

---

## 🪟 Instalação no Windows

### Passo 1: Instalar Python

1. **Baixe o Python**:
   - Acesse: https://www.python.org/downloads/
   - Baixe a versão 3.11+ (recomendado 3.13)

2. **Instale o Python**:
   - Execute o instalador
   - **CRUCIAL**: Marque "Add Python to PATH"
   - Selecione "Install for all users" (opcional)

3. **Verifique a instalação**:
   ```cmd
   python --version
   pip --version
   ```

### Passo 2: Instalar Git

1. **Baixe o Git**:
   - Acesse: https://git-scm.com/download/win
   - Baixe o instalador

2. **Instale o Git**:
   - Execute o instalador
   - Aceite as configurações padrão
   - Selecione "Use Git from the Windows Command Prompt"

3. **Verifique a instalação**:
   ```cmd
   git --version
   ```

### Passo 3: Instalar Node.js (Opcional)

1. **Baixe o Node.js**:
   - Acesse: https://nodejs.org/
   - Baixe a versão LTS

2. **Instale o Node.js**:
   - Execute o instalador
   - Aceite as configurações padrão

3. **Verifique a instalação**:
   ```cmd
   node --version
   npm --version
   ```

### Passo 4: Instalar MetaTrader 5

1. **Baixe o MT5**:
   - Acesse o site da sua corretora
   - Ou baixe diretamente do site da MetaQuotes

2. **Instale o MT5**:
   - Execute o instalador
   - Configure sua conta demo ou real

### Passo 5: Clonar o Projeto

1. **Abra o PowerShell ou CMD**:
   ```cmd
   # Navegue para o diretório desejado
   cd C:\Projetos

   # Clone o repositório
   git clone https://github.com/seu-usuario/EA_SCALPER_XAUUSD.git
   cd EA_SCALPER_XAUUSD
   ```

### Passo 6: Configurar Ambiente Virtual

1. **Crie o ambiente virtual**:
   ```cmd
   python -m venv venv

   # Ative o ambiente virtual
   venv\Scripts\activate
   ```

2. **Instale as dependências**:
   ```cmd
   # Upgrade do pip
   python -m pip install --upgrade pip

   # Instale as dependências básicas
   pip install httpx python-dotenv mcp pylint pytest pytest-json-report

   # Instale dependências adicionais
   pip install structlog pathspec pytest-asyncio mypy
   ```

### Passo 7: Configurar Variáveis de Ambiente

1. **Copie o arquivo .env**:
   ```cmd
   copy .env.example .env
   ```

2. **Edite o arquivo .env**:
   - Abra o Bloco de Notas ou VS Code
   - Configure suas chaves de API:
   ```env
   OPENROUTER_API_KEY=sua_chave_api_aqui
   DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
   ```

### Passo 8: Executar Scripts Windows

1. **Execute o script de configuração**:
   ```cmd
   # Via PowerShell
   .\scripts\windows\setup_environment.ps1

   # Ou via CMD
   .\scripts\windows\setup_environment.bat
   ```

2. **Se tiver problemas com execution policy**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

---

## 🐧 Instalação no Linux

### Passo 1: Atualizar o Sistema

```bash
# Para sistemas Debian/Ubuntu
sudo apt update && sudo apt upgrade -y

# Para sistemas Fedora/RHEL
sudo dnf update -y
```

### Passo 2: Instalar Python e Ferramentas

```bash
# Debian/Ubuntu
sudo apt install -y python3 python3-pip python3-venv git curl

# Fedora/RHEL
sudo dnf install -y python3 python3-pip git curl
```

### Passo 3: Instalar Node.js (Opcional)

```bash
# Via NodeSource (recomendado)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Ou via snap
sudo snap install node --classic
```

### Passo 4: Verificar Instalações

```bash
python3 --version
pip3 --version
git --version
node --version  # se instalado
```

### Passo 5: Clonar o Projeto

```bash
# Navegue para o diretório desejado
cd ~/Projetos

# Clone o repositório
git clone https://github.com/seu-usuario/EA_SCALPER_XAUUSD.git
cd EA_SCALPER_XAUUSD
```

### Passo 6: Configurar Ambiente Virtual

```bash
# Crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente virtual
source venv/bin/activate

# Upgrade do pip
python -m pip install --upgrade pip

# Instale as dependências
pip install httpx python-dotenv mcp pylint pytest pytest-json-report
pip install structlog pathspec pytest-asyncio mypy
```

### Passo 7: Configurar Variáveis de Ambiente

```bash
# Copie o arquivo .env
cp .env.example .env

# Edite o arquivo
nano .env
```

Configure suas chaves de API:
```env
OPENROUTER_API_KEY=sua_chave_api_aqui
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
```

### Passo 8: MetaTrader no Linux (Opcional)

O MetaTrader pode ser executado no Linux via Wine:

```bash
# Instale o Wine
sudo apt install -y wine64 wine32

# Configure o Wine
winecfg

# Instale o MetaTrader 5
wine mt5setup.exe
```

---

## 🍎 Instalação no macOS

### Passo 1: Instalar Homebrew

```bash
# Instale o Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Adicione o Homebrew ao PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Passo 2: Instalar Python e Ferramentas

```bash
# Instale Python 3.11+
brew install python@3.11

# Instale Git
brew install git

# Instale Node.js (opcional)
brew install node
```

### Passo 3: Verificar Instalações

```bash
python3.11 --version
pip3.11 --version
git --version
node --version  # se instalado
```

### Passo 4: Clonar o Projeto

```bash
# Navegue para o diretório desejado
cd ~/Projetos

# Clone o repositório
git clone https://github.com/seu-usuario/EA_SCALPER_XAUUSD.git
cd EA_SCALPER_XAUUSD
```

### Passo 5: Configurar Ambiente Virtual

```bash
# Crie o ambiente virtual
python3.11 -m venv venv

# Ative o ambiente virtual
source venv/bin/activate

# Upgrade do pip
python -m pip install --upgrade pip

# Instale as dependências
pip install httpx python-dotenv mcp pylint pytest pytest-json-report
pip install structlog pathspec pytest-asyncio mypy
```

### Passo 6: Configurar Variáveis de Ambiente

```bash
# Copie o arquivo .env
cp .env.example .env

# Edite o arquivo
nano .env
```

### Passo 7: MetaTrader no macOS

O MetaTrader pode ser instalado:
- Via Parallels Desktop (recomendado)
- Via CrossOver
- Via Boot Camp

---

## 💻 Configuração do MetaTrader

### Passo 1: Configurar Pasta de Dados

1. **Abra o MetaTrader 5**
2. **Vá em**: Arquivo → Abrir Pasta de Dados
3. **Navegue até**: MQL5 → Experts
4. **Copie os EAs** do projeto para esta pasta

### Passo 2: Habilitar Trading Automático

1. **No MetaTrader**:
   - Pressione F6 ou clique em "AutoTrading"
   - Certifique-se que o botão está verde
   - Verifique as permissões na aba "Ferramentas → Opções"

2. **Configure as opções**:
   ```
   Aba Expert Advisors:
   ✓ Permitir trading automatizado
   ✓ Permitir DLL imports
   ```

### Passo 3: Configurar Compilação

1. **Abra o MetaEditor** (F4)
2. **Compile os EAs**:
   - Abra cada arquivo .mq5
   - Pressione F7 ou clique em "Compile"
   - Verifique se não há erros

---

## ✅ Verificação da Instalação

### Teste 1: Verificar Ambiente Python

```bash
# Ative o ambiente virtual
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows

# Execute o script de verificação
python scripts/python/quick_test.py
```

### Teste 2: Verificar Proxy Server

```bash
# Inicie o proxy
python scripts/python/simple_trading_proxy.py

# Em outro terminal, teste o health check
curl http://localhost:4000/health
```

### Teste 3: Verificar MCP Servers

```bash
# Teste os servidores MCP
python -m pytest 🤖\ AI_AGENTS/MCP_Code_Checker/tests/ -v
```

### Teste 4: Verificar EAs no MetaTrader

1. **Abra o MetaTrader 5**
2. **Navegue até**: Navigator → Expert Advisors
3. **Verifique se os EAs aparecem na lista**
4. **Arraste um EA para um gráfico para teste**

### Teste 5: Verificar Scripts de Automatização

```bash
# Execute o script de classificação
python 🔧\ WORKSPACE/Development/Core/classificador_qualidade_maxima.py

# Execute o script de backup
python 🔧\ WORKSPACE/Development/Scripts/git_auto_backup.py
```

---

## 🚀 Próximos Passos

Após a instalação bem-sucedida:

1. **Leia o Guia de Configuração Inicial**: `/docs/installation/02-configuracao-inicial.md`
2. **Siga o Quick Start Guide**: `/docs/installation/06-quick-start.md`
3. **Estude a Documentação Completa**: Verifique os arquivos em `📋 DOCUMENTACAO_FINAL/`

---

## ❗ Solução de Problemas Comuns

### Problema: "python não reconhecido"
**Solução**: Adicione o Python ao PATH ou use `python3`

### Problema: "pip command not found"
**Solução**: Use `python -m pip` ou reinstale o Python

### Problema: Erro de permissão no Windows
**Solução**: Execute PowerShell como Administrador

### Problema: Proxy não inicia
**Solução**: Verifique se a porta 4000 está livre

### Problema: EAs não aparecem no MetaTrader
**Solução**: Verifique o caminho da pasta MQL5/Experts

---

## 📞 Suporte

Se encontrar problemas durante a instalação:

1. **Verifique os logs**: `logs/`
2. **Consulte o troubleshooting**: `/docs/installation/05-troubleshooting.md`
3. **Abra uma issue**: No repositório GitHub

---

## 📝 Checklist de Instalação

- [ ] Python 3.11+ instalado
- [ ] Git instalado
- [ ] Ambiente virtual criado
- [ ] Dependências Python instaladas
- [ ] Arquivo .env configurado
- [ ] MetaTrader instalado (se aplicável)
- [ ] EAs compilados
- [ ] Proxy server testado
- [ ] Scripts básicos executados
- [ ] Testes de verificação passaram

**Parabéns! Seu sistema EA_SCALPER_XAUUSD está pronto para uso.** 🎉