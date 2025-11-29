# Guia de Configuração de APIs - EA_SCALPER_XAUUSD

## Overview

Este documento fornece um guia completo para configurar e gerenciar as APIs utilizadas no projeto EA_SCALPER_XAUUSD, incluindo OpenRouter, LiteLLM, MCP (Model Context Protocol) e outras integrações.

## Sumário

1. [OpenRouter API](#openrouter-api)
2. [LiteLLM Configuration](#litellm-configuration)
3. [MCP (Model Context Protocol)](#mcp-model-context-protocol)
4. [GitHub Integration](#github-integration)
5. [Notification APIs](#notification-apis)
6. [Security Best Practices](#security-best-practices)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)

---

## OpenRouter API

### Visão Geral

OpenRouter é um gateway de API que fornece acesso a múltiplos modelos de linguagem através de uma interface unificada.

### Configuração Inicial

#### 1. Obter Chave de API

```bash
# Passo 1: Acesse https://openrouter.ai/
# Passo 2: Crie uma conta ou faça login
# Passo 3: Vá para Settings > API Keys
# Passo 4: Crie uma nova chave
# Passo 5: Copie a chave para seu .env
```

#### 2. Configurar Variáveis de Ambiente

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-1234567890abcdef1234567890abcdef1234567890abcdef1234567890
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD_Trading_Bot"
OPENROUTER_SITE_URL="https://github.com/seu-usuario/EA_SCALPER_XAUUSD"
```

#### 3. Configuração LiteLLM com OpenRouter

```python
# setup_openrouter.py
import os
from dotenv import load_dotenv
import litellm

def setup_openrouter():
    """Configura LiteLLM para usar OpenRouter"""

    # Carregar variáveis de ambiente
    load_dotenv()

    # Configurar API
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não encontrada")

    litellm.api_key = api_key
    litellm.api_base = "https://openrouter.ai/api/v1"

    # Configurar headers personalizados
    litellm.set_verbose = True  # Para debug

    return True

def test_openrouter_connection():
    """Testa conexão com OpenRouter"""
    try:
        response = litellm.completion(
            model="openrouter/anthropic/claude-3-5-sonnet",
            messages=[
                {"role": "user", "content": "Responda apenas 'OK' para teste"}
            ],
            max_tokens=5
        )
        print("✅ Conexão OpenRouter bem-sucedida")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

if __name__ == "__main__":
    if setup_openrouter():
        test_openrouter_connection()
```

### Modelos Suportados

#### Modelos Principais

| Modelo | Provider | Uso Recomendado | Cost (1M tokens) |
|--------|----------|------------------|------------------|
| claude-3-5-sonnet | Anthropic | Análise complexa | $15.00 |
| gpt-4o | OpenAI | Geral | $5.00 |
| claude-3-opus | Anthropic | Alta qualidade | $75.00 |
| gemini-pro | Google | Análise de dados | $0.50 |

#### Configuração de Modelos

```python
# model_config.py
DEFAULT_MODELS = {
    "analysis": "openrouter/anthropic/claude-3-5-sonnet",
    "generation": "openrouter/openai/gpt-4o",
    "validation": "openrouter/anthropic/claude-3-haiku",
    "trading": "openrouter/anthropic/claude-3-5-sonnet"
}

MODEL_LIMITS = {
    "openrouter/anthropic/claude-3-5-sonnet": {
        "max_tokens": 4096,
        "temperature": 0.1,
        "context_window": 200000
    },
    "openrouter/openai/gpt-4o": {
        "max_tokens": 4096,
        "temperature": 0.2,
        "context_window": 128000
    }
}
```

### Rate Limiting e Caching

```python
# rate_limiter.py
import time
from typing import Dict
import litellm

class OpenRouterRateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = []
        self.last_reset = time.time()

    def wait_if_needed(self):
        """Aguarda se necessário para respeitar rate limit"""
        now = time.time()

        # Limpar requisições antigas (mais de 1 minuto)
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < 60]

        # Se atingiu o limite, aguardar
        if len(self.requests) >= self.rpm:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.requests.append(now)

def make_request_with_rate_limit(model: str, messages: list, **kwargs):
    """Faz requisição com rate limiting"""
    limiter = OpenRouterRateLimiter()
    limiter.wait_if_needed()

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print(f"Erro na requisição: {e}")
        return None
```

---

## LiteLLM Configuration

### Visão Geral

LiteLLM é uma biblioteca que simplifica o acesso a múltiplos modelos de linguagem através de uma interface unificada.

### Instalação e Setup

```bash
# Instalar LiteLLM
pip install litellm

# Para cache Redis
pip install litellm[redis]

# Para proxy server
pip install litellm[proxy]
```

### Configuração Básica

```python
# litellm_config.py
import litellm
from dotenv import load_dotenv
import os

class LiteLLMManager:
    def __init__(self):
        load_dotenv()
        self.setup_configuration()

    def setup_configuration(self):
        """Configura LiteLLM"""

        # Configurar OpenRouter como padrão
        litellm.api_key = os.getenv('OPENROUTER_API_KEY')
        litellm.api_base = os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')

        # Configurar cache
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            litellm.cache = litellm.Cache(type="redis", redis_url=redis_url)
        else:
            litellm.cache = litellm.Cache(type="local")

        # Configurações de proxy
        litellm.drop_params = True  # Remove parâmetros não suportados
        litellm.set_verbose = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    def get_model_config(self, model_name: str) -> dict:
        """Retorna configuração para um modelo específico"""
        configs = {
            "openrouter/anthropic/claude-3-5-sonnet": {
                "model": model_name,
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.95,
                "stream": False
            },
            "openrouter/openai/gpt-4o": {
                "model": model_name,
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.95,
                "stream": False
            }
        }
        return configs.get(model_name, {})

    async def create_completion(self, model: str, messages: list, **kwargs):
        """Cria completion com tratamento de erro"""
        try:
            config = self.get_model_config(model)
            config.update(kwargs)

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                **config
            )
            return response

        except Exception as e:
            print(f"Erro na completion: {e}")
            return None
```

### Configuração de Cache

```python
# cache_config.py
import litellm
import os
from dotenv import load_dotenv

def setup_cache():
    """Configura cache LiteLLM"""
    load_dotenv()

    cache_type = os.getenv('CACHE_TYPE', 'local')

    if cache_type == 'redis':
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            litellm.cache = litellm.Cache(
                type="redis",
                redis_url=redis_url,
                ttl=os.getenv('RESPONSE_CACHE_TTL', 1800)
            )
            print("✅ Cache Redis configurado")
        else:
            print("⚠️ REDIS_URL não encontrada, usando cache local")
            setup_local_cache()
    else:
        setup_local_cache()

def setup_local_cache():
    """Configura cache local"""
    litellm.cache = litellm.Cache(
        type="local",
        ttl=os.getenv('RESPONSE_CACHE_TTL', 1800)
    )
    print("✅ Cache local configurado")

# Uso
setup_cache()
```

### Proxy Server LiteLLM

```yaml
# config.yaml
model_list:
  - model_name: "claude-3-5-sonnet"
    litellm_params:
      model: "openrouter/anthropic/claude-3-5-sonnet"
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: "https://openrouter.ai/api/v1"

  - model_name: "gpt-4o"
    litellm_params:
      model: "openrouter/openai/gpt-4o"
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: "https://openrouter.ai/api/v1"

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
```

```bash
# Iniciar proxy server
litellm --config config.yaml --port 4000
```

---

## MCP (Model Context Protocol)

### Visão Geral

MCP é um protocolo para conectar modelos de linguagem com ferramentas externas, incluindo integração com GitHub e outros serviços.

### Configuração GitHub MCP

#### 1. Instalar Dependências

```bash
# Para MCP com Docker
docker pull ghcr.io/github/github-mcp-server:latest

# Para desenvolvimento local
npm install -g @modelcontextprotocol/server-github
```

#### 2. Configurar MCP JSON

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
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_1234567890abcdef1234567890abcdef12345678",
        "GITHUB_TOOLSETS": "coding,review,analysis",
        "GITHUB_READ_ONLY": "false"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/project"
      ]
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/path/to/repo"
      ]
    }
  }
}
```

#### 3. Integração com Python

```python
# mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.session = None

    async def connect(self):
        """Conecta ao servidor MCP"""
        # Configurações do servidor GitHub
        if self.server_name == "github":
            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run", "-i", "--rm",
                    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "-e", "GITHUB_TOOLSETS",
                    "ghcr.io/github/github-mcp-server"
                ],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
                    "GITHUB_TOOLSETS": "coding,review,analysis"
                }
            )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                return session

    async def call_tool(self, tool_name: str, arguments: dict):
        """Executa uma ferramenta MCP"""
        if not self.session:
            await self.connect()

        result = await self.session.call_tool(tool_name, arguments)
        return result

# Uso
async def main():
    client = MCPClient("github")
    await client.connect()

    # Listar repositórios
    repos = await client.call_tool("github_list_repositories", {})
    print(repos)

if __name__ == "__main__":
    asyncio.run(main())
```

### Toolsets GitHub MCP

```python
# github_tools.py
GITHUB_TOOLSETS = {
    "coding": [
        "github_create_or_update_file",
        "github_get_file_contents",
        "github_search_code",
        "github_create_pull_request",
        "github_merge_pull_request"
    ],
    "review": [
        "github_list_pull_requests",
        "github_get_pull_request",
        "github_create_review_comment",
        "github_request_pull_request_review"
    ],
    "analysis": [
        "github_get_repository",
        "github_list_commits",
        "github_get_commit",
        "github_search_repositories",
        "github_get_repository_languages"
    ]
}

def get_available_tools(toolset: str = None):
    """Retorna ferramentas disponíveis por toolset"""
    if toolset:
        return GITHUB_TOOLSETS.get(toolset, [])

    # Retorna todas as ferramentas
    all_tools = []
    for tools in GITHUB_TOOLSETS.values():
        all_tools.extend(tools)
    return list(set(all_tools))
```

---

## GitHub Integration

### Personal Access Token Setup

#### 1. Criar Token no GitHub

```bash
# Passo 1: Vá para GitHub > Settings > Developer settings > Personal access tokens
# Passo 2: Tokens (classic) > Generate new token
# Passo 3: Configure as permissões:
#    - repo (Full control of private repositories)
#    - read:org (Read org and team membership)
#    - workflow (Update GitHub Action workflows)
# Passo 4: Copie o token gerado
```

#### 2. Configurar Variáveis

```bash
# .env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_1234567890abcdef1234567890abcdef12345678
GITHUB_TOOLSETS="coding,review,analysis"
GITHUB_READ_ONLY=false
```

### GitHub API Client

```python
# github_client.py
import os
from dotenv import load_dotenv
import requests
from typing import Dict, List, Optional

class GitHubClient:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def get_repository(self, owner: str, repo: str) -> Dict:
        """Obtém informações do repositório"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else None

    def list_commits(self, owner: str, repo: str, branch: str = "main") -> List[Dict]:
        """Lista commits do repositório"""
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {"sha": branch, "per_page": 100}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json() if response.status_code == 200 else []

    def create_pull_request(self, owner: str, repo: str, title: str,
                          head: str, base: str, body: str = "") -> Dict:
        """Cria um Pull Request"""
        if os.getenv('GITHUB_READ_ONLY', 'false').lower() == 'true':
            raise PermissionError("Modo read-only ativado")

        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json() if response.status_code == 201 else None

    def search_code(self, query: str, repo: str = None) -> List[Dict]:
        """Pesquisa código"""
        url = f"{self.base_url}/search/code"
        params = {"q": query}
        if repo:
            params["q"] += f" repo:{repo}"

        response = requests.get(url, headers=self.headers, params=params)
        return response.json().get("items", []) if response.status_code == 200 else []

# Uso
client = GitHubClient()
repo_info = client.get_repository("seu-usuario", "EA_SCALPER_XAUUSD")
commits = client.list_commits("seu-usuario", "EA_SCALPER_XAUUSD")
```

---

## Notification APIs

### Telegram Bot Setup

#### 1. Criar Bot no Telegram

```bash
# Passo 1: Converse com @BotFather no Telegram
# Passo 2: /newbot para criar um novo bot
# Passo 3: Siga as instruções para nomear e configurar
# Passo 4: Copie o token fornecido
# Passo 5: Obtenha seu chat ID conversando com @userinfobot
```

#### 2. Configurar Variáveis

```bash
# .env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

#### 3. Implementação do Cliente

```python
# telegram_client.py
import os
from dotenv import load_dotenv
import requests
import json

class TelegramClient:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envia mensagem para o Telegram"""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }

        try:
            response = requests.post(url, json=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Erro ao enviar mensagem Telegram: {e}")
            return False

    def send_trade_alert(self, symbol: str, action: str, price: float,
                        profit: float = None):
        """Envia alerta de trade"""
        message = f"""
🚀 <b>Alerta de Trade</b>

📊 <b>Ativo:</b> {symbol}
📈 <b>Ação:</b> {action}
💰 <b>Preço:</b> ${price:.2f}
        """

        if profit is not None:
            profit_emoji = "📈" if profit > 0 else "📉"
            message += f"\n{profit_emoji} <b>Profit:</b> ${profit:.2f}"

        self.send_message(message)

    def send_system_alert(self, level: str, message: str):
        """Envia alerta do sistema"""
        emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }

        emoji = emojis.get(level, "ℹ️")
        formatted_message = f"{emoji} <b>{level}</b>\n\n{message}"

        self.send_message(formatted_message)

# Uso
telegram = TelegramClient()
telegram.send_trade_alert("XAUUSD", "BUY", 2650.50, 25.30)
```

### Discord Webhook

```python
# discord_client.py
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

class DiscordClient:
    def __init__(self):
        load_dotenv()
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

    def send_embed(self, title: str, description: str, color: int = 0x00ff00):
        """Envia mensagem embed para Discord"""
        if not self.webhook_url:
            return False

        data = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }

        try:
            response = requests.post(self.webhook_url, json=data)
            return response.status_code == 204
        except Exception as e:
            print(f"Erro ao enviar mensagem Discord: {e}")
            return False

    def send_trade_notification(self, symbol: str, action: str, price: float,
                              profit: float = None):
        """Envia notificação de trade"""
        title = f"🚀 Trade Alert - {symbol}"
        description = f"**Action:** {action}\n**Price:** ${price:.2f}"

        if profit is not None:
            profit_color = 0x00ff00 if profit > 0 else 0xff0000
            description += f"\n**Profit:** ${profit:.2f}"
        else:
            profit_color = 0x0099ff

        self.send_embed(title, description, profit_color)

# Uso
discord = DiscordClient()
discord.send_trade_notification("XAUUSD", "SELL", 2651.75, -10.25)
```

---

## Security Best Practices

### 1. Gerenciamento de Chaves

```python
# security_manager.py
import os
import hashlib
import secrets
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self):
        self.encryption_key = self.load_or_generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def load_or_generate_key(self):
        """Carrega ou gera chave de criptografia"""
        key_file = "encryption.key"

        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_api_key(self, api_key: str) -> str:
        """Criptografa chave de API"""
        return self.cipher_suite.encrypt(api_key.encode()).decode()

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Descriptografa chave de API"""
        return self.cipher_suite.decrypt(encrypted_key.encode()).decode()

    def validate_api_key_format(self, key: str, provider: str) -> bool:
        """Valida formato da chave de API"""
        formats = {
            "openrouter": lambda k: k.startswith("sk-or-v1-") and len(k) >= 40,
            "github": lambda k: k.startswith("ghp_") and len(k) >= 40,
            "openai": lambda k: k.startswith("sk-") and len(k) >= 40,
            "telegram": lambda k: ":" in k and len(k) >= 35
        }

        validator = formats.get(provider.lower())
        return validator(key) if validator else False

    def rotate_api_key(self, current_key: str, new_key: str) -> bool:
        """Rotaciona chave de API"""
        if not self.validate_api_key_format(new_key, "openrouter"):
            return False

        # Lógica de rotação
        encrypted_current = self.encrypt_api_key(current_key)
        encrypted_new = self.encrypt_api_key(new_key)

        # Salvar em local seguro
        self.save_key_backup(encrypted_current, encrypted_new)

        return True

# Uso
security = SecurityManager()
encrypted_key = security.encrypt_api_key("sk-or-v1-123456...")
decrypted_key = security.decrypt_api_key(encrypted_key)
```

### 2. Rate Limiting e Throttling

```python
# rate_limiter.py
import time
import asyncio
from typing import Dict, List
from collections import defaultdict, deque

class AdvancedRateLimiter:
    def __init__(self):
        self.windows = defaultdict(deque)  # Janelas de tempo por endpoint
        self.limits = {
            "openrouter": {"requests": 60, "window": 60},  # 60 requests/minuto
            "github": {"requests": 5000, "window": 3600},  # 5000 requests/hora
            "telegram": {"requests": 30, "window": 60}     # 30 requests/minuto
        }

    async def acquire(self, endpoint: str):
        """Adquire permissão para fazer requisição"""
        limit = self.limits.get(endpoint, {"requests": 60, "window": 60})

        now = time.time()
        window_start = now - limit["window"]

        # Limpar requisições antigas
        while self.windows[endpoint] and self.windows[endpoint][0] < window_start:
            self.windows[endpoint].popleft()

        # Verificar limite
        if len(self.windows[endpoint]) >= limit["requests"]:
            # Calcular tempo de espera
            oldest_request = self.windows[endpoint][0]
            wait_time = limit["window"] - (now - oldest_request)

            if wait_time > 0:
                print(f"Rate limit atingido para {endpoint}. Aguardando {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                return await self.acquire(endpoint)  # Tentar novamente

        # Registrar requisição
        self.windows[endpoint].append(now)
        return True

    def get_remaining_requests(self, endpoint: str) -> int:
        """Retorna número de requisições restantes"""
        limit = self.limits.get(endpoint, {"requests": 60, "window": 60})
        now = time.time()
        window_start = now - limit["window"]

        # Contar requisições na janela atual
        recent_requests = sum(1 for req_time in self.windows[endpoint]
                            if req_time >= window_start)

        return max(0, limit["requests"] - recent_requests)

# Uso
rate_limiter = AdvancedRateLimiter()

async def make_api_request(endpoint: str, request_func):
    """Faz requisição com rate limiting"""
    await rate_limiter.acquire(endpoint)
    remaining = rate_limiter.get_remaining_requests(endpoint)

    if remaining < 5:
        print(f"⚠️ Apenas {remaining} requisições restantes para {endpoint}")

    return await request_func()
```

### 3. Audit Logging

```python
# audit_logger.py
import json
import os
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.ensure_log_directory()

    def ensure_log_directory(self):
        """Garante que diretório de logs existe"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log_api_call(self, endpoint: str, method: str, status: int,
                    response_time: float, user_id: str = None):
        """Registra chamada de API"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "api_call",
            "endpoint": endpoint,
            "method": method,
            "status_code": status,
            "response_time_ms": round(response_time * 1000, 2),
            "user_id": user_id
        }

        self.write_log(entry)

    def log_configuration_change(self, component: str, old_value: Any,
                               new_value: Any, user_id: str = None):
        """Registra mudança de configuração"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "config_change",
            "component": component,
            "old_value": self.sanitize_value(old_value),
            "new_value": self.sanitize_value(new_value),
            "user_id": user_id
        }

        self.write_log(entry)

    def log_security_event(self, event_type: str, details: Dict,
                          severity: str = "INFO"):
        """Registra evento de segurança"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "security",
            "security_event": event_type,
            "severity": severity,
            "details": details
        }

        self.write_log(entry)

    def sanitize_value(self, value: Any) -> str:
        """Sanitiza valores sensíveis para logging"""
        if isinstance(value, str) and any(keyword in value.lower()
                                        for keyword in ["key", "token", "secret"]):
            return "***REDACTED***"
        return str(value)

    def write_log(self, entry: Dict):
        """Escreve entrada no log"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Erro ao escrever audit log: {e}")

# Uso
audit = AuditLogger()
audit.log_api_call("/api/v1/chat", "POST", 200, 1.23, "user_123")
audit.log_security_event("api_key_rotation", {"user": "admin"}, "INFO")
```

---

## Testing and Validation

### 1. Suite de Testes de API

```python
# test_apis.py
import asyncio
import pytest
import os
from dotenv import load_dotenv

from litellm_config import LiteLLMManager
from github_client import GitHubClient
from telegram_client import TelegramClient
from rate_limiter import AdvancedRateLimiter

class APITestSuite:
    def __init__(self):
        load_dotenv()
        self.results = {}

    async def test_openrouter_connection(self):
        """Testa conexão com OpenRouter"""
        try:
            manager = LiteLLMManager()
            response = await manager.create_completion(
                model="openrouter/anthropic/claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )

            if response and response.choices:
                self.results["openrouter"] = "✅ PASS"
                return True
            else:
                self.results["openrouter"] = "❌ FAIL - No response"
                return False

        except Exception as e:
            self.results["openrouter"] = f"❌ FAIL - {str(e)}"
            return False

    def test_github_connection(self):
        """Testa conexão com GitHub"""
        try:
            client = GitHubClient()
            user_data = requests.get("https://api.github.com/user",
                                   headers=client.headers)

            if user_data.status_code == 200:
                self.results["github"] = "✅ PASS"
                return True
            else:
                self.results["github"] = f"❌ FAIL - {user_data.status_code}"
                return False

        except Exception as e:
            self.results["github"] = f"❌ FAIL - {str(e)}"
            return False

    def test_telegram_connection(self):
        """Testa conexão com Telegram"""
        try:
            client = TelegramClient()
            success = client.send_message("🧪 Test message from EA_SCALPER_XAUUSD")

            if success:
                self.results["telegram"] = "✅ PASS"
                return True
            else:
                self.results["telegram"] = "❌ FAIL - Message not sent"
                return False

        except Exception as e:
            self.results["telegram"] = f"❌ FAIL - {str(e)}"
            return False

    async def test_rate_limiting(self):
        """Testa rate limiting"""
        try:
            limiter = AdvancedRateLimiter()

            # Fazer múltiplas requisições rápidas
            start_time = time.time()
            for i in range(5):
                await limiter.acquire("openrouter")

            elapsed = time.time() - start_time

            if elapsed > 0.1:  # Deve haver algum atraso
                self.results["rate_limiting"] = "✅ PASS"
                return True
            else:
                self.results["rate_limiting"] = "❌ FAIL - No delay detected"
                return False

        except Exception as e:
            self.results["rate_limiting"] = f"❌ FAIL - {str(e)}"
            return False

    async def run_all_tests(self):
        """Executa todos os testes"""
        print("🧪 Iniciando suite de testes de APIs...")

        tests = [
            ("OpenRouter", self.test_openrouter_connection),
            ("GitHub", self.test_github_connection),
            ("Telegram", self.test_telegram_connection),
            ("Rate Limiting", self.test_rate_limiting)
        ]

        for test_name, test_func in tests:
            print(f"\n📋 Testando {test_name}...")

            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()

        self.print_results()

    def print_results(self):
        """Imprime resultados dos testes"""
        print("\n" + "="*50)
        print("📊 RESULTADOS DOS TESTES")
        print("="*50)

        for test_name, result in self.results.items():
            print(f"{test_name.title():<15} | {result}")

        passed = sum(1 for result in self.results.values() if "✅" in result)
        total = len(self.results)

        print(f"\nTotal: {passed}/{total} testes passaram")

        if passed == total:
            print("🎉 Todos os testes passaram!")
        else:
            print("⚠️ Alguns testes falharam. Verifique a configuração.")

# Uso
async def main():
    test_suite = APITestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Health Check API

```python
# health_check.py
from flask import Flask, jsonify
import asyncio
import threading
from datetime import datetime

app = Flask(__name__)

class HealthChecker:
    def __init__(self):
        self.status = {}
        self.last_check = None

    async def check_all_services(self):
        """Verifica status de todos os serviços"""
        test_suite = APITestSuite()
        await test_suite.run_all_tests()

        self.status = test_suite.results
        self.last_check = datetime.utcnow()

        return self.status

    def get_health_status(self):
        """Retorna status geral de saúde"""
        if not self.status:
            return {
                "status": "unknown",
                "timestamp": None,
                "services": {}
            }

        all_passed = all("✅" in result for result in self.status.values())

        return {
            "status": "healthy" if all_passed else "degraded",
            "timestamp": self.last_check.isoformat() if self.last_check else None,
            "services": self.status
        }

health_checker = HealthChecker()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    return jsonify(health_checker.get_health_status())

@app.route('/health/check', methods=['POST'])
def trigger_health_check():
    """Dispara verificação de saúde"""
    # Executar em background
    threading.Thread(
        target=asyncio.run,
        args=(health_checker.check_all_services(),)
    ).start()

    return jsonify({
        "message": "Health check initiated",
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Troubleshooting

### Problemas Comuns e Soluções

#### 1. Erro de Autenticação OpenRouter

**Erro:** `401 Unauthorized - Invalid API key`

**Causas Possíveis:**
- Chave de API incorreta
- Chave expirada
- Formato inválido

**Solução:**
```python
# debug_openrouter.py
import os
from dotenv import load_dotenv
import litellm

def debug_openrouter_auth():
    """Debug autenticação OpenRouter"""
    load_dotenv()

    api_key = os.getenv('OPENROUTER_API_KEY')
    print(f"API Key encontrada: {'Sim' if api_key else 'Não'}")

    if api_key:
        print(f"Comprimento da chave: {len(api_key)}")
        print(f"Prefixo: {api_key[:10]}...")
        print(f"Formato válido: {api_key.startswith('sk-or-v1-')}")

        # Teste com requisição simples
        try:
            litellm.api_key = api_key
            litellm.api_base = "https://openrouter.ai/api/v1"

            response = litellm.completion(
                model="openrouter/anthropic/claude-3-5-sonnet",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("✅ Autenticação bem-sucedida")

        except Exception as e:
            print(f"❌ Erro na autenticação: {e}")
    else:
        print("❌ OPENROUTER_API_KEY não encontrada no .env")

debug_openrouter_auth()
```

#### 2. Problemas de Conexão Redis

**Erro:** `Redis connection failed`

**Solução:**
```bash
# Verificar se Redis está rodando
redis-cli ping

# Iniciar Redis se necessário
redis-server

# Verificar configuração
redis-cli config get "*"
```

```python
# debug_redis.py
import redis
import os
from dotenv import load_dotenv

def debug_redis_connection():
    """Debug conexão Redis"""
    load_dotenv()

    redis_url = os.getenv('REDIS_URL')
    print(f"Redis URL: {redis_url}")

    if redis_url:
        try:
            # Tentar conexão
            r = redis.from_url(redis_url)

            # Testar ping
            pong = r.ping()
            print(f"Redis ping: {pong}")

            # Testar set/get
            r.set("test_key", "test_value")
            value = r.get("test_key")
            print(f"Test set/get: {value}")

            # Informações do servidor
            info = r.info()
            print(f"Redis version: {info['redis_version']}")
            print(f"Memory used: {info['used_memory_human']}")

        except Exception as e:
            print(f"❌ Erro na conexão Redis: {e}")
            print("\nSugestões:")
            print("1. Verifique se Redis está rodando: redis-server")
            print("2. Verifique URL no .env")
            print("3. Teste com: redis-cli ping")
    else:
        print("❌ REDIS_URL não configurada")

debug_redis_connection()
```

#### 3. Rate Limit Excedido

**Erro:** `429 Too Many Requests`

**Solução:**
```python
# fix_rate_limiting.py
import time
import asyncio
from rate_limiter import AdvancedRateLimiter

async def demonstrate_rate_limiting():
    """Demonstra funcionamento do rate limiting"""
    limiter = AdvancedRateLimiter()

    print("🚀 Testando rate limiting...")

    for i in range(10):
        start_time = time.time()
        await limiter.acquire("openrouter")
        elapsed = time.time() - start_time

        remaining = limiter.get_remaining_requests("openrouter")

        print(f"Request {i+1}: {elapsed:.3f}s, Remaining: {remaining}")

    print("✅ Rate limiting funcionando corretamente")

asyncio.run(demonstrate_rate_limiting())
```

#### 4. Problemas com MCP GitHub

**Erro:** `Docker container failed to start`

**Solução:**
```bash
# Verificar Docker
docker --version
docker ps

# Puxar imagem GitHub MCP
docker pull ghcr.io/github/github-mcp-server:latest

# Testar manualmente
docker run --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=seu_token ghcr.io/github/github-mcp-server
```

### Monitoramento e Diagnóstico

#### Dashboard de Status

```python
# status_dashboard.py
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

class StatusDashboard:
    def __init__(self):
        self.metrics_history = []

    def collect_metrics(self):
        """Coleta métricas de todos os serviços"""
        health_checker = HealthChecker()
        status = asyncio.run(health_checker.check_all_services())

        metrics = {
            "timestamp": datetime.utcnow(),
            "openrouter_status": "✅" in status.get("openrouter", ""),
            "github_status": "✅" in status.get("github", ""),
            "telegram_status": "✅" in status.get("telegram", ""),
            "rate_limiting_status": "✅" in status.get("rate_limiting", "")
        }

        self.metrics_history.append(metrics)

        # Manter apenas últimas 24 horas
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if m["timestamp"] > cutoff
        ]

        return metrics

    def generate_report(self):
        """Gera relatório de status"""
        if not self.metrics_history:
            return "Sem dados disponíveis"

        df = pd.DataFrame(self.metrics_history)

        report = f"""
📊 RELATÓRIO DE STATUS - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

🔍 Serviços:
├── OpenRouter: {'🟢' if df['openrouter_status'].all() else '🔴'} ({df['openrouter_status'].sum()}/{len(df)} checks OK)
├── GitHub: {'🟢' if df['github_status'].all() else '🔴'} ({df['github_status'].sum()}/{len(df)} checks OK)
├── Telegram: {'🟢' if df['telegram_status'].all() else '🔴'} ({df['telegram_status'].sum()}/{len(df)} checks OK)
└── Rate Limiting: {'🟢' if df['rate_limiting_status'].all() else '🔴'} ({df['rate_limiting_status'].sum()}/{len(df)} checks OK)

📈 Estatísticas:
├── Total de checks: {len(df)}
├── Período: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds()/3600:.1f} horas
└── Taxa de sucesso: {(df.select_dtypes(bool).mean().mean()*100):.1f}%
        """

        return report

    def plot_uptime(self):
        """Plota gráfico de uptime"""
        if not self.metrics_history:
            return

        df = pd.DataFrame(self.metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(12, 6))

        services = ['openrouter_status', 'github_status', 'telegram_status', 'rate_limiting_status']

        for service in services:
            plt.plot(df['timestamp'], df[service].astype(int),
                    label=service.replace('_status', '').title(),
                    marker='o', markersize=3)

        plt.title('Service Uptime Over Time')
        plt.xlabel('Time')
        plt.ylabel('Status (1=UP, 0=DOWN)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Uso
dashboard = StatusDashboard()
dashboard.collect_metrics()
print(dashboard.generate_report())
dashboard.plot_uptime()
```

Este guia completo de configuração de APIs cobre todos os aspectos necessários para configurar, gerenciar e manter as integrações de APIs do projeto EA_SCALPER_XAUUSD, incluindo best practices de segurança, troubleshooting e monitoramento.