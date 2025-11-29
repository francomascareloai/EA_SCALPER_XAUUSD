# Variáveis de Ambiente - EA_SCALPER_XAUUSD

## Overview

Este documento descreve todas as variáveis de ambiente utilizadas no projeto EA_SCALPER_XAUUSD, incluindo suas descrições, valores padrão, validações e exemplos de uso.

## Estrutura do Documento

1. [Variáveis de API](#variáveis-de-api)
2. [Variáveis de Configuração do Sistema](#variáveis-de-configuração-do-sistema)
3. [Variáveis de Cache e Performance](#variáveis-de-cache-e-performance)
4. [Variáveis de Segurança](#variáveis-de-segurança)
5. [Variáveis de Desenvolvimento](#variáveis-de-desenvolvimento)
6. [Variáveis de Integração Externa](#variáveis-de-integração-externa)
7. [Exemplos de Configuração](#exemplos-de-configuração)
8. [Troubleshooting](#troubleshooting)

---

## Variáveis de API

### OPENROUTER_API_KEY

**Descrição:** Chave de API para acessar os serviços do OpenRouter AI.

- **Tipo:** String
- **Obrigatória:** Sim
- **Valor Padrão:** N/A
- **Formato:** `sk-or-v1-[hash]`
- **Exemplo:** `sk-or-v1-SEU_HASH_AQUI`

**Validação:**
- Deve começar com `sk-or-v1-`
- Mínimo de 40 caracteres
- Apenas caracteres alfanuméricos

**Como Obter:**
1. Acesse https://openrouter.ai/keys
2. Crie uma conta ou faça login
3. Gere uma nova chave de API
4. Copie e cole no arquivo `.env`

**Segurança:**
- Nunca compartilhe esta chave
- Não commitar no controle de versão
- Rotacionar regularmente

### OPENROUTER_APP_NAME

**Descrição:** Nome da aplicação que será exibida nas requisições à API.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `"Trading Agent Organizer"`
- **Exemplo:** `"EA_SCALPER_XAUUSD_Trading_Bot"`

**Validação:**
- Máximo 100 caracteres
- Sem caracteres especiais exceto espaços e hífens

### OPENROUTER_SITE_URL

**Descrição:** URL do site associado à aplicação.

- **Tipo:** URL
- **Obrigatória:** Não
- **Valor Padrão:** `"https://github.com/your_repo"`
- **Exemplo:** `"https://github.com/seu-usuario/EA_SCALPER_XAUUSD"`

**Validação:**
- URL válida
- Protocolo HTTP ou HTTPS

### GITHUB_PERSONAL_ACCESS_TOKEN

**Descrição:** Token de acesso pessoal para integração com GitHub MCP.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Formato:** `ghp_[hash]`
- **Exemplo:** `ghp_SEU_HASH_AQUI`

**Validação:**
- Deve começar com `ghp_`
- Mínimo de 40 caracteres

**Permissões Recomendadas:**
- `repo` (controle total de repositórios)
- `read:org` (leitura da organização)

### OPENAI_API_KEY

**Descrição:** Chave de API para serviços OpenAI (uso alternativo).

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Formato:** `sk-[hash]`
- **Exemplo:** `sk-SEU_HASH_AQUI`

**Validação:**
- Deve começar com `sk-`
- Mínimo de 40 caracteres

### ANTHROPIC_API_KEY

**Descrição:** Chave de API para serviços Anthropic Claude.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Exemplo:** `sk-ant-SEU_HASH_AQUI`

**Validação:**
- Deve começar com `sk-ant-`
- Mínimo de 40 caracteres

---

## Variáveis de Configuração do Sistema

### OPENAI_API_BASE

**Descrição:** URL base para API OpenAI/LiteLLM proxy.

- **Tipo:** URL
- **Obrigatória:** Não
- **Valor Padrão:** `"http://localhost:4000"`
- **Exemplo:** `"http://localhost:4000"` ou `"https://api.openai.com/v1"`

**Validação:**
- URL válida
- Porta entre 1-65535

### DEFAULT_MODEL

**Descrição:** Modelo padrão para processamento de linguagem.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `"openrouter/anthropic/claude-3-5-sonnet"`
- **Opções Suportadas:**
  - `"openrouter/anthropic/claude-3-5-sonnet"`
  - `"openrouter/openai/gpt-4o"`
  - `"openrouter/anthropic/claude-3-opus"`
  - `"openrouter/google/gemini-pro"`

**Validação:**
- Deve ser um modelo suportado pelo provedor

### BACKUP_MODEL

**Descrição:** Modelo alternativo para fallback.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `"openrouter/openai/gpt-4o"`
- **Opções Suportadas:** Mesmas do `DEFAULT_MODEL`

**Validação:**
- Não pode ser igual ao `DEFAULT_MODEL`
- Deve ser um modelo suportado

---

## Variáveis de Cache e Performance

### PROMPT_CACHE_TTL

**Descrição:** Tempo de vida (TTL) do cache de prompts em segundos.

- **Tipo:** Integer
- **Obrigatória:** Não
- **Valor Padrão:** `3600` (1 hora)
- **Intervalo:** `60` a `86400` (1 minuto a 24 horas)
- **Exemplo:** `3600`

**Impacto na Performance:**
- Maior valor = menos requisições à API
- Menor valor = respostas mais atualizadas

### RESPONSE_CACHE_TTL

**Descrição:** Tempo de vida do cache de respostas em segundos.

- **Tipo:** Integer
- **Obrigatória:** Não
- **Valor Padrão:** `1800` (30 minutos)
- **Intervalo:** `30` a `7200` (30 segundos a 2 horas)
- **Exemplo:** `1800`

### REDIS_URL

**Descrição:** URL de conexão com servidor Redis para cache distribuído.

- **Tipo:** URL
- **Obrigatória:** Não
- **Valor Padrão:** N/A (usa cache local)
- **Formato:** `redis://[password@]host[:port][/db]`
- **Exemplo:** `redis://localhost:6379/0` ou `redis://:password@redis.example.com:6379/1`

**Validação:**
- URL Redis válida
- Servidor Redis acessível

### CACHE_TYPE

**Descrição:** Tipo de cache a ser utilizado.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `"local"`
- **Opções:** `"local"`, `"redis"`, `"memory"`
- **Exemplo:** `"redis"`

---

## Variáveis de Segurança

### GITHUB_TOOLSETS

**Descrição:** Configuração de toolsets GitHub MCP.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `""` (vazio)
- **Exemplo:** `"coding,review"`

**Validação:**
- Lista separada por vírgulas
- Apenas caracteres alfanuméricos e hífens

### GITHUB_READ_ONLY

**Descrição:** Modo somente leitura para integração GitHub.

- **Tipo:** Boolean
- **Obrigatória:** Não
- **Valor Padrão:** `""` (falso)
- **Exemplo:** `"true"` ou `"false"`

**Valores Aceitos:**
- `"true"` ou `"1"` para modo leitura
- `"false"` ou `"0"` para modo completo
- `""` (vazio) para padrão (falso)

### ENABLE_AUDIT_LOG

**Descrição:** Ativa logging de auditoria de segurança.

- **Tipo:** Boolean
- **Obrigatória:** Não
- **Valor Padrão:** `"true"`
- **Exemplo:** `"true"`

**Log Inclui:**
- Acesso a APIs
- Mudanças de configuração
- Operações de risco

### MAX_REQUEST_RATE

**Descrição:** Taxa máxima de requisições por minuto.

- **Tipo:** Integer
- **Obrigatória:** Não
- **Valor Padrão:** `60`
- **Intervalo:** `1` a `1000`
- **Exemplo:** `100`

---

## Variáveis de Desenvolvimento

### DEBUG_MODE

**Descrição:** Ativa modo debug para desenvolvimento.

- **Tipo:** Boolean
- **Obrigatória:** Não
- **Valor Padrão:** `"false"`
- **Exemplo:** `"true"`

**Funcionalidades Debug:**
- Logs detalhados
- Trace de requisições
- Informações de performance

### LOG_LEVEL

**Descrição:** Nível de logging detalhado.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** `"INFO"`
- **Opções:** `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Exemplo:** `"DEBUG"`

**Hierarquia:**
- `DEBUG`: Toda informação
- `INFO`: Informação geral
- `WARNING`: Avisos importantes
- `ERROR`: Apenas erros
- `CRITICAL`: Apenas erros críticos

### TESTING_MODE

**Descrição:** Ativa modo de testes automatizados.

- **Tipo:** Boolean
- **Obrigatória:** Não
- **Valor Padrão:** `"false"`
- **Exemplo:** `"true"`

**Comportamento:**
- Usa APIs de teste
- Limita consumo de recursos
- Gera relatórios detalhados

---

## Variáveis de Integração Externa

### TELEGRAM_BOT_TOKEN

**Descrição:** Token do bot Telegram para notificações.

- **Tipo:** String
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Formato:** `[number]:[hash]`
- **Exemplo:** `SEU_TOKEN_AQUI`

**Validação:**
- Formato válido de token Telegram
- Bot deve estar configurado

### TELEGRAM_CHAT_ID

**Descrição:** ID do chat para envio de notificações.

- **Tipo:** String/Integer
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Exemplo:** `"SEU_CHAT_ID_AQUI"`

**Validação:**
- ID numérico válido
- Bot deve ter acesso ao chat

### DISCORD_WEBHOOK_URL

**Descrição:** URL de webhook para integração com Discord.

- **Tipo:** URL
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Exemplo:** `"SEU_WEBHOOK_AQUI"`

**Validação:**
- URL Discord válida
- Webhook ativo

### SLACK_WEBHOOK_URL

**Descrição:** URL de webhook para integração com Slack.

- **Tipo:** URL
- **Obrigatória:** Não
- **Valor Padrão:** N/A
- **Exemplo:** `"SEU_WEBHOOK_AQUI"`

---

## Exemplos de Configuração

### Configuração Mínima

```bash
# .env - Configuração básica
OPENROUTER_API_KEY=sk-or-v1-SEU_HASH_AQUI
```

### Configuração Desenvolvimento

```bash
# .env - Ambiente de desenvolvimento
OPENROUTER_API_KEY=sk-or-v1-SEU_HASH_AQUI
DEBUG_MODE=true
LOG_LEVEL=DEBUG
TESTING_MODE=true
OPENAI_API_BASE=http://localhost:4000
```

### Configuração Produção

```bash
# .env - Ambiente de produção
OPENROUTER_API_KEY=sk-or-v1-SEU_HASH_AQUI
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
BACKUP_MODEL=openrouter/openai/gpt-4o
PROMPT_CACHE_TTL=7200
RESPONSE_CACHE_TTL=3600
REDIS_URL=redis://localhost:6379/0
ENABLE_AUDIT_LOG=true
MAX_REQUEST_RATE=120
TELEGRAM_BOT_TOKEN=SEU_TOKEN_AQUI
TELEGRAM_CHAT_ID=SEU_CHAT_ID_AQUI
```

### Configuração Completa

```bash
# .env - Configuração completa
# OpenRouter Configuration
OPENROUTER_API_KEY=sk-or-v1-SEU_HASH_AQUI
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD_Trading_Bot"
OPENROUTER_SITE_URL="https://github.com/seu-usuario/EA_SCALPER_XAUUSD"

# Model Configuration
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
BACKUP_MODEL=openrouter/openai/gpt-4o

# API Configuration
OPENAI_API_BASE=http://localhost:4000

# Cache Configuration
PROMPT_CACHE_TTL=3600
RESPONSE_CACHE_TTL=1800
REDIS_URL=redis://localhost:6379/0
CACHE_TYPE=redis

# Security Configuration
ENABLE_AUDIT_LOG=true
MAX_REQUEST_RATE=100

# GitHub Integration
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_SEU_HASH_AQUI
GITHUB_TOOLSETS="coding,review"
GITHUB_READ_ONLY=false

# Development Configuration
DEBUG_MODE=false
LOG_LEVEL=INFO
TESTING_MODE=false

# Notification Configuration
TELEGRAM_BOT_TOKEN=SEU_TOKEN_AQUI
TELEGRAM_CHAT_ID=SEU_CHAT_ID_AQUI
DISCORD_WEBHOOK_URL=SEU_WEBHOOK_AQUI
SLACK_WEBHOOK_URL=SEU_WEBHOOK_AQUI
```

---

## Troubleshooting

### Problemas Comuns

#### 1. Erro de API Key Inválida

```
❌ ERRO: OPENROUTER_API_KEY não encontrada no .env
📝 Crie um arquivo .env com: OPENROUTER_API_KEY=sua_chave_aqui
```

**Solução:**
1. Verifique se o arquivo `.env` existe no diretório raiz
2. Confirme que a variável está escrita corretamente
3. Valide se a chave está correta e ativa

#### 2. Conexão com API Falhando

```
❌ ERRO: Falha na conexão com OpenRouter
🔍 Verifique sua conexão de rede e a URL da API
```

**Solução:**
1. Teste conectividade com o servidor
2. Verifique a URL base configurada
3. Confirme se o proxy está funcionando

#### 3. Cache Não Funcionando

```
⚠️ AVISO: Cache Redis não disponível, usando cache local
💾 Desempenho pode ser reduzido
```

**Solução:**
1. Verifique se o Redis está rodando
2. Confirme a URL de conexão
3. Teste conectividade com o Redis

#### 4. Rate Limit Excedido

```
❌ ERRO: Limite de requisições excedido
⏱️ Aguarde antes de fazer novas requisições
```

**Solução:**
1. Ajuste `MAX_REQUEST_RATE`
2. Implemente backoff exponencial
3. Use cache para reduzir requisições

### Validação de Configuração

Use o script de validação para verificar sua configuração:

```python
# validate_env.py
import os
from dotenv import load_dotenv
import re

def validate_api_key(key, prefix):
    """Valida formato da chave de API"""
    if not key.startswith(prefix):
        return False
    return len(key) >= 40

def validate_url(url):
    """Valida formato da URL"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def main():
    load_dotenv()

    # Validar OpenRouter API Key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key and validate_api_key(api_key, 'sk-or-v1-'):
        print("✅ OPENROUTER_API_KEY válida")
    else:
        print("❌ OPENROUTER_API_KEY inválida")

    # Validar URL base
    api_base = os.getenv('OPENAI_API_BASE')
    if api_base and validate_url(api_base):
        print("✅ OPENAI_API_BASE válida")
    else:
        print("❌ OPENAI_API_BASE inválida")

    # Validar modelo padrão
    default_model = os.getenv('DEFAULT_MODEL')
    if default_model:
        print(f"✅ DEFAULT_MODEL configurado: {default_model}")
    else:
        print("⚠️ DEFAULT_MODEL não configurado")

if __name__ == "__main__":
    main()
```

### Boas Práticas

1. **Segurança:**
   - Nunca commitar arquivos `.env`
   - Usar senhas fortes e únicas
   - Rotacionar chaves regularmente
   - Limitar permissões de acesso

2. **Performance:**
   - Configurar cache adequado
   - Monitorar uso de APIs
   - Ajustar timeouts e rate limits

3. **Manutenção:**
   - Documentar alterações
   - Testar configurações em staging
   - Manter backup das configurações

4. **Monitoramento:**
   - Log de auditoria ativo
   - Alertas para falhas críticas
   - Métricas de performance

---

## Referências

- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [LiteLLM Documentation](https://litellm.ai/docs)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Security Guidelines](https://owasp.org/www-project-cheat-sheets/cheatsheets/Environment_Variable_Security_Cheat_Sheet.html)