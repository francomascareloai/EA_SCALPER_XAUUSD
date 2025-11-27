# Guia de Instalação - LiteLLM com GitHub Copilot e Claude

## 1. Instalação do LiteLLM

```bash
# Instalação básica
pip install litellm

# Instalação com extras (recomendado)
pip install 'litellm[proxy]'

# Para usar com Claude Responses API
pip install 'litellm[anthropic]'
```

## 2. Configurar Variáveis de Ambiente

### Windows (PowerShell)
```powershell
# GitHub Copilot - usa token do GitHub
$env:GITHUB_TOKEN = "seu_github_token_aqui"

# Anthropic Claude
$env:ANTHROPIC_API_KEY = "sk-ant-xxxxx"
```

### Windows (CMD)
```cmd
set GITHUB_TOKEN=seu_github_token_aqui
set ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### Arquivo .env (recomendado)
Crie um arquivo `.env` na raiz do projeto:
```env
GITHUB_TOKEN=seu_github_token_aqui
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

## 3. Obter Tokens

### GitHub Token (para Copilot)
1. Vá para: https://github.com/settings/tokens
2. Clique em "Generate new token (classic)"
3. Selecione os scopes: `copilot`, `read:user`
4. Copie o token gerado

### Anthropic API Key (para Claude)
1. Vá para: https://console.anthropic.com/
2. Crie uma conta ou faça login
3. Vá em "API Keys"
4. Crie uma nova chave

## 4. Uso Básico

### Testar GitHub Copilot via LiteLLM
```python
import litellm

# GitHub Copilot
response = litellm.completion(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Testar Claude com Responses API
```python
import litellm

# Claude com Responses API (novo formato)
response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
    # Para usar Responses API format
    response_format={"type": "text"}
)
print(response.choices[0].message.content)
```

## 5. Configuração do Proxy (Opcional)

Para usar o LiteLLM como proxy central:

```bash
# Iniciar o proxy
litellm --model anthropic/claude-sonnet-4-20250514 --port 8000
```

Ou com arquivo de configuração:
```bash
litellm --config config.yaml
```
