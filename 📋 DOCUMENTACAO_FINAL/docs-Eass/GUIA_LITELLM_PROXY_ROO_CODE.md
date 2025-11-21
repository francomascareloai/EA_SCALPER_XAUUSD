# 🎯 GUIA CONFIGURAÇÃO LITELLM PROXY + ROO CODE

## ✅ CONFIGURAÇÃO COMPLETA REALIZADA!

### 🚀 COMO INICIAR O PROXY:

#### Opção 1 - Script PowerShell (Recomendado):
```powershell
.\start_litellm_proxy.ps1
```

#### Opção 2 - Script Batch:
```cmd
start_litellm_proxy.bat
```

#### Opção 3 - Python Manager:
```bash
python litellm_proxy_manager.py
```

### 📡 INFORMAÇÕES DO PROXY:

- **🔗 URL Base:** `http://127.0.0.1:4000`
- **🔑 API Key:** `sk-litellm-proxy-key-12345`
- **🌐 Endpoint:** `http://127.0.0.1:4000/v1`
- **📊 Interface Admin:** `http://127.0.0.1:4000/ui`

### 🤖 MODELOS DISPONÍVEIS:

#### 1. Qwen 3 Coder Free:
- **Nome:** `qwen-coder`
- **Uso:** Análise de código, debugging, estrutura
- **Cache TTL:** 3600s (1 hora)

#### 2. DeepSeek R1 Free:
- **Nome:** `deepseek-r1` 
- **Uso:** FTMO compliance, organização, estratégias complexas
- **Cache TTL:** 1800s (30 minutos)

### 🔌 CONFIGURAÇÃO NO ROO CODE:

#### 1. Configurações de API:
```json
{
  "apiBaseUrl": "http://127.0.0.1:4000/v1",
  "apiKey": "sk-litellm-proxy-key-12345",
  "model": "qwen-coder"
}
```

#### 2. Para usar DeepSeek R1:
```json
{
  "apiBaseUrl": "http://127.0.0.1:4000/v1", 
  "apiKey": "sk-litellm-proxy-key-12345",
  "model": "deepseek-r1"
}
```

### 💾 PROMPT CACHING ATIVADO:

- ✅ **Cache Local:** Ativo por padrão
- ✅ **TTL Inteligente:** 1-3 horas por modelo
- ✅ **Rate Limiting:** 10 RPM / 1000 TPM
- ✅ **Headers OpenRouter:** Configurados automaticamente

### 🎯 FLUXO DE USO:

#### 1. Iniciar Proxy:
```bash
# Execute um dos scripts de início
.\start_litellm_proxy.ps1
```

#### 2. Configurar Roo Code:
- Base URL: `http://127.0.0.1:4000/v1`
- API Key: `sk-litellm-proxy-key-12345`
- Modelo: `qwen-coder` ou `deepseek-r1`

#### 3. Usar Normalmente:
- O proxy intercepta as chamadas
- Aplica prompt caching automaticamente
- Roteia para OpenRouter com suas credenciais
- Retorna resposta para Roo Code

### 📊 MONITORAMENTO:

#### Interface Admin:
- **URL:** http://127.0.0.1:4000/ui
- **Usuário:** admin
- **Senha:** trading123

#### Logs em Tempo Real:
- Requests/responses
- Cache hits/misses
- Rate limiting status
- Errors/warnings

### 🔧 CONFIGURAÇÕES AVANÇADAS:

#### Rate Limiting:
```yaml
# No arquivo litellm_config.yaml
rpm: 10    # Requests por minuto
tpm: 1000  # Tokens por minuto
```

#### Cache Customizado:
```yaml
cache_params:
  type: "local"  # ou "redis"
  ttl: 3600      # segundos
```

### ⚠️ TROUBLESHOOTING:

#### Proxy não inicia:
1. Verificar .env com OPENROUTER_API_KEY
2. Verificar porta 4000 livre
3. Ativar ambiente virtual primeiro

#### Roo Code não conecta:
1. Verificar URL: `http://127.0.0.1:4000/v1`
2. Verificar API key: `sk-litellm-proxy-key-12345`
3. Verificar se proxy está rodando

#### Rate limiting:
- Aguardar entre requisições
- Verificar logs do proxy
- Ajustar RPM/TPM no config

### 🎉 BENEFÍCIOS:

- ✅ **Prompt Caching** - Respostas instantâneas para prompts repetidos
- ✅ **Rate Limiting** - Proteção automática contra 429
- ✅ **Modelos Gratuitos** - Qwen 3 Coder + DeepSeek R1
- ✅ **Interface Unificada** - Um proxy para múltiplos modelos
- ✅ **Monitoramento** - Logs e métricas em tempo real
- ✅ **Compatibilidade** - OpenAI API padrão para Roo Code

### 🚀 PRÓXIMOS PASSOS:

1. **Iniciar proxy:** `.\start_litellm_proxy.ps1`
2. **Configurar Roo Code** com as credenciais do proxy
3. **Testar conexão** fazendo uma requisição
4. **Monitorar logs** na interface admin
5. **Aproveitar o prompt caching!**

---

**🎯 SISTEMA PRONTO! Agora você tem um proxy intermediário com prompt caching entre Roo Code e OpenRouter!**
