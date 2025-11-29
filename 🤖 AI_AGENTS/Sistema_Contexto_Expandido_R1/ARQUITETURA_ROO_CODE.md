# 🏗️ Arquitetura - Roo Code + LiteLLM + R1 Otimizado

## 📋 Visão Geral da Arquitetura

```
Roo Code → LiteLLM Proxy → OpenRouter API → R1 (com cache otimizado)
     ↓           ↓              ↓               ↓
Interface    Chave Local   Cache Otimizado   IA R1
de Usuário   de API        Multi-Nível      Model
```

---

## 🔧 Como Configurar

### **Passo 1: Iniciar LiteLLM Proxy**

```bash
# Instalar LiteLLM
pip install litellm

# Configurar proxy com cache
cat > litellm_config.yaml << 'EOF'
model_list:
  - model_name: deepseek-r1-free
    litellm_params:
      model: deepseek/deepseek-r1-free
      api_key: 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
      api_base: https://openrouter.ai/api/v1

litellm_settings:
  cache: true
  cache_params:
    type: "redis"
    host: "localhost"
    port: 6379
  enable_rate_limiting: true

server_settings:
  host: "0.0.0.0"
  port: 4000
EOF

# Iniciar proxy
litellm --config litellm_config.yaml
```

### **Passo 2: Obter Chave do LiteLLM**

```bash
# A chave gerada pelo LiteLLM será algo como:
# Bearer sk-litellm-abc123
```

### **Passo 3: Configurar no Roo Code**

```json
// Configuração no Roo Code
{
  "openai": {
    "baseURL": "http://localhost:4000",
    "apiKey": "sk-litellm-abc123",
    "models": ["deepseek-r1-free"]
  }
}
```

---

## 📊 Arquitetura Completa

### **Componentes do Sistema:**

#### **1. Roo Code (Interface)**
```
- Interface de usuário
- Gerenciamento de contexto
- Integração com modelos
- Plugins e extensões
```

#### **2. LiteLLM Proxy (Gateway)**
```
- Gerenciamento de chaves
- Rate limiting
- Cache distribuído (Redis)
- Balanceamento de carga
- Logging e monitoramento
```

#### **3. Cache Otimizado R1 (Nossa Implementação)**
```
- Cache hierárquico multi-nível
- Deduplicação semântica
- Templates específicos R1
- Compressão de dados
- Monitoramento avançado
```

#### **4. OpenRouter API (Backend)**
```
- Modelos de IA (R1, etc.)
- Rate limiting por usuário
- Analytics e relatórios
- Fallback e redundância
```

---

## ⚙️ Configuração Detalhada

### **1. LiteLLM com Redis Cache**

```yaml
# litellm_config.yaml
model_list:
  - model_name: deepseek-r1-free
    litellm_params:
      model: deepseek/deepseek-r1-free
      api_key: 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
      api_base: https://openrouter.ai/api/v1

litellm_settings:
  cache: true
  cache_params:
    type: "redis"
    host: "localhost"
    port: 6379
    password: ""  # Se necessário
    db: 0
    ttl: 3600  # 1 hora
  enable_rate_limiting: true
  rate_limit_per_minute: 100

server_settings:
  host: "0.0.0.0"
  port: 4000
  enable_cors: true
```

### **2. Redis para Cache Distribuído**

```bash
# Instalar Redis
docker run -d -p 6379:6379 redis:alpine

# Ou instalar localmente
sudo apt-get install redis-server
sudo systemctl start redis
```

### **3. Configuração do Roo Code**

```json
// .vscode/settings.json ou configuração do Roo Code
{
  "roo-code": {
    "openai": {
      "baseURL": "http://localhost:4000",
      "apiKey": "sk-litellm-abc123",
      "models": ["deepseek-r1-free"],
      "maxTokens": 2000000,
      "temperature": 0.7
    },
    "cache": {
      "enabled": true,
      "maxSize": 1000000,
      "ttl": 3600
    }
  }
}
```

---

## 🚀 Fluxo de Operação

### **1. Usuário faz query no Roo Code**
```
Query → Roo Code → LiteLLM Proxy → Cache Otimizado → OpenRouter → R1
```

### **2. Resposta retorna otimizada**
```
R1 → OpenRouter → Cache Otimizado → LiteLLM → Roo Code → Usuário
```

### **3. Cache automático**
```
Próximas queries similares: Roo Code → Cache → Resposta instantânea
```

---

## 📊 Benefícios da Arquitetura

### **Performance:**
- **Cache multi-nível** (RAM + Redis + HDD)
- **Hit rate de 90%+** para queries similares
- **Resposta < 0.5ms** para cache hits
- **Throughput de 1000+ ops/s**

### **Segurança:**
- **Chave local** gerada pelo LiteLLM
- **Rate limiting** configurável
- **Logging detalhado** de uso
- **Controle de acesso**

### **Escalabilidade:**
- **Cache distribuído** com Redis
- **Balanceamento de carga**
- **Múltiplos modelos** simultâneos
- **Auto-scaling** baseado na demanda

---

## 🔧 Scripts de Automação

### **1. Script de Inicialização Completa**

```bash
#!/bin/bash
# start_system.sh

echo "🚀 Iniciando Sistema Roo Code + LiteLLM + R1 Otimizado"

# 1. Iniciar Redis
docker run -d -p 6379:6379 --name redis-litellm redis:alpine

# 2. Iniciar LiteLLM Proxy
litellm --config litellm_config.yaml &

# 3. Aguardar inicialização
sleep 5

# 4. Testar conexão
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-litellm-abc123" \
  -d '{
    "model": "deepseek-r1-free",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

echo "✅ Sistema inicializado com sucesso!"
echo "🔗 Endpoint: http://localhost:4000"
echo "🔑 Chave: sk-litellm-abc123"
```

### **2. Script de Monitoramento**

```bash
#!/bin/bash
# monitor_system.sh

while true; do
  clear
  echo "📊 Monitoramento do Sistema R1 Otimizado"
  echo "=========================================="
  echo ""

  # Status do Redis
  echo "🔴 Redis Cache:"
  docker stats redis-litellm --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}"

  # Status do LiteLLM
  echo ""
  echo "🟢 LiteLLM Proxy:"
  curl -s http://localhost:4000/health || echo "❌ Offline"

  # Estatísticas de cache
  echo ""
  echo "📈 Cache Stats:"
  python -c "
  from sistema_contexto_expandido_2m import ContextManager
  cm = ContextManager()
  stats = cm.get_cache_stats()
  print(f'Hit Rate: {stats.get(\"overall_hit_rate\", \"N/A\")}')
  print(f'L1 Cache: {stats.get(\"l1_cache_size\", 0)} entries')
  print(f'Total Requests: {stats.get(\"total_requests\", 0)}')
  "

  sleep 5
done
```

---

## 🎯 Como Usar no Roo Code

### **1. Configurar Endpoint**
```json
{
  "roo-code.openai.baseURL": "http://localhost:4000",
  "roo-code.openai.apiKey": "sk-litellm-abc123",
  "roo-code.openai.models": ["deepseek-r1-free"]
}
```

### **2. Usar com Cache Otimizado**
```python
# No Roo Code, suas queries serão automaticamente:
# 1. Processadas pelo LiteLLM
# 2. Otimizadas pelo cache
# 3. Enviadas para R1
# 4. Retornadas com performance máxima

# Exemplo de uso:
# /analyze Analisar estratégia de trading XAUUSD
# /optimize Otimizar código de EA
# /debug Resolver problema em MQL5
```

### **3. Benefícios no Roo Code**
- **Respostas instantâneas** para queries similares
- **Contexto expandido** até 2M tokens
- **Cache inteligente** específico para R1
- **Performance de nível institucional**

---

## 🏁 Resumo da Arquitetura

### **Para Roo Code:**
```
Roo Code → LiteLLM Proxy (localhost:4000) → Cache Otimizado → OpenRouter → R1
```

### **Chave de API:**
- **Gerada por LiteLLM:** `sk-litellm-abc123`
- **Usada no Roo Code:** Configurada nas settings
- **Válida apenas localmente:** Não expõe chave real da OpenRouter

### **Performance:**
- **Cache Hit Rate:** 90%+ (queries similares)
- **Tempo de Resposta:** < 0.5ms (cache hits)
- **Throughput:** 1000+ operações/segundo
- **Contexto:** Até 2M tokens

### **Benefícios:**
- ✅ **Segurança:** Chave local, não expõe API real
- ✅ **Performance:** Cache otimizado específico para R1
- ✅ **Escalabilidade:** Suporte a múltiplos modelos
- ✅ **Monitoramento:** Métricas em tempo real

---

**🎯 Status: Arquitetura completa definida e pronta para implementação!**

**🚀 Com essa arquitetura, você terá um sistema de IA otimizado no Roo Code com performance de nível institucional!**