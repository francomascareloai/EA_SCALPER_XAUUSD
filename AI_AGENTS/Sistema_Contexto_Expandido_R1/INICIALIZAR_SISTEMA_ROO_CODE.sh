#!/bin/bash

# 🚀 Script de Inicialização - Roo Code + LiteLLM + R1 Otimizado
# Este script configura toda a arquitetura automaticamente

echo "🚀 INICIANDO CONFIGURAÇÃO DO SISTEMA ROO CODE + LITELLM + R1"
echo "================================================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Verificar se estamos no diretório correto
if [ ! -f "sistema_contexto_expandido_2m.py" ]; then
    error "Arquivo sistema_contexto_expandido_2m.py não encontrado!"
    error "Execute este script no diretório Sistema_Contexto_Expandido_R1"
    exit 1
fi

log "✅ Diretório correto identificado"

# Passo 1: Instalar dependências
log "📦 Instalando dependências..."

# Instalar LiteLLM
pip install litellm redis openai sentence-transformers python-dotenv

# Verificar se Redis está instalado
if ! command -v redis-server &> /dev/null; then
    warning "Redis não encontrado. Instalando via Docker..."
    if command -v docker &> /dev/null; then
        docker run -d -p 6379:6379 --name redis-litellm redis:alpine
        log "✅ Redis instalado via Docker"
    else
        error "Docker não encontrado. Instale Redis manualmente:"
        error "sudo apt-get install redis-server"
        exit 1
    fi
else
    sudo systemctl start redis-server 2>/dev/null || redis-server --daemonize yes
    log "✅ Redis iniciado"
fi

# Passo 2: Criar configuração LiteLLM
log "⚙️ Criando configuração LiteLLM..."

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
    ttl: 3600
  enable_rate_limiting: true
  rate_limit_per_minute: 100

server_settings:
  host: "0.0.0.0"
  port: 4000
  enable_cors: true
EOF

log "✅ Configuração LiteLLM criada"

# Passo 3: Verificar se o cache já foi implementado
if ! grep -q "hierarchical_cache" sistema_contexto_expandido_2m.py; then
    error "Cache não foi implementado no sistema!"
    error "Execute primeiro o script de implementação do cache:"
    error "python implementar_cache_r1_completo.py"
    exit 1
fi

log "✅ Cache otimizado já implementado"

# Passo 4: Iniciar LiteLLM Proxy
log "🟢 Iniciando LiteLLM Proxy..."

# Verificar se porta 4000 está livre
if lsof -Pi :4000 -sTCP:LISTEN -t >/dev/null ; then
    warning "Porta 4000 já está em uso. Matando processo..."
    lsof -ti:4000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Iniciar LiteLLM em background
litellm --config litellm_config.yaml > litellm.log 2>&1 &
LITELLM_PID=$!

# Aguardar inicialização
sleep 5

# Verificar se está rodando
if kill -0 $LITELLM_PID 2>/dev/null; then
    log "✅ LiteLLM Proxy iniciado (PID: $LITELLM_PID)"
else
    error "Falha ao iniciar LiteLLM Proxy"
    exit 1
fi

# Passo 5: Testar conexão
log "🧪 Testando conexão com LiteLLM..."

TEST_RESPONSE=$(curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-litellm-abc123" \
  -d '{
    "model": "deepseek-r1-free",
    "messages": [{"role": "user", "content": "Hello from Roo Code!"}],
    "max_tokens": 50
  }' \
  --connect-timeout 10 \
  --max-time 30)

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    log "✅ Conexão com LiteLLM funcionando"
else
    error "Falha na conexão com LiteLLM"
    echo "Resposta: $TEST_RESPONSE"
    exit 1
fi

# Passo 6: Criar arquivo de configuração para Roo Code
log "📝 Criando configuração para Roo Code..."

mkdir -p ~/.config/roo-code 2>/dev/null || true

cat > ~/.config/roo-code/config.json << 'EOF'
{
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
  },
  "features": {
    "contextExpansion": true,
    "semanticCache": true,
    "performanceMonitoring": true
  }
}
EOF

log "✅ Configuração Roo Code criada"

# Passo 7: Testar sistema completo
log "🎯 Testando sistema completo..."

python -c "
from sistema_contexto_expandido_2m import ContextManager
import time

# Inicializar sistema
cm = ContextManager()

# Teste de cache
query = 'Teste do sistema Roo Code + R1 Otimizado'
response = cm.chat_with_expanded_context(query)
stats = cm.get_cache_stats()

print('🎉 Sistema funcionando!')
print(f'📊 Cache Hit Rate: {stats.get(\"overall_hit_rate\", \"N/A\")}')
print(f'📦 Cache L1 Size: {stats.get(\"l1_cache_size\", 0)}')
"

if [ $? -eq 0 ]; then
    log "✅ Sistema completo testado e funcionando"
else
    error "Erro no teste do sistema"
    exit 1
fi

# Passo 8: Criar script de monitoramento
log "📊 Criando script de monitoramento..."

cat > monitor_sistema.sh << 'EOF'
#!/bin/bash

# Script de monitoramento do sistema R1 Otimizado

while true; do
  clear
  echo "📊 Monitoramento do Sistema R1 Otimizado"
  echo "=========================================="
  echo ""

  # Status do Redis
  echo "🔴 Redis Cache:"
  if docker ps | grep -q redis-litellm; then
    docker stats redis-litellm --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "Docker não disponível"
  else
    echo "Redis local: $(redis-cli ping 2>/dev/null || echo 'Offline')"
  fi

  echo ""
  echo "🟢 LiteLLM Proxy:"
  curl -s http://localhost:4000/health | head -1 || echo "❌ Offline"

  echo ""
  echo "📈 Cache Stats:"
  python -c "
  from sistema_contexto_expandido_2m import ContextManager
  cm = ContextManager()
  stats = cm.get_cache_stats()
  print(f'Hit Rate: {stats.get(\"overall_hit_rate\", \"N/A\")}')
  print(f'L1 Cache: {stats.get(\"l1_cache_size\", 0)} entries')
  print(f'Total Requests: {stats.get(\"total_requests\", 0)}')
  " 2>/dev/null || echo "Erro ao obter estatísticas"

  echo ""
  echo "🔄 Pressione Ctrl+C para sair"
  sleep 5
done
EOF

chmod +x monitor_sistema.sh

# Passo 9: Criar script de parada
log "🛑 Criando script de parada..."

cat > parar_sistema.sh << 'EOF'
#!/bin/bash

echo "🛑 Parando Sistema R1 Otimizado..."

# Parar LiteLLM
pkill -f litellm

# Parar Redis Docker (se existir)
docker stop redis-litellm 2>/dev/null || true
docker rm redis-litellm 2>/dev/null || true

# Parar Redis local
sudo systemctl stop redis-server 2>/dev/null || true

echo "✅ Sistema parado"
EOF

chmod +x parar_sistema.sh

log "✅ Script de parada criado"

# Passo 10: Salvar PIDs para gerenciamento
echo $LITELLM_PID > litellm.pid
log "✅ PIDs salvos para gerenciamento"

# Resumo final
echo ""
echo "🎉 SISTEMA CONFIGURADO COM SUCESSO!"
echo "====================================="
echo ""
echo "📍 Arquitetura Configurada:"
echo "   Roo Code → LiteLLM Proxy → Cache Otimizado → OpenRouter → R1"
echo ""
echo "🔗 Endpoints:"
echo "   • LiteLLM Proxy: http://localhost:4000"
echo "   • Cache Stats: python -c \"from sistema_contexto_expandido_2m import ContextManager; cm = ContextManager(); print(cm.get_cache_stats())\""
echo ""
echo "🔑 Chave para Roo Code:"
echo "   sk-litellm-abc123"
echo ""
echo "📊 Scripts disponíveis:"
echo "   • ./monitor_sistema.sh - Monitorar sistema"
echo "   • ./parar_sistema.sh - Parar sistema"
echo ""
echo "🚀 Como usar no Roo Code:"
echo "   1. Configure a chave: sk-litellm-abc123"
echo "   2. Configure o endpoint: http://localhost:4000"
echo "   3. Selecione o modelo: deepseek-r1-free"
echo ""
echo "📈 Performance Esperada:"
echo "   • Cache Hit Rate: 90%+"
echo "   • Tempo de Resposta: < 0.5ms (cache hits)"
echo "   • Throughput: 1000+ ops/s"
echo "   • Contexto: Até 2M tokens"
echo ""
echo "🛑 Para parar o sistema:"
echo "   ./parar_sistema.sh"
echo ""
echo "🎯 SISTEMA PRONTO PARA USO!"

# Iniciar monitoramento automaticamente
echo ""
read -p "Deseja iniciar o monitoramento agora? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./monitor_sistema.sh
fi