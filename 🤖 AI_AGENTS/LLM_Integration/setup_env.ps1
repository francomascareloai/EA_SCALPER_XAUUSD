# Script PowerShell para configuração robusta
Write-Host "🚀 Configurando LiteLLM com cache hierárquico..." -ForegroundColor Cyan

# 1. Verificar instalação do Python
$pythonVersion = python --version 2>&1
if (-not $?) {
    Write-Host "❌ Python não encontrado. Instale Python primeiro!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python encontrado: $pythonVersion" -ForegroundColor Green

# 2. Instalar/atualizar dependências globalmente
Write-Host "Instalando dependências..." -ForegroundColor Yellow
pip install --upgrade litellm diskcache
if (-not $?) {
    Write-Host "❌ Falha na instalação das dependências" -ForegroundColor Red
    exit 1
}

# 3. Criar diretório de cache
New-Item -ItemType Directory -Path "trading_cache" -Force | Out-Null

# 4. Testar o sistema
Write-Host "✅ Configuração completa! Testando o sistema..." -ForegroundColor Green
python litellm_prompt_cache.py

# 5. Mostrar instruções finais
Write-Host ""
Write-Host "✨ PRONTO PARA USAR! ✨" -ForegroundColor Cyan
Write-Host "Use o seguinte código para começar:"
Write-Host ""
Write-Host "from litellm_prompt_cache import LiteLLMWithCache"
Write-Host 'llm = LiteLLMWithCache(cache_dir="./trading_cache")'
Write-Host 'response = llm.query_llm("Análise de Fibonacci para XAUUSD M15")'
Write-Host ""