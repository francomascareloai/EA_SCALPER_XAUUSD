@echo off
REM Script de configuração completa para LiteLLM com cache

echo 🚀 Configurando ambiente para LiteLLM com cache hierárquico...

REM 1. Criar ambiente virtual
python -m venv .venv
if %errorlevel% neq 0 (
    echo ❌ Falha ao criar ambiente virtual
    exit /b 1
)

REM 2. Ativar ambiente
call .venv\Scripts\activate

REM 3. Instalar dependências
pip install litellm==1.0.0 diskcache==5.6.1
if %errorlevel% neq 0 (
    echo ❌ Falha ao instalar dependências
    exit /b 1
)

REM 4. Criar diretório de cache
mkdir trading_cache 2>nul

REM 5. Testar sistema
echo ✅ Ambiente configurado com sucesso!
echo Executando teste...
python litellm_prompt_cache.py

pause