@echo off
echo ==========================================
echo   INICIANDO LITELLM PROXY PARA ROO CODE
echo ==========================================
echo.
echo 🚀 Configurações:
echo 📡 URL: http://127.0.0.1:4000
echo 🔑 Key: sk-litellm-proxy-key-12345
echo 🤖 Modelos: qwen-coder, deepseek-r1
echo.
echo 💾 Prompt Caching: ATIVO
echo ⚡ Rate Limiting: Configurado
echo.
echo Para parar: Ctrl+C
echo ==========================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat

python -m litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000 --detailed_debug

pause
