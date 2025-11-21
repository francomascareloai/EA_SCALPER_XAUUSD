@echo off
echo ========================================
echo   AMBIENTE PYTHON CORRIGIDO COM SUCESSO
echo ========================================
echo.
echo Ativando ambiente virtual...
call "%~dp0.venv\Scripts\activate.bat"
echo.
echo Ambiente ativo! Use 'python' para executar scripts.
echo Para desativar, use 'deactivate'
echo.
echo Pacotes instalados:
pip list
echo.
cmd /k
