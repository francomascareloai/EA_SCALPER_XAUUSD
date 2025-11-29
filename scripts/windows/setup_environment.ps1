# Script para configurar ambiente Python
# Execute: .\setup_environment.ps1

Write-Host "=======================================" -ForegroundColor Green
Write-Host "  AMBIENTE PYTHON CORRIGIDO COM SUCESSO" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""

$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    Write-Host "Ativando ambiente virtual..." -ForegroundColor Yellow
    & $venvPath
    Write-Host ""
    Write-Host "Ambiente ativo! Use 'python' para executar scripts." -ForegroundColor Cyan
    Write-Host "Para desativar, use 'deactivate'" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Pacotes instalados:" -ForegroundColor Yellow
    pip list
} else {
    Write-Host "ERRO: Ambiente virtual não encontrado!" -ForegroundColor Red
    Write-Host "Execute primeiro: py -m venv .venv" -ForegroundColor Yellow
}
