#!/usr/bin/env pwsh
# LIMPEZA FINAL DOS ÚLTIMOS ARQUIVOS ÓRFÃOS
# Agente Organizador Expert - Toque Final
# Data: 24/08/2024

Write-Host "🎯 FINALIZANDO LIMPEZA DOS ÚLTIMOS ÓRFÃOS..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Yellow

$baseDir = "c:\Users\Admin\Documents\EA_SCALPER_XAUUSD"
$moved = 0
$cleaned = 0

Write-Host "`n🧹 MOVENDO ARQUIVOS RESTANTES..." -ForegroundColor Cyan

# Mover XAUUSD_ML_Complete_EA.mq5 para MAIN_EAS
$mlEA = Join-Path $baseDir "XAUUSD_ML_Complete_EA.mq5"
if (Test-Path $mlEA) {
    $destPath = Join-Path $baseDir "🚀 MAIN_EAS\DEVELOPMENT\XAUUSD_ML_Complete_EA.mq5"
    Move-Item $mlEA $destPath -Force
    Write-Host "✅ XAUUSD_ML_Complete_EA.mq5 -> MAIN_EAS/DEVELOPMENT" -ForegroundColor Green
    $moved++
}

# Mover prompt_arqiteto para DOCUMENTACAO_FINAL
$promptArq = Join-Path $baseDir "prompt_arqiteto"
if (Test-Path $promptArq) {
    $destPath = Join-Path $baseDir "📋 DOCUMENTACAO_FINAL\PROMPTS\prompt_arqiteto"
    Move-Item $promptArq $destPath -Force
    Write-Host "✅ prompt_arqiteto -> DOCUMENTACAO_FINAL/PROMPTS" -ForegroundColor Green
    $moved++
}

# Verificar se bmad-trading tem conteúdo útil, caso contrário mover
$bmadPath = Join-Path $baseDir "bmad-trading"
if (Test-Path $bmadPath) {
    $bmadContent = Get-ChildItem $bmadPath -Recurse
    if ($bmadContent.Count -gt 0) {
        $destPath = Join-Path $baseDir "🔧 WORKSPACE\bmad-trading"
        Move-Item $bmadPath $destPath -Force
        Write-Host "✅ bmad-trading -> WORKSPACE" -ForegroundColor Green
        $moved++
    }
}

Write-Host "`n📊 VERIFICANDO ESTRUTURA FINAL..." -ForegroundColor Cyan

# Contar itens em cada diretório principal
$mainDirs = @{
    "🚀 MAIN_EAS" = ""
    "📚 LIBRARY" = ""
    "🛠️ TOOLS" = ""
    "📊 DATA" = ""
    "📋 DOCUMENTACAO_FINAL" = ""
    "🔧 WORKSPACE" = ""
    "📋 METADATA" = ""
    "📊 TRADINGVIEW" = ""
    "🤖 AI_AGENTS" = ""
}

foreach ($dir in $mainDirs.Keys) {
    $fullPath = Join-Path $baseDir $dir
    if (Test-Path $fullPath) {
        $itemCount = (Get-ChildItem $fullPath -Recurse -File).Count
        Write-Host "📁 $dir`: $itemCount arquivos" -ForegroundColor White
    }
}

# Listar arquivos restantes na raiz (exceto arquivos do sistema)
$rootFiles = Get-ChildItem $baseDir -File | Where-Object { 
    -not $_.Name.StartsWith('.') -and
    -not $_.Name.StartsWith('🎯') -and
    -not $_.Name.StartsWith('🏆') -and
    -not $_.Name.StartsWith('📖') -and
    -not $_.Name.StartsWith('📊') -and
    -not $_.Name.EndsWith('.ps1') -and
    -not $_.Name.EndsWith('.code-workspace')
}

if ($rootFiles.Count -eq 0) {
    Write-Host "`n✅ RAIZ LIMPA! Nenhum arquivo órfão restante." -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Arquivos restantes na raiz:" -ForegroundColor Yellow
    $rootFiles | ForEach-Object { Write-Host "   - $($_.Name)" -ForegroundColor White }
}

Write-Host "`n📈 ESTATÍSTICAS FINAIS:" -ForegroundColor Yellow
Write-Host "✅ Arquivos movidos nesta limpeza: $moved" -ForegroundColor Green
Write-Host "✅ Diretórios principais: $($mainDirs.Keys.Count)" -ForegroundColor Green
Write-Host "✅ Arquivos órfãos na raiz: $($rootFiles.Count)" -ForegroundColor Green

if ($rootFiles.Count -eq 0 -and $moved -ge 1) {
    Write-Host "`n🎉 LIMPEZA TOTAL CONCLUÍDA COM SUCESSO!" -ForegroundColor Green
    Write-Host "🏆 PROJETO 100% ORGANIZADO E LIMPO!" -ForegroundColor Magenta
} elseif ($rootFiles.Count -eq 0) {
    Write-Host "`n✅ PROJETO JÁ ESTAVA PERFEITAMENTE LIMPO!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Alguns arquivos ainda precisam ser organizados manualmente." -ForegroundColor Yellow
}

Write-Host "`n🎯 LIMPEZA FINAL CONCLUÍDA!" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor Yellow
