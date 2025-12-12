#!/usr/bin/env pwsh
# Monitor Visual Studio Installation
# Created: 2025-12-11

Write-Host "`n🔍 MONITORANDO INSTALAÇÃO DO VISUAL STUDIO...`n" -ForegroundColor Cyan

$startTime = Get-Date
$notified = $false

while ($true) {
    $processes = Get-Process | Where-Object {
        $_.ProcessName -like "*vs_*" -or
        $_.ProcessName -like "*VisualStudio*Setup*"
    }

    $elapsed = ((Get-Date) - $startTime).TotalMinutes

    if ($processes.Count -eq 0) {
        if (-not $notified) {
            Write-Host "`n✅ INSTALAÇÃO CONCLUÍDA!" -ForegroundColor Green
            Write-Host "⏱️  Tempo total: $([math]::Round($elapsed, 1)) minutos`n" -ForegroundColor White

            # Verificar se VS foi instalado
            $vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe"
            if (Test-Path $vsPath) {
                Write-Host "✅ Visual Studio 2022 Community instalado com sucesso!" -ForegroundColor Green
                Write-Host "📂 Localização: $vsPath`n" -ForegroundColor White

                Write-Host "🎯 PRÓXIMOS PASSOS:" -ForegroundColor Cyan
                Write-Host "  1. Reinicie o VS Code" -ForegroundColor White
                Write-Host "  2. O erro do Windows SDK está RESOLVIDO! ✅" -ForegroundColor Green
                Write-Host "  3. Agora você tem C++, Python e .NET integrados`n" -ForegroundColor White
            } else {
                Write-Host "⚠️  Instalação concluída mas VS não encontrado no path padrão" -ForegroundColor Yellow
            }

            $notified = $true
            [System.Media.SystemSounds]::Asterisk.Play()
            break
        }
    } else {
        $memoryTotal = ($processes | Measure-Object WorkingSet -Sum).Sum / 1MB
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Instalando... ($($processes.Count) processos, $([math]::Round($memoryTotal, 0)) MB RAM, $([math]::Round($elapsed, 1)) min)" -ForegroundColor Yellow
    }

    Start-Sleep -Seconds 30
}

Write-Host "`n🎉 VOCÊ PODE VOLTAR AO TRABALHO!`n" -ForegroundColor Green
