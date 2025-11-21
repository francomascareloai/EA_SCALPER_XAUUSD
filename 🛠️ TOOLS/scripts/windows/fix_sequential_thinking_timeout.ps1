# 🔧 Sequential Thinking MCP Timeout Fix Script
# This script fixes the "context deadline exceeded" error

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "🔧 FIXING SEQUENTIAL THINKING MCP TIMEOUT" -ForegroundColor Green
Write-Host "=" * 50

$targetFile = "C:\Users\Admin\AppData\Roaming\Qoder\SharedClientCache\mcp.json"
$backupFile = "C:\Users\Admin\AppData\Roaming\Qoder\SharedClientCache\mcp_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$fixedConfigFile = "c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\mcp_config_fixed_sequential_thinking.json"

try {
    # Check if source file exists
    if (-not (Test-Path $targetFile)) {
        Write-Host "❌ Target file not found: $targetFile" -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path $fixedConfigFile)) {
        Write-Host "❌ Fixed config file not found: $fixedConfigFile" -ForegroundColor Red
        exit 1
    }

    # Create backup
    Write-Host "📋 Creating backup..." -ForegroundColor Yellow
    Copy-Item $targetFile $backupFile -Force
    Write-Host "✅ Backup created: $backupFile" -ForegroundColor Green

    # Check current configuration
    $currentConfig = Get-Content $targetFile -Raw | ConvertFrom-Json
    if ($currentConfig.mcpServers."sequential_thinking".args -contains "-y") {
        Write-Host "⚠️  Found problematic '-y' parameter in sequential_thinking config" -ForegroundColor Yellow
        
        # Apply fix
        Write-Host "🔧 Applying fix..." -ForegroundColor Yellow
        Copy-Item $fixedConfigFile $targetFile -Force
        
        Write-Host "✅ Configuration fixed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 CHANGES MADE:" -ForegroundColor Cyan
        Write-Host "   - Removed '-y' parameter from sequential_thinking args" -ForegroundColor White
        Write-Host "   - This eliminates the startup delay causing timeout" -ForegroundColor White
        Write-Host ""
        Write-Host "🚀 NEXT STEPS:" -ForegroundColor Yellow
        Write-Host "   1. Restart Qoder IDE completely" -ForegroundColor White
        Write-Host "   2. Sequential thinking should start without timeout" -ForegroundColor White
        Write-Host "   3. Test by using sequential thinking features" -ForegroundColor White
        Write-Host ""
        Write-Host "💡 EXPLANATION:" -ForegroundColor Green
        Write-Host "   The '-y' parameter was causing npm to take extra time" -ForegroundColor White
        Write-Host "   during package resolution, triggering the timeout." -ForegroundColor White
        Write-Host "   Removing it uses the global installation directly." -ForegroundColor White
        
    } else {
        Write-Host "✅ Sequential thinking configuration already correct!" -ForegroundColor Green
        Write-Host "   No '-y' parameter found - timeout should not occur" -ForegroundColor White
    }

} catch {
    Write-Host "❌ Error during fix: $($_.Exception.Message)" -ForegroundColor Red
    
    if (Test-Path $backupFile) {
        Write-Host "🔄 Restoring backup..." -ForegroundColor Yellow
        Copy-Item $backupFile $targetFile -Force
        Write-Host "✅ Backup restored" -ForegroundColor Green
    }
    
    exit 1
}

Write-Host ""
Write-Host "=" * 50
Write-Host "🎯 SEQUENTIAL THINKING MCP FIX COMPLETE!" -ForegroundColor Green