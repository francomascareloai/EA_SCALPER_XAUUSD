# 🏷️ GUIA DE NOMENCLATURA E AUTOMAÇÃO - AGENTE ORGANIZADOR

## 📋 SISTEMA DE NOMENCLATURA RIGOROSO

### 🎯 PADRÃO OBRIGATÓRIO
```
[PREFIX]_[NOME]v[MAJOR.MINOR][_ESPECIFICO].[EXT]
```

### 🔖 PREFIXOS OBRIGATÓRIOS

| Prefixo | Tipo | Descrição | Exemplo |
|---------|------|-----------|----------|
| **EA_** | Expert Advisor | Robôs de trading | `EA_Scalper_v2.1_XAUUSD.mq5` |
| **IND_** | Indicator | Indicadores técnicos | `IND_OrderBlocks_v1.0_SMC.mq4` |
| **SCR_** | Script | Scripts utilitários | `SCR_RiskCalc_v1.2_FTMO.mq5` |
| **STR_** | Strategy | Estratégias TradingView | `STR_Breakout_v1.0_GOLD.pine` |
| **LIB_** | Library | Bibliotecas/Funções | `LIB_ICT_Functions_v2.0.mqh` |

### 🎨 ESPECIFICADORES RECOMENDADOS

#### **Por Mercado:**
- `_FOREX` - Pares de moedas gerais
- `_XAUUSD` - Ouro específico
- `_XAGUSD` - Prata específico
- `_INDICES` - Índices (SPX500, NAS100)
- `_CRYPTO` - Criptomoedas
- `_MULTI` - Multi-mercado

#### **Por Estratégia:**
- `_SCALP` - Scalping
- `_GRID` - Grid/Martingale
- `_SMC` - Smart Money Concepts
- `_ICT` - Inner Circle Trader
- `_TREND` - Trend Following
- `_MEAN` - Mean Reversion

#### **Por Compliance:**
- `_FTMO` - Compatível FTMO
- `_PROP` - Prop firms gerais
- `_DEMO` - Apenas demo
- `_LIVE` - Aprovado para live

#### **Por Timeframe:**
- `_M1` - 1 minuto
- `_M5` - 5 minutos
- `_M15` - 15 minutos
- `_H1` - 1 hora
- `_H4` - 4 horas
- `_D1` - Diário
- `_MTF` - Multi-timeframe

---

## ✅ EXEMPLOS DE RENOMEAÇÃO

### 🔄 ANTES → DEPOIS

```
❌ Beast_EA_V4.mq4
✅ EA_Beast_v4.0_XAUUSD_SCALP.mq4

❌ FFCal_v1.0_Multi_1.mq4
✅ IND_FFCal_v1.0_FOREX_MULTI.mq4

❌ TrueScalper_Ron_MT4_v112.mq4
✅ EA_TrueScalper_v1.12_FOREX_SCALP.mq4

❌ GMACD2.mq4
✅ IND_GMACD_v2.0_TREND_MULTI.mq4

❌ PZ_ParabolicSar_EA.mq4
✅ EA_ParabolicSar_v1.0_TREND_MULTI.mq4

❌ my_custom_indicator.mq5
✅ IND_Custom_v1.0_SMC_XAUUSD.mq5

❌ scalper_v2.mq5
✅ EA_Scalper_v2.0_XAUUSD_FTMO.mq5
```

---

## 🤖 SCRIPTS DE AUTOMAÇÃO

### 📝 Script PowerShell: Renomeação Automática

```powershell
# rename_files_auto.ps1
# Script para renomeação automática baseada em padrões

function Rename-TradingFiles {
    param(
        [string]$SourcePath = ".",
        [switch]$DryRun = $false
    )
    
    $renameRules = @{
        # EAs patterns
        "*_EA*.mq*" = { param($file) 
            $newName = $file.Name -replace "_EA", "" -replace "EA_", ""
            "EA_$newName"
        }
        
        # Indicators patterns
        "*indicator*.mq*" = { param($file)
            $newName = $file.Name -replace "indicator", "" -replace "IND_", ""
            "IND_$newName"
        }
        
        # Version normalization
        "*[Vv][0-9]*" = { param($file)
            $file.Name -replace "[Vv]([0-9]+)", "v`$1.0"
        }
        
        # Remove spaces and special chars
        "* *" = { param($file)
            $file.Name -replace " ", "_" -replace "[^a-zA-Z0-9._-]", ""
        }
    }
    
    Get-ChildItem -Path $SourcePath -Recurse -File | ForEach-Object {
        $originalName = $_.Name
        $newName = $originalName
        
        foreach ($pattern in $renameRules.Keys) {
            if ($_.Name -like $pattern) {
                $newName = & $renameRules[$pattern] $_
                break
            }
        }
        
        if ($newName -ne $originalName) {
            $newPath = Join-Path $_.Directory $newName
            
            if ($DryRun) {
                Write-Host "WOULD RENAME: $originalName → $newName" -ForegroundColor Yellow
            } else {
                try {
                    Rename-Item -Path $_.FullName -NewName $newName
                    Write-Host "RENAMED: $originalName → $newName" -ForegroundColor Green
                } catch {
                    Write-Host "ERROR: $originalName - $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
    }
}

# Uso:
# Rename-TradingFiles -SourcePath "C:\Path\To\Trading\Files" -DryRun
# Rename-TradingFiles -SourcePath "C:\Path\To\Trading\Files"
```

### 📊 Script PowerShell: Análise de Duplicatas

```powershell
# find_duplicates.ps1
# Script para identificar arquivos duplicados

function Find-TradingDuplicates {
    param(
        [string]$SourcePath = ".",
        [string]$OutputFile = "duplicates_report.txt"
    )
    
    $files = Get-ChildItem -Path $SourcePath -Recurse -File -Include "*.mq4", "*.mq5", "*.ex4", "*.ex5", "*.pine"
    $duplicates = @{}
    
    # Group by similar names (ignoring version numbers)
    $files | ForEach-Object {
        $baseName = $_.BaseName -replace "[Vv]?[0-9]+([._][0-9]+)*", "" -replace "_[0-9]+$", ""
        
        if (-not $duplicates.ContainsKey($baseName)) {
            $duplicates[$baseName] = @()
        }
        $duplicates[$baseName] += $_
    }
    
    # Report duplicates
    $report = @()
    $report += "# RELATÓRIO DE DUPLICATAS - $(Get-Date)"
    $report += "="*50
    
    foreach ($group in $duplicates.Keys) {
        if ($duplicates[$group].Count -gt 1) {
            $report += ""
            $report += "## GRUPO: $group"
            $report += "Arquivos encontrados: $($duplicates[$group].Count)"
            
            $duplicates[$group] | Sort-Object LastWriteTime -Descending | ForEach-Object {
                $report += "  - $($_.Name) ($(Get-Date $_.LastWriteTime -Format 'yyyy-MM-dd HH:mm')) - $($_.DirectoryName)"
            }
            
            # Suggest which to keep
            $newest = $duplicates[$group] | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $report += "  ✅ MANTER: $($newest.Name) (mais recente)"
            
            $toDelete = $duplicates[$group] | Where-Object { $_.FullName -ne $newest.FullName }
            $toDelete | ForEach-Object {
                $report += "  ❌ DELETAR: $($_.Name)"
            }
        }
    }
    
    $report | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host "Relatório salvo em: $OutputFile" -ForegroundColor Green
}

# Uso:
# Find-TradingDuplicates -SourcePath "C:\Path\To\Trading\Files"
```

### 🗂️ Script PowerShell: Criação de Estrutura

```powershell
# create_structure.ps1
# Script para criar a nova estrutura de pastas

function New-TradingStructure {
    param(
        [string]$RootPath = "."
    )
    
    $structure = @(
        "CODIGO_FONTE_LIBRARY",
        "CODIGO_FONTE_LIBRARY\MQL4_Source",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\FTMO_Ready",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Scalping",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Grid_Martingale",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Trend_Following",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Mean_Reversion",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Misc",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators\SMC_ICT",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators\Volume_Analysis",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators\Trend_Analysis",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators\Oscillators",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Indicators\Custom",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Scripts",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Scripts\Risk_Management",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Scripts\Utilities",
        "CODIGO_FONTE_LIBRARY\MQL4_Source\Scripts\Analysis",
        "CODIGO_FONTE_LIBRARY\MQL5_Source",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\EAs",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\EAs\FTMO_Ready",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\EAs\Advanced_Scalping",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\EAs\Multi_Symbol",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\EAs\Others",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Indicators",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Indicators\Order_Blocks",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Indicators\Volume_Flow",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Indicators\Market_Structure",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Indicators\Custom",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Scripts",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Scripts\Risk_Tools",
        "CODIGO_FONTE_LIBRARY\MQL5_Source\Scripts\Analysis_Tools",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Indicators",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Indicators\SMC_Concepts",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Indicators\Volume_Analysis",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Indicators\Custom_Plots",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Strategies",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Strategies\Backtesting",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Strategies\Alert_Systems",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Libraries",
        "CODIGO_FONTE_LIBRARY\TradingView_Scripts\Pine_Script_Source\Libraries\Pine_Functions",
        "CODIGO_FONTE_LIBRARY\Unknown",
        "EA_FTMO_XAUUSD_ELITE",
        "DOCUMENTATION",
        "DEVELOPMENT",
        "TESTING_VALIDATION",
        "REPORTS_ANALYTICS",
        "BACKUP_ARCHIVE"
    )
    
    foreach ($folder in $structure) {
        $fullPath = Join-Path $RootPath $folder
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Host "✅ Criado: $folder" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Já existe: $folder" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "🎯 Estrutura criada com sucesso!" -ForegroundColor Cyan
    Write-Host "📁 Total de pastas: $($structure.Count)" -ForegroundColor Cyan
}

# Uso:
# New-TradingStructure -RootPath "C:\Path\To\New\Structure"
```

---

## 📋 CHECKLIST DE QUALIDADE

### ✅ Para cada arquivo renomeado:
- [ ] Prefixo correto aplicado
- [ ] Versão no formato v[MAJOR.MINOR]
- [ ] Especificador de mercado/estratégia
- [ ] Extensão preservada
- [ ] Sem espaços ou caracteres especiais
- [ ] Nome descritivo e claro

### ✅ Para cada pasta:
- [ ] Nome em inglês
- [ ] Hierarquia lógica respeitada
- [ ] Máximo 3 níveis de profundidade
- [ ] Sem duplicação de conceitos
- [ ] Categorização clara

### ✅ Para documentação:
- [ ] INDEX.md criado para cada categoria
- [ ] Tags aplicadas corretamente
- [ ] Status de teste documentado
- [ ] Compatibilidade FTMO indicada
- [ ] Descrição clara e concisa

---

## 🎯 COMANDOS RÁPIDOS

### 🔍 Buscar arquivos por padrão:
```powershell
# Buscar EAs
Get-ChildItem -Recurse -Filter "*EA*.mq*"

# Buscar indicadores SMC
Get-ChildItem -Recurse -Filter "*SMC*.mq*"

# Buscar arquivos FTMO
Get-ChildItem -Recurse -Filter "*FTMO*.mq*"

# Buscar duplicatas por nome base
Get-ChildItem -Recurse -Filter "*.mq*" | Group-Object {$_.BaseName -replace "[Vv]?[0-9]+.*", ""} | Where-Object Count -gt 1
```

### 📊 Estatísticas rápidas:
```powershell
# Contar arquivos por tipo
Get-ChildItem -Recurse -Filter "*.mq4" | Measure-Object | Select-Object Count
Get-ChildItem -Recurse -Filter "*.mq5" | Measure-Object | Select-Object Count
Get-ChildItem -Recurse -Filter "*.pine" | Measure-Object | Select-Object Count

# Listar pastas com mais arquivos
Get-ChildItem -Recurse -Directory | ForEach-Object { 
    [PSCustomObject]@{
        Folder = $_.Name
        FileCount = (Get-ChildItem $_.FullName -File).Count
    }
} | Sort-Object FileCount -Descending | Select-Object -First 10
```

---

## 🚀 PRÓXIMOS PASSOS

1. **Executar scripts de análise** para entender escopo completo
2. **Testar renomeação** em modo DryRun primeiro
3. **Criar backup** antes de qualquer modificação
4. **Migrar por prioridade**: FTMO → SMC → Scalping → Outros
5. **Validar estrutura** após cada fase de migração

---

*Guia criado pelo Agente Organizador - Especialista em Estruturação de Códigos Trading*