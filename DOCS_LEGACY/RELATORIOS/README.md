# 🚀 EA_SCALPER_XAUUSD - Projeto Organizado

## 📊 Visão Geral
Projeto de Expert Advisors para trading automatizado em XAUUSD, organizados com estrutura otimizada para performance e escalabilidade.

**Data de Organização:** 24/08/2025
**Versão da Estrutura:** 2.0

## 📁 Estrutura Principal

### 🚀 MAIN_EAS/
EAs principais do projeto com acesso direto:
- **PRODUCTION/**: EAs prontos para produção
- **DEVELOPMENT/**: EAs em desenvolvimento ativo
- **RELEASES/**: Candidatos a release

### 📚 LIBRARY/
Biblioteca organizada por tecnologia e categoria:
- **MQL5_Components/**: Componentes MQL5 (EAs, Indicators, Scripts, Include)
- **MQL4_Components/**: Componentes MQL4 (legado)
- **TradingView/**: Scripts Pine Script

### 📊 METADATA/
Metadados organizados por performance (máx. 500 arquivos/pasta):
- **EA_Metadata/**: Metadados dos Expert Advisors
  - FTMO_Compatible/ (prioridade máxima)
  - Scalping_Systems/
  - SMC_ICT_Systems/
  - Grid_Systems/
  - Trend_Following/
  - Archive/

### 🔧 WORKSPACE/
Ambiente de desenvolvimento:
- **Active_Development/**: Desenvolvimento em andamento
- **Testing/**: Testes e validação
- **Sandbox/**: Experimentos rápidos

### 🛠️ TOOLS/
Ferramentas e automação:
- **Build/**: Scripts de compilação
- **Testing/**: Ferramentas de teste
- **Automation/**: Scripts de automação

## 📈 Estatísticas do Projeto

### MAIN_EAS
- **Total de arquivos:** 4
- **Total de diretórios:** 8
- **Tipos de arquivo:** .mq5(4)

### LIBRARY
- **Total de arquivos:** 13451
- **Total de diretórios:** 13506
- **Tipos de arquivo:** .json(1), .txt(839), .pine(27), .py(4), .mqh(62), .mq5(546), .ex5(4), .mq4(11929), .ex4(39)

### WORKSPACE
- **Total de arquivos:** 0
- **Total de diretórios:** 4
- **Tipos de arquivo:** 

### METADATA
- **Total de arquivos:** 6364
- **Total de diretórios:** 6382
- **Tipos de arquivo:** .json(6364)

### TOOLS
- **Total de arquivos:** 13
- **Total de diretórios:** 13
- **Tipos de arquivo:** .py(10), .json(3)

## 🎯 Melhorias Implementadas

### ✅ Performance Otimizada
- Metadados reorganizados: **6364** arquivos
- Pastas vazias removidas: **598**
- Máximo 500 arquivos por diretório
- Acesso direto aos EAs principais

### ✅ Organização por Prioridade
1. **FTMO-compatible EAs** (HIGHEST)
2. **XAUUSD specialists + SMC/Order Blocks** (HIGH)
3. **General scalping + trend following** (MEDIUM)
4. **Grid/martingale + experimental** (LOW)

### ✅ Convenção de Nomenclatura
Padrão: `[TYPE]_[NAME]v[VERSION][SPECIFIC].[EXT]`

Exemplo: `EA_FTMO_Scalper_Elite_v2.12_XAUUSD.mq5`

## 🚀 Quick Start

### Compilar EAs Principais
```bash
# Windows
cd TOOLS/Build
compile_main_eas.bat

# Python
python TOOLS/Build/compile_main_eas.py
```

### Localizar Arquivos
- **EAs Principais:** `MAIN_EAS/PRODUCTION/`
- **Biblioteca:** `LIBRARY/MQL5_Components/EAs/`
- **Metadados:** `METADATA/EA_Metadata/`

## 📋 Índices de Referência
- **MASTER_INDEX.json**: Índice completo do projeto
- **LIBRARY/LIBRARY_INDEX.json**: Índice da biblioteca
- **METADATA/METADATA_INDEX.json**: Índice de metadados

---
**Última atualização:** 24/08/2025 14:24
