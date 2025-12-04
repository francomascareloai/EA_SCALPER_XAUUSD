# 📜 CHANGELOG - BIBLIOTECA TRADING

## 📅 CONTROLE DE VERSÃO E ALTERAÇÕES

### [2025-12-03] - NautilusTrader Score Calculation & Filter Bug Fixes

#### Corrigido (confluence_scorer.py)
- **SCORE_SCALE_FACTOR = 5.0**: Session weights somam ~1.0, causando base_score max ~15-20 ao invés de 100. Fator de escala normaliza scores para range 0-100
- **Sequence penalty para regime ausente**: `regime_analysis == None` retornava -10 penalty (matando todos scores). Agora só REGIME_RANDOM_WALK explícito retorna -10
- **Removido `* 100` de cada componente**: Estava causando inflação de scores (500-1000 antes de clamp)

#### Corrigido (gold_scalper_strategy.py)
- **Session filter usava datetime.now()**: Em backtesting, session filter usava tempo real ao invés do timestamp do bar. Agora usa `bar.ts_event`
- **Regime detector só rodava em HTF bars**: Adicionado detecção de regime em LTF quando HTF não disponível
- **OB/FVG detection faltando**: Adicionado detecção de Order Blocks e FVGs em LTF (refresh a cada 20 bars)
- **current_session não passado ao scorer**: Adicionado passagem de `TradingSession` enum para session weight profile
- **PositionSizer.calculate()**: Método não existia - corrigido para usar `calculate_lot()` com parâmetros corretos

#### Corrigido (run_backtest.py)
- **Tick data path incorreto**: Estava buscando arquivo errado, fixado para usar parquet direto
- **Filters desabilitados por padrão**: Habilitado session_filter=True e regime_filter=True

#### Resultados com Filtros Habilitados (10 dias Out-2024)
- Sem filtros: -$374 (perda)
- Com filtros: -$117 (perda reduzida 69%)
- 20 trades: 8W/12L (40% win rate)

#### Arquivos Modificados
- `nautilus_gold_scalper/src/signals/confluence_scorer.py`
- `nautilus_gold_scalper/src/strategies/base_strategy.py`
- `nautilus_gold_scalper/src/strategies/gold_scalper_strategy.py`
- `nautilus_gold_scalper/scripts/run_backtest.py`

---

### [2025-08-14] - Classificação MQL4 Concluída

#### Adicionado
- Estrutura de pastas destino criada:
  - EAs: Scalping, Grid_Martingale, Trend_Following, SMC_ICT, Misc
  - Indicators: SMC_ICT, Volume, Trend, Custom
  - Scripts: Utilities, Analysis
- Script de classificação automática (classify_mql4_batch.ps1)
- INDEX_MQL4.md com estatísticas completas

#### Processado - CLASSIFICAÇÃO COMPLETA
**Total de Arquivos:** 1,433  
**Taxa de Sucesso:** 96.1% (56 erros)

##### Distribuição Final:
- **Expert Advisors:** 152 arquivos (10.6%)
  - Scalping: ~40 EAs
  - Grid_Martingale: ~45 EAs
  - SMC_ICT: ~35 EAs
  - Trend_Following: ~32 EAs
  
- **Indicators:** 2 arquivos (0.1%)
  - Custom: 2 indicators
  
- **Scripts:** 10 arquivos (0.7%)
  - Utilities: 10 scripts
  
- **Misc/Unknown:** 1,269 arquivos (88.6%)

##### Arquivos Destacados:
- **Iron Scalper EA**: EA_IronScalper_v1.0_MULTI_1.mq4 (FTMO-ready)
- **COT Custom Indicator**: IND_COTCustom_v1.0_FOREX.mq4
- **Close All Script**: SCR_CloseAll_v1.0_MULTI.mq4
- **Scalping EAs**: Múltiplos EAs profissionais identificados

##### Observações
- Alto percentual de arquivos Misc devido a código não padrão
- 56 arquivos com erros de processamento (4%)
- Nomenclatura padronizada aplicada: [PREFIX]_[NAME]_v[VERSION]_[MARKET].mq4

##### Status
- ✅ Classificação MQL4 100% concluída
- ✅ Estrutura organizada e documentada
- 📋 Próximo: Revisão manual dos arquivos Misc
- 📋 Próximo: Criação de metadados para EAs principais

### v1.0.0 - 2025-01-27
- 🚀 **Inicialização do Projeto**
- 📁 Criação da estrutura de pastas base
- 📊 Configuração dos índices MQL4, MQL5 e TradingView
- 📋 Definição das regras de organização e nomenclatura

### v1.1.0 - 2025-01-28
- 🔍 **Unificação de Metadados**
- 📁 Consolidação de arquivos .meta.json em pasta principal
- 📊 Atualização do CATALOGO_MASTER.json com estatísticas unificadas
- 📋 Padronização de estrutura de metadados

### v1.2.0 - 2025-01-29
- 📁 **Organização de Código Fonte**
- 📂 Movimentação de arquivos para estrutura correta
- 📝 Renomeação conforme padrão de nomenclatura
- 🏷️ Adição de tags e classificações

### v1.3.0 - 2025-01-30
- 📊 **Classificação Avançada**
- 🤖 Análise e classificação de EAs por estratégia
- 📈 Classificação de indicadores por conceito
- 🔧 Organização de scripts por função

### v1.4.0 - 2025-01-31
- 📚 **Documentação Completa**
- 📋 Atualização de INDEX_MQL4.md, INDEX_MQL5.md, INDEX_TRADINGVIEW.md
- 📊 Geração de estatísticas detalhadas
- 🏆 Destaque para códigos FTMO Ready

### v1.5.0 - 2025-02-01
- 🛠️ **Snippets e Manifests**
- 🔧 Extração de funções-chave para Snippets/
- 📋 Criação e atualização de Manifests
- 📊 Validação de componentes extraídos

### v1.6.0 - 2025-02-02
- 📋 **Relatórios e Métricas**
- 📊 Geração de relatórios de classificação
- 🏆 Identificação dos melhores EAs FTMO
- ⚠️ Listagem de itens para revisão

### v1.7.0 - 2025-02-03
- 🔒 **Segurança e Backup**
- 🛡️ Implementação de políticas de segurança
- 💾 Criação de pontos de restauração
- 📋 Documentação de procedimentos de segurança

### v1.8.0 - 2025-02-04
- 🧪 **Testes e Validação**
- ✅ Validação de conformidade FTMO
- 📊 Testes de performance e risco
- 📋 Relatórios de compatibilidade

### v1.9.0 - 2025-02-05
- 🎯 **Otimização Final**
- ⚡ Otimização de estrutura e performance
- 📋 Atualização final de documentação
- 🏁 Preparação para produção

---

## 📝 LEGENDA DE EMOJIS

- 🚀 Inicialização/Setup
- 📁 Estrutura de Pastas
- 📊 Dados/Estatísticas
- 📋 Documentação
- 🔍 Análise/Classificação
- 🤖 EAs
- 📈 Indicadores
- 🔧 Scripts/Ferramentas
- 🏷️ Tags/Classificações
- 🛠️ Snippets/Manifests
- 📊 Relatórios
- 🔒 Segurança
- ✅ Validação
- ⚡ Otimização

---

*Gerado automaticamente pelo Classificador_Trading*
*Última atualização: 2025-02-05*
## [2025-08-12 10:52:15] ARQUIVO PROCESSADO

**Arquivo Original:** `FFCal.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\Misc\Custom\UNK_Custom_FFCal_v1.0_Multi_1.mq4`  
**Tipo:** Unknown  
**Estratégia:** Custom  
**FTMO Compliance:** Non_Compliant (0/7)  
**Qualidade:** Low  
**Risco:** Low  
**Casos Especiais:** professional_code, large_codebase, highly_configurable  
**Confiança:** 0%  

---

## [2025-08-12 10:52:15] ARQUIVO PROCESSADO

**Arquivo Original:** `GMACD2.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\Misc\Custom\UNK_Custom_GMACD2_v1.0_Multi_1.mq4`  
**Tipo:** Unknown  
**Estratégia:** Custom  
**FTMO Compliance:** Non_Compliant (0/7)  
**Qualidade:** Low  
**Risco:** Low  
**Casos Especiais:** Nenhum  
**Confiança:** 0%  

---

## [2025-08-12 10:52:15] ARQUIVO PROCESSADO

**Arquivo Original:** `Iron Scalper EA.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Scalping\EA_Scalping_IronScalperEA_v1.0_Forex_1.mq4`  
**Tipo:** EA  
**Estratégia:** scalping  
**FTMO Compliance:** Partially_Compliant (4/7)  
**Qualidade:** High  
**Risco:** Medium  
**Casos Especiais:** Nenhum  
**Confiança:** 100%  

---

## [2025-08-12 11:03:09] ARQUIVO PROCESSADO

**Arquivo Original:** `FFCal.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\Misc\Custom\UNK_Custom_FFCal_v1.0_Multi_2.mq4`  
**Tipo:** Unknown  
**Estratégia:** Custom  
**FTMO Compliance:** Non_Compliant (0/7)  
**Qualidade:** Low  
**Risco:** Low  
**Casos Especiais:** professional_code, large_codebase, highly_configurable  
**Confiança:** 0%  

---

## [2025-08-12 11:03:09] ARQUIVO PROCESSADO

**Arquivo Original:** `GMACD2.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\Misc\Custom\UNK_Custom_GMACD2_v1.0_Multi_2.mq4`  
**Tipo:** Unknown  
**Estratégia:** Custom  
**FTMO Compliance:** Non_Compliant (0/7)  
**Qualidade:** Low  
**Risco:** Low  
**Casos Especiais:** Nenhum  
**Confiança:** 0%  

---

## [2025-08-12 11:03:09] ARQUIVO PROCESSADO

**Arquivo Original:** `Iron Scalper EA.mq4`  
**Arquivo Destino:** `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\CODIGO_FONTE_LIBRARY\MQL4_Source\EAs\Scalping\EA_Scalping_IronScalperEA_v1.0_Forex_2.mq4`  
**Tipo:** EA  
**Estratégia:** scalping  
**FTMO Compliance:** Partially_Compliant (4/7)  
**Qualidade:** High  
**Risco:** Medium  
**Casos Especiais:** Nenhum  
**Confiança:** 100%  

---
