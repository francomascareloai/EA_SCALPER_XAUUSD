# 🤖 Expert Advisors em Produção

## 📋 Índice de EAs

### 🏆 FTMO Ready - Eas Validados

#### 1. EA_VolatilityOptimizedS_v1.0_MULTI
- **Status**: ✅ Produção
- **Conformidade**: FTMO Ready
- **Estratégia**: SMA Otimizada por Volatilidade
- **Ativos**: XAUUSD, EURUSD
- **Timeframe**: M15, H1
- **Documentação**: [Ver detalhes](ftmo-ready/volatility-optimized.md)

#### 2. EA_GoldScalpingAI_v1.0_XAUUSD
- **Status**: ✅ Produção
- **Conformidade**: FTMO Ready
- **Estratégia**: Scalping AI para Ouro
- **Ativos**: XAUUSD
- **Timeframe**: M5, M15
- **Documentação**: [Ver detalhes](ftmo-ready/gold-scaling-ai.md)

### ⚡ Scalping - Alta Frequência

#### 3. EA_Scalping3_v1.0_MULTI
- **Status**: ✅ Produção
- **Estratégia**: Scalping Multi-Indicador
- **Ativos**: Multiativo
- **Timeframe**: M1, M5
- **Documentação**: [Ver detalhes](scalping/scalping3.md)

#### 4. EA_RonzSLTPMT5_v1.0_MULTI
- **Status**: ✅ Produção
- **Estratégia**: Gestão Avançada SL/TP
- **Ativos**: Multiativo
- **Timeframe**: M5, M15
- **Documentação**: [Ver detalhes](scalping/ronz-sltp.md)

#### 5. EA_ControlPanel2_v1.0_MULTI
- **Status**: ✅ Produção
- **Estratégia**: Painel de Controle
- **Ativos**: Multiativo
- **Timeframe**: Todos
- **Documentação**: [Ver detalhes](scalping/control-panel.md)

### 🧪 Experimental - Em Teste

#### 6. EA_SMC_Star_v2.0_MULTI
- **Status**: 🧪 Experimental
- **Estratégia**: Smart Money Concepts
- **Ativos**: XAUUSD, EURUSD
- **Timeframe**: M15, H1
- **Documentação**: [Ver detalhes](experimental/smc-star.md)

#### 7. EA_TechnicalMaster25Strategies_v1.0_MULTI
- **Status**: 🧪 Experimental
- **Estratégia**: 25 Estratégias Técnicas
- **Ativos**: Multiativo
- **Timeframe**: M5, M15, H1
- **Documentação**: [Ver detalhes](experimental/technical-master.md)

---

## 📊 Comparativo de Performance

| EA | Win Rate | Profit Factor | Max DD | Status | Recomendação |
|----|----------|---------------|--------|--------|--------------|
| EA_VolatilityOptimizedS | 72% | 1.85 | 4.2% | ✅ Produção | ⭐⭐⭐⭐⭐ |
| EA_GoldScalpingAI | 68% | 1.65 | 3.8% | ✅ Produção | ⭐⭐⭐⭐ |
| EA_Scalping3 | 65% | 1.55 | 5.1% | ✅ Produção | ⭐⭐⭐ |
| EA_RonzSLTPMT5 | 70% | 1.75 | 4.5% | ✅ Produção | ⭐⭐⭐⭐ |
| EA_SMC_Star | - | - | - | 🧪 Experimental | ⭐⭐ |

---

## 🎯 Seleção por Perfil de Trader

### 🔰 Iniciante
- **EA Recomendado**: EA_VolatilityOptimizedS
- **Motivo**: Simples, robusto, FTMO ready
- **Configuração**: Padrão

### 📈 Intermediário
- **EA Recomendado**: EA_RonzSLTPMT5
- **Motivo**: Gestão avançada de risco
- **Configuração**: Semi-avançada

### 🚀 Avançado
- **EA Recomendado**: EA_TechnicalMaster25Strategies
- **Motivo**: Múltiplas estratégias
- **Configuração**: Avançada

---

## ⚙️ Parâmetros Comuns

### Gestão de Risco Padrão
```mql5
input double MaxRiskPerTrade = 1.0;    // 1% por trade
input double MaxDailyLoss = 5.0;       // 5% loss diário
input int MaxPositions = 3;            // Máx. 3 posições
input bool UseBreakEven = true;        // Break-even automático
```

### Configurações de SL/TP
```mql5
input int StopLossPoints = 100;        // 100 pips SL
input int TakeProfitPoints = 200;      // 200 pips TP
input int TrailingStopPoints = 50;     // 50 pips trailing
```

---

## 📝 Notas de Versão

### v1.0 (2025-01-18)
- Lançamento inicial da documentação
- 5 EAs em produção validados
- 2 EAs experimentais em teste

### Roadmap
- [ ] Validar mais EAs FTMO ready
- [ ] Implementar backtesting automatizado
- [ ] Adicionar mais estratégias experimentais

---

## 🔗 Links Úteis

- [Configurações Recomendadas](../configuracoes/recommended-settings.md)
- [Guia FTMO Compliance](../ftmo-risk/compliance-guide.md)
- [Estratégias de Trading](../estrategias/index.md)