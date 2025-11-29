# 🤖 PROPOSTA DE SUBAGENTES ESPECIALIZADOS - EA SCALPER XAUUSD

## 📋 **OVERVIEW**

Com base na análise profunda do projeto, proponho a criação de **12 subagentes especializados** para acelerar o desenvolvimento do robô de trading XAUUSD. Cada agente tem responsabilidades específicas e trabalha de forma coordenada.

---

## 🎯 **ARQUITETURA DE SUBAGENTES**

```
┌─────────────────────────────────────────────────────────────┐
│              COORDENADOR PRINCIPAL (VOCÊ)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐        ┌─────────┐
│ ANÁLISE │        │  BUILD  │        │  TESTE  │
└────┬────┘        └────┬────┘        └────┬────┘
     │                  │                   │
     └──────────────────┴───────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    DEPLOY         MONITOR         OPTIMIZE
```

---

## 🔧 **SUBAGENTES PROPOSTOS**

### **GRUPO 1: ANÁLISE E PESQUISA** 🔍

#### **1. AGENTE: Market Analyzer (Analista de Mercado)**
```yaml
Nome: market-analyzer
Tipo: Explore + bmm-market-researcher
Responsabilidade: Análise profunda de mercado XAUUSD
Ferramentas: Grep, Read, WebSearch, WebFetch
Tarefas:
  - Analisar padrões históricos do XAUUSD
  - Identificar níveis-chave de suporte/resistência
  - Pesquisar correlações com outros ativos
  - Analisar impacto de eventos econômicos
Output: Relatório de análise de mercado (JSON/MD)
Prioridade: P0 (Crítico)
```

**Como usar:**
```bash
# Invocar agente
Task(
  subagent_type="bmm-market-researcher",
  prompt="Analyze XAUUSD market structure for last 3 months, identify key levels and patterns for scalping strategy"
)
```

---

#### **2. AGENTE: Codebase Explorer (Explorador de Código)**
```yaml
Nome: codebase-explorer
Tipo: Explore + bmm-codebase-analyzer
Responsabilidade: Mapear e entender código existente
Ferramentas: Glob, Grep, Read, Bash
Tarefas:
  - Mapear todos os EAs na LIBRARY
  - Identificar melhores práticas de código
  - Detectar componentes reutilizáveis
  - Analisar padrões arquiteturais
Output: Mapa de código e componentes
Prioridade: P0 (Crítico)
```

**Como usar:**
```bash
Task(
  subagent_type="Explore",
  prompt="Explore LIBRARY folder, identify all scalping EAs with neural network implementation, analyze their architecture patterns",
  thoroughness="very thorough"
)
```

---

#### **3. AGENTE: Strategy Researcher (Pesquisador de Estratégias)**
```yaml
Nome: strategy-researcher
Tipo: bmm-market-researcher + bmm-trend-spotter
Responsabilidade: Pesquisar estratégias de scalping
Ferramentas: WebSearch, WebFetch, Read
Tarefas:
  - Pesquisar estratégias SMC para XAUUSD
  - Analisar EAs de sucesso no mercado
  - Identificar indicadores mais eficazes
  - Estudar risk management avançado
Output: Documento de estratégias recomendadas
Prioridade: P1 (Alto)
```

---

### **GRUPO 2: DESENVOLVIMENTO** 💻

#### **4. AGENTE: MQL5 Developer (Desenvolvedor MQL5)**
```yaml
Nome: mql5-developer
Tipo: general-purpose + code-reviewer
Responsabilidade: Desenvolver EA principal em MQL5
Ferramentas: Read, Write, Edit, Bash
Tarefas:
  - Criar EA_XAUUSD_Scalper_Elite_Unified.mq5
  - Implementar neural network nativa
  - Integrar Smart Money Concepts
  - Implementar risk management dinâmico
Output: EA principal compilável e testável
Prioridade: P0 (Crítico)
```

**Estrutura do EA:**
```mql5
// EA_XAUUSD_Scalper_Elite_Unified.mq5
#include <NeuralNetwork.mqh>
#include <SmartMoneyConcepts.mqh>
#include <RiskManager.mqh>
#include <MTFAnalyzer.mqh>

class CXAUUSDScalper {
  // Neural Network Engine
  CNeuralNetwork m_nn;

  // Smart Money Concepts
  COrderBlockDetector m_ob_detector;
  CFVGDetector m_fvg_detector;

  // Risk Management
  CRiskManager m_risk;

  // Multi-Timeframe
  CMTFAnalyzer m_mtf;
};
```

---

#### **5. AGENTE: Python AI Engineer (Engenheiro de IA Python)**
```yaml
Nome: python-ai-engineer
Tipo: ai-engineer
Responsabilidade: Desenvolver sistema AI em Python
Ferramentas: Write, Edit, Read, Bash
Tarefas:
  - Implementar KAN Networks para XAUUSD
  - Criar xLSTM analyzer
  - Desenvolver ensemble AI system
  - Integrar ONNX Runtime
Output: Módulos Python AI funcionais
Prioridade: P0 (Crítico)
```

**Estrutura AI:**
```python
# ai_core/
├── kan_network.py          # KAN implementation
├── xlstm_analyzer.py       # xLSTM time series
├── ensemble_system.py      # Ensemble AI
├── onnx_converter.py       # ONNX export
└── trading_environment.py  # RL environment
```

---

#### **6. AGENTE: Integration Specialist (Especialista em Integração)**
```yaml
Nome: integration-specialist
Tipo: network-engineer
Responsabilidade: Integrar MT5 ↔ Python
Ferramentas: Write, Edit, Bash, Read
Tarefas:
  - Implementar ZeroMQ bridge
  - Configurar WebSocket communication
  - Setup Redis cache layer
  - Criar shared memory buffer
Output: Sistema de comunicação funcional
Prioridade: P0 (Crítico)
```

**Componentes:**
```python
# integration/
├── zmq_bridge.py        # ZeroMQ MT5↔Python
├── websocket_server.py  # WebSocket real-time
├── redis_cache.py       # Cache layer
└── message_protocol.py  # Serialization
```

---

### **GRUPO 3: TESTES E QUALIDADE** 🧪

#### **7. AGENTE: Test Engineer (Engenheiro de Testes)**
```yaml
Nome: test-engineer
Tipo: bmm-test-coverage-analyzer
Responsabilidade: Criar e executar testes
Ferramentas: Write, Bash, Read, Edit
Tarefas:
  - Criar testes unitários para EA
  - Implementar testes de integração
  - Configurar backtesting framework
  - Validar FTMO compliance
Output: Suite de testes completa
Prioridade: P1 (Alto)
```

**Estrutura de Testes:**
```
tests/
├── unit/
│   ├── test_neural_network.py
│   ├── test_risk_manager.py
│   └── test_smc_detector.py
├── integration/
│   ├── test_zmq_bridge.py
│   └── test_ai_pipeline.py
└── backtest/
    ├── test_xauusd_strategy.py
    └── test_ftmo_compliance.py
```

---

#### **8. AGENTE: Quality Assurance (Garantia de Qualidade)**
```yaml
Nome: qa-specialist
Tipo: code-reviewer + mcp-testing-engineer
Responsabilidade: Revisar código e garantir qualidade
Ferramentas: Read, Grep, Bash
Tarefas:
  - Code review de todos os módulos
  - Validar padrões de código
  - Verificar segurança e performance
  - Garantir compliance FTMO
Output: Relatório de qualidade
Prioridade: P1 (Alto)
```

---

#### **9. AGENTE: Performance Optimizer (Otimizador de Performance)**
```yaml
Nome: performance-optimizer
Tipo: database-optimizer + general-purpose
Responsabilidade: Otimizar performance do sistema
Ferramentas: Bash, Read, Edit
Tarefas:
  - Analisar latência do sistema
  - Otimizar queries e processamento
  - Melhorar uso de memória/CPU
  - Atingir target de <5ms latência
Output: Sistema otimizado
Prioridade: P1 (Alto)
```

---

### **GRUPO 4: DEPLOY E OPERAÇÕES** 🚀

#### **10. AGENTE: DevOps Engineer (Engenheiro DevOps)**
```yaml
Nome: devops-engineer
Tipo: cloud-architect
Responsabilidade: Setup infraestrutura e deploy
Ferramentas: Bash, Write, Edit
Tarefas:
  - Configurar ambiente de desenvolvimento
  - Setup VPS/Cloud para produção
  - Implementar CI/CD pipeline
  - Configurar monitoring
Output: Infraestrutura automatizada
Prioridade: P2 (Médio)
```

**Infraestrutura:**
```yaml
# docker-compose.yml
services:
  mt5-bridge:
    build: ./mt5-bridge
    ports: ["5555:5555"]

  ai-core:
    build: ./ai-core
    gpus: all

  redis:
    image: redis:latest

  postgres:
    image: postgres:14

  monitoring:
    image: grafana/grafana
```

---

#### **11. AGENTE: Monitoring Specialist (Especialista em Monitoramento)**
```yaml
Nome: monitoring-specialist
Tipo: business-analyst
Responsabilidade: Monitorar sistema em produção
Ferramentas: WebFetch, Bash, Write
Tarefas:
  - Criar dashboard de métricas
  - Setup alertas de performance
  - Monitorar trades em tempo real
  - Analisar KPIs de trading
Output: Dashboard de monitoramento
Prioridade: P2 (Médio)
```

**Métricas Monitoradas:**
```python
# Métricas de Trading
- Win Rate (%)
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Daily PnL

# Métricas Técnicas
- Latência (ms)
- CPU/Memory usage
- AI inference time
- Order execution speed
```

---

#### **12. AGENTE: Documentation Writer (Escritor de Documentação)**
```yaml
Nome: doc-writer
Tipo: bmm-document-reviewer
Responsabilidade: Criar e manter documentação
Ferramentas: Write, Read, Edit
Tarefas:
  - Documentar arquitetura do sistema
  - Criar guia de uso do EA
  - Documentar APIs e integrações
  - Manter changelog atualizado
Output: Documentação completa
Prioridade: P2 (Médio)
```

---

## 🎯 **COORDENAÇÃO DE SUBAGENTES**

### **Fluxo de Trabalho Recomendado:**

```
FASE 1 - ANÁLISE (Semana 1)
├── Market Analyzer: Analisar XAUUSD
├── Codebase Explorer: Mapear código
└── Strategy Researcher: Pesquisar estratégias
    │
    ▼
FASE 2 - DESENVOLVIMENTO (Semana 2-3)
├── MQL5 Developer: Criar EA principal
├── Python AI Engineer: Implementar AI
└── Integration Specialist: Integrar sistemas
    │
    ▼
FASE 3 - TESTES (Semana 4)
├── Test Engineer: Criar testes
├── QA Specialist: Revisar código
└── Performance Optimizer: Otimizar
    │
    ▼
FASE 4 - DEPLOY (Semana 5)
├── DevOps Engineer: Deploy
├── Monitoring Specialist: Monitorar
└── Doc Writer: Documentar
```

---

## 📊 **MATRIZ DE RESPONSABILIDADES (RACI)**

| Tarefa | Market Analyzer | Codebase Explorer | MQL5 Dev | Python AI | Integration | Test Eng | QA | Perf Opt | DevOps | Monitor | Doc |
|--------|----------------|-------------------|----------|-----------|-------------|----------|----|---------| -------|---------|-----|
| Análise Mercado | **R** | C | I | I | I | - | - | - | - | I | C |
| Mapear Código | I | **R** | C | C | C | I | C | I | - | - | C |
| Criar EA MQL5 | C | I | **R** | C | C | A | A | I | - | - | C |
| Implementar AI | C | I | C | **R** | C | A | A | I | - | - | C |
| Integração | I | I | C | C | **R** | A | A | I | C | - | C |
| Testes | - | - | I | I | I | **R** | A | C | - | I | C |
| Code Review | - | I | A | A | A | C | **R** | C | - | - | C |
| Otimização | - | I | I | I | I | C | C | **R** | - | I | C |
| Deploy | - | - | I | I | I | - | - | - | **R** | C | C |
| Monitoramento | I | - | - | - | - | - | - | I | C | **R** | C |
| Documentação | C | C | A | A | A | A | A | A | A | A | **R** |

**Legenda RACI:**
- **R** = Responsible (Responsável)
- **A** = Accountable (Aprovador)
- **C** = Consulted (Consultado)
- **I** = Informed (Informado)

---

## 🚀 **COMO INVOCAR OS SUBAGENTES**

### **Exemplo 1: Análise de Mercado**
```python
# Invocar Market Analyzer
Task(
    description="Analyze XAUUSD market",
    subagent_type="bmm-market-researcher",
    prompt="""
    Analyze XAUUSD market for the last 6 months:
    1. Identify key support/resistance levels
    2. Analyze correlation with USD index
    3. Study impact of Federal Reserve decisions
    4. Recommend best trading sessions (London/NY)
    5. Provide statistical analysis of volatility patterns

    Output: Comprehensive market analysis report in Markdown format
    """
)
```

### **Exemplo 2: Desenvolvimento MQL5**
```python
# Invocar MQL5 Developer
Task(
    description="Create unified EA",
    subagent_type="general-purpose",
    prompt="""
    Create EA_XAUUSD_Scalper_Elite_Unified.mq5 with:

    Requirements:
    - Native neural network (64→32→16→3 architecture)
    - Smart Money Concepts (Order Blocks + FVG detection)
    - Dynamic risk management (1% base risk)
    - Multi-timeframe analysis (M5, M15, H1, D1)
    - Trailing stop system
    - FTMO compliance (max 5% drawdown)

    Code should be:
    - Well-commented
    - Modular (separate classes)
    - Optimized for performance
    - Ready for backtesting

    Output: Complete MQL5 file ready to compile
    """
)
```

### **Exemplo 3: Implementação AI**
```python
# Invocar Python AI Engineer
Task(
    description="Implement KAN network",
    subagent_type="ai-engineer",
    prompt="""
    Implement KAN (Kolmogorov-Arnold Network) for XAUUSD trading:

    Architecture:
    - Input: 64 features (price, indicators, volume, etc.)
    - Hidden layers: KAN with spline-based activation
    - Output: 3 classes (BUY, SELL, HOLD)

    Features to extract:
    - OHLCV data (last 50 candles)
    - Technical indicators (RSI, MACD, ATR, etc.)
    - Smart Money Concepts (OB, FVG proximity)
    - Multi-timeframe trend alignment

    Training:
    - Use last 2 years of XAUUSD M5 data
    - Train/val/test split: 70/15/15
    - Early stopping on validation loss
    - Export to ONNX format for MT5 integration

    Output: Python module with trained KAN model
    """
)
```

---

## 📊 **MÉTRICAS DE SUCESSO DOS AGENTES**

### **KPIs por Agente:**

| Agente | KPI Principal | Target | Medição |
|--------|--------------|--------|---------|
| Market Analyzer | Qualidade insights | 90%+ | Review score |
| Codebase Explorer | Cobertura código | 100% | Files mapped |
| MQL5 Developer | EA funcional | 100% | Compile success |
| Python AI Engineer | Precisão modelo | >90% | Validation acc |
| Integration Spec | Latência | <10ms | Benchmark |
| Test Engineer | Code coverage | >80% | pytest-cov |
| QA Specialist | Bugs encontrados | 0 críticos | Issues |
| Performance Opt | Latência final | <5ms | Profiling |
| DevOps Engineer | Uptime | 99.9% | Monitoring |
| Monitor Specialist | Alert response | <5min | Avg time |
| Doc Writer | Completude | 100% | Sections |

---

## 🎯 **PLANO DE EXECUÇÃO IMEDIATO**

### **HOJE (Dia 1):**
1. ✅ Análise profunda completa
2. ✅ Proposta de subagentes criada
3. ⏳ **Próximo**: Invocar **Market Analyzer**
4. ⏳ **Próximo**: Invocar **Codebase Explorer**

### **AMANHÃ (Dia 2):**
1. ⏳ Invocar **Strategy Researcher**
2. ⏳ Iniciar **MQL5 Developer**
3. ⏳ Iniciar **Python AI Engineer**

### **SEMANA 1:**
1. ⏳ Completar análise de mercado
2. ⏳ Mapear completamente código
3. ⏳ Criar EA base (skeleton)
4. ⏳ Implementar KAN network base

---

## 💡 **DICAS DE USO DOS SUBAGENTES**

### **1. Seja Específico nos Prompts**
✅ **BOM**: "Analyze XAUUSD support/resistance on H1 for last 3 months"
❌ **RUIM**: "Analyze XAUUSD"

### **2. Use Múltiplos Agentes em Paralelo**
```python
# Invocar 3 agentes simultaneamente
Task(..., subagent_type="bmm-market-researcher"),
Task(..., subagent_type="Explore"),
Task(..., subagent_type="ai-engineer")
```

### **3. Encadeie Resultados**
```python
# Passo 1: Analisar
result1 = Market_Analyzer.analyze()

# Passo 2: Usar resultado para desenvolver
MQL5_Developer.create_ea(based_on=result1)
```

### **4. Revise Sempre**
Após cada agente, **revise o output** antes de prosseguir.

---

## 🎉 **CONCLUSÃO**

Com estes **12 subagentes especializados**, você terá um **exército de especialistas** trabalhando no seu projeto EA_SCALPER_XAUUSD!

**Próximo passo recomendado:**
Invocar os 3 primeiros agentes para começar a análise e mapeamento:
1. **Market Analyzer**
2. **Codebase Explorer**
3. **Strategy Researcher**

**Deseja que eu invoque algum destes agentes agora?** 🚀

---

*Documento criado em: 19/10/2025*
*Método BMAD aplicado com sucesso!* ✅
