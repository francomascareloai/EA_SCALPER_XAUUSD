# 🔌 GUIA COMPLETO - PESQUISA E CONFIGURAÇÃO DE MCPs

## 📋 **OVERVIEW**

Este documento contém **prompts estruturados** para pesquisar, avaliar e configurar os **melhores MCPs (Model Context Protocol servers)** para cada etapa do projeto EA XAUUSD Scalper Elite.

**Objetivo:** Identificar e configurar MCPs ideais para os 12 subagentes usando Claude Code + Codex.

---

## 🎯 **ESTRUTURA DE PESQUISA**

```
FASE 1: MCPs para ANÁLISE E PESQUISA
├── Market research & data
├── Codebase exploration
└── Strategy research

FASE 2: MCPs para DESENVOLVIMENTO
├── MQL5 development
├── Python AI/ML development
└── System integration

FASE 3: MCPs para TESTES
├── Testing & QA
├── Performance analysis
└── Security & compliance

FASE 4: MCPs para DEPLOY & OPS
├── Infrastructure & deployment
├── Monitoring & observability
└── Documentation
```

---

# 📊 PROMPT 1: MCPs PARA PESQUISA DE MERCADO E DADOS

## **CONTEXTO**
Subagentes: Market Analyzer, Strategy Researcher
Necessidades: Busca de dados financeiros, análise de mercado XAUUSD, papers acadêmicos, notícias

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Market Research & Financial Data

## OBJETIVO
Identificar os melhores MCPs para pesquisa de mercado financeiro, análise de XAUUSD, e estratégias de trading.

## CATEGORIAS A PESQUISAR

### 1. WEB SEARCH MCPs
Pesquise e compare:
- **Brave Search MCP**
  - Capabilities
  - Rate limits
  - Free vs Paid
  - Quality para financial research
  - Instalação no Claude Code

- **Perplexity MCP**
  - API availability
  - Search quality para trading
  - Cost
  - Integration difficulty
  - Real-time data?

- **Tavily MCP**
  - Especialização em research
  - Depth of search
  - Pricing model
  - Claude Code compatibility
  - Best use cases

- **SearXNG MCP** (self-hosted)
  - Setup complexity
  - Privacy benefits
  - Performance
  - Maintenance burden

**Compare em tabela:**
| MCP | Search Quality | Real-time Data | Cost | Setup Difficulty | Best For |
|-----|----------------|----------------|------|------------------|----------|

**Recomende:** Top 2 para market research

### 2. FINANCIAL DATA MCPs
Pesquise:
- **Yahoo Finance MCP**
  - XAUUSD data availability
  - Historical data depth
  - Free tier limits
  - API reliability

- **Alpha Vantage MCP**
  - Forex data quality
  - Intraday data (M5)
  - Free vs Premium
  - Rate limits

- **Twelve Data MCP**
  - Gold/Forex coverage
  - Real-time capabilities
  - Pricing
  - API stability

- **FMP (Financial Modeling Prep) MCP**
  - Commodities data
  - Technical indicators
  - Websocket support
  - Cost structure

**Recomende:** Best MCP for XAUUSD M5 data

### 3. NEWS & SENTIMENT MCPs
Pesquise:
- **NewsAPI MCP**
  - Financial news coverage
  - XAUUSD relevant sources
  - Free tier
  - Update frequency

- **Finnhub MCP**
  - Market news quality
  - Economic calendar
  - Sentiment analysis
  - Integration ease

**Recomende:** Best for gold market news

### 4. RESEARCH PAPER MCPs
- **Arxiv MCP**
  - Trading/ML papers
  - Search capabilities
  - PDF access

- **Semantic Scholar MCP**
  - Academic quality
  - Citation network
  - API limits

**Recomende:** Best for trading research papers

## DELIVERABLE
Para cada categoria, forneça:
1. Top 2 MCPs recomendados
2. Installation command for Claude Code
3. Configuration example (.roo/mcp.json)
4. Usage examples
5. Cost analysis (monthly estimate)

## VALIDATION CRITERIA
- ✅ Works with Claude Code
- ✅ Relevant for XAUUSD trading
- ✅ Reliable API
- ✅ Reasonable cost
- ✅ Good documentation
```

---

# 🔍 PROMPT 2: MCPs PARA CODEBASE EXPLORATION

## **CONTEXTO**
Subagente: Codebase Explorer
Necessidades: Análise de código MQL4/MQL5, search em 150+ EAs, pattern detection

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Code Analysis & Exploration

## OBJETIVO
Identificar MCPs ideais para análise de codebase (150+ EAs MQL4/MQL5), search de código, e pattern detection.

## CATEGORIAS A PESQUISAR

### 1. CODE SEARCH MCPs
- **Sourcegraph MCP**
  - Code search quality
  - MQL5 language support
  - Self-hosted vs Cloud
  - Claude Code integration

- **GitHub MCP**
  - Repository search
  - Code browsing
  - Rate limits (5000/hour)
  - Free tier adequacy

- **Searchcode MCP**
  - Multi-language support
  - Search precision
  - Setup complexity

**Compare:** Which is best for searching 150+ MQL5 files?

### 2. CODE ANALYSIS MCPs
- **Tree-sitter MCP**
  - AST parsing
  - MQL5 grammar support
  - Performance
  - Use cases

- **CodeQL MCP**
  - Static analysis
  - Security scanning
  - Custom queries
  - Complexity

- **SonarQube MCP**
  - Code quality metrics
  - MQL5 support
  - Self-hosted requirement
  - Integration effort

**Recomende:** Best for MQL5 code quality analysis

### 3. DOCUMENTATION MCPs
- **Devdocs MCP**
  - MQL5 reference docs
  - Offline capability
  - Search speed

- **MDN MCP** (if applicable)
  - Relevance for trading
  - Quality of docs

**Recomende:** Best MCP for MQL5 documentation access

## DELIVERABLE
1. Top MCP for code search in local codebase
2. Top MCP for code quality analysis
3. Configuration for Claude Code
4. Example queries for MQL5 code

## VALIDATION
- ✅ Handles 150+ files efficiently
- ✅ Supports MQL5 syntax
- ✅ Works locally (no internet dependency)
- ✅ Fast search (<1s)
```

---

# 💻 PROMPT 3: MCPs PARA DESENVOLVIMENTO MQL5

## **CONTEXTO**
Subagente: MQL5 Developer
Necessidades: MQL5 reference, MetaTrader 5 docs, compilation, debugging

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para MQL5 Development

## OBJETIVO
MCPs que auxiliam desenvolvimento em MQL5, acesso a documentação MetaTrader, e debugging.

## CATEGORIAS A PESQUISAR

### 1. MQL5 DOCUMENTATION MCPs
- **MQL5 Community MCP** (se existir)
  - Official docs access
  - Example codes
  - Forum integration

- **Custom Documentation MCP**
  - Build own MCP with MQL5 docs
  - Effort required
  - Benefits

**Recomende:** Best way to access MQL5 docs in Claude Code

### 2. COMPILATION & TESTING MCPs
- **Filesystem MCP** (local)
  - Write MQL5 files
  - Trigger compilation
  - Read compiler output

- **Terminal/Shell MCP**
  - Execute MetaEditor
  - Run MT5 for testing
  - Automation possibilities

**Recomende:** Setup for MQL5 compile workflow

### 3. VERSION CONTROL MCPs
- **Git MCP**
  - Commit MQL5 code
  - Branch management
  - Integration quality

- **GitHub MCP**
  - Push to remote
  - Pull requests
  - CI/CD triggers

**Recomende:** Best VCS workflow for MQL5 development

## DELIVERABLE
1. Documentation access solution
2. Compilation workflow with MCPs
3. Git workflow configuration
4. Complete mcp.json example

## VALIDATION
- ✅ Seamless MQL5 development
- ✅ Fast doc lookup
- ✅ Efficient compilation
- ✅ Version control integrated
```

---

# 🧠 PROMPT 4: MCPs PARA AI/ML DEVELOPMENT

## **CONTEXTO**
Subagente: Python AI Engineer
Necessidades: ML libraries, papers, model registry, GPU access

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para AI/ML Development (KAN Networks, xLSTM)

## OBJETIVO
MCPs para desenvolvimento de modelos AI/ML, acesso a papers, experimentação.

## CATEGORIAS A PESQUISAR

### 1. ML RESEARCH MCPs
- **Arxiv MCP**
  - KAN Networks papers
  - xLSTM papers
  - Search by topic
  - PDF download

- **Papers with Code MCP**
  - Implementation search
  - Benchmarks
  - State-of-the-art models

- **Hugging Face MCP**
  - Model hub access
  - Datasets
  - Pre-trained models
  - ONNX export support

**Recomende:** Best for ML research & model discovery

### 2. EXPERIMENTATION MCPs
- **Weights & Biases MCP**
  - Experiment tracking
  - API integration
  - Visualization
  - Cost

- **MLflow MCP**
  - Model registry
  - Self-hosted option
  - Tracking capabilities

- **Neptune MCP**
  - Experiment management
  - Comparison with W&B

**Recomende:** Best for tracking KAN/xLSTM experiments

### 3. DATA PROCESSING MCPs
- **PostgreSQL MCP**
  - Historical XAUUSD data
  - Query optimization
  - Connection management

- **Redis MCP**
  - Feature caching
  - Real-time data
  - Performance

- **DuckDB MCP**
  - Analytical queries
  - In-memory processing
  - Speed for ML

**Recomende:** Best database MCP for ML data pipeline

### 4. COMPUTE MCPs
- **Docker MCP**
  - Container management
  - GPU containers
  - Deployment

- **Kubernetes MCP** (se necessário)
  - Scaling ML workloads
  - Complexity vs benefits

**Recomende:** Container orchestration strategy

## DELIVERABLE
1. Research workflow (papers → implementation)
2. Experiment tracking setup
3. Data pipeline MCPs configuration
4. Complete AI/ML MCP stack

## VALIDATION
- ✅ Access to latest ML papers (2024-2025)
- ✅ Efficient experiment tracking
- ✅ Fast data access for training
- ✅ GPU support if needed
```

---

# 🔗 PROMPT 5: MCPs PARA INTEGRAÇÃO E COMUNICAÇÃO

## **CONTEXTO**
Subagente: Integration Specialist
Necessidades: ZeroMQ testing, WebSocket, Redis, message protocols

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para System Integration (MT5↔Python)

## OBJETIVO
MCPs para testar e validar comunicação entre MT5 e Python via ZeroMQ/WebSocket.

## CATEGORIAS A PESQUISAR

### 1. NETWORK TESTING MCPs
- **cURL MCP** / **HTTP Client MCP**
  - Test WebSocket endpoints
  - API testing
  - Performance measurement

- **Postman MCP** (se existir)
  - API collections
  - Automated testing

**Recomende:** Best for testing communication protocols

### 2. MESSAGE QUEUE MCPs
- **Redis MCP**
  - Pub/Sub testing
  - Cache validation
  - Performance monitoring

- **RabbitMQ MCP** (alternative to ZeroMQ)
  - Message broker
  - Reliability
  - Comparison with ZeroMQ

**Recomende:** Message queue monitoring solution

### 3. MONITORING MCPs
- **Prometheus MCP**
  - Metrics collection
  - Latency tracking
  - Alert configuration

- **Grafana MCP**
  - Dashboard creation
  - Visualization
  - Integration with Prometheus

**Recomende:** Monitoring stack for <5ms latency target

### 4. DEBUGGING MCPs
- **Browser MCP** (Puppeteer)
  - Test WebSocket connections
  - Debug UI interfaces
  - Automated testing

- **Terminal MCP**
  - Network diagnostics (ping, netstat)
  - Process monitoring
  - Log tailing

**Recomende:** Debugging toolkit for integration issues

## DELIVERABLE
1. Network testing workflow
2. Message queue monitoring setup
3. Latency tracking configuration
4. Debugging MCP stack

## VALIDATION
- ✅ Can measure latency accurately
- ✅ Monitor ZeroMQ/WebSocket health
- ✅ Debug integration issues efficiently
- ✅ Real-time visibility into communication
```

---

# 🧪 PROMPT 6: MCPs PARA TESTES E QA

## **CONTEXTO**
Subagentes: Test Engineer, QA Specialist
Necessidades: Test execution, coverage, quality metrics

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Testing & Quality Assurance

## OBJETIVO
MCPs para executar testes, medir coverage, garantir qualidade de código.

## CATEGORIAS A PESQUISAR

### 1. TEST EXECUTION MCPs
- **Filesystem MCP**
  - Read test files
  - Write test reports
  - Manage fixtures

- **Terminal/Shell MCP**
  - Run pytest
  - Execute MQL5 tests
  - Parse output

- **GitHub Actions MCP**
  - Trigger CI/CD
  - Get test results
  - Workflow status

**Recomende:** Test execution workflow

### 2. CODE COVERAGE MCPs
- **Coverage.py MCP** (via Terminal)
  - Measure Python coverage
  - Generate reports
  - Integration with pytest

- **SonarQube MCP**
  - Code quality metrics
  - Coverage visualization
  - Technical debt

**Recomende:** Coverage measurement solution

### 3. SECURITY SCANNING MCPs
- **Snyk MCP**
  - Dependency scanning
  - Vulnerability detection
  - Fix recommendations

- **Bandit MCP** (via Terminal)
  - Python security issues
  - Static analysis

**Recomende:** Security scanning for trading EA

### 4. PERFORMANCE TESTING MCPs
- **Locust MCP** (if exists)
  - Load testing
  - Latency measurement

- **Custom profiling MCP**
  - Profile Python code
  - Memory usage
  - Bottleneck detection

**Recomende:** Performance testing approach

## DELIVERABLE
1. Test automation workflow
2. Coverage reporting setup
3. Security scanning configuration
4. Performance profiling toolkit

## VALIDATION
- ✅ >80% code coverage target
- ✅ Security vulnerabilities detected
- ✅ Performance bottlenecks identified
- ✅ Automated test execution
```

---

# ⚡ PROMPT 7: MCPs PARA PERFORMANCE OPTIMIZATION

## **CONTEXTO**
Subagente: Performance Optimizer
Necessidades: Profiling, benchmarking, latency analysis

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Performance Optimization (<5ms target)

## OBJETIVO
MCPs para profiling, benchmarking, e otimização de latência do sistema.

## CATEGORIAS A PESQUISAR

### 1. PROFILING MCPs
- **cProfile MCP** (via Terminal)
  - Python profiling
  - Hotspot detection
  - Visualization

- **py-spy MCP**
  - Sampling profiler
  - Low overhead
  - Real-time profiling

- **perf MCP** (Linux)
  - System-level profiling
  - CPU performance
  - Cache analysis

**Recomende:** Best profiling stack for <5ms target

### 2. MEMORY ANALYSIS MCPs
- **memory_profiler MCP**
  - Line-by-line memory
  - Memory leaks
  - Optimization opportunities

- **Valgrind MCP** (if applicable)
  - C++ DLL profiling
  - Memory errors

**Recomende:** Memory optimization approach

### 3. BENCHMARKING MCPs
- **Timeit MCP** (built-in)
  - Micro-benchmarks
  - Function timing

- **Benchmark.js MCP** (if relevant)
  - Cross-platform benchmarks

- **Custom latency MCP**
  - End-to-end latency
  - ZeroMQ roundtrip time
  - AI inference time

**Recomende:** Benchmarking methodology for latency

### 4. SYSTEM MONITORING MCPs
- **psutil MCP** (via Python)
  - CPU usage
  - Memory usage
  - Network I/O

- **htop/top MCP** (via Terminal)
  - Real-time process monitoring

**Recomende:** Real-time monitoring during optimization

## DELIVERABLE
1. Profiling workflow
2. Memory analysis setup
3. Benchmarking framework
4. Continuous monitoring configuration

## VALIDATION
- ✅ Identify <5ms bottlenecks
- ✅ Memory optimization validated
- ✅ Benchmark reproducibility
- ✅ Real-time performance visibility
```

---

# 🚀 PROMPT 8: MCPs PARA DEVOPS E DEPLOYMENT

## **CONTEXTO**
Subagente: DevOps Engineer
Necessidades: Infrastructure, CI/CD, containers, deployment

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para DevOps & Deployment

## OBJETIVO
MCPs para infraestrutura, deployment, e operações do EA em produção.

## CATEGORIAS A PESQUISAR

### 1. INFRASTRUCTURE MCPs
- **AWS MCP**
  - EC2 management
  - S3 storage
  - Cost tracking
  - Integration ease

- **Docker MCP**
  - Container build
  - Image management
  - Registry operations

- **Terraform MCP**
  - Infrastructure as Code
  - State management
  - Cloud provider support

**Recomende:** Infrastructure management stack

### 2. CI/CD MCPs
- **GitHub Actions MCP**
  - Workflow triggers
  - Status checks
  - Artifact management

- **GitLab CI MCP**
  - Alternative to GH Actions
  - Self-hosted option
  - Features comparison

- **Jenkins MCP**
  - Traditional CI/CD
  - Flexibility
  - Complexity

**Recomende:** Best CI/CD for MQL5 + Python project

### 3. CONTAINER ORCHESTRATION MCPs
- **Kubernetes MCP**
  - Cluster management
  - Scaling
  - Complexity vs benefits for small project

- **Docker Compose MCP**
  - Multi-container apps
  - Simplicity
  - Production suitability

**Recomende:** Container orchestration strategy

### 4. SECRETS MANAGEMENT MCPs
- **Vault MCP** (HashiCorp)
  - Secrets storage
  - Rotation
  - Complexity

- **Environment MCP** (simple)
  - .env files
  - Security limitations

**Recomende:** Secrets management for API keys

## DELIVERABLE
1. Infrastructure setup
2. CI/CD pipeline configuration
3. Container orchestration choice
4. Secrets management solution

## VALIDATION
- ✅ Automated deployment
- ✅ Secure secrets management
- ✅ Scalability if needed
- ✅ Cost-effective
```

---

# 📊 PROMPT 9: MCPs PARA MONITORING E OBSERVABILITY

## **CONTEXTO**
Subagente: Monitoring Specialist
Necessidades: Metrics, logs, traces, alerts, dashboards

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Monitoring & Observability (99.9% uptime)

## OBJETIVO
MCPs para monitorar EA em produção, garantir 99.9%+ uptime, alertas.

## CATEGORIAS A PESQUISAR

### 1. METRICS MCPs
- **Prometheus MCP**
  - Time-series metrics
  - Scraping configuration
  - Storage retention
  - Query language (PromQL)

- **InfluxDB MCP**
  - Alternative to Prometheus
  - Time-series database
  - Performance comparison

- **Datadog MCP**
  - SaaS monitoring
  - Cost
  - Feature richness

**Recomende:** Metrics collection solution

### 2. LOGGING MCPs
- **Elasticsearch MCP**
  - Log aggregation
  - Search capabilities
  - Resource requirements

- **Loki MCP** (Grafana)
  - Log aggregation (lighter than ES)
  - Integration with Grafana
  - Cost-effectiveness

- **CloudWatch MCP** (AWS)
  - Managed logging
  - Cost
  - Lock-in

**Recomende:** Logging solution for trading EA

### 3. VISUALIZATION MCPs
- **Grafana MCP**
  - Dashboard creation
  - Alerting
  - Plugin ecosystem
  - Self-hosted vs Cloud

- **Kibana MCP** (with ES)
  - Log visualization
  - Comparison with Grafana

**Recomende:** Visualization platform

### 4. ALERTING MCPs
- **PagerDuty MCP**
  - Incident management
  - On-call rotations
  - Cost

- **Alertmanager MCP** (Prometheus)
  - Alert routing
  - Grouping
  - Integration options

- **Slack MCP**
  - Simple alerting
  - Webhook integration
  - Free tier

**Recomende:** Alerting strategy for critical issues

### 5. TRACING MCPs (if needed)
- **Jaeger MCP**
  - Distributed tracing
  - Latency analysis
  - Setup complexity

**Evaluate:** Is tracing needed for <5ms latency debugging?

## DELIVERABLE
1. Metrics collection setup
2. Logging aggregation configuration
3. Dashboard templates
4. Alerting rules and channels

## VALIDATION
- ✅ Real-time trading metrics visible
- ✅ Logs searchable and aggregated
- ✅ Alerts for critical events (drawdown, errors)
- ✅ <1 min alert response time
```

---

# 📚 PROMPT 10: MCPs PARA DOCUMENTAÇÃO

## **CONTEXTO**
Subagente: Documentation Writer
Necessidades: Doc generation, diagrams, knowledge base

## **PROMPT PARA PESQUISA:**

```markdown
# PESQUISA: MCPs para Documentation

## OBJETIVO
MCPs para gerar e manter documentação técnica do projeto.

## CATEGORIAS A PESQUISAR

### 1. DOCUMENTATION GENERATION MCPs
- **Sphinx MCP** (Python)
  - API docs from docstrings
  - Integration with code
  - Output formats

- **Doxygen MCP** (C++)
  - If needed for C++ DLLs
  - MQL5 support?

- **MkDocs MCP**
  - Markdown-based docs
  - Theming
  - Simplicity

**Recomende:** Documentation framework

### 2. DIAGRAM MCPs
- **Mermaid MCP**
  - Diagram as code
  - Architecture diagrams
  - Sequence diagrams
  - Integration

- **PlantUML MCP**
  - UML diagrams
  - Complexity vs Mermaid

- **Draw.io MCP** (if exists)
  - Visual diagramming
  - Export options

**Recomende:** Diagramming solution

### 3. KNOWLEDGE BASE MCPs
- **Notion MCP**
  - Knowledge management
  - Collaboration
  - API limitations

- **Confluence MCP**
  - Enterprise wiki
  - Cost
  - Feature set

- **Obsidian MCP** (local)
  - Markdown notes
  - Graph view
  - Privacy

**Recomende:** Knowledge base for project

### 4. API DOCUMENTATION MCPs
- **Swagger/OpenAPI MCP**
  - API spec
  - Interactive docs
  - Code generation

- **Redoc MCP**
  - OpenAPI rendering
  - Customization

**Recomende:** API documentation if building APIs

## DELIVERABLE
1. Documentation framework choice
2. Diagram workflow
3. Knowledge base setup
4. API docs configuration (if applicable)

## VALIDATION
- ✅ Easy to maintain
- ✅ Versioned with code
- ✅ Searchable
- ✅ Clear and comprehensive
```

---

# 🎯 PROMPT 11: MATRIZ COMPLETA DE MCPs POR SUBAGENTE

## **PROMPT SÍNTESE:**

```markdown
# PESQUISA FINAL: MCP Matrix - Complete Mapping

## OBJETIVO
Criar matriz completa de MCPs recomendados para cada subagente, com prioridades.

## TAREFA
Com base em todas as pesquisas anteriores (Prompts 1-10), crie:

### 1. TABELA MATRIZ
| Subagente | MCPs Essenciais (P0) | MCPs Importantes (P1) | MCPs Nice-to-Have (P2) |
|-----------|----------------------|-----------------------|------------------------|
| 1. Market Analyzer | ... | ... | ... |
| 2. Codebase Explorer | ... | ... | ... |
| 3. Strategy Researcher | ... | ... | ... |
| 4. MQL5 Developer | ... | ... | ... |
| 5. Python AI Engineer | ... | ... | ... |
| 6. Integration Specialist | ... | ... | ... |
| 7. Test Engineer | ... | ... | ... |
| 8. QA Specialist | ... | ... | ... |
| 9. Performance Optimizer | ... | ... | ... |
| 10. DevOps Engineer | ... | ... | ... |
| 11. Monitoring Specialist | ... | ... | ... |
| 12. Documentation Writer | ... | ... | ... |

### 2. SHARED MCPs
Liste MCPs que beneficiam múltiplos agentes:
- **Filesystem MCP:** Usado por X, Y, Z
- **Terminal MCP:** Usado por A, B, C
- **Git MCP:** Usado por ...

### 3. INSTALLATION PRIORITY
Ordem de instalação recomendada:
1. [MCP_1] - Justificativa
2. [MCP_2] - Justificativa
...

### 4. CONFIGURATION FILE
Forneça `.roo/mcp.json` completo com todos MCPs recomendados:
```json
{
  "mcpServers": {
    "brave-search": { ... },
    "perplexity": { ... },
    ...
  }
}
```

### 5. COST ANALYSIS
| MCP | Cost Type | Monthly Estimate | Free Tier Adequate? |
|-----|-----------|------------------|---------------------|
| ... | ... | ... | ... |

**Total estimated monthly cost:** $XXX

### 6. QUICK START GUIDE
Steps to configure all MCPs:
1. Install MCP X: `command`
2. Configure API keys
3. Test MCP: `how to validate`
...

## DELIVERABLE
Complete MCP configuration strategy ready for implementation.
```

---

# 📋 TEMPLATE DE CONFIGURAÇÃO

## **ARQUIVO: `.roo/mcp.json` TEMPLATE**

```json
{
  "mcpServers": {
    "// === RESEARCH & DATA ===": {},

    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    },

    "perplexity": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-perplexity"],
      "env": {
        "PERPLEXITY_API_KEY": "your-perplexity-key"
      }
    },

    "// === CODE ANALYSIS ===": {},

    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"
      }
    },

    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/franco/projetos/EA_SCALPER_XAUUSD"]
    },

    "// === DEVELOPMENT ===": {},

    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    },

    "// === AI/ML ===": {},

    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/xauusd"
      }
    },

    "// === MONITORING ===": {},

    "prometheus": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-prometheus"],
      "env": {
        "PROMETHEUS_URL": "http://localhost:9090"
      }
    },

    "// === UTILITIES ===": {},

    "time": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-time"]
    },

    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

---

# 🚀 ORDEM DE EXECUÇÃO DOS PROMPTS

## **RECOMENDAÇÃO DE WORKFLOW:**

```
FASE 1 - PESQUISA BÁSICA (Execute primeiro)
├── Prompt 1: Market Research MCPs (30 min)
├── Prompt 2: Codebase Exploration MCPs (20 min)
└── Prompt 3: MQL5 Development MCPs (20 min)

FASE 2 - PESQUISA AVANÇADA
├── Prompt 4: AI/ML MCPs (40 min)
├── Prompt 5: Integration MCPs (30 min)
└── Prompt 6: Testing MCPs (25 min)

FASE 3 - PESQUISA OPERACIONAL
├── Prompt 7: Performance MCPs (25 min)
├── Prompt 8: DevOps MCPs (35 min)
└── Prompt 9: Monitoring MCPs (30 min)

FASE 4 - SÍNTESE
├── Prompt 10: Documentation MCPs (20 min)
└── Prompt 11: MCP Matrix (60 min - FINAL)

TOTAL ESTIMADO: ~5-6 horas de pesquisa
```

---

# ✅ CHECKLIST DE PESQUISA

- [ ] Prompt 1 executado - Market Research MCPs
- [ ] Prompt 2 executado - Codebase MCPs
- [ ] Prompt 3 executado - MQL5 Development MCPs
- [ ] Prompt 4 executado - AI/ML MCPs
- [ ] Prompt 5 executado - Integration MCPs
- [ ] Prompt 6 executado - Testing MCPs
- [ ] Prompt 7 executado - Performance MCPs
- [ ] Prompt 8 executado - DevOps MCPs
- [ ] Prompt 9 executado - Monitoring MCPs
- [ ] Prompt 10 executado - Documentation MCPs
- [ ] Prompt 11 executado - MCP Matrix Final
- [ ] `.roo/mcp.json` criado
- [ ] MCPs instalados e testados
- [ ] Custo mensal validado
- [ ] Documentação de MCPs completa

---

# 🎓 GUIA DE USO

## **Como usar estes prompts:**

1. **Escolha onde pesquisar:**
   - Perplexity (recomendado para deep research)
   - Claude Code (com web search MCP)
   - ChatGPT (com browsing)
   - Manual research (Google + docs)

2. **Execute os prompts em ordem**
   - Comece pelos essenciais (1-3)
   - Continue com avançados (4-6)
   - Finalize com operacionais (7-10)
   - Sintetize com matriz (11)

3. **Documente resultados**
   - Crie arquivo para cada prompt
   - Cole os relatórios de pesquisa
   - Atualize o MCP matrix

4. **Configure MCPs**
   - Edite `.roo/mcp.json`
   - Instale MCPs selecionados
   - Teste cada MCP
   - Valide funcionamento

---

**AGORA VOCÊ TEM 11 PROMPTS ESTRUTURADOS PARA PESQUISAR TODOS OS MCPs NECESSÁRIOS! 🚀**

*Guia criado em: 19/10/2025*
*Projeto: EA XAUUSD Scalper Elite*
*Total de prompts: 11*
