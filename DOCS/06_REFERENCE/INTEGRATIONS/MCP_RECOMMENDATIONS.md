# 🔧 MCP RECOMMENDATIONS - AGENT TEAM ARSENAL
## Síntese da Pesquisa ARGUS (300% Obsessiva)

**Data**: 2025-11-29
**Fonte**: wong2/awesome-mcp-servers (99KB, 500+ MCPs analisados)
**Metodologia**: Triangulação multi-fonte (GitHub + Perplexity + Brave)

---

## 📊 MATRIZ DE MCPs POR AGENTE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MCP ARSENAL COMPLETO                             │
├─────────────────────────────────────────────────────────────────────────┤
│  🔥 CRUCIBLE (Estratégia)     → Finance + Market Data + Research        │
│  🛡️ SENTINEL (Risco)          → Database + Calculator + Monitoring      │
│  ⚒️ FORGE (Código)            → Code Execution + GitHub + Testing       │
│  🔮 ORACLE (Backtest)         → QuantConnect + Database + TimeSeries    │
│  🔍 ARGUS (Pesquisa)          → Search + Research + Academic            │
│  📦 TODOS                     → Core Infrastructure                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔥 CRUCIBLE - Estratégia & Mercado

### Tier 1 - CRÍTICOS (Instalar Primeiro)

| MCP | Descrição | GitHub | Prioridade |
|-----|-----------|--------|------------|
| **Financial Datasets** | Stock market API para AI agents | [financial-datasets/mcp-server](https://github.com/financial-datasets/mcp-server) | 🔴 CRÍTICO |
| **Twelve Data** | Real-time & historical market data | [twelvedata/mcp](https://github.com/twelvedata/mcp) | 🔴 CRÍTICO |
| **CoinGecko** | Crypto price & market data (200+ chains, 8M+ tokens) | [docs.coingecko.com](https://docs.coingecko.com/reference/mcp-server/) | 🟡 ALTO |
| **Octagon** | Real-time investment research | [OctagonAI/octagon-mcp-server](https://github.com/OctagonAI/octagon-mcp-server) | 🟡 ALTO |

### Tier 2 - IMPORTANTES

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **Trade Agent** | Execute stock & crypto trades | [Trade-Agent/trade-agent-mcp](https://github.com/Trade-Agent/trade-agent-mcp) |
| **Token Metrics** | Crypto trading signals & predictions | [token-metrics/mcp](https://github.com/token-metrics/mcp) |
| **DexPaprika** | DEX analytics 20+ blockchains | [coinpaprika/dexpaprika-mcp](https://github.com/coinpaprika/dexpaprika-mcp) |
| **CoinCap** | Real-time crypto market data | [QuantGeekDev/coincap-mcp](https://github.com/QuantGeekDev/coincap-mcp) |
| **Hive Intelligence** | DeFi & Web3 analytics | [hive-intel/hive-crypto-mcp](https://github.com/hive-intel/hive-crypto-mcp) |

### Tier 3 - OPCIONAIS

| MCP | Descrição |
|-----|-----------|
| **Armor Crypto** | Multi-blockchain DeFi, staking, swap |
| **Norman Finance** | Accounting & taxes |
| **LunchMoney** | Personal finance & budgeting |

---

## 🛡️ SENTINEL - Risco & FTMO

### Tier 1 - CRÍTICOS

| MCP | Descrição | GitHub | Prioridade |
|-----|-----------|--------|------------|
| **Calculator** | Precise numerical calculations | [githejie/mcp-server-calculator](https://github.com/githejie/mcp-server-calculator) | 🔴 CRÍTICO |
| **PostgreSQL** | Database queries com RBAC | [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 🔴 CRÍTICO |
| **Memory** | Knowledge graph persistent memory | [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 🟡 ALTO |

### Tier 2 - IMPORTANTES

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **ClickHouse** | Time-series database queries | [ClickHouse/mcp-clickhouse](https://github.com/ClickHouse/mcp-clickhouse) |
| **Grafana** | Dashboards, incidents, metrics | [grafana/mcp-grafana](https://github.com/grafana/mcp-grafana) |
| **Logfire** | OpenTelemetry traces & metrics | [pydantic/logfire-mcp](https://github.com/pydantic/logfire-mcp) |
| **Scout APM** | Performance & error data | [scoutapm.com/mcp](https://www.scoutapm.com/mcp) |

### Para Cálculo de Risco

```python
# MCPs recomendados para SENTINEL calcular:
# - Position sizing (Kelly Criterion)
# - Drawdown tracking
# - FTMO compliance monitoring
# - Risk/Reward ratios

MCPs_RISCO = [
    "Calculator",      # Cálculos precisos de lot size
    "PostgreSQL",      # Histórico de trades
    "ClickHouse",      # Time-series de equity
    "Memory",          # Persistir estados de risco
]
```

---

## ⚒️ FORGE - Código & Arquitetura

### Tier 1 - CRÍTICOS

| MCP | Descrição | GitHub | Prioridade |
|-----|-----------|--------|------------|
| **E2B** | Run code in secure sandboxes | [e2b-dev/mcp-server](https://github.com/e2b-dev/mcp-server) | 🔴 CRÍTICO |
| **GitHub Official** | GitHub's official MCP | [github/github-mcp-server](https://github.com/github/github-mcp-server) | 🔴 CRÍTICO |
| **Filesystem** | Secure file operations | [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 🔴 CRÍTICO |
| **Git** | Read, search, manipulate Git repos | [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 🔴 CRÍTICO |

### Tier 2 - IMPORTANTES

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **ForeverVM** | Python code sandbox | [jamsocket/forevervm](https://github.com/jamsocket/forevervm/tree/main/javascript/mcp-server) |
| **Riza** | Arbitrary code execution | [riza-io/riza-mcp](https://github.com/riza-io/riza-mcp) |
| **YepCode** | Execute LLM-generated code | [yepcode/mcp-server-js](https://github.com/yepcode/mcp-server-js) |
| **SonarQube** | Code analysis & quality | [SonarSource/sonarqube-mcp-server](https://github.com/SonarSource/sonarqube-mcp-server) |
| **Semgrep** | Security scanning | [semgrep/mcp](https://github.com/semgrep/mcp) |
| **Digma** | Code observability via OTEL | [digma-ai/digma-mcp-server](https://github.com/digma-ai/digma-mcp-server) |

### Tier 3 - UTILIDADES

| MCP | Descrição |
|-----|-----------|
| **JetBrains** | Work with JetBrains IDEs |
| **Octocode** | GitHub code search & analysis |
| **llm-context** | Share code context with LLMs |
| **Sourcerer** | Semantic code search |
| **DeepView** | Analyze large codebases (1M context) |

---

## 🔮 ORACLE - Backtest & Validação

### Tier 1 - CRÍTICOS

| MCP | Descrição | GitHub | Prioridade |
|-----|-----------|--------|------------|
| **QuantConnect** | Backtest & live-trading workflows | [QuantConnect/mcp-server](https://github.com/QuantConnect/mcp-server) | 🔴 CRÍTICO |
| **PostgreSQL** | Backtest data storage | Reference servers | 🔴 CRÍTICO |
| **ClickHouse** | Time-series analytics | [ClickHouse/mcp-clickhouse](https://github.com/ClickHouse/mcp-clickhouse) | 🟡 ALTO |

### Tier 2 - IMPORTANTES

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **Twelve Data** | Historical market data | [twelvedata/mcp](https://github.com/twelvedata/mcp) |
| **Financial Datasets** | Stock market historical data | [financial-datasets/mcp-server](https://github.com/financial-datasets/mcp-server) |
| **Calculator** | Monte Carlo calculations | [githejie/mcp-server-calculator](https://github.com/githejie/mcp-server-calculator) |
| **Vega-Lite** | Data visualization | [isaacwasserman/mcp-vegalite-server](https://github.com/isaacwasserman/mcp-vegalite-server) |
| **ECharts** | Chart generation | [hustcc/mcp-echarts](https://github.com/hustcc/mcp-echarts) |

### Para Walk-Forward Analysis

```python
# Pipeline de WFA com MCPs:
WFA_PIPELINE = {
    "data_fetch": "Twelve Data / Financial Datasets",
    "storage": "PostgreSQL / ClickHouse",
    "backtest": "QuantConnect",
    "monte_carlo": "Calculator + E2B (Python)",
    "visualization": "Vega-Lite / ECharts",
}
```

---

## 🔍 ARGUS - Pesquisa & Inteligência

### Tier 1 - CRÍTICOS

| MCP | Descrição | GitHub | Prioridade |
|-----|-----------|--------|------------|
| **Perplexity** | Real-time web research | [ppl-ai/modelcontextprotocol](https://github.com/ppl-ai/modelcontextprotocol) | 🔴 CRÍTICO |
| **Exa** | AI-native search engine | [exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server) | 🔴 CRÍTICO |
| **DeepResearch** | Lightning-fast deep research | [OctagonAI/octagon-deep-research-mcp](https://github.com/OctagonAI/octagon-deep-research-mcp) | 🔴 CRÍTICO |
| **Context7** | Up-to-date docs for any prompt | [upstash/context7-mcp](https://github.com/upstash/context7-mcp) | 🟡 ALTO |

### Tier 2 - IMPORTANTES

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **Vectorize** | RAG, Deep Research, file extraction | [vectorize-io/vectorize-mcp-server](https://github.com/vectorize-io/vectorize-mcp-server/) |
| **Brave Search** | Web search API | [punkpeye/brave-search](https://github.com/punkpeye/brave-search) |
| **Kagi Search** | Premium search API | [kagisearch/kagimcp](https://github.com/kagisearch/kagimcp) |
| **Firecrawl** | Web data extraction | [mendableai/firecrawl-mcp-server](https://github.com/mendableai/firecrawl-mcp-server) |
| **Bright Data** | Web scraping at scale | [brightdata/brightdata-mcp](https://github.com/brightdata/brightdata-mcp) |

### Tier 3 - ACADEMIC/PAPERS

| MCP | Descrição | GitHub |
|-----|-----------|--------|
| **Latex MCP** | Compile latex, download/organize papers | [Yeok-c/latex-mcp-server](https://github.com/Yeok-c/latex-mcp-server) |
| **Inkeep** | RAG search over content | [inkeep/mcp-server-python](https://github.com/inkeep/mcp-server-python) |
| **Graphlit** | Ingest & search content | [graphlit/graphlit-mcp-server](https://github.com/graphlit/graphlit-mcp-server) |

---

## 📦 CORE INFRASTRUCTURE (TODOS OS AGENTES)

### Reference Servers (Oficiais Anthropic)

| MCP | Descrição | Uso |
|-----|-----------|-----|
| **Filesystem** | Secure file operations | Todos os agentes |
| **Git** | Repository operations | FORGE, ORACLE |
| **Memory** | Persistent memory | SENTINEL, CRUCIBLE |
| **Sequential Thinking** | Problem-solving | TODOS |
| **Fetch** | Web content fetching | ARGUS |
| **Time** | Timezone conversion | ORACLE (backtest times) |

### Database Layer

| MCP | Uso Recomendado |
|-----|-----------------|
| **PostgreSQL** | Primary data storage |
| **ClickHouse** | Time-series (ticks, equity curve) |
| **SQLite** | Local/lightweight queries |
| **MongoDB** | Document storage (configs) |
| **Chroma** | Vector search (RAG) |

---

## 🚀 QUICK START - TOP 10 MCPs ESSENCIAIS

```bash
# Os 10 MCPs mais importantes para começar:

1. 🔴 Financial Datasets   → Market data
2. 🔴 Twelve Data          → Historical + real-time
3. 🔴 QuantConnect         → Backtesting
4. 🔴 E2B                  → Code execution
5. 🔴 GitHub Official      → Repo management
6. 🔴 PostgreSQL           → Data storage
7. 🔴 Perplexity           → Web research
8. 🔴 Calculator           → Risk calculations
9. 🟡 ClickHouse           → Time-series
10. 🟡 DeepResearch        → Deep analysis
```

---

## 📋 INSTALAÇÃO TÍPICA

### Para Claude Desktop (claude_desktop_config.json)

```json
{
  "mcpServers": {
    "financial-datasets": {
      "command": "npx",
      "args": ["-y", "@anthropic/financial-datasets-mcp"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user:pass@host/db"]
    },
    "e2b": {
      "command": "npx",
      "args": ["-y", "e2b-mcp-server"],
      "env": {
        "E2B_API_KEY": "your-key"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

---

## 🎯 MATRIX DE FUNCIONALIDADES

| Funcionalidade | MCP Primário | MCP Backup |
|----------------|--------------|------------|
| **Market Data Real-time** | Twelve Data | Financial Datasets |
| **Market Data Historical** | Financial Datasets | Twelve Data |
| **Crypto Data** | CoinGecko | CoinCap |
| **Backtest Engine** | QuantConnect | E2B + Custom |
| **Code Execution** | E2B | ForeverVM, Riza |
| **Web Research** | Perplexity | Exa, Brave |
| **Deep Research** | DeepResearch | Vectorize |
| **Database SQL** | PostgreSQL | ClickHouse |
| **Database NoSQL** | MongoDB | Fireproof |
| **Vector Search** | Chroma | Milvus |
| **File Operations** | Filesystem | Git |
| **Calculations** | Calculator | E2B Python |
| **Visualization** | Vega-Lite | ECharts |

---

## 📝 NOTAS IMPORTANTES

### APIs com Custo
- **Twelve Data**: Free tier limitado, paid para prod
- **QuantConnect**: Free tier disponível
- **E2B**: Free tier com limites
- **Perplexity**: API key necessária

### APIs Gratuitas
- **CoinCap**: Totalmente gratuito
- **CoinGecko**: Free tier generoso
- **Financial Datasets**: Check pricing
- **Todos Reference Servers**: Gratuitos

### Segurança
- NUNCA colocar API keys em código
- Usar variáveis de ambiente
- PostgreSQL: sempre usar RBAC
- Filesystem: limitar diretórios

---

## 🔄 PRÓXIMOS PASSOS

1. [ ] Instalar MCPs Core (Filesystem, Git, Memory)
2. [ ] Configurar Financial Datasets + Twelve Data
3. [ ] Setup PostgreSQL para storage
4. [ ] Testar E2B para code execution
5. [ ] Configurar Perplexity para ARGUS
6. [ ] Integrar QuantConnect para ORACLE
7. [ ] Documentar configs específicas por agente

---

**Versão**: 1.0
**Última Atualização**: 2025-11-29
**Pesquisa por**: ARGUS Research Analyst
