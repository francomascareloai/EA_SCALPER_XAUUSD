# 📚 ÍNDICE COMPLETO - DOCUMENTAÇÃO MCP

## 🎯 **DOCUMENTOS MCP CRIADOS**

**Data:** 19/10/2025
**Projeto:** EA XAUUSD Scalper Elite
**Objetivo:** Configurar MCPs ideais para 12 subagentes

---

## 📂 **ARQUIVOS DISPONÍVEIS**

### **1. MCP_RESEARCH_PROMPTS.md** (27 KB) ⭐ **PRINCIPAL**
**Localização:** `/docs/MCP_RESEARCH_PROMPTS.md`

**Conteúdo:**
- 11 prompts estruturados para pesquisa completa
- Cobertura de todas as categorias de MCPs
- Template de configuração `.roo/mcp.json`
- Checklist de pesquisa
- Ordem de execução recomendada

**Prompts incluídos:**
1. MCPs para Pesquisa de Mercado e Dados
2. MCPs para Codebase Exploration
3. MCPs para Desenvolvimento MQL5
4. MCPs para AI/ML Development
5. MCPs para Integração e Comunicação
6. MCPs para Testes e QA
7. MCPs para Performance Optimization
8. MCPs para DevOps e Deployment
9. MCPs para Monitoring e Observability
10. MCPs para Documentação
11. Matriz Completa de MCPs por Subagente

**Quando usar:** Para pesquisa sistemática e completa de todos os MCPs necessários

---

### **2. MCP_RESEARCH_PERPLEXITY_TAVILY.md** (13 KB)
**Localização:** `/docs/MCP_RESEARCH_PERPLEXITY_TAVILY.md`

**Conteúdo:**
- Guia específico para usar Perplexity + Tavily
- Formato otimizado de queries
- Exemplos práticos por categoria
- Workflow recomendado (Perplexity → Tavily → Decisão)
- Checklist de pesquisa por categoria
- Tempo estimado: 5.5 horas

**Quando usar:** Como guia prático durante a execução da pesquisa com Perplexity e Tavily

---

## 🚀 **COMO USAR ESTA DOCUMENTAÇÃO**

### **PASSO 1: ENTENDER O ESCOPO**
Leia primeiro:
```bash
cat docs/MCP_RESEARCH_PROMPTS.md
```
- Entenda as 11 categorias de MCPs
- Veja a matriz de subagentes
- Compreenda o workflow geral

**Tempo:** 30 minutos

---

### **PASSO 2: EXECUTAR A PESQUISA**

#### **OPÇÃO A: Pesquisa Completa (Recomendado)**

**Usar:** `MCP_RESEARCH_PROMPTS.md`

1. **Abra Perplexity** (ou ferramenta de pesquisa)
2. **Copie PROMPT 1** do documento
3. **Execute pesquisa** e documente resultados
4. **Repita para PROMPTS 2-11**
5. **Sintetize com PROMPT 11** (Matriz final)

**Tempo:** 5-6 horas
**Resultado:** Lista completa de MCPs validados

---

#### **OPÇÃO B: Pesquisa Rápida por Categoria**

**Usar:** `MCP_RESEARCH_PERPLEXITY_TAVILY.md`

1. **Escolha categoria** (ex: Market Research)
2. **Use query Perplexity** do guia
3. **Valide com Tavily** (configuração específica)
4. **Documente decisão**
5. **Próxima categoria**

**Tempo:** 1-2 horas (para categorias prioritárias)
**Resultado:** MCPs essenciais configurados rapidamente

---

### **PASSO 3: CONFIGURAR MCPs**

Após pesquisa, use o template:
```bash
# Editar arquivo de configuração
nano /home/franco/projetos/EA_SCALPER_XAUUSD/.roo/mcp.json

# Usar template do documento como base
# (disponível em MCP_RESEARCH_PROMPTS.md)
```

**Exemplo de configuração:**
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    }
  }
}
```

---

### **PASSO 4: TESTAR MCPs**

Após configurar cada MCP:
```bash
# Reiniciar Claude Code para carregar MCPs
# Testar MCP específico (exemplo):
# 1. Abrir nova conversa
# 2. Pedir: "Use o Brave Search MCP para pesquisar XAUUSD market analysis"
# 3. Verificar se MCP responde corretamente
```

---

## 📊 **CATEGORIAS DE MCPs**

### **GRUPO 1: PESQUISA E ANÁLISE**
Subagentes beneficiados: Market Analyzer, Strategy Researcher

**MCPs esperados:**
- Web Search (Brave, Perplexity, Tavily)
- Financial Data (Yahoo Finance, Alpha Vantage)
- News (NewsAPI, Finnhub)
- Papers (Arxiv, Semantic Scholar)

**Prioridade:** 🔴 ALTA

---

### **GRUPO 2: DESENVOLVIMENTO**
Subagentes beneficiados: MQL5 Developer, Python AI Engineer

**MCPs esperados:**
- Filesystem (local file access)
- Git (version control)
- GitHub (repository management)
- Terminal (command execution)
- PostgreSQL/Redis (databases)

**Prioridade:** 🔴 ALTA

---

### **GRUPO 3: INTEGRAÇÃO E TESTES**
Subagentes beneficiados: Integration Specialist, Test Engineer, QA

**MCPs esperados:**
- HTTP Client (API testing)
- Redis (message queue)
- Terminal (test execution)
- GitHub Actions (CI/CD)

**Prioridade:** 🟡 MÉDIA

---

### **GRUPO 4: PERFORMANCE E OPS**
Subagentes beneficiados: Performance Optimizer, DevOps, Monitoring

**MCPs esperados:**
- Prometheus (metrics)
- Grafana (visualization)
- Docker (containers)
- AWS/Cloud (infrastructure)

**Prioridade:** 🟡 MÉDIA

---

### **GRUPO 5: DOCUMENTAÇÃO**
Subagente beneficiado: Documentation Writer

**MCPs esperados:**
- Filesystem (write docs)
- Git (version docs)
- Mermaid (diagrams)

**Prioridade:** 🟢 BAIXA (pode vir depois)

---

## ⚡ **QUICK START - MCPs ESSENCIAIS**

Se você tem tempo limitado, comece com estes MCPs:

### **TOP 5 MCPs PRIORITÁRIOS:**

1. **Filesystem MCP** (P0 - Crítico)
   - Usado por: TODOS os subagentes
   - Permite: Ler/escrever arquivos localmente
   - Configuração: Simples (sem API key)

2. **Git MCP** (P0 - Crítico)
   - Usado por: MQL5 Dev, Python Dev, QA
   - Permite: Commits, branches, version control
   - Configuração: Simples

3. **Brave Search MCP** ou **Perplexity MCP** (P0 - Crítico)
   - Usado por: Market Analyzer, Strategy Researcher
   - Permite: Pesquisa web de alta qualidade
   - Configuração: Requer API key

4. **GitHub MCP** (P1 - Alto)
   - Usado por: Todos os desenvolvedores
   - Permite: Repository management, CI/CD
   - Configuração: Requer GitHub token

5. **PostgreSQL MCP** (P1 - Alto)
   - Usado por: Python AI Engineer
   - Permite: Armazenar/query dados XAUUSD
   - Configuração: Requer database setup

---

## 🎯 **WORKFLOW COMPLETO**

```
┌─────────────────────────────────────────┐
│  FASE 1: PESQUISA (5-6h)                │
├─────────────────────────────────────────┤
│ 1. Executar 11 prompts                  │
│ 2. Documentar resultados                │
│ 3. Criar matriz de decisão              │
│ 4. Priorizar MCPs (P0, P1, P2)         │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  FASE 2: CONFIGURAÇÃO (2-3h)           │
├─────────────────────────────────────────┤
│ 1. Criar .roo/mcp.json                  │
│ 2. Obter API keys necessárias           │
│ 3. Instalar MCPs escolhidos             │
│ 4. Testar cada MCP                      │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  FASE 3: VALIDAÇÃO (1h)                │
├─────────────────────────────────────────┤
│ 1. Testar MCPs com subagentes           │
│ 2. Verificar latência/performance       │
│ 3. Documentar uso de cada MCP           │
│ 4. Criar troubleshooting guide          │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  FASE 4: OTIMIZAÇÃO (ongoing)          │
├─────────────────────────────────────────┤
│ 1. Adicionar MCPs conforme necessário   │
│ 2. Remover MCPs não utilizados          │
│ 3. Otimizar custos (free tier)          │
│ 4. Atualizar documentação               │
└─────────────────────────────────────────┘

TOTAL ESTIMADO: 8-10 horas
```

---

## 📋 **CHECKLIST GERAL**

### **PESQUISA:**
- [ ] Lido MCP_RESEARCH_PROMPTS.md
- [ ] Lido MCP_RESEARCH_PERPLEXITY_TAVILY.md
- [ ] Escolhida ferramenta de pesquisa (Perplexity/Tavily/Ambas)
- [ ] Executados prompts 1-11 (ou categorias prioritárias)
- [ ] Resultados documentados
- [ ] MCPs priorizados (P0/P1/P2)

### **CONFIGURAÇÃO:**
- [ ] Arquivo `.roo/mcp.json` criado
- [ ] API keys obtidas
- [ ] MCPs P0 instalados
- [ ] MCPs P1 instalados
- [ ] MCPs P2 (opcional) avaliados

### **VALIDAÇÃO:**
- [ ] Cada MCP testado individualmente
- [ ] MCPs funcionando com subagentes
- [ ] Latência aceitável
- [ ] Custos dentro do orçamento

### **DOCUMENTAÇÃO:**
- [ ] Lista final de MCPs documentada
- [ ] Guias de uso criados
- [ ] Troubleshooting conhecido documentado
- [ ] Custos mensais calculados

---

## 💰 **ANÁLISE DE CUSTOS ESPERADA**

### **MCPs Gratuitos:**
- Filesystem (local)
- Git (local)
- Terminal (local)
- Time (local)
- Memory (local)

**Custo:** $0/mês

### **MCPs Free Tier (adequado):**
- GitHub (5000 requests/hour)
- Brave Search (free tier)
- Arxiv (free)
- Devdocs (free)

**Custo:** $0/mês

### **MCPs Pagos (opcionais):**
- Perplexity API (~$20/mês)
- Alpha Vantage Premium (~$50/mês)
- Weights & Biases (~$50/mês)
- Datadog (~$15/host/mês)

**Custo estimado:** $0-135/mês (depende das escolhas)

---

## 🎓 **RECURSOS ADICIONAIS**

### **Documentação Oficial MCP:**
- https://modelcontextprotocol.io/
- https://github.com/modelcontextprotocol/servers

### **Claude Code MCP Setup:**
- https://docs.anthropic.com/claude/docs/model-context-protocol

### **Community MCPs:**
- GitHub topic: `mcp-server`
- Awesome MCP list: (procurar no GitHub)

---

## 🚀 **PRÓXIMOS PASSOS**

**AGORA:**
1. ✅ Ler este índice
2. ⏳ Escolher: Pesquisa Completa OU Rápida
3. ⏳ Executar pesquisa com prompts
4. ⏳ Configurar MCPs escolhidos

**DEPOIS:**
1. ⏳ Testar MCPs com subagentes
2. ⏳ Validar performance
3. ⏳ Otimizar custos
4. ⏳ Documentar uso

---

## 📊 **RESUMO EXECUTIVO**

**Documentos criados:** 2
**Prompts de pesquisa:** 11
**Categorias cobertas:** 9
**Subagentes beneficiados:** 12
**Tempo de pesquisa:** 5-6 horas
**Tempo de configuração:** 2-3 horas
**Custo mensal estimado:** $0-135 (configurável)

---

**VOCÊ AGORA TEM UM SISTEMA COMPLETO PARA PESQUISAR E CONFIGURAR TODOS OS MCPs NECESSÁRIOS! 🎉**

---

*Índice criado em: 19/10/2025*
*Documentos: MCP_RESEARCH_PROMPTS.md + MCP_RESEARCH_PERPLEXITY_TAVILY.md*
*Status: Pronto para uso*
