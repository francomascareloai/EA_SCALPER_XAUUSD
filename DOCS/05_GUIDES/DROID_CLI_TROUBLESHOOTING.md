# Droid CLI - Troubleshooting Guide

**Data:** 2025-12-07  
**Problema Reportado:** Droid CLI crashando/saindo da conversa sozinho  
**Versão:** droid 0.32.1 | exe v1.3.3.0

---

## 🚨 Sintomas Observados

- ✅ Sessões do `droid` terminam inesperadamente
- ✅ Terminal com nome "criador prompts" (rodando `droid`)
- ✅ Conversas são interrompidas sem aviso
- ✅ Possível perda de contexto entre sessões

---

## 🔍 Causas Raiz Identificadas

### 1. **Token Limit Overflow** (Mais Provável)
- **Problema:** Conversas muito longas excedem limite de tokens
- **Evidência:** Droid CLI usa Claude API com limites de contexto
- **Impacto:** Sessão termina abruptamente ao atingir o limite

### 2. **MCP Server Hangs**
- **Problema:** Servidores MCP (como `sequential-thinking`, `memory`, etc) travam
- **Evidência:** Terminal sem resposta ou timeouts
- **Impacto:** Droid CLI aguarda resposta e pode crashar

### 3. **Memory Leaks**
- **Problema:** Acúmulo de memória ao longo da sessão
- **Evidência:** Performance degradando progressivamente
- **Impacto:** Sistema mata o processo quando memória excede

### 4. **Context Window Bloat**
- **Problema:** Arquivo AGENTS.md (142KB) + histórico de conversa = context overflow
- **Evidência:** AGENTS.md tem 3133 linhas, muitos droids/skills carregados
- **Impacto:** Context window fica MUITO grande, causando crashes

---

## ✅ Soluções Recomendadas

### **Solução 1: Limitar Tamanho das Sessões** (IMEDIATO)

```powershell
# Usar `droid exec` para tarefas pontuais ao invés de sessões longas
droid exec "tarefa específica aqui"

# Forçar checkpoint e resumo a cada 20-30 mensagens
# (use Ctrl+C para sair, depois droid --resume para voltar)
```

**Vantagem:** Evita acúmulo de contexto  
**Desvantagem:** Perde continuidade

---

### **Solução 2: Otimizar AGENTS.md** (CRÍTICO)

**Problema:** Arquivo AGENTS.md está MUITO grande (3133 linhas, ~142KB)

**Ação Requerida:**
```markdown
1. Criar versões "NANO" dos droids (como já existe nautilus-nano.md)
2. Separar strategic_intelligence em arquivo próprio
3. Usar @references seletivos ao invés de carregar tudo
```

**Exemplo de Otimização:**
```xml
<!-- ANTES: 142KB carregado sempre -->
<agents>
  <!-- 18 droids completos com todos os detalhes -->
</agents>

<!-- DEPOIS: ~30KB + carregamento sob demanda -->
<agents>
  <agent name="FORGE" file=".factory/droids/forge-mql5-architect.md" />
  <agent name="SENTINEL" file=".factory/droids/sentinel-apex-guardian.md" />
  <!-- Apenas referências, carrega conforme necessário -->
</agents>
```

---

### **Solução 3: Monitorar MCP Servers** (PREVENTIVO)

```powershell
# Verificar status dos servidores MCP
droid mcp

# Se algum estiver travado, restart:
# (droid geralmente faz isso automaticamente, mas pode falhar)
```

**MCPs Críticos a Monitorar:**
- `sequential-thinking` - Usado para raciocínio profundo
- `memory` - Knowledge graph
- `mql5-docs` / `mql5-books` - RAG databases

---

### **Solução 4: Usar Sessões Resumíveis** (WORKAROUND)

```powershell
# Sempre usar --resume ao invés de novo droid
droid --resume  # Retoma última sessão

# OU especificar session ID
droid --resume <session-id>
```

**Checkpoint Manual a Cada 20 Msgs:**
1. `Ctrl+C` para sair
2. Droid salva estado automaticamente
3. `droid --resume` para voltar

---

## 🛠️ Debugging Avançado

### Verificar Logs do Droid

```powershell
# Localizar diretório de logs (geralmente em %APPDATA% ou %LOCALAPPDATA%)
Get-ChildItem "$env:LOCALAPPDATA" -Recurse -Filter "*.log" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*droid*" }

# OU verificar em:
# C:\Users\Admin\.droid\logs\
# C:\Users\Admin\AppData\Local\droid\
# C:\Users\Admin\AppData\Roaming\droid\
```

### Rodar Droid em Modo Debug

```powershell
# Se existir flag de debug
droid --debug "teste de conexão"

# OU com verbose
droid --verbose
```

### Verificar Token Usage

```powershell
# Durante conversa, perguntar ao droid:
"Quantos tokens foram usados até agora nesta sessão?"

# Se >80% do limite: CHECKPOINT IMEDIATO
```

---

## 📊 Métricas de Saúde da Sessão

| Métrica | Valor Ideal | Valor Crítico | Ação |
|---------|-------------|---------------|------|
| **Mensagens** | <30 | >50 | Checkpoint |
| **Tokens Usados** | <80% | >90% | Reiniciar |
| **Tempo de Resposta** | <5s | >15s | MCP hangs |
| **Memória (droid.exe)** | <500MB | >1GB | Restart |

---

## 🔧 Ações Corretivas Específicas

### Para o Problema Atual

```powershell
# 1. Sair de todas as sessões droid ativas
Get-Process droid -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Limpar cache (se existir)
Remove-Item "$env:TEMP\droid-*" -Recurse -Force -ErrorAction SilentlyContinue

# 3. Iniciar nova sessão com contexto mínimo
droid "modo compacto: usar apenas informações essenciais"
```

### Otimização de AGENTS.md (Próxima Sessão)

**Criar arquivo de referência:**
```xml
<!-- AGENTS_REFERENCES.md -->
<agents_index>
  <agent id="FORGE" description="Code/MQL5/Python" load="on-demand" />
  <agent id="SENTINEL" description="Risk/DD/Apex" load="on-demand" />
  <agent id="CRUCIBLE" description="Strategy/SMC" load="on-demand" />
  <!-- Apenas índice, detalhes carregados sob demanda -->
</agents_index>
```

---

## 📝 Checklist de Troubleshooting

Quando droid crashar novamente:

- [ ] **Quantas mensagens na sessão?** (se >30, checkpoint)
- [ ] **Rodando `droid --resume` ou `droid` novo?** (sempre prefer resume)
- [ ] **Algum MCP server travado?** (verificar `droid mcp`)
- [ ] **AGENTS.md foi modificado recentemente?** (se sim, pode ter bug XML)
- [ ] **Memória do processo droid.exe?** (Task Manager)
- [ ] **Último comando antes do crash?** (repetível?)

---

## 🚀 Melhorias Futuras Recomendadas

1. **Lazy Loading de Droids**
   - Não carregar todos os 18 droids sempre
   - Carregar apenas quando invocados

2. **Compression de Strategic Intelligence**
   - Seção muito grande (~40% do AGENTS.md)
   - Separar em arquivo próprio com @reference

3. **Session Health Monitoring**
   - Droid avisar quando contexto atingir 80%
   - Auto-checkpoint a cada 25 mensagens

4. **MCP Watchdog**
   - Auto-restart de MCP servers que travarem
   - Timeout mais agressivo (5s → 2s)

---

## 📞 Contato e Suporte

- **Documentação Oficial:** https://docs.factory.ai/factory-cli
- **GitHub Issues:** https://github.com/factory-ai/factory-cli/issues
- **Local:** Este guia em `DOCS/05_GUIDES/DROID_CLI_TROUBLESHOOTING.md`

---

## ⚡ TL;DR - Quick Fix

```powershell
# SE DROID CRASHANDO AGORA:
1. Ctrl+C na sessão atual
2. droid --resume  # Retoma de onde parou
3. A cada 20 mensagens: Ctrl+C + resume novamente

# SOLUÇÃO PERMANENTE (PRÓXIMA SESSÃO):
1. Criar AGENTS_NANO.md (versão compacta)
2. Usar sessões mais curtas (<30 msgs)
3. Sempre usar --resume ao invés de novo droid
```

---

**Status:** 🟢 Diagnóstico Completo | 🟡 Solução Temporária Disponível | 🔴 Otimização AGENTS.md Pendente
