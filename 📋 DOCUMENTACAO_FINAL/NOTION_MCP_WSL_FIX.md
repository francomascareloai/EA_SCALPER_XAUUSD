# 🔧 Como Resolver OAuth do Notion MCP no WSL

## 🎯 Problema
O Notion MCP não consegue receber o callback OAuth porque o WSL não está aceitando conexões localhost do Windows.

## ✅ SOLUÇÃO DEFINITIVA (Recomendada)

### 1. Configure WSL Network Mirroring

**No Windows, edite o arquivo:** `C:\Users\<seu_usuario>\.wslconfig`

Adicione estas linhas:
```ini
[wsl2]
networkingMode=mirrored
localhostForwarding=true
```

### 2. Reinicie o WSL

No PowerShell do Windows:
```powershell
wsl --shutdown
```

Depois abra o WSL novamente.

### 3. Faça login no Notion MCP

```bash
codex mcp login notion
```

Agora o callback OAuth deve funcionar! 🎉

---

## 🔄 ALTERNATIVAS

### Opção 1: Usar socat (temporário)

1. Instale o socat:
```bash
sudo apt-get install -y socat
```

2. Em um terminal separado, redirecione a porta:
```bash
sudo socat TCP-LISTEN:80,fork TCP:127.0.0.1:34281
```

3. Execute o login:
```bash
codex mcp login notion
```

### Opção 2: Desabilitar Notion temporariamente

Edite `/home/franco/.codex/config.toml` e mude:
```toml
[mcp_servers.notion]
enabled = false
```

Você ainda terá acesso a:
- ✅ Brave Search
- ✅ Tavily Search  
- ✅ Perplexity Search
- ✅ DocFork MCP
- ✅ Context7
- ✅ E muitos outros...

---

## 📝 Notas

- O token de integração (`ntn_582550274831...`) **NÃO funciona** porque o Notion MCP **exige OAuth**
- O problema é específico do WSL e afeta qualquer MCP que use OAuth com callback local
- A solução de Network Mirroring resolve permanentemente para todos os MCPs

---

## 🆘 Se nada funcionar

Entre em contato com o suporte do Codex ou abra uma issue:
- GitHub: https://github.com/codexstanford/codex-cli/issues
- Discord: https://discord.gg/codex

---

**Status atual:** Notion MCP desabilitado até configuração WSL
**Próximo passo:** Configurar `.wslconfig` com network mirroring
