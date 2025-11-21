# EA_SCALPER_XAUUSD

Sistema completo para desenvolvimento e operação de Expert Advisors focados em **XAUUSD (ouro)** no MetaTrader 5. O repositório reúne EAs modulares, bibliotecas MQL4/MQL5 reutilizáveis, dados de backtest, automações com LLMs e ferramentas auxiliares (proxies, agentes MCP, prompts personalizados).

## Objetivo

Fornecer uma base única para pesquisa, desenvolvimento, testes e operação de estratégias de scalping em XAUUSD, mantendo código organizado, reprodutível e fácil de evoluir.

## Principais blocos

- `🚀 MAIN_EAS/` – Expert Advisors principais. Versões de produção (FTMO-ready), desenvolvimento e módulos experimentais.
- `📚 LIBRARY/` – Componentes compartilhados MQL4/MQL5 (indicadores, utilitários, templates e estratégias reutilizáveis).
- `📊 DATA/` – Dados de mercado, resultados de backtests e artefatos do TradingView.
- `🤖 AI_AGENTS/` – Integrações com agentes/LLMs (MCP, backtest runner, automações de prompts).
- `🛠️ TOOLS/` – Ferramentas auxiliares (ex.: `CLIPROXY/CLIProxyAPI` para proxy de modelos, scripts de suporte).
- `🔧 WORKSPACE/` – Configurações de IDE, testes e artefatos de trabalho.
- Guias gerais: `ORGANIZATION.md` (mapa da estrutura) e `📖 GUIA_ORGANIZACAO_COMPLETO.md`.

## Uso rápido

### MetaTrader 5
1) Abra o MetaEditor e carregue o EA desejado (ex.: `🚀 MAIN_EAS/PRODUCTION/EA_AUTONOMOUS_XAUUSD_ELITE_v3.0_Modular.mq5`).
2) Compile e anexe ao par **XAUUSD** no timeframe apropriado conforme a estratégia.
3) Ajuste os parâmetros do EA conforme sua corretora/risco. Coloque `.ex5` e outros binários no `.gitignore` (já recomendado).

### Ferramentas e automações
- Proxy LLM: `bash CLIPROXY/CLIProxyAPI/start_cliproxyapi.sh` (usa `config.yaml` no mesmo diretório).
- Prompts locais do Codex CLI: exporte `CODEX_HOME="$PWD/.codex"` ou use o alias já registrado em `~/.bashrc`.
- Ambiente Python/LLM: veja requisitos em `🤖 AI_AGENTS/LLM_Integration/requirements.txt` e demais scripts em `🛠️ TOOLS/scripts/python/`.

## Convenções

- Commits seguem **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `chore:` …).
- Arquivos grandes/binaries devem ficar fora do versionamento (use git-lfs se inevitável).
- Estrutura detalhada e caminhos consolidados estão em `ORGANIZATION.md` e `CLAUDE.md`.

## Estado atual

O projeto passou por uma grande reorganização para concentrar EAs, bibliotecas e ferramentas em pastas temáticas. Se algum script ou pipeline ainda referenciar caminhos antigos, ajuste para os novos diretórios listados acima.

