# Estrutura Organizada do Projeto

Este guia descreve a nova organizacao dos arquivos que estavam soltos na raiz do repositorio. Nada foi removido; apenas movido para pastas tematicas para facilitar navegacao e manutencao.

- `docs/` Documentacao, relatorios e anotações gerais.
  - `docs/reports/` reservado para relatorios futuros.
- `configs/` Arquivos de configuracao (YAML/TOML/JSON). Observacao: `.env` e `.env.example` permanecem na raiz para compatibilidade com ferramentas que esperam esses arquivos no diretorio atual.
- `scripts/python/` Scripts Python utilitarios e entrypoints (ex.: agentes, proxies, setups).
- `scripts/windows/` Scripts `.bat` e `.ps1` (Windows/PowerShell).
- `tests/` Testes automatizados em Python.
- `workspace/` Arquivos de configuracao do editor/IDE (VS Code workspace).

Como executar apos a reorganizacao:

- Scripts Python (exemplo): `python scripts/python/simple_trading_proxy.py`
- Testes (exemplo com pytest): `python -m pytest tests -q`

Pastas existentes nao alteradas:

`BMAD-METHOD`, `Include`, `LIBRARY`, `LLM_Integration`, `MAIN_EAS`, `MCP`, `MULTI_AGENT_TRADING_SYSTEM` e as pastas com icones (ex.: `📊`, `📋`, `📚`, `🔧`, `🚀`, `🤖`).

Se algum import ou automacao depender de caminhos antigos, ajuste os caminhos para refletir os novos locais ou execute os scripts diretamente pelos novos caminhos conforme exemplos acima.

