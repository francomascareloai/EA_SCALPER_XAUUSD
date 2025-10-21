#!/bin/bash
# Instalação dos MCPs Essenciais para Desenvolvimento Autônomo de EA

echo "🤖 Instalando MCPs para Desenvolvimento Autônomo de EA XAUUSD..."

# MCPs Essenciais
echo "📥 Instalando MCP Fetch (já resolvido)..."
uvx --version

echo "📥 Instalando GitHub MCP..."
uvx mcp-server-github --help

echo "📥 Instalando Sequential Thinking MCP..."
uvx mcp-server-sequential-thinking --help

echo "📥 Instalando Context7 MCP..."
uvx mcp-server-context7 --help

echo "📥 Instalando Playwright MCP..."
uvx mcp-server-playwright --help

echo "✅ Instalação concluída!"
echo "📋 Para usar, copie o conteúdo de claude_desktop_config_autonomous_ea.json"
echo "    para seu arquivo de configuração do Claude Desktop."

# Verificação
echo "🔍 Verificando MCPs instalados..."
uvx --help | grep -i "available packages" || echo "Use 'uvx <package>' para testar cada MCP"

echo "🚀 Sistema pronto para desenvolvimento autônomo de EA!"