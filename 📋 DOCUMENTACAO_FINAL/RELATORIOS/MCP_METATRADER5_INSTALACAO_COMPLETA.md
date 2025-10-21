# MCP MetaTrader5 Server - Guia de Instalação e Uso

## ✅ Status da Instalação
O MCP MetaTrader5 Server foi **instalado com sucesso** em seu workspace!

## 📁 Localização dos Arquivos
- **Servidor MCP**: `mcp-metatrader5-server/`
- **Scripts de inicialização**: 
  - `iniciar_mt5_mcp_server.bat` (Windows Batch)
  - `iniciar_mt5_mcp_server.ps1` (PowerShell)
- **Configuração Claude**: `claude_desktop_config_mt5.json`

## 🚀 Como Iniciar o Servidor

### Opção 1: Script PowerShell (Recomendado)
```powershell
.\iniciar_mt5_mcp_server.ps1
```

### Opção 2: Script Batch
```cmd
iniciar_mt5_mcp_server.bat
```

### Opção 3: Manual
```powershell
cd mcp-metatrader5-server
C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe run.py dev
```

## 🌐 Acesso ao Servidor
- **URL**: http://127.0.0.1:8000
- **Status**: O servidor está rodando em segundo plano (Terminal ID: 23442a4a-4a54-4b95-8281-2014e3bb1089)

## 🔗 Integração com Claude Desktop

1. Localize o arquivo de configuração do Claude Desktop:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Adicione o conteúdo do arquivo `claude_desktop_config_mt5.json` ao seu arquivo de configuração.

3. Reinicie o Claude Desktop para carregar a nova configuração.

## 📚 Funcionalidades Disponíveis

### Gerenciamento de Conexão
- `initialize()`: Inicializar o terminal MT5
- `login(account, password, server)`: Login em conta de trading
- `shutdown()`: Fechar conexão com MT5

### Dados de Mercado
- `get_symbols()`: Obter todos os símbolos disponíveis
- `get_symbol_info(symbol)`: Informações sobre um símbolo específico
- `copy_rates_from_pos()`: Obter barras de preço
- `copy_ticks_from_pos()`: Obter ticks de preço

### Trading
- `order_send(request)`: Enviar ordens para o servidor de trading
- `positions_get()`: Obter posições abertas
- `orders_get()`: Obter ordens ativas
- `history_orders_get()`: Obter histórico de ordens

## ⚠️ Pré-requisitos

1. **MetaTrader 5**: Deve estar instalado no sistema
2. **Conta MT5**: Conta demo ou real para testes
3. **Python 3.11+**: Já configurado no ambiente virtual
4. **Dependências**: Já instaladas (MetaTrader5, pandas, numpy, fastmcp, etc.)

## 🔧 Troubleshooting

### Problema: "Não é possível conectar ao MT5"
- Verifique se o MetaTrader 5 está instalado e rodando
- Confirme se a API está habilitada nas configurações do MT5

### Problema: "Servidor não responde"
- Verifique se o servidor está rodando na porta 8000
- Teste o acesso em http://127.0.0.1:8000

### Problema: "Erro de importação"
- Verifique se todas as dependências estão instaladas
- Execute: `pip install -e .` no diretório do projeto

## 📖 Recursos Adicionais

- **GitHub**: https://github.com/Qoyyuum/mcp-metatrader5-server
- **Documentação MCP**: https://modelcontextprotocol.io/
- **MetaTrader5 Python**: https://www.mql5.com/en/docs/python_metatrader5

## 🎯 Próximos Passos

1. **Teste a conexão**: Acesse http://127.0.0.1:8000 
2. **Configure o Claude**: Adicione a configuração MCP
3. **Teste com MT5**: Certifique-se de que o MetaTrader 5 está funcionando
4. **Experimente as funções**: Use as APIs disponíveis para trading e análise

---
*Instalação concluída em: 21 de agosto de 2025*
*Versão do MCP MetaTrader5 Server: 0.1.4*
