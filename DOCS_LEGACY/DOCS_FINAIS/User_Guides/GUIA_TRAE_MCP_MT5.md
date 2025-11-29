# 🚀 MCP MetaTrader5 Server para Trae

## ✅ Configuração Específica para Trae

O MCP MetaTrader5 Server foi instalado e está configurado para uso com o **Trae**!

### 📋 Arquivo de Configuração
- **Arquivo**: `trae_mcp_config_mt5.json`
- **Localização**: `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\`

### 🔧 Configuração do Trae

```json
{
  "mcpServers": {
    "MetaTrader 5 MCP Server": {
      "command": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe",
      "args": [
        "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/run.py",
        "dev",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
      ],
      "env": {
        "PYTHONPATH": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/src"
      }
    }
  }
}
```

### 🚀 Como Usar com Trae

#### 1. Iniciar o Servidor MCP
```powershell
# Opção 1: Script PowerShell
.\iniciar_mt5_mcp_server.ps1

# Opção 2: Script Batch  
iniciar_mt5_mcp_server.bat

# Opção 3: Manual
cd mcp-metatrader5-server
C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe run.py dev
```

#### 2. Configurar Trae
1. No Trae, vá para configurações de MCP Servers
2. Importe ou adicione a configuração do arquivo `trae_mcp_config_mt5.json`
3. O servidor será acessível em: `http://127.0.0.1:8000`

#### 3. Testar Conexão
```bash
# Testar se o servidor está respondendo
curl http://127.0.0.1:8000

# Ou use o script de teste
C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe teste_mt5_mcp.py
```

### 🛠️ Funcionalidades Disponíveis no Trae

#### 🔌 Gerenciamento de Conexão
```python
# Conectar ao MT5
initialize()

# Fazer login
login(account=123456, password="senha", server="servidor")

# Desconectar
shutdown()
```

#### 📊 Obter Dados de Mercado
```python
# Listar símbolos disponíveis
get_symbols()

# Informações de um símbolo específico
get_symbol_info("EURUSD")

# Obter dados de preços (barras)
copy_rates_from_pos(symbol="EURUSD", timeframe=15, start_pos=0, count=100)

# Obter ticks
copy_ticks_from_pos(symbol="EURUSD", start_pos=0, count=1000)
```

#### 💰 Trading e Ordens
```python
# Enviar ordem de compra
order_send({
    "action": "TRADE_ACTION_DEAL",
    "symbol": "EURUSD", 
    "volume": 0.1,
    "type": "ORDER_TYPE_BUY",
    "price": 1.1000,
    "deviation": 20,
    "magic": 123456,
    "comment": "Ordem via Trae"
})

# Verificar posições abertas
positions_get()

# Obter ordens ativas
orders_get()

# Histórico de negociações
history_orders_get()
```

### 🌐 Endpoints Disponíveis

O servidor MCP MetaTrader5 expõe as seguintes funcionalidades via HTTP:

- **Base URL**: `http://127.0.0.1:8000`
- **Status**: Verificar se o servidor está ativo
- **Tools**: Todas as funções MT5 disponíveis
- **Resources**: Documentação e guias integrados

### ⚡ Comandos Úteis

```powershell
# Verificar se o servidor está rodando
Get-Process python | Where-Object {$_.ProcessName -eq "python"}

# Parar o servidor (se necessário)
Stop-Process -Name "python" -Force

# Reiniciar o servidor
.\iniciar_mt5_mcp_server.ps1
```

### 🔍 Troubleshooting

#### Problema: Servidor não inicia
```powershell
# Verificar dependências
C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe teste_mt5_mcp.py

# Verificar porta em uso
netstat -an | findstr ":8000"
```

#### Problema: Trae não conecta
1. Verifique se o servidor está rodando em `http://127.0.0.1:8000`
2. Confirme a configuração JSON no Trae
3. Verifique logs do servidor

#### Problema: MT5 não conecta
1. Certifique-se de que o MetaTrader 5 está instalado
2. Verifique se a API está habilitada no MT5
3. Teste com o script: `teste_mt5_mcp.py`

### 📈 Exemplos Práticos para Trae

#### Análise de Mercado
```python
# Obter dados do EURUSD últimas 24 horas
rates = copy_rates_from_pos("EURUSD", 60, 0, 24)  # H1, 24 barras

# Calcular médias móveis
sma_20 = rates['close'][-20:].mean()
sma_50 = rates['close'][-50:].mean()

# Sinal de compra/venda
if sma_20 > sma_50:
    print("Sinal de COMPRA")
else:
    print("Sinal de VENDA")
```

#### Trading Automatizado
```python
# Sistema simples de trading
symbol = "EURUSD"
volume = 0.1

# Obter preço atual
tick = get_symbol_info_tick(symbol)
current_price = tick['bid']

# Estratégia simples (exemplo)
if current_price > moving_average:
    # Ordem de compra
    order_send({
        "action": "TRADE_ACTION_DEAL",
        "symbol": symbol,
        "volume": volume,
        "type": "ORDER_TYPE_BUY",
        "price": tick['ask'],
        "sl": current_price - 0.0050,  # Stop Loss
        "tp": current_price + 0.0100,  # Take Profit
        "deviation": 10,
        "magic": 123456,
        "comment": "Auto Buy via Trae"
    })
```

---
**✨ Seu MCP MetaTrader5 Server está configurado e pronto para uso com Trae!**

*Configure o Trae com o arquivo `trae_mcp_config_mt5.json` e comece a usar as funcionalidades de trading automatizado.*
