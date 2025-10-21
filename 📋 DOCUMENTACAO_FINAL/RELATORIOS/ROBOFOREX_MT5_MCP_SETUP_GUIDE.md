# 🏢 **RoboForex MetaTrader 5 MCP Configuration Guide**

## 🎯 **Configuração para Testes com RoboForex**

Este guia mostra como configurar o **MetaTrader 5 MCP** especificamente para usar o **RoboForex** como broker para testes do seu EA XAUUSD.

---

## 📋 **Pré-requisitos**

### **1. Conta RoboForex Demo** 📝
```bash
# Visite: https://www.roboforex.com/demo-account/
# Registre uma conta demo
# Dados necessários:
- Login: 123456789 (exemplo)
- Senha: SuaSenha123
- Servidor: RoboForex-Demo
```

### **2. MetaTrader 5 RoboForex** 💻
```bash
# Download do MT5 RoboForex:
# https://www.roboforex.com/trading-platforms/metatrader-5/
# Instale e configure com suas credenciais
```

---

## ⚙️ **Configuração do MCP**

### **1. Configuração RoboForex** 🔧

O arquivo `config/roboforex_config.json` já foi criado com as configurações específicas:

```json
{
  "broker_config": {
    "name": "RoboForex",
    "server_name": "RoboForex-Demo",
    "company": "RoboForex Ltd"
  },
  "connection_settings": {
    "server": "RoboForex-Demo",
    "timeout": 10000,
    "retry_attempts": 3
  },
  "symbol_settings": {
    "xauusd": {
      "symbol": "XAUUSD",
      "min_lot": 0.01,
      "max_lot": 100.0,
      "contract_size": 100
    }
  },
  "ftmo_compliance": {
    "enabled": true,
    "daily_loss_limit": 5.0,
    "total_loss_limit": 10.0,
    "hedging_prohibited": true
  }
}
```

### **2. Credenciais Seguras** 🔐

Use o script de setup para salvar suas credenciais:

```python
# Execute: python setup_roboforex_mt5.py
# E forneça suas credenciais RoboForex

from setup_roboforex_mt5 import RoboForexSetup

setup = RoboForexSetup()

# Salvar credenciais (substitua pelos seus dados reais)
setup.save_credentials(
    login=123456789,        # Seu login RoboForex
    password="SuaSenha123", # Sua senha
    server="RoboForex-Demo" # Servidor demo
)
```

### **3. Configuração MCP Atualizada** 📊

Sua configuração MCP agora inclui RoboForex específico:

```json
{
  "mcpServers": {
    "metatrader5_roboforex": {
      "command": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/.venv/Scripts/python.exe",
      "args": [
        "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/run.py",
        "dev",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--broker", "roboforex"
      ],
      "env": {
        "PYTHONPATH": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/src",
        "MT5_BROKER": "RoboForex",
        "MT5_SERVER": "RoboForex-Demo",
        "MT5_CONFIG_PATH": "C:/Users/Admin/Documents/EA_SCALPER_XAUUSD/mcp-metatrader5-server/config/roboforex_config.json"
      }
    }
  }
}
```

---

## 🧪 **Testando a Configuração**

### **1. Teste de Conexão** 🔌

```python
# Execute o teste de conexão
python setup_roboforex_mt5.py

# Ou teste programaticamente:
import asyncio
from setup_roboforex_mt5 import RoboForexSetup

async def test():
    setup = RoboForexSetup()
    await setup.test_connection(
        login=123456789,        # Seu login
        password="SuaSenha123", # Sua senha  
        server="RoboForex-Demo"
    )

asyncio.run(test())
```

### **2. Validação Esperada** ✅

O teste deve mostrar:

```bash
🔌 Testing RoboForex MT5 Connection...
✅ Connection successful!
📊 Setting up XAUUSD symbol...
✅ XAUUSD configured successfully

💰 RoboForex Trading Conditions:
  🏢 Broker: RoboForex
  🖥️ Server: RoboForex-Demo
  📊 Leverage: 1:100
  💱 Currency: USD
  📱 Account Type: Demo

🥇 XAUUSD Specifications:
  📈 Spread: 20 points
  📏 Min Lot: 0.01
  📏 Max Lot: 100.0
  💎 Contract Size: 100
  💰 Tick Value: $1.0
  🔢 Digits: 2

📡 Connection Quality Report:
  🔌 Status: connected
  ⚡ Average Latency: 15.23ms

✅ FTMO Compliance Check:
  🎯 Overall Compliant: ✅
  📊 Leverage OK: ✅
  💱 Currency OK: ✅
  🔄 Netting OK: ✅
```

---

## 🤖 **Como o Agente Autônomo Usará RoboForex**

### **1. Conexão Automática** 🔌

```python
# O agente se conectará automaticamente ao RoboForex
from roboforex_mt5_connector import RoboForexMT5Connector

async def autonomous_connection():
    connector = RoboForexMT5Connector()
    
    # Conectar ao RoboForex Demo
    if await connector.connect(login, password, "RoboForex-Demo"):
        print("✅ Connected to RoboForex MT5")
        
        # Setup XAUUSD para trading
        await connector.setup_xauusd_symbol()
        
        # Verificar condições de trading
        conditions = await connector.get_roboforex_trading_conditions()
        return conditions
    
    return None
```

### **2. Validação FTMO Contínua** ✅

```python
# Validação automática de compliance FTMO
async def continuous_ftmo_validation():
    connector = RoboForexMT5Connector()
    
    while True:
        # Verificar compliance FTMO
        compliance = await connector.validate_ftmo_compliance()
        
        if not compliance.get("ftmo_compliant"):
            print("⚠️ FTMO compliance issue detected!")
            # Tomar ações corretivas
            await handle_compliance_issue(compliance)
        
        await asyncio.sleep(60)  # Verificar a cada minuto
```

### **3. Monitoramento de Qualidade** 📡

```python
# Monitoramento contínuo da qualidade da conexão
async def monitor_connection_quality():
    connector = RoboForexMT5Connector()
    
    quality = await connector.test_connection_quality()
    
    # Verificar latência
    avg_latency = float(quality["latency"]["average"].replace("ms", ""))
    
    if avg_latency > 100:  # Se latência > 100ms
        print("⚠️ High latency detected, adjusting strategy...")
        # Ajustar parâmetros de trading para latência alta
```

---

## 🚀 **Execução Completa**

### **1. Iniciar MCP Server RoboForex** 

```bash
# No terminal, execute:
cd C:\Users\Admin\Documents\EA_SCALPER_XAUUSD
python mcp-metatrader5-server/run.py dev --host 127.0.0.1 --port 8000 --broker roboforex
```

### **2. Atualizar Configuração Qoder** 

```bash
# Execute o script de instalação:
./install_mcps_qoder.ps1

# Ou copie manualmente:
# qoder_mcp_config_complete.json → C:\Users\Admin\AppData\Roaming\Qoder\SharedClientCache\mcp.json
```

### **3. Reiniciar Qoder IDE** 🔄

Reinicie o Qoder IDE para carregar a nova configuração RoboForex.

---

## 📊 **Exemplo de Uso pelo Agente**

```python
async def autonomous_ea_development_roboforex():
    """Desenvolvimento autônomo de EA usando RoboForex"""
    
    # 1. Conectar ao RoboForex
    connector = RoboForexMT5Connector()
    await connector.connect(login, password, "RoboForex-Demo")
    
    # 2. Configurar XAUUSD
    await connector.setup_xauusd_symbol()
    
    # 3. Obter dados multi-timeframe
    h4_data = await get_rates("XAUUSD", mt5.TIMEFRAME_H4, 500)
    h1_data = await get_rates("XAUUSD", mt5.TIMEFRAME_H1, 1000)
    m15_data = await get_rates("XAUUSD", mt5.TIMEFRAME_M15, 2000)
    
    # 4. Analisar padrões específicos do RoboForex
    roboforex_conditions = await connector.get_roboforex_trading_conditions()
    spread = roboforex_conditions["xauusd"]["spread"]
    
    # 5. Ajustar estratégia para spread RoboForex
    strategy = adjust_strategy_for_spread(spread)
    
    # 6. Executar backtesting
    backtest_results = await run_backtest_roboforex(strategy)
    
    # 7. Validar FTMO compliance
    ftmo_valid = await connector.validate_ftmo_compliance()
    
    if ftmo_valid["ftmo_compliant"]:
        print("🚀 EA ready for RoboForex trading!")
        return True
    else:
        print("❌ FTMO compliance failed, refining strategy...")
        return False
```

---

## ⚠️ **Considerações Importantes**

### **🔐 Segurança**
- **NUNCA** committe credenciais reais no código
- Use variáveis de ambiente para dados sensíveis
- Teste sempre em conta demo primeiro

### **📊 Diferenças RoboForex**
- **Spread XAUUSD**: Tipicamente 20-30 points
- **Execution**: Market execution
- **Leverage**: Até 1:100 para FTMO compliance
- **Trading Hours**: 24/5 com gap de fim de semana

### **✅ FTMO Compliance**
- Max 5% daily loss
- Max 10% total loss
- No hedging allowed
- No martingale strategies
- Weekend holding restrictions

---

## 🏆 **Resultado Final**

Com esta configuração, seu agente autônomo pode:

1. **🔌 Conectar** automaticamente ao RoboForex MT5
2. **📊 Analisar** condições específicas do broker
3. **🧪 Testar** estratégias no ambiente RoboForex
4. **✅ Validar** compliance FTMO continuamente
5. **🚀 Executar** trades com parâmetros otimizados
6. **📈 Monitorar** performance em tempo real

**🎯 Seu sistema está agora configurado especificamente para RoboForex e pronto para desenvolvimento autônomo de EA XAUUSD!**

---

## 📞 **Suporte**

Se encontrar problemas:

1. **🔧 Teste a conexão**: `python setup_roboforex_mt5.py`
2. **📊 Verifique logs**: `logs/roboforex_mt5.log`
3. **🔄 Reinicie MT5**: Feche e abra o terminal RoboForex
4. **📝 Valide credenciais**: Teste login manual no MT5

---

*Configurado para RoboForex MetaTrader 5*  
*Data: 2025-08-22*  
*Sistema: EA_SCALPER_XAUUSD*