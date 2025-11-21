# 📝 Exemplos de Configuração - EA_SCALPER_XAUUSD

## 🎯 Visão Geral

Este documento contém exemplos práticos de configuração para diferentes cenários de uso do sistema EA_SCALPER_XAUUSD.

---

## 📋 Índice
1. [Configuração para Iniciantes](#config-iniciantes)
2. [Configuração para Traders Intermediários](#config-intermediarios)
3. [Configuração para Traders Avançados](#config-avancados)
4. [Configuração para Contas FTMO](#config-ftmo)
5. [Configuração para Multi-Moedas](#config-multi)
6. [Configuração para Backtest](#config-backtest)
7. [Configuração para Produção](#config-producao)

---

## 🌱 Configuração para Iniciantes

### Arquivo .env Básico

```env
# ================================
# CONFIGURAÇÃO BÁSICA - INICIANTES
# ================================

# OpenRouter API
OPENROUTER_API_KEY=sk-or-v1-sua_chave_api_aqui
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD Iniciante"
DEFAULT_MODEL="deepseek/deepseek-r1-0528:free"
BACKUP_MODEL="meta-llama/llama-3.1-8b-instruct:free"

# Proxy Server
PROXY_HOST=127.0.0.1
PROXY_PORT=4000
PROXY_RATE_LIMIT=3.0  # Mais conservador

# Cache
PROMPT_CACHE_TTL=3600
RESPONSE_CACHE_TTL=1800
ENABLE_CACHE=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/iniciante.log
ENABLE_DEBUG=false

# MetaTrader (opcional, configurar via MT5)
# MT5_LOGIN=seu_login_demo
# MT5_PASSWORD=sua_senha_demo
# MT5_SERVER=seu_servidor_demo
```

### Parâmetros do EA para Iniciantes

```mql5
// ================================
// CONFIGURAÇÃO EA - INICIANTES
// ================================

// Risk Management (Ultra Conservador)
input group "Risk Management - Iniciante"
input double LotSize = 0.01;           // Lote fixo mínimo
input int StopLoss = 300;              // 30 pips de segurança
input int TakeProfit = 600;            // 60 pips (2:1 risk/reward)
input double MaxDrawdown = 5.0;        // Máximo 5% de drawdown
input double MaxDailyLoss = 50.0;      // Máximo $50 de perda diária

// Trading Schedule (Horário Seguro)
input group "Horário de Trading"
input int StartHour = 10;              // Início 10:00 (mais volume)
input int EndHour = 18;                // Fim 18:00 (evitar notícias)
input bool MondayTrading = true;       // Segunda sim
input bool TuesdayTrading = true;      // Terça sim
input bool WednesdayTrading = true;    // Quarta sim
input bool ThursdayTrading = true;     // Quinta sim
input bool FridayTrading = false;      // Sexta não (riscos)

// Indicadores (Simples e Confirmados)
input group "Indicadores Básicos"
input bool UseMAFilter = true;         // Filtro de média móvel
input int MA_Period = 20;              // Média de 20 períodos
input bool UseRSIFilter = true;        // Filtro RSI
input int RSI_Period = 14;             // RSI padrão
input int RSI_Overbought = 70;         // Sobrecompra
input int RSI_Oversold = 30;           // Sobrevenda

// Controle de Posições
input group "Controle de Posições"
input int MaxPositions = 1;            // Máximo 1 posição por vez
input int MagicNumber = 12345;         // Magic único
input bool EnableTrading = true;       // Habilitar trading
input string TradeComment = "EA_Iniciante"; // Comentário nos trades

// Notificações (Básicas)
input group "Notificações"
input bool EnableAlerts = true;        // Alertas sonoros
input bool EnablePushNotifications = false; // Sem push no início
```

### Script de Inicialização para Iniciantes

```bash
#!/bin/bash
# start_iniciante.sh

echo "🚀 Iniciando EA_SCALPER_XAUUSD - Modo Iniciante"
echo "================================================"

# 1. Ativar ambiente virtual
echo "📦 Ativando ambiente virtual..."
source venv/bin/activate

# 2. Verificar configuração
echo "🔍 Verificando configuração..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print(f'✅ API Key configurada: {bool(os.getenv(\"OPENROUTER_API_KEY\"))}')
print(f'✅ Proxy porta: {os.getenv(\"PROXY_PORT\", \"4000\")}')
print(f'✅ Log level: {os.getenv(\"LOG_LEVEL\", \"INFO\")}')
"

# 3. Iniciar proxy
echo "🌐 Iniciando proxy server..."
python scripts/python/simple_trading_proxy.py &
PROXY_PID=$!
echo "Proxy iniciado com PID: $PROXY_PID"

# 4. Aguardar inicialização
sleep 3

# 5. Testar conexão
echo "🧪 Testando conexões..."
curl -s http://localhost:4000/health | python -m json.tool

# 6. Criar log do dia
echo "📝 Criando log do dia..."
mkdir -p logs
echo "$(date): Sistema iniciado em modo iniciante" >> logs/iniciante_$(date +%Y%m%d).log

echo "================================================"
echo "✅ Sistema pronto para uso!"
echo "📊 Abra o MetaTrader 5 e ative o EA"
echo "🛑 Para parar: kill $PROXY_PID"
echo "================================================"
```

---

## 💼 Configuração para Traders Intermediários

### Arquivo .env Intermediário

```env
# ================================
# CONFIGURAÇÃO - INTERMEDIÁRIOS
# ================================

# OpenRouter API (Modelos Melhores)
OPENROUTER_API_KEY=sk-or-v1-sua_chave_api_aqui
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD Intermediário"
DEFAULT_MODEL="openrouter/anthropic/claude-3-5-sonnet"
BACKUP_MODEL="openrouter/openai/gpt-4o"
FAST_MODEL="openrouter/meta-llama/llama-3.1-70b-instruct"

# Proxy Server (Otimizado)
PROXY_HOST=0.0.0.0
PROXY_PORT=4000
PROXY_RATE_LIMIT=2.0
PROXY_CACHE_SIZE=2000

# Cache Avançado
PROMPT_CACHE_TTL=7200
RESPONSE_CACHE_TTL=3600
ENABLE_CACHE=true
CACHE_CLEANUP_INTERVAL=3600

# Logging Detalhado
LOG_LEVEL=DEBUG
LOG_FILE=logs/intermediario.log
ENABLE_DEBUG=true
PERFORMANCE_LOGGING=true

# Análise Avançada
ENABLE_MARKET_ANALYSIS=true
ENABLE_SIGNAL_CONFIRMATION=true
ENABLE_RISK_ASSESSMENT=true

# Integrações
ENABLE_GITHUB_BACKUP=true
ENABLE_CLOUD_SYNC=false
ENABLE_EMAIL_NOTIFICATIONS=true

# MetaTrader
MT5_AUTO_RECONNECT=true
MT5_HEARTBEAT_INTERVAL=30
```

### Parâmetros do EA Intermediário

```mql5
// ================================
// CONFIGURAÇÃO EA - INTERMEDIÁRIOS
// ================================

// Risk Management (Moderado)
input group "Risk Management - Intermediário"
input double LotSize = 0.02;           // Lote maior
input bool UseDynamicLot = true;       // Lote dinâmico
input double LotPercent = 1.0;         // 1% do equity por operação
input int StopLoss = 200;              // 20 pips
input int TakeProfit = 400;            // 40 pips
input double MaxDrawdown = 8.0;        // Máximo 8%
input double MaxDailyLoss = 100.0;     // Máximo $100 diário
input bool UseTrailingStop = true;     // Trailing stop
input int TrailingStop = 150;          // 15 pips trailing

// Trading Schedule (Estendido)
input group "Horário de Trading"
input int StartHour = 8;               // Início 8:00
input int EndHour = 20;                // Fim 20:00
input bool MondayTrading = true;
input bool TuesdayTrading = true;
input bool WednesdayTrading = true;
input bool ThursdayTrading = true;
input bool FridayTrading = true;       // Sexta sim (com cuidado)
input bool AvoidNews = true;           // Evitar notícias
input int NewsAvoidanceMins = 30;      // Evitar 30 min antes/depois

// Indicadores Múltiplos
input group "Indicadores Avançados"
input bool UseMAFilter = true;
input int MA_Period_Fast = 10;
input int MA_Period_Slow = 20;
input bool UseRSIFilter = true;
input int RSI_Period = 14;
input bool UseMACD = true;             // Adicionar MACD
input int MACD_Fast = 12;
input int MACD_Slow = 26;
input int MACD_Signal = 9;
input bool UseVolumeFilter = true;     // Filtro de volume
input double MinVolumeRatio = 1.2;     // Volume mínimo 1.2x média

// Estratégia Avançada
input group "Estratégia"
input bool UseMultiTimeframe = true;   // Confirmar em M15
input int HigherTimeframe = 15;        // M15 para confirmação
input bool UseSupportResistance = true; // S/R levels
input int LookbackBars = 100;          // Análise de 100 barras
input bool UseBreakoutStrategy = true; // Estratégia de breakout

// Controle de Posições
input group "Controle de Posições"
input int MaxPositions = 2;            // Até 2 posições
input int MagicNumber = 54321;
input bool EnableHedging = false;      // Sem hedge
input double MinDistanceBetweenTrades = 50; // Distância mínima
input bool EnablePyramiding = false;   // Sem pirâmide

// Notificações Avançadas
input group "Notificações"
input bool EnableAlerts = true;
input bool EnablePushNotifications = true;
input bool EnableEmailNotifications = true;
input double AlertProfitThreshold = 50.0; // Alertar ao atingir $50
input double AlertLossThreshold = 30.0;   // Alertar na perda de $30
```

### Script de Monitoramento Intermediário

```python
#!/usr/bin/env python3
# monitor_intermediario.py

import time
import MetaTrader5 as mt5
import json
import logging
from datetime import datetime
from pathlib import Path

class IntermediarioMonitor:
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()

    def setup_logging(self):
        """Configurar logging detalhado"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/intermediario_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Carregar configuração"""
        with open('config/intermediario.json', 'r') as f:
            return json.load(f)

    def check_trading_conditions(self):
        """Verificar condições de trading"""
        try:
            if not mt5.initialize():
                self.logger.error("Falha ao inicializar MT5")
                return False

            account = mt5.account_info()
            current_dd = abs(account.balance - account.equity) / account.balance * 100

            if current_dd > self.config['max_drawdown']:
                self.logger.warning(f"Drawdown alto: {current_dd:.2f}%")
                self.send_alert(f"Drawdown alert: {current_dd:.2f}%")

            # Verificar posições abertas
            positions = mt5.positions_get(symbol="XAUUSD")
            if positions:
                total_profit = sum(pos.profit for pos in positions)
                self.logger.info(f"Posições abertas: {len(positions)}, P&L: {total_profit:.2f}")

                # Alertar se perda diária exceder limite
                if total_profit < -self.config['max_daily_loss']:
                    self.logger.critical(f"Perda diária excedida: {total_profit:.2f}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Erro ao verificar condições: {e}")
            return False

    def analyze_performance(self):
        """Analisar performance do dia"""
        try:
            # Obter histórico de trades do dia
            today = datetime.now().date()
            from_date = datetime.combine(today, datetime.min.time())
            to_date = datetime.combine(today, datetime.max.time())

            history = mt5.history_deals_get(from_date, to_date)
            if not history:
                return

            # Calcular métricas
            profitable_trades = [d for d in history if d.profit > 0]
            losing_trades = [d for d in history if d.profit < 0]

            win_rate = len(profitable_trades) / len(history) * 100 if history else 0
            avg_win = sum(d.profit for d in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(d.profit for d in losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            performance_data = {
                'date': today.isoformat(),
                'total_trades': len(history),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_profit': sum(d.profit for d in history)
            }

            # Salvar relatório
            with open(f'data/reports/performance_{today.strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(performance_data, f, indent=2)

            self.logger.info(f"Performance atualizada: Win Rate {win_rate:.1f}%, PF {profit_factor:.2f}")

        except Exception as e:
            self.logger.error(f"Erro na análise: {e}")

    def send_alert(self, message):
        """Enviar alerta"""
        # Implementar envio de email/push notification
        self.logger.info(f"ALERT: {message}")

    def run(self):
        """Executar monitoramento"""
        self.logger.info("Iniciando monitoramento intermediário")

        while True:
            try:
                if self.check_trading_conditions():
                    self.analyze_performance()
                else:
                    self.logger.warning("Condições de trading inadequadas")

                time.sleep(60)  # Verificar a cada minuto

            except KeyboardInterrupt:
                self.logger.info("Monitoramento interrompido")
                break
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(30)

if __name__ == "__main__":
    monitor = IntermediarioMonitor()
    monitor.run()
```

---

## 🏆 Configuração para Traders Avançados

### Arquivo .env Avançado

```env
# ================================
# CONFIGURAÇÃO AVANÇADA - PROFISSIONAIS
# ================================

# OpenRouter API (Top Models)
OPENROUTER_API_KEY=sk-or-v1-sua_chave_api_aqui
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD Professional"
DEFAULT_MODEL="openrouter/anthropic/claude-3-opus"
BACKUP_MODEL="openrouter/openai/gpt-4-turbo"
FAST_MODEL="openrouter/meta-llama/llama-3.1-405b-instruct"
ULTRA_FAST_MODEL="openrouter/mistralai/mistral-7b-instruct:free"

# High-Performance Proxy
PROXY_HOST=0.0.0.0
PROXY_PORT=4000
PROXY_WORKERS=4
PROXY_RATE_LIMIT=1.0
PROXY_CACHE_SIZE=5000
PROXY_ENABLE_PERSISTENCE=true

# Cache Avançado com Redis
PROMPT_CACHE_TTL=14400
RESPONSE_CACHE_TTL=7200
ENABLE_CACHE=true
CACHE_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Logging Nível Enterprise
LOG_LEVEL=TRACE
LOG_FILE=logs/advanced.log
ENABLE_DEBUG=true
PERFORMANCE_LOGGING=true
NETWORK_LOGGING=true
DATABASE_LOGGING=true

# Machine Learning
ENABLE_ML_PREDICTIONS=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_NEURAL_NETWORKS=true
MODEL_UPDATE_INTERVAL=3600

# High-Frequency Features
ENABLE_TICK_ANALYSIS=true
ENABLE_ORDER_BOOK_ANALYSIS=true
ENABLE_MARKET_DEPTH=true
MICROSTRUCTURE_ANALYSIS=true

# Advanced Integrations
ENABLE_GITHUB_BACKUP=true
ENABLE_CLOUD_SYNC=true
ENABLE_EMAIL_NOTIFICATIONS=true
ENABLE_SLACK_NOTIFICATIONS=true
ENABLE_TELEGRAM_NOTIFICATIONS=true

# Enterprise Features
ENABLE_LOAD_BALANCING=true
ENABLE_AUTO_SCALING=true
ENABLE_FAULT_TOLERANCE=true
ENABLE_REDIS_CLUSTER=true

# MetaTrader Enterprise
MT5_AUTO_RECONNECT=true
MT5_HEARTBEAT_INTERVAL=15
MT5_ENABLE_TICK_STREAM=true
MT5_ENABLE_BOOK_STREAM=true
MT5_CONNECTION_TIMEOUT=5
MT5_REQUEST_TIMEOUT=10
```

### Parâmetros do EA Avançado

```mql5
// ================================
// CONFIGURAÇÃO EA - AVANÇADO
// ================================

// Risk Management Professional
input group "Risk Management - Professional"
input double LotSize = 0.05;           // Lote maior
input bool UseDynamicLot = true;       // Lote dinâmico avançado
input double LotPercent = 2.0;         // 2% por operação
input bool UseVolatilityBasedLot = true; // Baseado em volatilidade
input double VolatilityMultiplier = 1.5;
input int StopLoss = 150;              // 15 pips
input int TakeProfit = 300;            // 30 pips
input double MaxDrawdown = 15.0;       // Máximo 15%
input double MaxDailyLoss = 500.0;     // Máximo $500 diário
input bool UseTrailingStop = true;     // Trailing stop avançado
input int TrailingStop = 100;          // 10 pips
input bool UseBreakeven = true;        // Break-even automático
input int BreakevenPoints = 150;       // BE após 15 pips

// Trading Schedule 24/5
input group "Horário de Trading Profissional"
input int StartHour = 0;               // Início meia-noite
input int EndHour = 23;                // Fim 23:00
input bool MondayTrading = true;
input bool TuesdayTrading = true;
input bool WednesdayTrading = true;
input bool ThursdayTrading = true;
input bool FridayTrading = true;
input bool WeekendTrading = false;      // Fim de semana não
input bool UseAsianSession = true;     // Sessão asiática
input bool UseEuropeanSession = true;  // Sessão europeia
input bool UseAmericanSession = true;  // Sessão americana
input bool AvoidNews = true;           // Evitar notícias importantes
input int NewsAvoidanceMins = 60;      // Evitar 1 hora

// Indicadores Profissionais
input group "Indicadores Profissionais"
input bool UseMAFilter = true;
input int MA_Period_Fast = 5;
input int MA_Period_Medium = 20;
input int MA_Period_Slow = 50;
input bool UseRSIFilter = true;
input int RSI_Period = 14;
input bool UseMACD = true;
input bool UseStochastic = true;       // Adicionar Stochastic
input bool UseBollingerBands = true;   // Adicionar Bollinger Bands
input bool UseATR = true;              // Average True Range
input int ATR_Period = 14;
input bool UseVolumeProfile = true;    // Volume Profile
input bool UseOrderFlow = true;        // Order Flow Analysis

// Estratégia Multi-Camadas
input group "Estratégia Multi-Camadas"
input bool UseMultiTimeframe = true;
input int MTF_Timeframe1 = 15;         // M15
input int MTF_Timeframe2 = 60;         // H1
input int MTF_Timeframe3 = 240;        // H4
input bool UseNeuralSignals = true;    // Sinais de rede neural
input bool UseSentimentAnalysis = true; // Análise de sentimento
input bool UseCorrelationAnalysis = true; // Análise de correlação
input string CorrelatedPairs = "EURUSD,GBPUSD"; // Pares correlacionados
input bool UseMarketRegimeDetection = true; // Detecção de regime de mercado

// Risk Management Avançado
input group "Risk Management Avançado"
input bool UseKellyCriterion = true;   // Kelly Criterion
input double KellyFraction = 0.25;     // Fração de Kelly
input bool UsePortfolioHeat = true;    // Portfolio Heat
input double MaxPortfolioHeat = 20.0;  // Máximo 20% heat
input bool UseCorrelationLimit = true; // Limite de correlação
input double MaxCorrelation = 0.7;     // Máximo 70% correlação
input bool UseVaRLimit = true;         // Value at Risk
input double MaxDailyVaR = 2.0;        // Máximo 2% VaR diário

// Execução Avançada
input group "Execução Avançada"
input bool UseLimitOrders = true;      // Ordens limitadas
input int OrderTimeout = 10;           // Timeout 10 segundos
input bool UsePartialFills = true;     // Preenchimentos parciais
input double PartialFillPercent = 50.0; // 50% preenchimento
input bool UseIcebergOrders = true;    // Ordens Iceberg
input int IcebergSlices = 3;           // 3 fatias

// Machine Learning Integration
input group "Machine Learning"
input bool UseMLPrediction = true;     // Previsão ML
input string MLPredictionModel = "ensemble_v2"; // Modelo ensemble
input bool UseReinforcementLearning = true; // Reinforcement Learning
input bool UseAdaptiveParameters = true; // Parâmetros adaptativos
input int AdaptationInterval = 100;    // A cada 100 trades

// Controle Profissional
input group "Controle Profissional"
input int MaxPositions = 5;            // Até 5 posições
input bool EnableHedging = true;       // Permitir hedge
input bool EnableNetting = false;      // Não netting
input int MagicNumber = 99999;
input string TradeComment = "EA_Advanced";
input bool EnableAutoRestart = true;   // Restart automático
input int RestartInterval = 3600;      // Restart a cada hora
```

---

## 🏢 Configuração para Contas FTMO

### Requisitos FTMO Específicos

**Máximo Drawdown**: 10% (5% para Challenge)
**Meta de Lucro**: 10% (5% para Challenge)
**Trading Days**: Mínimo 10 dias (4 para Challenge)
**Maximum Loss**: 10% (5% para Challenge)

### Configuração .env FTMO

```env
# ================================
# CONFIGURAÇÃO FTMO - RULES COMPLIANT
# ================================

# OpenRouter (Models de alta precisão)
OPENROUTER_API_KEY=sk-or-v1-sua_chave_api_aqui
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD FTMO"
DEFAULT_MODEL="openrouter/anthropic/claude-3-opus"
BACKUP_MODEL="openrouter/openai/gpt-4-turbo"

# FTMO Specific Settings
FTMO_MODE=true
FTMO_CHALLENGE=false  # Mudar para true se estiver no challenge
FTMO_MAX_DRAWDOWN=5.0  # 5% para challenge, 10% para funded
FTMO_PROFIT_TARGET=5.0  # 5% para challenge, 10% para funded
FTMO_MIN_TRADING_DAYS=4
FTMO_MAX_DAILY_LOSS=2.5  # 2.5% diário máximo
FTMO_MAX_LOSS=5.0  # 5% total máximo

# Risk Management Extremo
ENABLE_HARD_STOPS=true
EMERGENCY_STOP_DRAWDOWN=4.0  # Parar em 4% drawdown
EMERGENCY_STOP_LOSS=4.5  # Parar em 4.5% perda
DAILY_LOSS_LIMIT=2.0  # 2% perda diária

# Logging Detalhado para Compliance
FTMO_LOGGING=true
TRADE_LOGGING=true
COMPLIANCE_LOGGING=true
AUDIT_TRAIL=true

# MetaTrader FTMO
MT5_FUTURES_SYMBOL=XAUUSD.f
MT5_MIN_LOT_SIZE=0.01
MT5_MAX_LOT_SIZE=1.0
MT5_TICK_SIZE=0.01
MT5_TICK_VALUE=1.0
```

### Parâmetros do EA FTMO

```mql5
// ================================
// CONFIGURAÇÃO EA - FTMO COMPLIANT
// ================================

// FTMO Risk Management (Extremamente Conservador)
input group "FTMO Risk Management"
input double LotSize = 0.01;           // Lote mínimo
input bool UseDynamicLot = true;       // Lote dinâmico ultra conservador
input double LotPercent = 0.5;         // 0.5% por operação
input int StopLoss = 300;              // 30 pips (grande segurança)
input int TakeProfit = 600;            // 60 pips (2:1 ratio)
input double MaxDrawdown = 4.0;        // Máximo 4% (abaixo do limite)
input double MaxDailyLoss = 2.0;       // Máximo 2% diário
input double MaxTotalLoss = 4.5;       // Máximo 4.5% total
input bool UseTrailingStop = false;    // Sem trailing stop (previsibilidade)
input bool UseBreakeven = true;        // Break-even em 1:1
input int BreakevenPoints = 300;       // BE após 30 pips

// FTMO Trading Schedule (Horários seguros)
input group "FTMO Trading Schedule"
input int StartHour = 9;               // Início 9:00 (abertura London)
input int EndHour = 19;                // Fim 19:00 (antes do close NY)
input bool MondayTrading = true;
input bool TuesdayTrading = true;
input bool WednesdayTrading = true;
input bool ThursdayTrading = true;
input bool FridayTrading = false;      // Sexta não (risco de weekend gap)
input bool AvoidNews = true;           // Evitar todas as notícias
input int NewsAvoidanceMins = 120;     // Evitar 2 horas antes/depois

// FTMO Strategy (Consistente e previsível)
input group "FTMO Strategy"
input bool UseMAFilter = true;         // Apenas média móvel
input int MA_Period = 50;              // Média longa para sinais mais confiáveis
input bool UseRSIFilter = true;        // RSI para confirmação
input int RSI_Period = 21;             // RSI longo para menos ruído
input int RSI_Overbought = 65;         // Mais conservador
input int RSI_Oversold = 35;           // Mais conservador
input bool UseVolumeFilter = true;     // Filtro de volume obrigatório
input double MinVolumeRatio = 1.5;     // Volume mínimo 1.5x média

// FTMO Position Management
input group "FTMO Position Management"
input int MaxPositions = 2;            // Máximo 2 posições
input bool SameDirectionOnly = true;   // Mesma direção apenas
input double MinDistanceBetweenTrades = 100; // 100 pips mínimo
input bool MartingaleDisabled = true;  // Sem martingale
input bool HedgingDisabled = true;     // Sem hedge
input bool PyramidingDisabled = true;  // Sem pyramiding
input int MagicNumber = 88888;         // Magic FTMO

// FTMO Compliance
input group "FTMO Compliance"
input bool EnableTrading = true;
input bool EnableEmergencyStop = true; // Parada de emergência
input double EmergencyStopLevel = 4.0; // Parar em 4% drawdown
input bool RequireMinBars = true;      // Requerir mínimo de barras
input int MinBarsRequired = 50;        // Mínimo 50 barras
input bool MaxTradePerDay = true;      // Limitar trades por dia
input int MaxTradesPerDay = 10;        // Máximo 10 trades/dia
input bool ConsistentLotSize = true;   // Lote consistente

// FTMO Monitoring
input group "FTMO Monitoring"
input bool EnableFTMOAlerts = true;    // Alertas específicos FTMO
input bool LogAllTrades = true;        // Log todos os trades
input bool GenerateFTMOReport = true;  // Gerar relatório FTMO
input string FTMOReportPath = "reports/ftmo/"; // Caminho dos relatórios
```

### Script de Monitoramento FTMO

```python
#!/usr/bin/env python3
# ftmo_monitor.py

import MetaTrader5 as mt5
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class FTMOComplianceMonitor:
    def __init__(self):
        self.ftmo_config = {
            'max_drawdown': 5.0,  # 5% para challenge
            'max_daily_loss': 2.5,
            'max_total_loss': 5.0,
            'profit_target': 5.0,
            'min_trading_days': 4,
            'max_trades_per_day': 10
        }
        self.setup_logging()

    def setup_logging(self):
        """Setup logging específico FTMO"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - FTMO_MONITOR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ftmo_compliance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_ftmo_compliance(self):
        """Verificar compliance FTMO"""
        try:
            if not mt5.initialize():
                return False

            account = mt5.account_info()

            # Verificar drawdown
            current_dd = abs(account.balance - account.equity) / account.balance * 100
            if current_dd > self.ftmo_config['max_drawdown']:
                self.logger.critical(f"ALERTA FTMO: Drawdown {current_dd:.2f}% > limite {self.ftmo_config['max_drawdown']}%")
                return False

            # Verificar perda diária
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_deals = mt5.history_deals_get(today_start, datetime.now())
            daily_loss = sum(d.profit for d in today_deals if d.profit < 0)
            daily_loss_percent = abs(daily_loss) / account.balance * 100

            if daily_loss_percent > self.ftmo_config['max_daily_loss']:
                self.logger.critical(f"ALERTA FTMO: Perda diária {daily_loss_percent:.2f}% > limite {self.ftmo_config['max_daily_loss']}%")
                return False

            # Verificar meta de lucro
            if account.equity > account.balance * 1.05:  # 5% de lucro
                self.logger.info(f"Meta FTMO alcançada! Lucro: {(account.equity/account.balance-1)*100:.2f}%")

            return True

        except Exception as e:
            self.logger.error(f"Erro na verificação FTMO: {e}")
            return False

    def generate_ftmo_report(self):
        """Gerar relatório de compliance FTMO"""
        try:
            account = mt5.account_info()

            # Calcular métricas FTMO
            current_dd = abs(account.balance - account.equity) / account.balance * 100
            profit_pct = (account.equity / account.balance - 1) * 100

            # Obter trades do período
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            history = mt5.history_deals_get(start_date, end_date)

            # Calcular dias de trading
            trading_days = len(set(deal.time for deal in history))

            report = {
                'timestamp': datetime.now().isoformat(),
                'account_info': {
                    'balance': account.balance,
                    'equity': account.equity,
                    'profit': profit_pct,
                    'drawdown': current_dd
                },
                'ftmo_metrics': {
                    'current_drawdown': current_dd,
                    'max_allowed_drawdown': self.ftmo_config['max_drawdown'],
                    'profit_target_achieved': profit_pct >= self.ftmo_config['profit_target'],
                    'trading_days_count': trading_days,
                    'min_trading_days_required': self.ftmo_config['min_trading_days'],
                    'compliance_status': current_dd < self.ftmo_config['max_drawdown']
                },
                'recommendations': self.get_ftmo_recommendations(current_dd, profit_pct, trading_days)
            }

            # Salvar relatório
            Path('reports/ftmo').mkdir(parents=True, exist_ok=True)
            with open(f'reports/ftmo/ftmo_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Relatório FTMO gerado: Compliance {report['ftmo_metrics']['compliance_status']}")

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório FTMO: {e}")

    def get_ftmo_recommendations(self, drawdown, profit, trading_days):
        """Obter recomendações FTMO"""
        recommendations = []

        if drawdown > 3.0:
            recommendations.append("REDUZIR RISCO: Drawdown próximo do limite")

        if profit > 4.0:
            recommendations.append("META PRÓXIMA: Considere reduzir exposição")

        if trading_days < self.ftmo_config['min_trading_days']:
            recommendations.append("AUMENTAR FREQUÊNCIA: Mais dias de trading necessários")

        if len(recommendations) == 0:
            recommendations.append("EXCELLENTE: Sistema operando dentro dos limites FTMO")

        return recommendations

    def run_monitoring(self):
        """Executar monitoramento contínuo"""
        self.logger.info("Iniciando monitoramento FTMO")

        while True:
            try:
                if not self.check_ftmo_compliance():
                    self.logger.critical("Sistema não está em compliance FTMO!")
                    # Implementar ação de emergência

                self.generate_ftmo_report()
                time.sleep(300)  # Verificar a cada 5 minutos

            except KeyboardInterrupt:
                self.logger.info("Monitoramento FTMO interrompido")
                break
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(60)

if __name__ == "__main__":
    monitor = FTMOComplianceMonitor()
    monitor.run_monitoring()
```

---

## 🌍 Configuração para Multi-Moedas

### Configuração .env Multi-Moedas

```env
# ================================
# CONFIGURAÇÃO MULTI-MOEDAS
# ================================

# Ativar modo multi-moedas
MULTI_CURRENCY_MODE=true
SYMBOLS=XAUUSD,EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD
PRIMARY_SYMBOL=XAUUSD
HEDGE_SYMBOLS=EURUSD,GBPUSD

# Configurações por símbolo
XAUUSD_LOT_SIZE=0.01
XAUUSD_STOP_LOSS=200
XAUUSD_TAKE_PROFIT=400
EURUSD_LOT_SIZE=0.1
EURUSD_STOP_LOSS=20
EURUSD_TAKE_PROFIT=40
GBPUSD_LOT_SIZE=0.1
GBPUSD_STOP_LOSS=25
GBPUSD_TAKE_PROFIT=50

# Correlação entre pares
CORRELATION_ANALYSIS=true
MAX_CORRELATED_POSITIONS=2
CORRELATION_THRESHOLD=0.7

# Portfolio Management
PORTFOLIO_HEAT_LIMIT=20.0
MAX_TOTAL_EXPOSURE=5.0
CURRENCY_DISTRIBUTION=EQUAL  # EQUAL, WEIGHTED, CUSTOM

# Risk Management Multi
ENABLE_DAILY_LOSS_PER_SYMBOL=true
SYMBOL_DAILY_LOSS_LIMIT=50.0
ENABLE_TOTAL_DAILY_LOSS=true
TOTAL_DAILY_LOSS_LIMIT=200.0

# Timeframes por símbolo
XAUUSD_TIMEFRAME=M5
EURUSD_TIMEFRAME=M15
GBPUSD_TIMEFRAME=M15
USDJPY_TIMEFRAME=M30
```

### Script de Gerenciamento Multi-Moedas

```python
#!/usr/bin/env python3
# multi_currency_manager.py

import MetaTrader5 as mt5
import json
import time
from datetime import datetime
from typing import Dict, List

class MultiCurrencyManager:
    def __init__(self):
        self.symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        self.symbol_configs = self.load_symbol_configs()
        self.setup_logging()

    def load_symbol_configs(self):
        """Carregar configurações por símbolo"""
        return {
            "XAUUSD": {
                "lot_size": 0.01,
                "stop_loss": 200,
                "take_profit": 400,
                "timeframe": mt5.TIMEFRAME_M5,
                "max_positions": 2
            },
            "EURUSD": {
                "lot_size": 0.1,
                "stop_loss": 20,
                "take_profit": 40,
                "timeframe": mt5.TIMEFRAME_M15,
                "max_positions": 1
            },
            "GBPUSD": {
                "lot_size": 0.1,
                "stop_loss": 25,
                "take_profit": 50,
                "timeframe": mt5.TIMEFRAME_M15,
                "max_positions": 1
            },
            "USDJPY": {
                "lot_size": 0.1,
                "stop_loss": 20,
                "take_profit": 40,
                "timeframe": mt5.TIMEFRAME_M30,
                "max_positions": 1
            }
        }

    def setup_logging(self):
        """Setup logging multi-moedas"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - MULTI_CURRENCY - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/multi_currency.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_portfolio_heat(self):
        """Calcular portfolio heat"""
        try:
            account = mt5.account_info()
            total_exposure = 0.0

            for symbol in self.symbols:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    symbol_exposure = sum(pos.volume * pos.price_current for pos in positions)
                    total_exposure += symbol_exposure

            portfolio_heat = (total_exposure / account.balance) * 100
            return portfolio_heat

        except Exception as e:
            self.logger.error(f"Erro ao calcular portfolio heat: {e}")
            return 0.0

    def check_correlation_risk(self):
        """Verificar risco de correlação"""
        # Implementar análise de correlação entre posições
        correlated_positions = []

        for symbol in self.symbols:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                # Lógica de correlação (simplificada)
                if symbol in ["EURUSD", "GBPUSD"] and len(positions) > 0:
                    correlated_positions.append(symbol)

        return len(correlated_positions)

    def distribute_risk(self):
        """Distribuir risco entre moedas"""
        account = mt5.account_info()
        risk_per_symbol = account.balance * 0.01  # 1% por símbolo

        risk_allocation = {}
        for symbol in self.symbols:
            config = self.symbol_configs[symbol]
            risk_allocation[symbol] = {
                "max_risk": risk_per_symbol,
                "lot_size": config["lot_size"],
                "max_positions": config["max_positions"]
            }

        return risk_allocation

    def monitor_all_symbols(self):
        """Monitorar todos os símbolos"""
        portfolio_status = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_heat": self.get_portfolio_heat(),
            "correlated_positions": self.check_correlation_risk(),
            "symbols_status": {}
        }

        for symbol in self.symbols:
            positions = mt5.positions_get(symbol=symbol)
            symbol_profit = sum(pos.profit for pos in positions) if positions else 0.0

            portfolio_status["symbols_status"][symbol] = {
                "positions": len(positions) if positions else 0,
                "profit": symbol_profit,
                "config": self.symbol_configs[symbol]
            }

        # Salvar status
        with open(f'data/reports/portfolio_status_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(portfolio_status, f, indent=2)

        self.logger.info(f"Portfolio Heat: {portfolio_status['portfolio_heat']:.2f}%")

        return portfolio_status

    def run_manager(self):
        """Executar gerenciador multi-moedas"""
        self.logger.info("Iniciando gerenciador multi-moedas")

        while True:
            try:
                status = self.monitor_all_symbols()

                # Verificar limites
                if status["portfolio_heat"] > 20.0:
                    self.logger.warning(f"Portfolio heat alto: {status['portfolio_heat']:.2f}%")

                if status["correlated_positions"] > 2:
                    self.logger.warning(f"Muitas posições correlacionadas: {status['correlated_positions']}")

                time.sleep(60)  # Verificar a cada minuto

            except KeyboardInterrupt:
                self.logger.info("Gerenciador multi-moedas interrompido")
                break
            except Exception as e:
                self.logger.error(f"Erro no gerenciador: {e}")
                time.sleep(30)

if __name__ == "__main__":
    manager = MultiCurrencyManager()
    manager.run_manager()
```

---

## 📊 Configuração para Backtest

### Configuração .env Backtest

```env
# ================================
# CONFIGURAÇÃO BACKTEST
# ================================

# Modo Backtest
BACKTEST_MODE=true
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2023-12-31
BACKTEST_SYMBOL=XAUUSD
BACKTEST_TIMEFRAME=M5

# Dados Históricos
HISTORICAL_DATA_SOURCE=MT5  # MT5, CSV, API
DATA_QUALITY=HIGH
FILL_GAPS=true
USE_TICK_DATA=true

# Configurações de Backtest
INITIAL_BALANCE=10000.0
BACKTEST_LOT_SIZE=0.01
BACKTEST_SPREAD=20  # 2 pips spread fixo
BACKTEST_COMMISSION=7  # $7 por lote padrão
BACKTEST_SWAP_SHORT=-2.5
BACKTEST_SWAP_LONG=-1.5

# Otimização
OPTIMIZATION_ENABLED=true
OPTIMIZATION_CRITERIA=MAX_PROFIT_FACTOR  # MAX_PROFIT, MAX_SHARPE, MIN_DRAWDOWN
OPTIMIZATION_GENETIC=true
OPTIMIZATION_POPULATION=100
OPTIMIZATION_GENERATIONS=50

# Análise de Resultados
GENERATE_DETAILED_REPORT=true
GENERATE_TRADE_LIST=true
GENERATE_EQUITY_CURVE=true
GENERATE_DRAWDOWN_CHART=true
GENERATE_MONTHLY_BREAKDOWN=true

# Validação
FORWARD_TEST_ENABLED=true
FORWARD_TEST_PERIOD=3  # 3 meses
MONTE_CARLO_SIMULATION=true
MONTE_CARLO_RUNS=1000

# Exportação
EXPORT_TO_EXCEL=true
EXPORT_TO_PDF=true
EXPORT_CHARTS=true
```

### Script de Backtest Automatizado

```python
#!/usr/bin/env python3
# automated_backtest.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from pathlib import Path

class AutomatedBacktest:
    def __init__(self):
        self.config = self.load_backtest_config()
        self.results = {}

    def load_backtest_config(self):
        """Carregar configuração de backtest"""
        return {
            "symbol": "XAUUSD",
            "timeframe": mt5.TIMEFRAME_M5,
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 12, 31),
            "initial_balance": 10000.0,
            "lot_size": 0.01,
            "spread": 20,  # points
            "commission": 7  # per lot
        }

    def get_historical_data(self):
        """Obter dados históricos"""
        try:
            # Obter candles do período
            utc_from = self.config["start_date"]
            utc_to = self.config["end_date"]

            rates = mt5.copy_rates_range(
                self.config["symbol"],
                self.config["timeframe"],
                utc_from,
                utc_to
            )

            if rates is None or len(rates) == 0:
                raise ValueError("Não foi possível obter dados históricos")

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            self.logger.info(f"Obtidos {len(df)} candles para backtest")
            return df

        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos: {e}")
            return None

    def simulate_strategy(self, df):
        """Simular estratégia de trading"""
        # Inicializar variáveis
        balance = self.config["initial_balance"]
        equity = balance
        positions = []
        trades = []

        # Configuração da estratégia (simplificada)
        ma_period = 20
        rsi_period = 14

        # Calcular indicadores
        df['ma'] = df['close'].rolling(window=ma_period).mean()
        df['rsi'] = self.calculate_rsi(df['close'], rsi_period)

        for i in range(ma_period, len(df)):
            current_bar = df.iloc[i]

            # Verificar posições abertas
            open_positions = [pos for pos in positions if pos['close_time'] is None]

            # Sinais de entrada
            if len(open_positions) == 0:  # Sem posições abertas
                # Sinal de compra
                if (current_bar['close'] > current_bar['ma'] and
                    current_bar['rsi'] < 70 and
                    current_bar['volume'] > df['volume'].rolling(20).mean().iloc[i]):

                    trade = self.open_trade(
                        'BUY', current_bar, i, balance, equity
                    )
                    if trade:
                        positions.append(trade)

                # Sinal de venda
                elif (current_bar['close'] < current_bar['ma'] and
                      current_bar['rsi'] > 30 and
                      current_bar['volume'] > df['volume'].rolling(20).mean().iloc[i]):

                    trade = self.open_trade(
                        'SELL', current_bar, i, balance, equity
                    )
                    if trade:
                        positions.append(trade)

            # Verificar fechamento de posições
            for pos in open_positions:
                if self.should_close_position(pos, current_bar, i):
                    closed_trade = self.close_trade(pos, current_bar, i)
                    if closed_trade:
                        trades.append(closed_trade)
                        balance += closed_trade['profit']
                        equity = balance

        return trades

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def open_trade(self, direction, bar, bar_index, balance, equity):
        """Abrir posição simulada"""
        lot_size = self.config["lot_size"]
        spread_cost = self.config["spread"] * 0.1 * lot_size  # $0.1 per point
        commission = self.config["commission"] * lot_size

        trade = {
            'open_time': bar['time'],
            'open_price': bar['close'],
            'direction': direction,
            'lot_size': lot_size,
            'spread_cost': spread_cost,
            'commission': commission,
            'close_time': None,
            'close_price': None,
            'profit': 0.0
        }

        return trade

    def should_close_position(self, position, current_bar, bar_index):
        """Verificar se deve fechar posição"""
        # Simplificado: TP ou SL
        pip_value = 0.1  # $0.1 per pip for 0.01 lot
        points_diff = abs(current_bar['close'] - position['open_price']) * 10000

        if position['direction'] == 'BUY':
            profit_points = (current_bar['close'] - position['open_price']) * 10000
            if profit_points >= 40 or profit_points <= -20:  # TP 40pips, SL 20pips
                return True
        else:  # SELL
            profit_points = (position['open_price'] - current_bar['close']) * 10000
            if profit_points >= 40 or profit_points <= -20:
                return True

        return False

    def close_trade(self, position, current_bar, bar_index):
        """Fechar posição simulada"""
        position['close_time'] = current_bar['time']
        position['close_price'] = current_bar['close']

        # Calcular profit
        pip_value = 0.1  # $0.1 per pip for 0.01 lot
        if position['direction'] == 'BUY':
            points = (position['close_price'] - position['open_price']) * 10000
        else:
            points = (position['open_price'] - position['close_price']) * 10000

        position['profit'] = (points * pip_value) - position['spread_cost'] - position['commission']

        return position

    def analyze_results(self, trades):
        """Analisar resultados do backtest"""
        if not trades:
            return {"error": "Nenhum trade executado"}

        # Métricas básicas
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] < 0]

        win_rate = len(profitable_trades) / total_trades * 100
        total_profit = sum(t['profit'] for t in profitable_trades)
        total_loss = sum(t['profit'] for t in losing_trades)
        net_profit = sum(t['profit'] for t in trades)
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0

        # Métricas avançadas
        avg_win = total_profit / len(profitable_trades) if profitable_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        largest_win = max(t['profit'] for t in profitable_trades) if profitable_trades else 0
        largest_loss = min(t['profit'] for t in losing_trades) if losing_trades else 0

        # Drawdown
        balance_curve = [self.config["initial_balance"]]
        running_balance = self.config["initial_balance"]

        for trade in trades:
            running_balance += trade['profit']
            balance_curve.append(running_balance)

        peak = balance_curve[0]
        max_drawdown = 0
        current_dd = 0

        for balance in balance_curve:
            if balance > peak:
                peak = balance
                current_dd = 0
            else:
                current_dd = (peak - balance) / peak * 100
                max_drawdown = max(max_drawdown, current_dd)

        # Sharpe Ratio (simplificado)
        returns = [t['profit'] / self.config["initial_balance"] for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return != 0 else 0

        analysis = {
            "backtest_period": {
                "start": self.config["start_date"].isoformat(),
                "end": self.config["end_date"].isoformat(),
                "duration_days": (self.config["end_date"] - self.config["start_date"]).days
            },
            "performance_metrics": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "net_profit": net_profit,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "return_percentage": (net_profit / self.config["initial_balance"]) * 100
            },
            "trade_statistics": {
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "avg_trade": net_profit / total_trades if total_trades > 0 else 0
            },
            "monthly_breakdown": self.calculate_monthly_breakdown(trades)
        }

        return analysis

    def calculate_monthly_breakdown(self, trades):
        """Calcular分解 mensal"""
        monthly_data = {}

        for trade in trades:
            month = trade['open_time'].strftime('%Y-%m')
            if month not in monthly_data:
                monthly_data[month] = {
                    "trades": 0,
                    "profit": 0.0,
                    "wins": 0
                }

            monthly_data[month]["trades"] += 1
            monthly_data[month]["profit"] += trade['profit']
            if trade['profit'] > 0:
                monthly_data[month]["wins"] += 1

        # Calcular win rate mensal
        for month, data in monthly_data.items():
            data["win_rate"] = (data["wins"] / data["trades"]) * 100 if data["trades"] > 0 else 0

        return monthly_data

    def generate_report(self, analysis, trades):
        """Gerar relatório completo"""
        report_dir = Path('reports/backtest')
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Salvar análise em JSON
        with open(report_dir / f'backtest_analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=2)

        # Salvar trades em CSV
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv(report_dir / f'trades_{timestamp}.csv', index=False)

        # Gerar gráfico da equity curve
        self.plot_equity_curve(trades, report_dir / f'equity_curve_{timestamp}.png')

        # Gerar relatório HTML
        self.generate_html_report(analysis, trades, report_dir / f'backtest_report_{timestamp}.html')

        self.logger.info(f"Relatório gerado em {report_dir}")

    def plot_equity_curve(self, trades, filepath):
        """Plotar equity curve"""
        balance_curve = [self.config["initial_balance"]]
        running_balance = self.config["initial_balance"]

        for trade in trades:
            running_balance += trade['profit']
            balance_curve.append(running_balance)

        plt.figure(figsize=(12, 6))
        plt.plot(balance_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Balance ($)')
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()

    def generate_html_report(self, analysis, trades, filepath):
        """Gerar relatório HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - EA_SCALPER_XAUUSD</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report - EA_SCALPER_XAUUSD</h1>

            <h2>Performance Metrics</h2>
            <div class="metric">
                <strong>Total Trades:</strong> {analysis['performance_metrics']['total_trades']}
            </div>
            <div class="metric">
                <strong>Win Rate:</strong> {analysis['performance_metrics']['win_rate']:.2f}%
            </div>
            <div class="metric">
                <strong>Net Profit:</strong> ${analysis['performance_metrics']['net_profit']:.2f}
            </div>
            <div class="metric">
                <strong>Profit Factor:</strong> {analysis['performance_metrics']['profit_factor']:.2f}
            </div>
            <div class="metric">
                <strong>Max Drawdown:</strong> {analysis['performance_metrics']['max_drawdown']:.2f}%
            </div>

            <h2>Monthly Breakdown</h2>
            <table>
                <tr>
                    <th>Month</th>
                    <th>Trades</th>
                    <th>Profit</th>
                    <th>Win Rate</th>
                </tr>
        """

        for month, data in analysis['monthly_breakdown'].items():
            profit_class = "positive" if data['profit'] > 0 else "negative"
            html_content += f"""
                <tr>
                    <td>{month}</td>
                    <td>{data['trades']}</td>
                    <td class="{profit_class}">${data['profit']:.2f}</td>
                    <td>{data['win_rate']:.1f}%</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(filepath, 'w') as f:
            f.write(html_content)

    def run_backtest(self):
        """Executar backtest completo"""
        self.logger.info("Iniciando backtest automatizado")

        # Obter dados históricos
        df = self.get_historical_data()
        if df is None:
            return False

        # Simular estratégia
        trades = self.simulate_strategy(df)

        # Analisar resultados
        analysis = self.analyze_results(trades)

        # Gerar relatório
        self.generate_report(analysis, trades)

        self.logger.info(f"Backtest concluído: {len(trades)} trades, Profit: ${analysis['performance_metrics']['net_profit']:.2f}")

        return True

if __name__ == "__main__":
    # Inicializar MT5
    if not mt5.initialize():
        print("Falha ao inicializar MetaTrader 5")
        exit()

    try:
        backtest = AutomatedBacktest()
        backtest.run_backtest()
    finally:
        mt5.shutdown()
```

---

## 🏭 Configuração para Produção

### Configuração .env Produção

```env
# ================================
# CONFIGURAÇÃO PRODUÇÃO
# ================================

# Production Mode
PRODUCTION_MODE=true
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# High Availability
ENABLE_LOAD_BALANCING=true
ENABLE_FAILOVER=true
ENABLE_AUTO_RESTART=true
RESTART_ON_CRASH=true
MAX_RESTART_ATTEMPTS=5
RESTART_DELAY=30

# Monitoring & Alerting
MONITORING_ENABLED=true
ALERT_EMAIL_ENABLED=true
ALERT_SMS_ENABLED=false
ALERT_SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
CRITICAL_ALERT_THRESHOLD=5.0
WARNING_ALERT_THRESHOLD=3.0

# Database
DATABASE_TYPE=postgresql
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ea_production
DATABASE_USER=ea_user
DATABASE_PASSWORD=secure_password
DATABASE_SSL=true
DATABASE_POOL_SIZE=10

# Cache (Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1
REDIS_PASSWORD=redis_password
REDIS_CLUSTER_ENABLED=true
REDIS_CLUSTER_NODES=redis-node1:6379,redis-node2:6379,redis-node3:6379

# Security
ENABLE_API_AUTH=true
API_SECRET_KEY=your_super_secret_key_here
JWT_EXPIRATION=3600
RATE_LIMIT_PER_IP=1000
ENABLE_IP_WHITELIST=true
ALLOWED_IPS=127.0.0.1,192.168.1.0/24

# Performance
ENABLE_PROFILING=false
METRICS_ENABLED=true
METRICS_PORT=9090
ENABLE_GZIP=true
CACHE_STATIC_FILES=true
STATIC_FILE_CACHE_TTL=86400

# Backup
AUTO_BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
BACKUP_ENCRYPTION=true
BACKUP_ENCRYPTION_KEY=backup_encryption_key
CLOUD_BACKUP_ENABLED=true
CLOUD_PROVIDER=aws_s3
AWS_S3_BUCKET=ea-backups
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key

# Trading Security
MAX_POSITION_SIZE=1.0
MAX_DAILY_LOSS=1000.0
EMERGENCY_STOP_ENABLED=true
MANUAL_OVERRIDE_ENABLED=true
TWO_FACTOR_AUTH_REQUIRED=true
AUDIT_TRAIL_ENABLED=true
```

### Script de Produção

```python
#!/usr/bin/env python3
# production_server.py

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List

class ProductionServer:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.running = True

    def setup_logging(self):
        """Configurar logging de produção"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PROD_SERVER - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/ea_scalper/production.log'),
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    '/var/log/ea_scalper/production.log',
                    maxBytes=10485760,  # 10MB
                    backupCount=5
                )
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Carregar configuração de produção"""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        self.config = {
            "debug": os.getenv('DEBUG', 'false').lower() == 'true',
            "max_restart_attempts": int(os.getenv('MAX_RESTART_ATTEMPTS', '5')),
            "restart_delay": int(os.getenv('RESTART_DELAY', '30')),
            "emergency_stop_enabled": os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true',
            "monitoring_enabled": os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
        }

    async def start_services(self):
        """Iniciar todos os serviços"""
        services = [
            self.start_proxy_server,
            self.start_mcp_servers,
            self.start_monitoring,
            self.start_trading_engine,
            self.start_backup_service
        ]

        tasks = []
        for service in services:
            task = asyncio.create_task(service())
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Erro nos serviços: {e}")
            await self.emergency_stop()

    async def start_proxy_server(self):
        """Iniciar proxy server"""
        self.logger.info("Iniciando proxy server...")

        while self.running:
            try:
                # Lógica do proxy server
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Erro no proxy server: {e}")
                await self.restart_service("proxy_server")

    async def start_mcp_servers(self):
        """Iniciar MCP servers"""
        self.logger.info("Iniciando MCP servers...")

        mcp_servers = [
            "code_checker",
            "github",
            "metatrader5"
        ]

        for server in mcp_servers:
            try:
                # Iniciar cada MCP server
                await asyncio.create_task(self.run_mcp_server(server))
            except Exception as e:
                self.logger.error(f"Erro ao iniciar MCP server {server}: {e}")

    async def start_monitoring(self):
        """Iniciar monitoramento"""
        if not self.config["monitoring_enabled"]:
            return

        self.logger.info("Iniciando sistema de monitoramento...")

        while self.running:
            try:
                await self.check_system_health()
                await self.check_trading_status()
                await self.check_resource_usage()
                await asyncio.sleep(60)  # Verificar a cada minuto
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")

    async def start_trading_engine(self):
        """Iniciar engine de trading"""
        self.logger.info("Iniciando trading engine...")

        while self.running:
            try:
                # Lógica do trading engine
                await self.execute_trading_logic()
                await asyncio.sleep(5)  # Verificar a cada 5 segundos
            except Exception as e:
                self.logger.error(f"Erro no trading engine: {e}")
                await self.handle_trading_error(e)

    async def start_backup_service(self):
        """Iniciar serviço de backup"""
        self.logger.info("Iniciando serviço de backup...")

        while self.running:
            try:
                # Verificar se é hora de backup
                if self.is_backup_time():
                    await self.perform_backup()
                await asyncio.sleep(3600)  # Verificar a cada hora
            except Exception as e:
                self.logger.error(f"Erro no serviço de backup: {e}")

    async def check_system_health(self):
        """Verificar saúde do sistema"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "resources": {}
        }

        # Verificar serviços
        services = ["proxy", "mcp_servers", "trading_engine", "database", "redis"]
        for service in services:
            health_status["services"][service] = await self.ping_service(service)

        # Verificar recursos
        health_status["resources"] = await self.check_resources()

        # Verificar se algum serviço está crítico
        critical_services = [
            s for s, status in health_status["services"].items()
            if status == "critical"
        ]

        if critical_services:
            await self.send_alert(f"Serviços críticos detectados: {critical_services}")

        # Salvar status
        self.save_health_status(health_status)

    async def execute_trading_logic(self):
        """Executar lógica de trading"""
        # Verificar se deve trading
        if not self.should_trade():
            return

        # Obter sinais
        signals = await self.get_trading_signals()

        # Validar sinais
        validated_signals = await self.validate_signals(signals)

        # Executar trades
        for signal in validated_signals:
            await self.execute_trade(signal)

    async def emergency_stop(self):
        """Parada de emergência"""
        self.logger.critical("Iniciando parada de emergência!")

        # Parar todos os trades
        await self.stop_all_trading()

        # Fechar posições abertas
        await self.close_all_positions()

        # Notificar administradores
        await self.send_emergency_notification()

        # Salvar estado atual
        await self.save_system_state()

        self.running = False

    async def restart_service(self, service_name):
        """Reiniciar serviço específico"""
        self.logger.warning(f"Reiniciando serviço: {service_name}")

        # Implementar lógica de restart
        await asyncio.sleep(self.config["restart_delay"])

    def setup_signal_handlers(self):
        """Configurar handlers de sinal"""
        def signal_handler(signum, frame):
            self.logger.info(f"Sinal recebido: {signum}")
            asyncio.create_task(self.graceful_shutdown())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def graceful_shutdown(self):
        """Desligamento gracioso"""
        self.logger.info("Iniciando desligamento gracioso...")

        # Parar de aceitar novos trades
        await self.stop_new_trades()

        # Aguardar trades abertos fecharem
        await self.wait_for_open_trades()

        # Salvar estado final
        await self.save_final_state()

        # Desligar serviços
        self.running = False

        self.logger.info("Desligamento concluído")

    async def run(self):
        """Executar servidor de produção"""
        self.setup_signal_handlers()

        try:
            await self.start_services()
        except KeyboardInterrupt:
            await self.graceful_shutdown()
        except Exception as e:
            self.logger.error(f"Erro fatal: {e}")
            await self.emergency_stop()

if __name__ == "__main__":
    server = ProductionServer()
    asyncio.run(server.run())
```

---

## 📋 Como Usar os Exemplos

### 1. Escolha o perfil adequado
- **Iniciante**: Use configurações ultra-conservadoras
- **Intermediário**: Balanceie risco e retorno
- **Avançado**: Use estratégias mais sofisticadas
- **FTMO**: Siga regras estritas de compliance

### 2. Adapte as configurações
- Modifique valores conforme seu capital
- Ajuste horários para seu timezone
- Personalize parâmetros de risco

### 3. Teste antes de usar
- Sempre faça backtest primeiro
- Use conta demo inicialmente
- Monitore resultados por várias semanas

### 4. Mantenha backups
- Salve configurações antigas
- Documente mudanças
- Tenha plano de rollback

Estes exemplos servem como ponto de partida. Adapte-os conforme suas necessidades específicas e tolerância a risco.