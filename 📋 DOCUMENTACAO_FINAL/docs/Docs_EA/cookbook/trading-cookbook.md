# Trading Cookbook - EA_SCALPER_XAUUSD
=====================================

Coleção de receitas práticas e padrões de uso para o sistema EA_SCALPER_XAUUSD.

## Índice

1. [Receitas Básicas](#receitas-básicas)
2. [Estratégias de Trading](#estratégias-de-trading)
3. [Gestão de Risco](#gestão-de-risco)
4. **Integração com IA** ([continuação...](#integração-com-ia))
5. **Backtesting e Otimização** ([continuação...](#backtesting-e-otimização))
6. **Monitoramento e Alertas** ([continuação...](#monitoramento-e-alertas))
7. **Soluções de Problemas** ([continuação...](#soluções-de-problemas))

---

## Receitas Básicas

### 🎯 Setup Rápido de Conexão MT5

**Problema:** Precisa conectar rapidamente ao MetaTrader 5.

```python
import asyncio
from ea_scalper_sdk import MT5Client
from dotenv import load_dotenv

async def quick_mt5_setup():
    """Setup rápido e confiável MT5"""

    # Carregar configuração
    load_dotenv()

    # Criar cliente
    client = MT5Client()

    # Conectar com retry
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            success = await client.connect(
                login=int(os.getenv('MT5_LOGIN')),
                password=os.getenv('MT5_PASSWORD'),
                server=os.getenv('MT5_SERVER')
            )

            if success:
                print("✅ Conectado ao MT5!")

                # Verificar informações básicas
                account = await client.get_account_info()
                print(f"📊 Saldo: ${account['balance']:.2f}")

                # Verificar XAUUSD
                symbol = await client.get_symbol_info("XAUUSD")
                if symbol:
                    print(f"💎 XAUUSD: Spread {symbol['spread']} pts")

                return client

        except Exception as e:
            print(f"❌ Tentativa {attempt + 1} falhou: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2)

    raise Exception("Não foi possível conectar ao MT5")

# Uso
client = asyncio.run(quick_mt5_setup())
```

### 📊 Coleta de Dados Multi-Timeframe

**Problema:** Precisa coletar dados de múltiplos timeframes de forma eficiente.

```python
async def collect_multi_timeframe_data(symbol="XAUUSD"):
    """Coleta dados de múltiplos timeframes"""

    timeframes = {
        "M5": 500,    # Últimas 500 barras de 5 minutos
        "M15": 300,   # Últimas 300 barras de 15 minutos
        "H1": 200,    # Últimas 200 barras de 1 hora
        "H4": 100,    # Últimas 100 barras de 4 horas
        "D1": 50      # Últimas 50 barras diárias
    }

    data = {}

    for tf, count in timeframes.items():
        try:
            bars = await client.get_bars(symbol, tf, count)
            if bars:
                data[tf] = {
                    'bars': bars,
                    'count': len(bars),
                    'latest': bars[-1],
                    'price_change': (bars[-1]['close'] - bars[0]['close']) / bars[0]['close'] * 100
                }
                print(f"✅ {tf}: {len(bars)} barras")
            else:
                print(f"❌ {tf}: Sem dados")

        except Exception as e:
            print(f"❌ {tf}: Erro - {e}")

    return data

# Uso
market_data = asyncio.run(collect_multi_timeframe_data())
```

### ⚡ Monitoramento de Ticks em Tempo Real

**Problema:** Precisa monitorar ticks em tempo real para alta frequência.

```python
import asyncio
from collections import deque

class TickMonitor:
    """Monitor de ticks em tempo real"""

    def __init__(self, symbol="XAUUSD", max_ticks=100):
        self.symbol = symbol
        self.max_ticks = max_ticks
        self.ticks = deque(maxlen=max_ticks)
        self.is_running = False

    async def start_monitoring(self, interval=1):
        """Inicia monitoramento de ticks"""

        self.is_running = True
        print(f"🔄 Monitorando {self.symbol}...")

        while self.is_running:
            try:
                # Obter ticks recentes
                new_ticks = await client.get_ticks(self.symbol, 10)

                for tick in new_ticks:
                    # Adicionar apenas ticks novos
                    if not self.ticks or tick['time'] > self.ticks[-1]['time']:
                        self.ticks.append(tick)

                        # Detectar mudanças significativas
                        if len(self.ticks) >= 2:
                            price_change = tick['bid'] - self.ticks[-2]['bid']
                            if abs(price_change) > 0.5:  # Mudança > 50 cents
                                direction = "📈" if price_change > 0 else "📉"
                                print(f"{direction} {self.symbol}: {tick['bid']:.2f} (Δ{price_change:+.2f})")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"❌ Erro no monitoramento: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Para monitoramento"""
        self.is_running = False

    def get_latest_price(self):
        """Obtém preço mais recente"""
        return self.ticks[-1] if self.ticks else None

    def calculate_spread(self):
        """Calcula spread atual"""
        if self.ticks:
            tick = self.ticks[-1]
            return tick['ask'] - tick['bid']
        return 0

# Uso
monitor = TickMonitor()

# Iniciar monitoramento em background
monitor_task = asyncio.create_task(monitor.start_monitoring())

# Deixar rodar por 30 segundos
await asyncio.sleep(30)

# Parar monitoramento
monitor.stop()
await monitor_task
```

---

## Estratégias de Trading

### 🎯 Estratégia de Scalping Baseada em RSI

**Problema:** Implementar estratégia de scalping simples e eficaz.

```python
import asyncio
import talib

class RSIScalpingStrategy:
    """Estratégia de scalping baseada em RSI"""

    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_open = False

    async def analyze_and_trade(self):
        """Analisa e executa trades baseados em RSI"""

        # Obter barras recentes
        bars = await client.get_bars("XAUUSD", "M5", 100)

        if len(bars) < self.rsi_period:
            return

        # Calcular RSI
        closes = [bar['close'] for bar in bars]
        rsi = talib.RSI(np.array(closes), timeperiod=self.rsi_period)
        current_rsi = rsi[-1]
        current_price = bars[-1]['close']

        print(f"📊 RSI: {current_rsi:.1f} | Preço: ${current_price:.2f}")

        # Verificar posições abertas
        positions = await client.get_positions("XAUUSD")

        # Lógica de trading
        if not positions:
            if current_rsi < self.oversold and not self.position_open:
                # Sinal de compra
                await self.place_buy_order(current_price)
                self.position_open = True

            elif current_rsi > self.overbought and not self.position_open:
                # Sinal de venda
                await self.place_sell_order(current_price)
                self.position_open = True

        else:
            # Verificar Take Profit ou Stop Loss
            for position in positions:
                await self.manage_position(position, current_price)

    async def place_buy_order(self, current_price):
        """Coloca ordem de compra"""

        # Calcular SL/TP
        atr = await self.calculate_atr()
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 2.0)

        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_BUY",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "magic_number": 12345,
            "comment": "RSI Scalper Buy"
        }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Compra executada: ${result['execution_price']:.2f}")
            print(f"🛡️ SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")

    async def place_sell_order(self, current_price):
        """Coloca ordem de venda"""

        # Calcular SL/TP
        atr = await self.calculate_atr()
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - (atr * 2.0)

        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": "MARKET_SELL",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "magic_number": 12345,
            "comment": "RSI Scalper Sell"
        }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Venda executada: ${result['execution_price']:.2f}")
            print(f"🛡️ SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")

    async def manage_position(self, position, current_price):
        """Gerencia posição aberta"""

        profit = position['profit']

        # Trailing stop
        if profit > 30:  # Se lucro > $30
            new_sl = position['open_price'] + 20 if position['type'] == 'BUY' else position['open_price'] - 20

            if position['type'] == 'BUY' and new_sl > position['stop_loss']:
                await client.modify_position(position['ticket'], stop_loss=new_sl)
                print(f"📏 Trailing stop ajustado para ${new_sl:.2f}")

            elif position['type'] == 'SELL' and new_sl < position['stop_loss']:
                await client.modify_position(position['ticket'], stop_loss=new_sl)
                print(f"📏 Trailing stop ajustado para ${new_sl:.2f}")

        # Fechar se reached TP ou SL
        if profit >= 100 or profit <= -50:
            await client.close_position(position['ticket'])
            print(f"🔄 Posição {position['ticket']} fechada - Lucro: ${profit:.2f}")
            self.position_open = False

    async def calculate_atr(self, period=14):
        """Calcula ATR para determinação de SL/TP"""

        bars = await client.get_bars("XAUUSD", "H1", period + 1)

        if len(bars) < period + 1:
            return 2.0  # Default

        # Calcular True Ranges
        trs = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)

        # ATR é média dos True Ranges
        atr = sum(trs) / len(trs)
        return atr

# Uso
strategy = RSIScalpingStrategy(rsi_period=14, oversold=25, overbought=75)

# Executar estratégia continuamente
async def run_strategy():
    while True:
        await strategy.analyze_and_trade()
        await asyncio.sleep(60)  # Verificar a cada minuto

# asyncio.run(run_strategy())
```

### 📈 Estratégia de Trend Following com EMAs

**Problema:** Implementar estratégia que segue tendências usando médias móveis.

```python
class EMATrendStrategy:
    """Estratégia de seguimento de tendência com EMAs"""

    def __init__(self, ema_fast=9, ema_slow=21, ema_trend=50):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.trend_direction = None

    async def analyze_trend(self):
        """Analisa direção da tendência principal"""

        bars = await client.get_bars("XAUUSD", "H4", 100)

        if len(bars) < self.ema_trend:
            return None

        closes = [bar['close'] for bar in bars]

        # Calcular EMAs
        ema_fast = talib.EMA(np.array(closes), timeperiod=self.ema_fast)
        ema_slow = talib.EMA(np.array(closes), timeperiod=self.ema_slow)
        ema_trend = talib.EMA(np.array(closes), timeperiod=self.ema_trend)

        current_fast = ema_fast[-1]
        current_slow = ema_slow[-1]
        current_trend = ema_trend[-1]
        current_price = closes[-1]

        # Determinar tendência
        if current_fast > current_slow > current_trend:
            trend = "bullish"
        elif current_fast < current_slow < current_trend:
            trend = "bearish"
        else:
            trend = "neutral"

        # Detectar mudança de tendência
        if self.trend_direction != trend:
            self.trend_direction = trend
            print(f"📈 Mudança de tendência detectada: {trend.upper()}")

        return {
            'trend': trend,
            'price': current_price,
            'ema_fast': current_fast,
            'ema_slow': current_slow,
            'ema_trend': current_trend,
            'fast_above_slow': current_fast > current_slow,
            'price_above_fast': current_price > current_fast
        }

    async def find_entry_point(self, trend_analysis):
        """Encontra ponto de entrada baseado na tendência"""

        if trend_analysis['trend'] == "neutral":
            return None

        # Para tendência de alta, esperar pullback
        if trend_analysis['trend'] == "bullish":
            # Verificar se preço está abaixo da EMA rápida (pullback)
            if not trend_analysis['price_above_fast']:
                # Verificar se está começando a subir
                m15_bars = await client.get_bars("XAUUSD", "M15", 10)

                if len(m15_bars) >= 3:
                    recent_closes = [bar['close'] for bar in m15_bars[-3:]]

                    # Se últimas 2 barras estão subindo
                    if recent_closes[-1] > recent_closes[-2] > recent_closes[-3]:
                        return {
                            'action': 'BUY',
                            'entry_price': trend_analysis['price'],
                            'confidence': 75,
                            'reasoning': 'Tendência de alta com pullback para EMA rápida'
                        }

        # Para tendência de baixa, esperar rally
        elif trend_analysis['trend'] == "bearish":
            # Verificar se preço está acima da EMA rápida (rally)
            if trend_analysis['price_above_fast']:
                # Verificar se está começando a cair
                m15_bars = await client.get_bars("XAUUSD", "M15", 10)

                if len(m15_bars) >= 3:
                    recent_closes = [bar['close'] for bar in m15_bars[-3:]]

                    # Se últimas 2 barras estão caindo
                    if recent_closes[-1] < recent_closes[-2] < recent_closes[-3]:
                        return {
                            'action': 'SELL',
                            'entry_price': trend_analysis['price'],
                            'confidence': 75,
                            'reasoning': 'Tendência de baixa com rally para EMA rápida'
                        }

        return None

    async def execute_trend_trade(self, signal):
        """Executa trade baseado no sinal de tendência"""

        if not signal:
            return

        # Calcular SL/TP baseado no ATR
        atr = await self.calculate_atr()

        if signal['action'] == 'BUY':
            stop_loss = signal['entry_price'] - (atr * 2.0)
            take_profit = signal['entry_price'] + (atr * 4.0)  # RR 1:2

            order_data = {
                "symbol": "XAUUSD",
                "volume": 0.02,
                "order_type": "LIMIT_BUY",
                "price": signal['entry_price'],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "magic_number": 54321,
                "comment": "EMA Trend Buy"
            }

        else:  # SELL
            stop_loss = signal['entry_price'] + (atr * 2.0)
            take_profit = signal['entry_price'] - (atr * 4.0)

            order_data = {
                "symbol": "XAUUSD",
                "volume": 0.02,
                "order_type": "LIMIT_SELL",
                "price": signal['entry_price'],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "magic_number": 54321,
                "comment": "EMA Trend Sell"
            }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Ordem de {signal['action']} colocada")
            print(f"💰 Entry: ${signal['entry_price']:.2f}")
            print(f"🛡️ SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")
            print(f"📊 Confiança: {signal['confidence']}%")
            print(f"💡 {signal['reasoning']}")

    async def run_trend_strategy(self):
        """Executa estratégia completa de tendência"""

        while True:
            try:
                # Analisar tendência
                trend_analysis = await self.analyze_trend()

                if trend_analysis:
                    print(f"📊 Tendência: {trend_analysis['trend'].upper()}")
                    print(f"💰 Preço: ${trend_analysis['price']:.2f}")

                    # Encontrar ponto de entrada
                    signal = await self.find_entry_point(trend_analysis)

                    if signal:
                        await self.execute_trend_trade(signal)
                    else:
                        print("⏸️ Aguardando ponto de entrada...")

                await asyncio.sleep(300)  # Verificar a cada 5 minutos

            except Exception as e:
                print(f"❌ Erro na estratégia: {e}")
                await asyncio.sleep(60)

# Uso
strategy = EMATrendStrategy(ema_fast=9, ema_slow=21, ema_trend=50)
# asyncio.run(strategy.run_trend_strategy())
```

### 🔄 Estratégia de Mean Reversion

**Problema:** Implementar estratégia que explora reversões para a média.

```python
class MeanReversionStrategy:
    """Estratégia de reversão para a média"""

    def __init__(self, lookback_period=20, std_multiplier=2.0):
        self.lookback_period = lookback_period
        self.std_multiplier = std_multiplier

    async def calculate_bands(self):
        """Calcula bandas de Bollinger para reversão"""

        bars = await client.get_bars("XAUUSD", "M15", self.lookback_period + 10)

        if len(bars) < self.lookback_period:
            return None

        closes = [bar['close'] for bar in bars]

        # Calcular banda média e desvio padrão
        recent_closes = closes[-self.lookback_period:]
        mean = sum(recent_closes) / len(recent_closes)

        # Calcular desvio padrão
        variance = sum((price - mean) ** 2 for price in recent_closes) / len(recent_closes)
        std = variance ** 0.5

        # Bandas
        upper_band = mean + (std * self.std_multiplier)
        lower_band = mean - (std * self.std_multiplier)
        current_price = closes[-1]

        # Posição do preço em relação às bandas (0-1)
        if upper_band != lower_band:
            position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            position = 0.5

        return {
            'current_price': current_price,
            'mean': mean,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'std': std,
            'position': position,  # 0=lower_band, 1=upper_band
            'distance_from_mean': abs(current_price - mean) / std
        }

    async def check_reversal_signal(self, bands):
        """Verifica sinal de reversão"""

        if not bands:
            return None

        current_price = bands['current_price']
        position = bands['position']
        distance_from_mean = bands['distance_from_mean']

        # Sinal de sobrecompra (possível reversão para baixo)
        if position > 0.9 and distance_from_mean > 2.0:
            # Verificar se está começando a reverter
            bars = await client.get_bars("XAUUSD", "M5", 5)

            if len(bars) >= 3:
                recent_closes = [bar['close'] for bar in bars[-3:]]

                # Se está fazendo topo e começando a cair
                if (recent_closes[-2] > recent_closes[-3] and
                    recent_closes[-1] < recent_closes[-2]):

                    return {
                        'action': 'SELL',
                        'entry_price': current_price,
                        'confidence': min(90, 60 + distance_from_mean * 10),
                        'reasoning': f'Sobrecompra extrema (posição: {position:.2f})'
                    }

        # Sinal de sobrevenda (possível reversão para cima)
        elif position < 0.1 and distance_from_mean > 2.0:
            # Verificar se está começando a reverter
            bars = await client.get_bars("XAUUSD", "M5", 5)

            if len(bars) >= 3:
                recent_closes = [bar['close'] for bar in bars[-3:]]

                # Se está fazendo fundo e começando a subir
                if (recent_closes[-2] < recent_closes[-3] and
                    recent_closes[-1] > recent_closes[-2]):

                    return {
                        'action': 'BUY',
                        'entry_price': current_price,
                        'confidence': min(90, 60 + distance_from_mean * 10),
                        'reasoning': f'Sobrevenda extrema (posição: {position:.2f})'
                    }

        return None

    async def execute_reversal_trade(self, signal, bands):
        """Executa trade de reversão"""

        # SL e TP baseados nas bandas
        if signal['action'] == 'BUY':
            stop_loss = bands['lower_band'] * 0.995  # Um pouco abaixo da banda inferior
            take_profit = bands['mean']  # Voltar para a média

        else:  # SELL
            stop_loss = bands['upper_band'] * 1.005  # Um pouco acima da banda superior
            take_profit = bands['mean']  # Voltar para a média

        # Calcular risk/reward
        risk = abs(signal['entry_price'] - stop_loss)
        reward = abs(take_profit - signal['entry_price'])
        rr_ratio = reward / risk if risk > 0 else 0

        # Apenas executar se RR for razoável
        if rr_ratio < 1.0:
            print(f"⚠️ Risk/Reward muito baixo: 1:{rr_ratio:.2f}")
            return

        order_data = {
            "symbol": "XAUUSD",
            "volume": 0.01,
            "order_type": f"MARKET_{signal['action']}",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "magic_number": 98765,
            "comment": f"Mean Reversion {signal['action']}"
        }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Trade de reversão executado: {signal['action']}")
            print(f"💰 Entry: ${signal['entry_price']:.2f}")
            print(f"🛡️ SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")
            print(f"📊 R/R: 1:{rr_ratio:.2f} | Confiança: {signal['confidence']}%")
            print(f"💡 {signal['reasoning']}")

    async def run_mean_reversion(self):
        """Executa estratégia de reversão à média"""

        while True:
            try:
                # Calcular bandas
                bands = await self.calculate_bands()

                if bands:
                    print(f"📊 Preço: ${bands['current_price']:.2f}")
                    print(f"📈 Posição: {bands['position']:.2f} (0=banda inferior, 1=banda superior)")
                    print(f"📏 Distância da média: {bands['distance_from_mean']:.2f} std")

                    # Verificar sinal de reversão
                    signal = await self.check_reversal_signal(bands)

                    if signal:
                        await self.execute_reversal_trade(signal, bands)
                    else:
                        print("⏸️ Aguardando sinal de reversão...")

                await asyncio.sleep(60)  # Verificar a cada minuto

            except Exception as e:
                print(f"❌ Erro na estratégia: {e}")
                await asyncio.sleep(30)

# Uso
strategy = MeanReversionStrategy(lookback_period=20, std_multiplier=2.0)
# asyncio.run(strategy.run_mean_reversion())
```

---

## Gestão de Risco

### 🛡️ Gestor de Risco FTMO-Compliant

**Problema:** Implementar gestão de risco rigorosa conforme regras FTMO.

```python
from datetime import datetime, time as dt_time

class FTMORiskManager:
    """Gestor de risco FTMO-compliant"""

    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.max_daily_loss = 0.05  # 5%
        self.max_total_loss = 0.10  # 10%
        self.daily_start_balance = initial_balance
        self.last_reset_date = datetime.now().date()

    def reset_daily(self):
        """Reseta controles diários"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_start_balance = self.current_balance
            self.last_reset_date = today
            print(f"📅 Controles diários resetados - Saldo inicial: ${self.daily_start_balance:.2f}")

    async def check_daily_loss(self, current_balance):
        """Verifica perda diária"""

        self.reset_daily()

        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance

        if daily_loss >= self.max_daily_loss:
            print(f"🚨 ALERTA: Perda diária de {daily_loss*100:.1f}% excede limite!")
            print("🛑 TRADING SUSPENSO - Aguardar próximo dia")
            return False

        return True

    async def check_total_loss(self, current_balance):
        """Verifica perda total"""

        total_loss = (self.initial_balance - current_balance) / self.initial_balance

        if total_loss >= self.max_total_loss:
            print(f"🚨 ALERTA: Perda total de {total_loss*100:.1f}% excede limite!")
            print("🛑 TRADING SUSPENSO - Limite máximo atingido")
            return False

        return True

    async def calculate_position_size(self, stop_loss_pips, account_balance):
        """Calcula tamanho da posição baseado no risco"""

        # Risco de 1% do capital
        risk_amount = account_balance * 0.01

        # Obter valor do pip para XAUUSD
        symbol_info = await client.get_symbol_info("XAUUSD")
        pip_value = symbol_info.get('trade_tick_value', 10)

        # Calcular tamanho da posição
        position_size = risk_amount / (stop_loss_pips * pip_value)

        # Limites mínimo e máximo
        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 1.0)

        position_size = max(min_lot, min(position_size, max_lot))

        return round(position_size, 2)

    async def validate_trade(self, account_balance, positions_count, new_trade_risk):
        """Valida se novo trade é permitido"""

        # Verificar perdas
        if not await self.check_daily_loss(account_balance):
            return False, "Perda diária excessiva"

        if not await self.check_total_loss(account_balance):
            return False, "Perda total excessiva"

        # Verificar número máximo de posições
        if positions_count >= 2:
            return False, "Número máximo de posições atingido"

        # Verificar risco total em posições abertas
        total_risk_percent = (new_trade_risk / account_balance) * 100
        if total_risk_percent > 2.0:  # Máximo 2% por trade
            return False, f"Risco muito alto: {total_risk_percent:.1f}%"

        return True, "Trade permitido"

    async def get_risk_metrics(self, account_balance, positions):
        """Obtém métricas de risco atuais"""

        self.reset_daily()

        daily_loss = (self.daily_start_balance - account_balance) / self.daily_start_balance
        total_loss = (self.initial_balance - account_balance) / self.initial_balance

        # Calcular risco total em posições abertas
        total_exposure = 0
        for position in positions:
            sl_distance = abs(position['open_price'] - position['stop_loss'])
            position_risk = position['volume'] * sl_distance * 100  # Simplificado
            total_exposure += position_risk

        exposure_percent = (total_exposure / account_balance) * 100

        return {
            'daily_loss_percent': daily_loss * 100,
            'total_loss_percent': total_loss * 100,
            'daily_loss_limit': self.max_daily_loss * 100,
            'total_loss_limit': self.max_total_loss * 100,
            'total_exposure_percent': exposure_percent,
            'max_positions': 2,
            'current_positions': len(positions),
            'risk_status': 'OK' if daily_loss < 0.8 * self.max_daily_loss else 'WARNING'
        }

# Uso
risk_manager = FTMORiskManager(initial_balance=10000)

async def check_and_place_trade():
    """Verifica risco e coloca ordem se seguro"""

    account_info = await client.get_account_info()
    current_balance = account_info['balance']

    positions = await client.get_positions("XAUUSD")

    # Exemplo: calcular risco para novo trade
    stop_loss_pips = 50  # 50 pips
    new_trade_risk = current_balance * 0.01  # 1% do saldo

    # Validar trade
    can_trade, reason = await risk_manager.validate_trade(
        current_balance, len(positions), new_trade_risk
    )

    if can_trade:
        print("✅ Trade aprovado pelo gestor de risco")

        # Calcular tamanho da posição
        position_size = await risk_manager.calculate_position_size(stop_loss_pips, current_balance)
        print(f"📊 Tamanho da posição: {position_size} lotes")

        # Colocar ordem...

    else:
        print(f"❌ Trade bloqueado: {reason}")

    # Exibir métricas de risco
    risk_metrics = await risk_manager.get_risk_metrics(current_balance, positions)
    print(f"📊 Perda diária: {risk_metrics['daily_loss_percent']:.2f}% (limite: {risk_metrics['daily_loss_limit']:.1f}%)")
    print(f"📊 Perda total: {risk_metrics['total_loss_percent']:.2f}% (limite: {risk_metrics['total_loss_limit']:.1f}%)")
    print(f"📊 Exposição: {risk_metrics['total_exposure_percent']:.2f}%")
    print(f"📊 Status: {risk_metrics['risk_status']}")

# asyncio.run(check_and_place_trade())
```

### 📊 Monitor de Drawdown em Tempo Real

**Problema:** Monitorar drawdown em tempo real e tomar ações preventivas.

```python
class DrawdownMonitor:
    """Monitor de drawdown em tempo real"""

    def __init__(self, initial_balance=10000, warning_threshold=0.03, stop_threshold=0.05):
        self.initial_balance = initial_balance
        self.warning_threshold = warning_threshold  # 3%
        self.stop_threshold = stop_threshold  # 5%
        self.peak_balance = initial_balance
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.drawdown_history = []

    def update_balance(self, current_balance):
        """Atualiza balanço e calcula drawdown"""

        # Atualizar pico se necessário
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            print(f"📈 Novo pico de equity: ${self.peak_balance:.2f}")

        # Calcular drawdown atual
        self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

        # Atualizar drawdown máximo
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

        # Adicionar ao histórico
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'balance': current_balance,
            'drawdown': self.current_drawdown
        })

        # Manter apenas últimas 1000 entradas
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]

        return self.current_drawdown

    def check_alerts(self):
        """Verifica se há alertas de drawdown"""

        if self.current_drawdown >= self.stop_threshold:
            return {
                'level': 'CRITICAL',
                'message': f'Drawdown crítico: {self.current_drawdown*100:.1f}% - TRADING PARADO',
                'action': 'STOP_TRADING'
            }

        elif self.current_drawdown >= self.warning_threshold:
            return {
                'level': 'WARNING',
                'message': f'Drawdown elevado: {self.current_drawdown*100:.1f}% - Reduzir risco',
                'action': 'REDUCE_RISK'
            }

        return {
            'level': 'OK',
            'message': f'Drawdown normal: {self.current_drawdown*100:.1f}%',
            'action': 'CONTINUE'
        }

    def get_drawdown_stats(self):
        """Obtém estatísticas detalhadas de drawdown"""

        if not self.drawdown_history:
            return {}

        # Calcular tempo de recuperação médio
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None

        for entry in self.drawdown_history:
            if entry['drawdown'] > 0.001 and not in_drawdown:
                in_drawdown = True
                drawdown_start = entry['timestamp']
            elif entry['drawdown'] < 0.001 and in_drawdown:
                in_drawdown = False
                if drawdown_start:
                    recovery_time = entry['timestamp'] - drawdown_start
                    drawdown_periods.append(recovery_time.total_seconds() / 3600)  # Horas

        avg_recovery_hours = sum(drawdown_periods) / len(drawdown_periods) if drawdown_periods else 0

        return {
            'current_drawdown': self.current_drawdown * 100,
            'max_drawdown': self.max_drawdown * 100,
            'peak_balance': self.peak_balance,
            'avg_recovery_hours': avg_recovery_hours,
            'total_drawdown_periods': len(drawdown_periods),
            'current_status': self.check_alerts()['level']
        }

    async def start_monitoring(self, check_interval=30):
        """Inicia monitoramento contínuo"""

        print("🔄 Iniciando monitoramento de drawdown...")
        print(f"⚠️ Alerta em: {self.warning_threshold*100:.1f}%")
        print(f"🛑 Parada em: {self.stop_threshold*100:.1f}%")

        while True:
            try:
                # Obter balanço atual
                account_info = await client.get_account_info()
                current_balance = account_info['balance']

                # Atualizar drawdown
                dd = self.update_balance(current_balance)

                # Verificar alertas
                alert = self.check_alerts()

                if alert['level'] == 'CRITICAL':
                    print(f"🚨 {alert['message']}")
                    # Aqui você implementaria a parada do trading
                    break

                elif alert['level'] == 'WARNING':
                    print(f"⚠️ {alert['message']}")

                else:
                    print(f"✅ {alert['message']}")

                await asyncio.sleep(check_interval)

            except Exception as e:
                print(f"❌ Erro no monitoramento: {e}")
                await asyncio.sleep(60)

# Uso
monitor = DrawdownMonitor(initial_balance=10000, warning_threshold=0.03, stop_threshold=0.05)

# Iniciar monitoramento em background
# monitor_task = asyncio.create_task(monitor.start_monitoring())
```

### 🎯 Calculador de Posição Dinâmico

**Problema:** Calcular tamanho da posição dinamicamente baseado em múltiplos fatores.

```python
class DynamicPositionSizer:
    """Calculador de tamanho de posição dinâmico"""

    def __init__(self, base_risk=0.01, max_risk=0.02):
        self.base_risk = base_risk  # 1% base
        self.max_risk = max_risk    # 2% máximo
        self.volatility_multiplier = 1.0
        self.confidence_multiplier = 1.0

    async def calculate_optimal_size(self, account_info, signal_strength, market_volatility):
        """Calcula tamanho ótimo da posição"""

        balance = account_info['balance']
        equity = account_info['equity']

        # Ajustar risco baseado na força do sinal
        risk_multiplier = self.calculate_risk_multiplier(signal_strength, market_volatility)

        # Calcular risco em dinheiro
        risk_percent = min(self.base_risk * risk_multiplier, self.max_risk)
        risk_amount = balance * risk_percent

        # Ajustar baseado no drawdown atual
        current_drawdown = (balance - equity) / balance if equity < balance else 0
        if current_drawdown > 0.02:  # Se drawdown > 2%
            risk_amount *= (1 - current_drawdown * 10)  # Reduzir risco

        # Obter informações do símbolo
        symbol_info = await client.get_symbol_info("XAUUSD")
        pip_value = symbol_info.get('trade_tick_value', 10)

        # Calcular tamanho da posição para risco de 100 pips
        position_size = risk_amount / (100 * pip_value)

        # Aplicar limites
        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 1.0)

        position_size = max(min_lot, min(position_size, max_lot))

        return {
            'volume': round(position_size, 2),
            'risk_percent': risk_percent * 100,
            'risk_amount': risk_amount,
            'multiplier': risk_multiplier,
            'reasoning': self.get_position_reasoning(signal_strength, market_volatility, risk_multiplier)
        }

    def calculate_risk_multiplier(self, signal_strength, market_volatility):
        """Calcula multiplicador de risco baseado em condições"""

        multiplier = 1.0

        # Ajustar baseado na força do sinal (0-100)
        if signal_strength > 80:
            multiplier *= 1.3  # Aumentar risco para sinais muito fortes
        elif signal_strength > 60:
            multiplier *= 1.1  # Aumentar risco moderadamente
        elif signal_strength < 40:
            multiplier *= 0.7  # Reduzir risco para sinais fracos

        # Ajustar baseado na volatilidade
        if market_volatility > 30:  # Muito volátil
            multiplier *= 0.8  # Reduzir risco
        elif market_volatility < 10:  # Pouca volatilidade
            multiplier *= 0.9  # Reduzir ligeiramente

        # Ajustar baseado na hora do dia
        current_hour = datetime.now().hour
        if 13 <= current_hour <= 17:  # Sessão NY (mais volátil)
            multiplier *= 0.9
        elif 8 <= current_hour <= 12:  # Sessão Londres
            multiplier *= 1.0  # Normal
        else:
            multiplier *= 0.8  # Outras sessões

        return max(0.5, min(1.5, multiplier))  # Limitar entre 0.5x e 1.5x

    def get_position_reasoning(self, signal_strength, market_volatility, multiplier):
        """Gera explicação para o tamanho da posição"""

        reasons = []

        if signal_strength > 80:
            reasons.append("Sinal muito forte")
        elif signal_strength < 40:
            reasons.append("Sinal fraco")

        if market_volatility > 30:
            reasons.append("Alta volatilidade")
        elif market_volatility < 10:
            reasons.append("Baixa volatilidade")

        if multiplier > 1.2:
            reasons.append("Risco aumentado")
        elif multiplier < 0.8:
            reasons.append("Risco reduzido")

        return " | ".join(reasons) if reasons else "Condições normais"

# Uso
position_sizer = DynamicPositionSizer(base_risk=0.01, max_risk=0.02)

async def calculate_trade_size():
    """Exemplo de uso do calculador dinâmico"""

    # Obter informações da conta
    account_info = await client.get_account_info()

    # Simular força do sinal e volatilidade
    signal_strength = 75  # 0-100
    market_volatility = 25  # Percentual

    # Calcular tamanho ótimo
    result = await position_sizer.calculate_optimal_size(
        account_info, signal_strength, market_volatility
    )

    print(f"📊 Tamanho calculado: {result['volume']} lotes")
    print(f"💰 Risco: {result['risk_percent']:.1f}% (${result['risk_amount']:.2f})")
    print(f"📈 Multiplicador: {result['multiplier']:.2f}x")
    print(f"💡 {result['reasoning']}")

# asyncio.run(calculate_trade_size())
```

---

## Integração com IA

### 🤖 Estratégia com Análise de IA em Tempo Real

**Problema:** Integrar análise de IA para tomada de decisões em tempo real.

```python
import json
from datetime import datetime, timedelta

class AIEnhancedTradingStrategy:
    """Estratégia de trading com análise de IA em tempo real"""

    def __init__(self):
        self.llm_client = LLMClient(base_url="http://localhost:4000")
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 minutos

    async def get_ai_market_analysis(self, market_data):
        """Obtém análise de mercado da IA"""

        # Verificar cache
        cache_key = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        try:
            # Preparar dados para IA
            analysis_prompt = self.prepare_ai_prompt(market_data)

            # Chamar API LiteLLM
            response = await self.llm_client.chat_completion(
                model="deepseek-r1-free",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um analista técnico especialista em XAUUSD. Forneça análise concisa e acionável."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Processar resposta
            ai_analysis = self.parse_ai_response(response['choices'][0]['message']['content'])

            # Salvar no cache
            self.analysis_cache[cache_key] = ai_analysis

            return ai_analysis

        except Exception as e:
            print(f"❌ Erro na análise IA: {e}")
            return None

    def prepare_ai_prompt(self, market_data):
        """Prepara prompt para análise de IA"""

        # Extrair dados principais
        h1_data = market_data.get('H1', {})
        current_price = h1_data.get('current_price', 0)

        # Obter indicadores principais
        indicators = self.calculate_key_indicators(market_data)

        # Obter contexto de mercado
        market_context = self.get_market_context()

        prompt = f"""
        ANALISE XAUUSD - {datetime.now().strftime('%Y-%m-%d %H:%M')}

        DADOS ATUAIS:
        - Preço: ${current_price:.2f}
        - Sessão: {market_context['session']}
        - Volatilidade: {market_context['volatility']:.1f}%

        INDICADORES TÉCNICOS:
        - RSI H1: {indicators['rsi_h1']:.1f}
        - MACD H1: {indicators['macd_h1_signal']}
        - EMA 9/21: {indicators['ema_signal']}
        - Bollinger Bands: {indicators['bb_position']}

        CONTEXTO:
        - Notícias recentes: {market_context['news_sentiment']}
        - Dólar index: {market_context['dxy_trend']}
        - Juros: {market_context['interest_rate_env']}

        FORNEÇA ANÁLISE EM JSON:
        {{
            "overall_sentiment": "bullish/bearish/neutral",
            "confidence": 0-100,
            "key_levels": {{
                "support": [nível1, nível2],
                "resistance": [nível1, nível2]
            }},
            "trade_setup": {{
                "action": "BUY/SELL/HOLD",
                "entry_zone": "preço_sugerido",
                "stop_loss": "sl_sugerido",
                "take_profit": "tp_sugerido",
                "risk_reward": 1.5
            }},
            "catalysts": ["fator1", "fator2"],
            "risks": ["risco1", "risco2"],
            "time_horizon": "scalping/swing/position"
        }}
        """

        return prompt

    def calculate_key_indicators(self, market_data):
        """Calcula indicadores principais para IA"""

        indicators = {}

        # RSI H1
        h1_bars = market_data.get('H1', [])
        if len(h1_bars) >= 14:
            closes = [bar['close'] for bar in h1_bars]
            indicators['rsi_h1'] = self.calculate_rsi(closes)

        # MACD H1
        if len(h1_bars) >= 26:
            indicators['macd_h1_signal'] = self.get_macd_signal(closes)

        # EMA Signal
        if len(h1_bars) >= 21:
            indicators['ema_signal'] = self.get_ema_signal(closes)

        # Bollinger Bands Position
        if len(h1_bars) >= 20:
            indicators['bb_position'] = self.get_bb_position(closes)

        return indicators

    def get_market_context(self):
        """Obtém contexto de mercado atual"""

        current_hour = datetime.now().hour

        # Determinar sessão
        if 0 <= current_hour < 8:
            session = "Asian"
        elif 8 <= current_hour < 13:
            session = "London"
        elif 13 <= current_hour < 17:
            session = "Overlap"
        else:
            session = "NY"

        # Simular outros dados (em implementação real, viria de APIs)
        return {
            'session': session,
            'volatility': 20.5,  # Simulado
            'news_sentiment': "neutral",  # Simulado
            'dxy_trend': "sideways",  # Simulado
            'interest_rate_env': "stable"  # Simulado
        }

    def parse_ai_response(self, response_text):
        """Parseia resposta da IA"""

        try:
            # Tentar extrair JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)

                # Validar estrutura
                required_keys = ['overall_sentiment', 'confidence', 'trade_setup']
                for key in required_keys:
                    if key not in analysis:
                        analysis[key] = None

                return analysis
            else:
                # Se não encontrar JSON, criar estrutura básica
                return {
                    'overall_sentiment': 'neutral',
                    'confidence': 50,
                    'trade_setup': {'action': 'HOLD'},
                    'raw_response': response_text
                }

        except json.JSONDecodeError:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0,
                'trade_setup': {'action': 'HOLD'},
                'error': 'Failed to parse AI response'
            }

    async def execute_ai_trade(self, ai_analysis, current_price):
        """Executa trade baseado na análise da IA"""

        if not ai_analysis or ai_analysis.get('confidence', 0) < 70:
            print("⏸️ IA indica aguardar - confiança baixa")
            return

        trade_setup = ai_analysis.get('trade_setup', {})
        action = trade_setup.get('action', 'HOLD')

        if action == 'HOLD':
            print("⏸️ IA indica HOLD")
            return

        # Validar e ajustar preços
        entry_price = self.validate_price(trade_setup.get('entry_zone'), current_price)
        stop_loss = self.validate_price(trade_setup.get('stop_loss'), current_price)
        take_profit = self.validate_price(trade_setup.get('take_profit'), current_price)

        if not all([entry_price, stop_loss, take_profit]):
            print("❌ Preços inválidos na análise IA")
            return

        # Calcular risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 1.0:
            print(f"⚠️ Risk/Reward desfavorável: 1:{rr_ratio:.2f}")
            return

        # Colocar ordem
        await self.place_ai_order(action, entry_price, stop_loss, take_profit, ai_analysis)

    def validate_price(self, price, current_price):
        """Valida e ajusta preço"""

        if not price:
            return None

        # Se preço for relativo (ex: "current + 10")
        if isinstance(price, str):
            if '+' in price:
                offset = float(price.split('+')[1])
                return current_price + offset
            elif '-' in price:
                offset = float(price.split('-')[1])
                return current_price - offset

        # Se preço for absoluto
        try:
            return float(price)
        except:
            return None

    async def place_ai_order(self, action, entry_price, stop_loss, take_profit, ai_analysis):
        """Coloca ordem baseada na análise IA"""

        # Calcular tamanho da posição
        account_info = await client.get_account_info()
        position_size = await self.calculate_ai_position_size(account_info, stop_loss, entry_price)

        order_data = {
            "symbol": "XAUUSD",
            "volume": position_size,
            "order_type": f"MARKET_{action}",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "magic_number": 88888,
            "comment": f"AI Trade - {ai_analysis.get('overall_sentiment', 'UNKNOWN')}"
        }

        result = await client.place_order(order_data)

        if result['success']:
            print(f"✅ Trade IA executado: {action}")
            print(f"💰 Entry: ${result['execution_price']:.2f}")
            print(f"🛡️ SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")
            print(f"🤖 Confiança IA: {ai_analysis.get('confidence', 0)}%")
            print(f"📊 Sentimento: {ai_analysis.get('overall_sentiment', 'N/A')}")

            # Registrar trade para análise posterior
            await self.record_ai_trade(result, ai_analysis)

    async def calculate_ai_position_size(self, account_info, stop_loss, entry_price):
        """Calcula tamanho da posição para trades IA"""

        # Baseado no risco da IA
        confidence = 75  # Default
        risk_amount = account_info['balance'] * (0.005 + (confidence / 100) * 0.01)  # 0.5% a 1.5%

        # Calcular posição
        sl_distance = abs(entry_price - stop_loss)
        symbol_info = await client.get_symbol_info("XAUUSD")
        pip_value = symbol_info.get('trade_tick_value', 10)

        position_size = risk_amount / (sl_distance * 100 * pip_value)
        position_size = max(0.01, min(position_size, 0.1))  # Entre 0.01 e 0.1

        return round(position_size, 2)

    async def record_ai_trade(self, trade_result, ai_analysis):
        """Registra trade IA para análise posterior"""

        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'ticket': trade_result['order_ticket'],
            'ai_sentiment': ai_analysis.get('overall_sentiment'),
            'ai_confidence': ai_analysis.get('confidence'),
            'ai_reasoning': ai_analysis.get('catalysts', []),
            'execution_price': trade_result['execution_price']
        }

        # Salvar em arquivo (simplificado)
        try:
            with open('ai_trades_history.json', 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
        except:
            pass  # Silently fail

    async def run_ai_strategy(self):
        """Executa estratégia completa com IA"""

        print("🤖 Iniciando estratégia com IA...")

        while True:
            try:
                # Coletar dados de mercado
                market_data = await self.collect_market_data()

                if not market_data:
                    print("❌ Falha na coleta de dados")
                    await asyncio.sleep(60)
                    continue

                # Obter análise da IA
                ai_analysis = await self.get_ai_market_analysis(market_data)

                if ai_analysis:
                    print(f"🧠 Análise IA: {ai_analysis.get('overall_sentiment', 'N/A')}")
                    print(f"📊 Confiança: {ai_analysis.get('confidence', 0)}%")

                    current_price = market_data.get('H1', {}).get('current_price', 0)
                    await self.execute_ai_trade(ai_analysis, current_price)

                await asyncio.sleep(300)  # Verificar a cada 5 minutos

            except Exception as e:
                print(f"❌ Erro na estratégia IA: {e}")
                await asyncio.sleep(60)

    async def collect_market_data(self):
        """Coleta dados de mercado para IA"""

        # Dados de múltiplos timeframes
        timeframes = ["M5", "M15", "H1", "H4"]
        data = {}

        for tf in timeframes:
            bars = await client.get_bars("XAUUSD", tf, 100)
            if bars:
                data[tf] = {
                    'bars': bars,
                    'current_price': bars[-1]['close'],
                    'volume': sum(bar['volume'] for bar in bars[-20:])
                }

        return data if len(data) >= 3 else None

# Funções auxiliares
def calculate_rsi(closes, period=14):
    """Calcula RSI simplificado"""
    if len(closes) < period + 1:
        return 50

    gains = []
    losses = []

    for i in range(1, len(closes[-period-1:])):
        change = closes[-i] - closes[-i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_macd_signal(closes):
    """Obtém sinal MACD simplificado"""
    if len(closes) < 26:
        return "neutral"

    # Simplificado - em implementação real usar talib
    return "bullish" if closes[-1] > closes[-2] else "bearish"

def get_ema_signal(closes):
    """Obtém sinal de EMA simplificado"""
    if len(closes) < 21:
        return "neutral"

    # Simplificado
    ema9 = sum(closes[-9:]) / 9
    ema21 = sum(closes[-21:]) / 21

    return "bullish" if ema9 > ema21 else "bearish"

def get_bb_position(closes):
    """Obtém posição nas Bollinger Bands"""
    if len(closes) < 20:
        return "middle"

    # Simplificado
    mean = sum(closes[-20:]) / 20
    std = (sum((x - mean) ** 2 for x in closes[-20:]) / 20) ** 0.5

    current_price = closes[-1]

    if current_price > mean + std:
        return "upper"
    elif current_price < mean - std:
        return "lower"
    else:
        return "middle"

# Uso
ai_strategy = AIEnhancedTradingStrategy()
# asyncio.run(ai_strategy.run_ai_strategy())
```

---

## Backtesting e Otimização

### 📊 Sistema de Backtesting com Dados Reais

**Problema:** Implementar backtesting realista com dados históricos do MT5.

```python
class RealisticBacktester:
    """Backtester realista com dados do MT5"""

    def __init__(self, initial_balance=10000, commission=7, spread=2):
        self.initial_balance = initial_balance
        self.commission = commission  # $7 por lote padrão
        self.spread = spread  # 2 pips spread médio
        self.slippage = 1  # 1 pip slippage médio

    async def run_historical_backtest(self, strategy, start_date, end_date):
        """Executa backtest com dados históricos reais"""

        print(f"🧪 Iniciando backtest: {start_date} a {end_date}")

        # Coletar dados históricos
        historical_data = await self.collect_historical_data(start_date, end_date)

        if not historical_data:
            raise Exception("Não foi possível coletar dados históricos")

        # Inicializar estado
        balance = self.initial_balance
        equity = balance
        positions = []
        trades = []

        # Processar cada barra
        bars = historical_data['H1']
        for i, bar in enumerate(bars):
            # Atualizar equity
            equity = self.update_equity(balance, positions, bar)

            # Verificar fechamento de posições
            positions = self.check_position_closures(positions, bar, trades, balance)
            balance = sum(trade.profit for trade in trades) + self.initial_balance

            # Gerar sinal da estratégia
            if i % 5 == 0:  # Verificar a cada 5 barras
                signal = await self.generate_backtest_signal(strategy, historical_data, i)

                if signal and len(positions) < 2:
                    position = self.execute_backtest_trade(signal, bar, balance)
                    if position:
                        positions.append(position)

        # Fechar posições restantes
        for position in positions:
            trade = self.close_position(position, bars[-1])
            trades.append(trade)

        # Calcular resultados finais
        results = self.calculate_backtest_results(trades, balance, equity)

        return results

    async def collect_historical_data(self, start_date, end_date):
        """Coleta dados históricos do MT5"""

        # Obter dados H1
        h1_bars = await client.get_bars("XAUUSD", "H1", 5000)

        # Filtrar por período
        filtered_bars = []
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        for bar in h1_bars:
            bar_time = datetime.fromisoformat(bar['time'])
            if start_dt <= bar_time <= end_dt:
                filtered_bars.append(bar)

        return {
            'H1': filtered_bars,
            'start_date': start_date,
            'end_date': end_date,
            'total_bars': len(filtered_bars)
        }

    def update_equity(self, balance, positions, current_bar):
        """Atualiza equity considerando posições abertas"""

        floating_pnl = 0
        current_price = current_bar['close']

        for position in positions:
            if position['type'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['volume'] * 100
            else:  # SELL
                pnl = (position['entry_price'] - current_price) * position['volume'] * 100

            floating_pnl += pnl

        return balance + floating_pnl

    def check_position_closures(self, positions, bar, trades, balance):
        """Verifica se posições devem ser fechadas"""

        current_price = bar['close']
        remaining_positions = []

        for position in positions:
            should_close = False
            close_price = current_price
            close_reason = "MARKET_CLOSE"

            # Verificar take profit
            if position['type'] == 'BUY' and current_price >= position['take_profit']:
                should_close = True
                close_price = position['take_profit']
                close_reason = "TAKE_PROFIT"
            elif position['type'] == 'SELL' and current_price <= position['take_profit']:
                should_close = True
                close_price = position['take_profit']
                close_reason = "TAKE_PROFIT"

            # Verificar stop loss
            elif position['type'] == 'BUY' and current_price <= position['stop_loss']:
                should_close = True
                close_price = position['stop_loss']
                close_reason = "STOP_LOSS"
            elif position['type'] == 'SELL' and current_price >= position['stop_loss']:
                should_close = True
                close_price = position['stop_loss']
                close_reason = "STOP_LOSS"

            if should_close:
                # Fechar posição
                trade = self.close_position(position, bar, close_price, close_reason)
                trades.append(trade)
            else:
                remaining_positions.append(position)

        return remaining_positions

    def close_position(self, position, bar, close_price=None, close_reason="MARKET_CLOSE"):
        """Fecha uma posição e calcula P&L"""

        if close_price is None:
            close_price = bar['close']

        # Adicionar slippage
        if position['type'] == 'BUY':
            close_price -= self.slippage * 0.01
        else:
            close_price += self.slippage * 0.01

        # Calcular P&L
        if position['type'] == 'BUY':
            gross_pnl = (close_price - position['entry_price']) * position['volume'] * 100
        else:  # SELL
            gross_pnl = (position['entry_price'] - close_price) * position['volume'] * 100

        # Subtrair comissão
        commission = self.commission * position['volume']
        net_pnl = gross_pnl - commission

        # Criar registro do trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': bar['time'],
            'type': position['type'],
            'volume': position['volume'],
            'entry_price': position['entry_price'],
            'exit_price': close_price,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'close_reason': close_reason,
            'duration_bars': bar['index'] - position['entry_bar']
        }

        return trade

    async def generate_backtest_signal(self, strategy, historical_data, bar_index):
        """Gera sinal da estratégia para backtest"""

        # Preparar dados até o índice atual
        current_data = {}
        for tf, bars in historical_data.items():
            if tf == 'start_date' or tf == 'end_date' or tf == 'total_bars':
                continue
            current_data[tf] = bars[:bar_index + 1]

        try:
            # Gerar sinal
            state = await strategy.analyze_market(current_data)
            signal = await strategy.generate_signal(state)

            return signal

        except Exception as e:
            # Silently fail em backtest
            return None

    def execute_backtest_trade(self, signal, bar, balance):
        """Executa trade no backtest"""

        # Calcular tamanho da posição
        risk_amount = balance * 0.01  # 1% de risco
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        volume = risk_amount / (sl_distance * 100)

        # Limitar volume
        volume = max(0.01, min(volume, 1.0))

        # Adicionar spread ao entry price
        if signal.action == 'BUY':
            entry_price = signal.entry_price + self.spread * 0.01
        else:
            entry_price = signal.entry_price - self.spread * 0.01

        position = {
            'type': signal.action,
            'volume': volume,
            'entry_price': entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'entry_time': bar['time'],
            'entry_bar': bar['index']
        }

        return position

    def calculate_backtest_results(self, trades, final_balance, final_equity):
        """Calcula resultados finais do backtest"""

        if not trades:
            return {
                'total_trades': 0,
                'net_profit': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        # Métricas básicas
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t['net_pnl'] > 0])
        losing_trades = total_trades - profitable_trades

        win_rate = (profitable_trades / total_trades) * 100

        # Profit/Loss
        total_profit = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
        total_loss = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0))
        net_profit = total_profit - total_loss

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Drawdown (simplificado)
        max_drawdown = 0
        peak = self.initial_balance

        for trade in trades:
            current_balance = self.initial_balance + trade['net_pnl']
            if current_balance > peak:
                peak = current_balance

            drawdown = peak - current_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Sharpe ratio (simplificado)
        returns = [t['net_pnl'] / self.initial_balance for t in trades]
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': (max_drawdown / self.initial_balance) * 100,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': final_balance,
            'return_percent': (final_balance / self.initial_balance - 1) * 100,
            'avg_trade': net_profit / total_trades if total_trades > 0 else 0,
            'trades': trades
        }

# Uso
async def run_complete_backtest():
    """Executa backtest completo"""

    backtester = RealisticBacktester()

    # Definir período de teste
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    # Criar estratégia simples para teste
    from simple_trading_bot import SimpleTradingBot
    strategy = SimpleTradingBot()

    # Executar backtest
    results = await backtester.run_historical_backtest(
        strategy, start_date, end_date
    )

    # Exibir resultados
    print(f"\n📊 RESULTADOS DO BACKTEST:")
    print(f"Período: {start_date} a {end_date}")
    print(f"Total de trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Lucro líquido: ${results['net_profit']:.2f}")
    print(f"Retorno: {results['return_percent']:.1f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Max drawdown: {results['max_drawdown_percent']:.1f}%")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

# asyncio.run(run_complete_backtest())
```

---

## Monitoramento e Alertas

### 📱 Sistema de Alertas Multi-Canal

**Problema:** Implementar sistema de alertas para múltiplos canais (email, Telegram, etc.).

```python
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class AlertManager:
    """Gerenciador de alertas multi-canal"""

    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': os.getenv('ALERT_EMAIL'),
            'sender_password': os.getenv('ALERT_EMAIL_PASSWORD'),
            'recipient_emails': os.getenv('ALERT_RECIPIENTS', '').split(',')
        }

        self.telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_ids': os.getenv('TELEGRAM_CHAT_IDS', '').split(',')
        }

        self.alert_history = []
        self.cooldown_period = 300  # 5 minutos entre alertas do mesmo tipo

    async def send_alert(self, alert_type, message, priority='medium'):
        """Envia alerta para todos os canais configurados"""

        # Verificar cooldown
        if not self.should_send_alert(alert_type):
            return

        # Preparar alerta
        alert = {
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'channels': []
        }

        # Enviar por email
        if self.email_config['sender_email']:
            try:
                await self.send_email_alert(alert)
                alert['channels'].append('email')
            except Exception as e:
                print(f"❌ Falha no email: {e}")

        # Enviar por Telegram
        if self.telegram_config['bot_token']:
            try:
                await self.send_telegram_alert(alert)
                alert['channels'].append('telegram')
            except Exception as e:
                print(f"❌ Falha no Telegram: {e}")

        # Registrar alerta
        self.alert_history.append(alert)

        # Manter apenas últimos 100 alertas
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        print(f"📢 Alerta enviado: {alert_type} - {message}")

    def should_send_alert(self, alert_type):
        """Verifica se deve enviar alerta (cooldown)"""

        if not self.alert_history:
            return True

        # Procurar último alerta do mesmo tipo
        for alert in reversed(self.alert_history):
            if alert['type'] == alert_type:
                time_diff = (datetime.now() - alert['timestamp']).seconds
                return time_diff > self.cooldown_period

        return True

    async def send_email_alert(self, alert):
        """Envia alerta por email"""

        if not self.email_config['sender_email']:
            return

        # Preparar mensagem
        subject = f"[EA ALERT] {alert['type'].upper()} - Priority: {alert['priority'].upper()}"

        body = f"""
        EA_SCALPER_XAUUSD Alert

        Type: {alert['type']}
        Priority: {alert['priority']}
        Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

        Message:
        {alert['message']}

        ---
        This is an automated message from EA_SCALPER_XAUUSD
        """

        # Enviar para cada destinatário
        for recipient in self.email_config['recipient_emails']:
            if recipient.strip():
                await self.send_single_email(recipient, subject, body)

    async def send_single_email(self, recipient, subject, body):
        """Envia email para um único destinatário"""

        msg = MIMEMultipart()
        msg['From'] = self.email_config['sender_email']
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['sender_email'], self.email_config['sender_password'])

        text = msg.as_string()
        server.sendmail(self.email_config['sender_email'], recipient, text)
        server.quit()

    async def send_telegram_alert(self, alert):
        """Envia alerta por Telegram"""

        if not self.telegram_config['bot_token']:
            return

        # Formatar mensagem
        emoji = {
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴',
            'critical': '🚨'
        }.get(alert['priority'], '📢')

        message = f"""
        {emoji} *EA_SCALPER_XAUUSD ALERT*

        *Type:* {alert['type']}
        *Priority:* {alert['priority'].upper()}
        *Time:* {alert['timestamp'].strftime('%H:%M:%S')}

        {alert['message']}
        """

        # Enviar para cada chat
        for chat_id in self.telegram_config['chat_ids']:
            if chat_id.strip():
                await self.send_telegram_message(chat_id, message)

    async def send_telegram_message(self, chat_id, message):
        """Envia mensagem para um chat do Telegram"""

        url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"

        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        response = requests.post(url, data=data)
        response.raise_for_status()

# Uso
alert_manager = AlertManager()

async def setup_trading_alerts():
    """Configura alertas de trading"""

    # Alerta de execução de trade
    async def trade_executed_alert(trade_info):
        message = f"""
        Trade Executed: {trade_info['type'].upper()}
        Symbol: XAUUSD
        Volume: {trade_info['volume']} lots
        Entry: ${trade_info['entry_price']:.2f}
        SL: ${trade_info['stop_loss']:.2f}
        TP: ${trade_info['take_profit']:.2f}
        """

        await alert_manager.send_alert('TRADE_EXECUTED', message, 'medium')

    # Alerta de drawdown elevado
    async def drawdown_alert(drawdown_percent):
        message = f"""
        Drawdown Alert
        Current Drawdown: {drawdown_percent:.1f}%

        Status: {'⚠️ WARNING' if drawdown_percent < 4 else '🚨 CRITICAL'}
        """

        priority = 'high' if drawdown_percent < 4 else 'critical'
        await alert_manager.send_alert('HIGH_DRAWDOWN', message, priority)

    # Alerta de meta diária alcançada
    async def daily_target_alert(profit_amount):
        message = f"""
        Daily Target Achieved! 🎉

        Profit: ${profit_amount:.2f}
        Time: {datetime.now().strftime('%H:%M:%S')}

        Great job today!
        """

        await alert_manager.send_alert('DAILY_TARGET', message, 'low')

# Exemplo de uso
# await alert_manager.send_alert('TEST', 'Sistema de alertas funcionando!', 'low')
```

### 📈 Dashboard de Monitoramento em Tempo Real

**Problema:** Criar dashboard para monitorar performance em tempo real.

```python
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

class TradingDashboard:
    """Dashboard de monitoramento de trading"""

    def __init__(self):
        self.performance_data = []
        self.trades_history = []
        self.alerts_history = []

    async def update_dashboard(self, client):
        """Atualiza dados do dashboard"""

        try:
            # Obter informações da conta
            account_info = await client.get_account_info()

            # Obter posições abertas
            positions = await client.get_positions("XAUUSD")

            # Obter histórico recente
            today_trades = await self.get_today_trades(client)

            # Calcular métricas
            metrics = self.calculate_metrics(account_info, positions, today_trades)

            # Atualizar dados
            self.performance_data.append({
                'timestamp': datetime.now(),
                'balance': account_info['balance'],
                'equity': account_info['equity'],
                'margin': account_info['margin'],
                'free_margin': account_info['free_margin'],
                'open_positions': len(positions),
                'daily_pnl': sum(t.get('profit', 0) for t in today_trades)
            })

            # Manter apenas últimas 1000 entradas
            if len(self.performance_data) > 1000:
                self.performance_data = self.performance_data[-1000:]

            return metrics

        except Exception as e:
            print(f"❌ Erro ao atualizar dashboard: {e}")
            return None

    def calculate_metrics(self, account_info, positions, today_trades):
        """Calcula métricas de performance"""

        balance = account_info['balance']
        equity = account_info['equity']
        margin = account_info['margin']

        # Métricas básicas
        daily_pnl = sum(t.get('profit', 0) for t in today_trades)
        floating_pnl = equity - balance

        # Métricas de risco
        margin_level = (equity / margin * 100) if margin > 0 else 0
        free_margin_percent = (account_info['free_margin'] / equity * 100) if equity > 0 else 0

        # Performance intraday
        if self.performance_data:
            daily_high = max(p['balance'] for p in self.performance_data)
            daily_low = min(p['balance'] for p in self.performance_data)
            daily_range = daily_high - daily_low
        else:
            daily_high = daily_low = daily_range = balance

        return {
            'account': {
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'free_margin': account_info['free_margin'],
                'margin_level': margin_level,
                'free_margin_percent': free_margin_percent
            },
            'performance': {
                'daily_pnl': daily_pnl,
                'floating_pnl': floating_pnl,
                'daily_high': daily_high,
                'daily_low': daily_low,
                'daily_range': daily_range,
                'daily_return': (equity - balance + daily_pnl) / balance * 100 if balance > 0 else 0
            },
            'positions': {
                'count': len(positions),
                'total_volume': sum(p.get('volume', 0) for p in positions),
                'total_pnl': sum(p.get('profit', 0) for p in positions)
            },
            'trading': {
                'trades_today': len(today_trades),
                'winning_trades': len([t for t in today_trades if t.get('profit', 0) > 0]),
                'losing_trades': len([t for t in today_trades if t.get('profit', 0) < 0]),
                'win_rate': 0  # Calculado abaixo
            }
        }

    def generate_report(self, metrics):
        """Gera relatório textual do dashboard"""

        if not metrics:
            return "Dados não disponíveis"

        report = f"""
        📊 EA_SCALPER_XAUUSD DASHBOARD
        {'='*50}
        🕐 Atualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        💰 CONTA:
        • Saldo: ${metrics['account']['balance']:.2f}
        • Equity: ${metrics['account']['equity']:.2f}
        • Margem: ${metrics['account']['margin']:.2f}
        • Margem Livre: ${metrics['account']['free_margin']:.2f}
        • Nível Margem: {metrics['account']['margin_level']:.1f}%

        📈 PERFORMANCE:
        • PnL Diário: ${metrics['performance']['daily_pnl']:.2f}
        • PnL Flutuante: ${metrics['performance']['floating_pnl']:.2f}
        • Retorno Diário: {metrics['performance']['daily_return']:.2f}%
        • Máxima Diária: ${metrics['performance']['daily_high']:.2f}
        • Mínima Diária: ${metrics['performance']['daily_low']:.2f}

        📊 POSIÇÕES:
        • Posições Abertas: {metrics['positions']['count']}
        • Volume Total: {metrics['positions']['total_volume']:.2f} lotes
        • PnL Aberto: ${metrics['positions']['total_pnl']:.2f}

        🔄 TRADING:
        • Trades Hoje: {metrics['trading']['trades_today']}
        • Trades Vencedores: {metrics['trading']['winning_trades']}
        • Trades Perdedores: {metrics['trading']['losing_trades']}
        """

        # Calcular win rate
        if metrics['trading']['trades_today'] > 0:
            win_rate = (metrics['trading']['winning_trades'] / metrics['trading']['trades_today']) * 100
            report += f"• Win Rate: {win_rate:.1f}%\n"

        # Status
        margin_level = metrics['account']['margin_level']
        if margin_level < 200:
            status = "🔴 MARGEM BAIXA"
        elif margin_level < 500:
            status = "🟡 MARGEM MODERADA"
        else:
            status = "✅ MARGEM SEGURA"

        report += f"\n🚦 STATUS: {status}"

        return report

    def generate_equity_chart(self, hours=24):
        """Gera gráfico da curva de equity"""

        if not self.performance_data:
            return None

        # Filtrar dados do período
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [p for p in self.performance_data if p['timestamp'] > cutoff_time]

        if not recent_data:
            return None

        # Criar DataFrame
        df = pd.DataFrame(recent_data)

        # Criar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity', linewidth=2)
        plt.plot(df['timestamp'], df['balance'], label='Balance', linewidth=1, alpha=0.7)

        plt.title(f'Curva de Equity - Últimas {hours} horas')
        plt.xlabel('Tempo')
        plt.ylabel('Valor ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Salvar gráfico
        filename = f"equity_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return filename

    async def save_dashboard_data(self):
        """Salva dados do dashboard em arquivo"""

        data = {
            'last_update': datetime.now().isoformat(),
            'performance_data': [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'balance': p['balance'],
                    'equity': p['equity'],
                    'open_positions': p['open_positions'],
                    'daily_pnl': p['daily_pnl']
                } for p in self.performance_data[-100:]  # Últimos 100 registros
            ],
            'trades_history': self.trades_history[-50:]  # Últimos 50 trades
        }

        filename = f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return filename

# Função principal de monitoramento
async def run_dashboard(client, update_interval=60):
    """Executa dashboard contínuo"""

    dashboard = TradingDashboard()

    print("📊 Iniciando dashboard de monitoramento...")

    while True:
        try:
            # Atualizar dados
            metrics = await dashboard.update_dashboard(client)

            if metrics:
                # Gerar relatório
                report = dashboard.generate_report(metrics)

                # Limpar tela e mostrar relatório
                os.system('clear' if os.name == 'posix' else 'cls')
                print(report)

                # Gerar gráfico a cada hora
                if datetime.now().minute == 0:
                    chart_file = dashboard.generate_equity_chart(24)
                    if chart_file:
                        print(f"\n📈 Gráfico salvo: {chart_file}")

                # Salvar dados a cada 6 horas
                if datetime.now().hour % 6 == 0 and datetime.now().minute == 0:
                    data_file = await dashboard.save_dashboard_data()
                    print(f"\n💾 Dados salvos: {data_file}")

            await asyncio.sleep(update_interval)

        except Exception as e:
            print(f"❌ Erro no dashboard: {e}")
            await asyncio.sleep(30)

# Uso
# dashboard = TradingDashboard()
# asyncio.run(run_dashboard(client, update_interval=60))
```

---

## Solução de Problemas

### 🔧 Guia de Troubleshooting Comum

**Problema 1: Conexão MT5 Falha**
```python
async def troubleshoot_mt5_connection():
    """Diagnóstico de problemas de conexão MT5"""

    print("🔧 Diagnosticando conexão MT5...")

    # 1. Verificar se MT5 está aberto
    try:
        import psutil
        mt5_running = any("terminal64.exe" in p.info['name'].lower() for p in psutil.process_iter(['name']))
        print(f"✅ MT5 aberto: {mt5_running}")

        if not mt5_running:
            print("❌ MT5 não está aberto! Abra o MetaTrader 5.")
            return False
    except:
        print("⚠️ Não foi possível verificar se MT5 está aberto")

    # 2. Tentar inicialização MT5
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            print("✅ MT5 inicializado com sucesso")
            mt5.shutdown()
        else:
            print(f"❌ Falha na inicialização MT5: {mt5.last_error()}")
            return False
    except Exception as e:
        print(f"❌ Erro ao importar/inicializar MT5: {e}")
        return False

    # 3. Verificar variáveis de ambiente
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Variáveis de ambiente faltando: {missing_vars}")
        return False
    else:
        print("✅ Variáveis de ambiente configuradas")

    # 4. Testar conexão completa
    try:
        client = MT5Client()
        success = await client.connect(
            login=int(os.getenv('MT5_LOGIN')),
            password=os.getenv('MT5_PASSWORD'),
            server=os.getenv('MT5_SERVER')
        )

        if success:
            print("✅ Conexão MT5 bem-sucedida!")
            await client.disconnect()
            return True
        else:
            print("❌ Falha na autenticação MT5")
            return False

    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

# asyncio.run(troubleshoot_mt5_connection())
```

**Problema 2: Símbolo XAUUSD Não Encontrado**
```python
async def troubleshoot_xauusd_symbol():
    """Diagnóstico de problemas com símbolo XAUUSD"""

    print("🔧 Diagnosticando símbolo XAUUSD...")

    # Lista de possíveis nomes para XAUUSD
    possible_symbols = [
        "XAUUSD",
        "XAUUSD.",
        "XAUUSD.m",
        "XAUUSD_TDS",
        "GOLD",
        "GOLD."
    ]

    try:
        client = MT5Client()
        await client.connect_from_env()

        found_symbols = []

        for symbol in possible_symbols:
            symbol_info = await client.get_symbol_info(symbol)
            if symbol_info:
                found_symbols.append({
                    'name': symbol,
                    'spread': symbol_info['spread'],
                    'volume_min': symbol_info['volume_min'],
                    'volume_max': symbol_info['volume_max'],
                    'trade_mode': symbol_info.get('trade_mode', 'UNKNOWN')
                })
                print(f"✅ Encontrado: {symbol}")
            else:
                print(f"❌ Não encontrado: {symbol}")

        if found_symbols:
            print(f"\n📊 Símbolos disponíveis:")
            for sym in found_symbols:
                trade_modes = {
                    0: "Desabilitado",
                    1: "Completo",
                    2: "Apenas Long",
                    3: "Apenas Short",
                    4: "Close Only"
                }

                mode = trade_modes.get(sym['trade_mode'], 'Desconhecido')

                print(f"  • {sym['name']}:")
                print(f"    - Spread: {sym['spread']} pts")
                print(f"    - Volume: {sym['volume_min']}-{sym['volume_max']}")
                print(f"    - Modo: {mode}")

            # Recomendar melhor símbolo
            best_symbol = min(found_symbols, key=lambda x: x['spread'])
            print(f"\n💡 Recomendação: Usar '{best_symbol['name']}' (spread mais baixo)")

        else:
            print("❌ Nenhum símbolo XAUUSD encontrado!")
            print("\nSoluções possíveis:")
            print("1. Verifique se sua conta RoboForex permite trading de XAUUSD")
            print("2. Adicione XAUUSD ao Market Watch no MT5")
            print("3. Contate o suporte do broker")

        await client.disconnect()
        return len(found_symbols) > 0

    except Exception as e:
        print(f"❌ Erro diagnóstico: {e}")
        return False

# asyncio.run(troubleshoot_xauusd_symbol())
```

**Problema 3: LiteLLM Proxy Não Responde**
```python
async def troubleshoot_litellm_proxy():
    """Diagnóstico de problemas com LiteLLM proxy"""

    print("🔧 Diagnosticando LiteLLM proxy...")

    # 1. Verificar se proxy está rodando
    import requests

    try:
        response = requests.get("http://localhost:4000/health", timeout=5)
        if response.status_code == 200:
            print("✅ LiteLLM proxy está rodando")
        else:
            print(f"❌ Proxy respondeu com status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ LiteLLM proxy não está rodando na porta 4000")
        print("\nSolução:")
        print("1. Inicie o proxy: litellm --config litellm_config.yaml --port 4000")
        print("2. Verifique se a porta 4000 não está em uso")
        print("3. Verifique o arquivo de configuração litellm_config.yaml")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar proxy: {e}")
        return False

    # 2. Verificar configuração
    try:
        with open('litellm_config.yaml', 'r') as f:
            config_content = f.read()

        if 'model_list' in config_content and 'api_key' in config_content:
            print("✅ Arquivo de configuração parece válido")
        else:
            print("❌ Arquivo de configuração inválido")
            return False

    except FileNotFoundError:
        print("❌ Arquivo litellm_config.yaml não encontrado")
        return False

    # 3. Testar chamada API
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_key"
        }

        data = {
            "model": "deepseek-r1-free",
            "messages": [{"role": "user", "content": "Teste"}],
            "max_tokens": 10
        }

        response = requests.post(
            "http://localhost:4000/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            print("✅ Chamada API funcionou")
            return True
        else:
            print(f"❌ Erro na chamada API: {response.status_code}")
            print(f"Resposta: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Erro na chamada API: {e}")
        print("\nPossíveis causas:")
        print("1. Chave de API inválida ou não configurada")
        print("2. Modelo não disponível")
        print("3. Problemas de conexão com o provedor da API")
        return False

# asyncio.run(troubleshoot_litellm_proxy())
```

---

## Conclusão

Este cookbook cobre as operações mais comuns e cenários práticos para o sistema EA_SCALPER_XAUUSD. As receitas aqui apresentadas podem ser adaptadas e combinadas para criar soluções personalizadas.

### Melhores Práticas

1. **Sempre teste** em conta demo antes de usar capital real
2. **Monitore performance** continuamente
3. **Implemente stop losses** rigorosos
4. **Diversifique estratégias** para reduzir risco
5. **Mantenha registros** detalhados de todas as operações
6. **Atualize sistemas** regularmente
7. **Faça backups** de configurações e dados

### Suporte

- **Documentação completa**: `/docs/api-reference/`
- **Exemplos práticos**: `/docs/examples/`
- **Tutoriais detalhados**: `/docs/tutorials/`
- **Issues**: Reporte problemas no GitHub

Continue explorando e adaptando estas receitas para suas necessidades específicas de trading!