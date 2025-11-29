# Tutorial 02: Desenvolvimento Avançado de Estratégias
===================================================

## Overview

Este tutorial avançado ensina a desenvolver estratégias de trading sofisticadas para XAUUSD usando o sistema EA_SCALPER_XAUUSD. Você aprenderá a criar, testar e otimizar estratégias FTMO-compliant.

## Pré-requisitos

- Tutorial 01 concluído (ambiente configurado)
- Conhecimento básico de trading técnico
- Noções de Python programação
- Compreensão de gestão de risco

## Conceitos Fundamentais

### 1. Filosofia de Trading XAUUSD

XAUUSD (Gold/USD) tem características únicas:
- **Alta volatilidade** durante sessões de Londres/NY
- **Correlação inversa** com DXY (Dólar Index)
- **Sensibilidade** a notícias econômicas
- **Liquidez** alta 24/5 (exceto fins de semana)

### 2. Abordagem Multi-Timeframe

Estratégia eficaz usa múltiplos timeframes:
- **H4**: Tendência principal
- **H1**: Direção do movimento
- **M15**: Timing de entrada
- **M5**: Precisão de execução

### 3. Princípios FTMO-Compliant

- Máximo 5% perda diária
- Máximo 10% perda total
- Sem hedging
- Sem martingale
- Consistência de lucros

## Estrutura da Estratégia Avançada

### 1. Framework de Desenvolvimento

Crie arquivo `strategy_framework.py`:

```python
import asyncio
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import logging

from ea_scalper_sdk import MT5Client

logger = logging.getLogger("StrategyFramework")

@dataclass
class Signal:
    """Estrutura de sinal de trading"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasoning: str
    indicators: Dict[str, float]
    timestamp: datetime

@dataclass
class MarketState:
    """Estado atual do mercado"""
    price: float
    trend: str  # bullish, bearish, neutral
    volatility: float
    volume: float
    momentum: float
    support_levels: List[float]
    resistance_levels: List[float]
    session: str  # Asian, London, NY, Overlap

class BaseStrategy(ABC):
    """Classe base para estratégias de trading"""

    def __init__(self, name: str, symbol: str = "XAUUSD"):
        self.name = name
        self.symbol = symbol
        self.mt5_client: Optional[MT5Client] = None
        self.is_active = False
        self.trading_hours = {
            "start": dt_time(8, 0),  # 8:00 GMT
            "end": dt_time(20, 0)     # 20:00 GMT
        }
        self.max_positions = 2
        self.risk_per_trade = 1.0  # 1% por trade

    @abstractmethod
    async def analyze_market(self, market_data: Dict) -> MarketState:
        """Análise do mercado - implementação específica da estratégia"""
        pass

    @abstractmethod
    async def generate_signal(self, state: MarketState) -> Optional[Signal]:
        """Geração de sinais - implementação específica da estratégia"""
        pass

    async def initialize(self, mt5_client: MT5Client):
        """Inicializa a estratégia"""
        self.mt5_client = mt5_client
        self.is_active = True
        logger.info(f"✅ Estratégia {self.name} inicializada")

    async def should_trade(self) -> Tuple[bool, str]:
        """Verifica se deve operar agora"""
        if not self.is_active:
            return False, "Estratégia inativa"

        current_time = datetime.now().time()
        if not (self.trading_hours["start"] <= current_time <= self.trading_hours["end"]):
            return False, "Fora do horário de trading"

        # Verificar posições abertas
        positions = await self.mt5_client.get_positions(self.symbol)
        if len(positions) >= self.max_positions:
            return False, f"Máximo de {self.max_positions} posições atingido"

        return True, "Condições favoráveis"

    async def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calcula tamanho da posição baseado no risco"""
        try:
            account = await self.mt5_client.get_account_info()
            balance = account.get('balance', 10000)

            risk_amount = balance * (self.risk_per_trade / 100)

            # Obter valor do pip para XAUUSD
            symbol_info = await self.mt5_client.get_symbol_info(self.symbol)
            pip_value = symbol_info.get('trade_tick_value', 10)

            position_size = risk_amount / (stop_loss_pips * pip_value)

            # Limitar entre 0.01 e 1.0
            position_size = max(0.01, min(position_size, 1.0))
            return round(position_size, 2)

        except Exception as e:
            logger.error(f"❌ Erro no cálculo de posição: {e}")
            return 0.01

    async def execute_trade(self, signal: Signal) -> bool:
        """Executa ordem baseada no sinal"""
        try:
            if signal.action == "HOLD" or signal.confidence < 70:
                return False

            # Calcular tamanho da posição
            sl_pips = abs(signal.entry_price - signal.stop_loss) * 100
            volume = await self.calculate_position_size(sl_pips)

            order_type = "MARKET_BUY" if signal.action == "BUY" else "MARKET_SELL"

            order_data = {
                "symbol": self.symbol,
                "volume": volume,
                "order_type": order_type,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "magic_number": hash(self.name) % 100000,
                "comment": f"{self.name} v1.0"
            }

            result = await self.mt5_client.place_order(order_data)

            if result['success']:
                logger.info(f"✅ {signal.action} {volume} lotes executado")
                logger.info(f"💰 Entry: {result['execution_price']}")
                logger.info(f"🛡️ SL: {signal.stop_loss}")
                logger.info(f"🎯 TP: {signal.take_profit}")
                return True
            else:
                logger.error(f"❌ Falha na execução: {result['message']}")
                return False

        except Exception as e:
            logger.error(f"❌ Erro na execução: {e}")
            return False

    async def run_cycle(self):
        """Executa um ciclo completo da estratégia"""
        try:
            # Verificar se deve operar
            can_trade, reason = await self.should_trade()
            if not can_trade:
                logger.info(f"⏸️ {reason}")
                return

            # Coletar dados de mercado
            market_data = await self.collect_market_data()
            if not market_data:
                logger.warning("⚠️ Falha na coleta de dados")
                return

            # Analisar mercado
            state = await self.analyze_market(market_data)

            # Gerar sinal
            signal = await self.generate_signal(state)

            if signal:
                logger.info(f"📊 Sinal gerado: {signal.action}")
                logger.info(f"🎯 Confiança: {signal.confidence}%")
                logger.info(f"💰 R/R: 1:{signal.risk_reward:.1f}")

                # Executar trade
                await self.execute_trade(signal)

        except Exception as e:
            logger.error(f"❌ Erro no ciclo: {e}")

    async def collect_market_data(self) -> Optional[Dict]:
        """Coleta dados de múltiplos timeframes"""
        try:
            timeframes = ["M5", "M15", "H1", "H4"]
            data = {}

            for tf in timeframes:
                bars = await self.mt5_client.get_bars(self.symbol, tf, 200)
                if bars:
                    data[tf] = bars

            # Obter ticks recentes
            ticks = await self.mt5_client.get_ticks(self.symbol, 50)
            if ticks:
                data['ticks'] = ticks

            return data if len(data) >= 3 else None

        except Exception as e:
            logger.error(f"❌ Erro na coleta de dados: {e}")
            return None

    def get_current_session(self) -> str:
        """Determina sessão de trading atual"""
        current_hour = datetime.now().hour

        if 0 <= current_hour < 8:
            return "Asian"
        elif 8 <= current_hour < 13:
            return "London"
        elif 13 <= current_hour < 17:
            return "Overlap"  # London/NY
        elif 17 <= current_hour < 22:
            return "NY"
        else:
            return "Asian"
```

### 2. Implementação da Estratégia Advanced Scalping

Crie arquivo `advanced_scalping_strategy.py`:

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import talib

from strategy_framework import BaseStrategy, Signal, MarketState

class AdvancedXAUUSDScalper(BaseStrategy):
    """Estratégia avançada de scalping para XAUUSD"""

    def __init__(self):
        super().__init__("AdvancedXAUUSDScalper")
        self.indicators_weights = {
            'rsi': 0.2,
            'macd': 0.15,
            'bbands': 0.15,
            'ema': 0.2,
            'volume': 0.1,
            'volatility': 0.1,
            'session': 0.1
        }
        self.atr_period = 14
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2.0

    async def analyze_market(self, market_data: Dict) -> MarketState:
        """Análise técnica multi-timeframe avançada"""

        try:
            # Obter dados principais (H1)
            h1_bars = market_data.get('H1', [])
            if len(h1_bars) < 100:
                raise ValueError("Dados insuficientes")

            # Preparar dados
            closes = np.array([bar['close'] for bar in h1_bars])
            highs = np.array([bar['high'] for bar in h1_bars])
            lows = np.array([bar['low'] for bar in h1_bars])
            volumes = np.array([bar['volume'] for bar in h1_bars])

            current_price = closes[-1]

            # Calcular indicadores técnicos
            indicators = await self.calculate_all_indicators(closes, highs, lows, volumes)

            # Determinar tendência principal
            trend = self.determine_trend(indicators, h1_bars)

            # Calcular volatilidade
            volatility = self.calculate_volatility(closes)

            # Calcular momentum
            momentum = self.calculate_momentum(closes)

            # Encontrar níveis de suporte/resistência
            support_levels, resistance_levels = self.find_key_levels(h1_bars)

            # Determinar sessão atual
            session = self.get_current_session()

            return MarketState(
                price=current_price,
                trend=trend,
                volatility=volatility,
                volume=volumes[-1] if len(volumes) > 0 else 0,
                momentum=momentum,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                session=session
            )

        except Exception as e:
            logger.error(f"❌ Erro na análise de mercado: {e}")
            raise

    async def calculate_all_indicators(self, closes: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray, volumes: np.ndarray) -> Dict:
        """Calcula todos os indicadores técnicos"""

        indicators = {}

        # EMAs
        indicators['ema_fast'] = talib.EMA(closes, timeperiod=self.ema_fast)[-1]
        indicators['ema_slow'] = talib.EMA(closes, timeperiod=self.ema_slow)[-1]

        # RSI
        indicators['rsi'] = talib.RSI(closes, timeperiod=self.rsi_period)[-1]

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes)
        indicators['macd'] = macd[-1]
        indicators['macd_signal'] = macd_signal[-1]
        indicators['macd_histogram'] = macd_hist[-1]

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
        indicators['bb_upper'] = bb_upper[-1]
        indicators['bb_middle'] = bb_middle[-1]
        indicators['bb_lower'] = bb_lower[-1]

        # ATR
        indicators['atr'] = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)[-1]

        # Estocástico
        slowk, slowd = talib.STOCH(highs, lows, closes)
        indicators['stoch_k'] = slowk[-1]
        indicators['stoch_d'] = slowd[-1]

        # ADX (trend strength)
        indicators['adx'] = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

        # CCI
        indicators['cci'] = talib.CCI(highs, lows, closes, timeperiod=14)[-1]

        # Volume analysis
        if len(volumes) > 20:
            indicators['volume_sma'] = np.mean(volumes[-20:])
            indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
        else:
            indicators['volume_sma'] = volumes[-1] if len(volumes) > 0 else 0
            indicators['volume_ratio'] = 1.0

        return indicators

    def determine_trend(self, indicators: Dict, bars: List) -> str:
        """Determina tendência baseada em múltiplos fatores"""

        # Análise de EMAs
        ema_signal = 0
        if indicators['ema_fast'] > indicators['ema_slow']:
            ema_signal = 1
        elif indicators['ema_fast'] < indicators['ema_slow']:
            ema_signal = -1

        # Análise de MACD
        macd_signal = 0
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
            macd_signal = 1
        elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
            macd_signal = -1

        # Análise de preço vs Bollinger Bands
        bb_signal = 0
        current_price = bars[-1]['close']
        if current_price > indicators['bb_upper']:
            bb_signal = 1  # Sobrecompra forte
        elif current_price < indicators['bb_lower']:
            bb_signal = -1  # Sobrevenda forte

        # Análise ADX (força da tendência)
        trend_strength = indicators['adx']
        if trend_strength < 20:
            return "neutral"  # Sem tendência definida

        # Combinar sinais
        total_signal = ema_signal + macd_signal

        if total_signal >= 1:
            return "bullish"
        elif total_signal <= -1:
            return "bearish"
        else:
            return "neutral"

    def calculate_volatility(self, closes: np.ndarray, period: int = 20) -> float:
        """Calcula volatilidade como desvio padrão dos retornos"""
        if len(closes) < period + 1:
            return 0.0

        returns = np.diff(closes[-period:]) / closes[-period:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada
        return volatility * 100  # Em percentagem

    def calculate_momentum(self, closes: np.ndarray, period: int = 10) -> float:
        """Calcula momentum como taxa de mudança percentual"""
        if len(closes) < period:
            return 0.0

        current_price = closes[-1]
        past_price = closes[-period]
        momentum = ((current_price - past_price) / past_price) * 100
        return momentum

    def find_key_levels(self, bars: List, lookback: int = 100) -> tuple:
        """Encontra níveis importantes de suporte e resistência"""

        if len(bars) < lookback:
            lookback = len(bars)

        highs = [bar['high'] for bar in bars[-lookback:]]
        lows = [bar['low'] for bar in bars[-lookback:]]

        # Encontrar picos e fundos significativos
        resistance_levels = []
        support_levels = []

        # Resistência: topos que se repetem
        high_counts = {}
        for high in highs:
            rounded = round(high, 2)
            high_counts[rounded] = high_counts.get(rounded, 0) + 1

        for price, count in high_counts.items():
            if count >= 3:  # Pelo menos 3 topos no mesmo nível
                resistance_levels.append(price)

        # Suporte: fundos que se repetem
        low_counts = {}
        for low in lows:
            rounded = round(low, 2)
            low_counts[rounded] = low_counts.get(rounded, 0) + 1

        for price, count in low_counts.items():
            if count >= 3:  # Pelo menos 3 fundos no mesmo nível
                support_levels.append(price)

        # Ordenar e limitar
        resistance_levels = sorted(resistance_levels, reverse=True)[:3]
        support_levels = sorted(support_levels)[:3]

        return support_levels, resistance_levels

    async def generate_signal(self, state: MarketState) -> Optional[Signal]:
        """Gera sinal de trading baseado na análise completa"""

        try:
            # Obter indicadores atualizados
            market_data = await self.collect_market_data()
            h1_bars = market_data.get('H1', [])
            closes = np.array([bar['close'] for bar in h1_bars])
            highs = np.array([bar['high'] for bar in h1_bars])
            lows = np.array([bar['low'] for bar in h1_bars])
            volumes = np.array([bar['volume'] for bar in h1_bars])

            indicators = await self.calculate_all_indicators(closes, highs, lows, volumes)

            # Calcular score de cada componente
            scores = {}

            # RSI Score
            rsi = indicators['rsi']
            if 30 <= rsi <= 70:
                if rsi < 40:
                    scores['rsi'] = 70 + (40 - rsi)  # Oversold
                elif rsi > 60:
                    scores['rsi'] = 70 - (rsi - 60)  # Overbought
                else:
                    scores['rsi'] = 50  # Neutral
            else:
                scores['rsi'] = 30  # Extremos (risco)

            # MACD Score
            macd_hist = indicators['macd_histogram']
            if macd_hist > 0:
                scores['macd'] = min(90, 60 + abs(macd_hist) * 1000)
            else:
                scores['macd'] = max(10, 40 - abs(macd_hist) * 1000)

            # EMA Score
            ema_fast = indicators['ema_fast']
            ema_slow = indicators['ema_slow']
            current_price = state.price

            if current_price > ema_fast > ema_slow:
                scores['ema'] = 80
            elif current_price > ema_fast > ema_slow * 0.998:
                scores['ema'] = 65
            elif current_price < ema_fast < ema_slow:
                scores['ema'] = 20
            elif current_price < ema_fast < ema_slow * 1.002:
                scores['ema'] = 35
            else:
                scores['ema'] = 50

            # Bollinger Bands Score
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            bb_width = (bb_upper - bb_lower) / state.price

            if current_price <= bb_lower:
                scores['bbands'] = 80  # Oversold
            elif current_price >= bb_upper:
                scores['bbands'] = 20  # Overbought
            else:
                # Posição dentro das bandas
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                scores['bbands'] = 50 - (bb_position - 0.5) * 40

            # Volume Score
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 1.5:
                scores['volume'] = 70  # Alto volume
            elif volume_ratio > 1.2:
                scores['volume'] = 60
            elif volume_ratio < 0.5:
                scores['volume'] = 30  # Baixo volume
            else:
                scores['volume'] = 50

            # Volatility Score
            if state.volatility > 30:
                scores['volatility'] = 20  # Muito volátil (risco)
            elif state.volatility > 20:
                scores['volatility'] = 40
            elif state.volatility < 10:
                scores['volatility'] = 30  # Pouca volatilidade
            else:
                scores['volatility'] = 70  # Ideal

            # Session Score
            session_scores = {
                "Overlap": 80,  # Melhor sessão
                "London": 70,
                "NY": 60,
                "Asian": 40
            }
            scores['session'] = session_scores.get(state.session, 30)

            # Calcular score total ponderado
            total_score = sum(scores[key] * self.indicators_weights[key] for key in scores)

            # Determinar direção do sinal
            if total_score > 65:
                action = "BUY"
            elif total_score < 35:
                action = "SELL"
            else:
                action = "HOLD"

            if action == "HOLD":
                return None

            # Calcular níveis de SL/TP
            atr = indicators['atr']
            volatility_adjustment = max(0.5, min(2.0, state.volatility / 20))

            if action == "BUY":
                stop_loss = state.price - (atr * 1.5 * volatility_adjustment)
                take_profit = state.price + (atr * 2.5 * volatility_adjustment)
            else:  # SELL
                stop_loss = state.price + (atr * 1.5 * volatility_adjustment)
                take_profit = state.price - (atr * 2.5 * volatility_adjustment)

            # Ajustar SL/TP baseado em níveis chave
            if action == "BUY" and state.resistance_levels:
                nearest_resistance = min(state.resistance_levels, key=lambda x: abs(x - state.price))
                if nearest_resistance > state.price:
                    take_profit = min(take_profit, nearest_resistance - 10)

            if action == "SELL" and state.support_levels:
                nearest_support = min(state.support_levels, key=lambda x: abs(x - state.price))
                if nearest_support < state.price:
                    take_profit = max(take_profit, nearest_support + 10)

            # Calcular risk/reward
            risk = abs(state.price - stop_loss)
            reward = abs(take_profit - state.price)
            risk_reward = reward / risk if risk > 0 else 1.0

            # Ajustar confiança baseada no risk/reward
            if risk_reward < 1.2:
                total_score *= 0.8  # Penalizar RR baixo

            confidence = max(50, min(95, total_score))

            # Gerar reasoning
            reasoning = self.generate_reasoning(scores, indicators, state, action)

            return Signal(
                action=action,
                confidence=confidence,
                entry_price=state.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                reasoning=reasoning,
                indicators={
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd_histogram'],
                    'atr': atr,
                    'volume_ratio': indicators['volume_ratio']
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ Erro na geração de sinal: {e}")
            return None

    def generate_reasoning(self, scores: Dict, indicators: Dict, state: MarketState, action: str) -> str:
        """Gera explicação detalhada do sinal"""

        reasons = []

        # Principais fatores
        top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        if action == "BUY":
            reasons.append("SINAL DE COMPRA GERADO")
        else:
            reasons.append("SINAL DE VENDA GERADO")

        reasons.append(f"Condição de mercado: {state.trend.upper()}")

        # Adicionar principais fatores
        for factor, score in top_factors:
            if score > 70:
                status = "Favorável"
            elif score < 40:
                status = "Desfavorável"
            else:
                status = "Neutro"

            factor_names = {
                'rsi': 'RSI',
                'macd': 'MACD',
                'ema': 'Médias Móveis',
                'bbands': 'Bollinger Bands',
                'volume': 'Volume',
                'volatility': 'Volatilidade',
                'session': 'Sessão de Trading'
            }

            reasons.append(f"{factor_names.get(factor, factor)}: {status} ({score:.0f}%)")

        # Adicionar contexto técnico
        if indicators['rsi'] < 30:
            reasons.append(f"RSI em sobrevenda ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > 70:
            reasons.append(f"RSI em sobrecompra ({indicators['rsi']:.1f})")

        if indicators['macd_histogram'] > 0:
            reasons.append("MACD com momentum positivo")
        else:
            reasons.append("MACD com momentum negativo")

        # Adicionar informação da sessão
        reasons.append(f"Sessão atual: {state.session}")

        return " | ".join(reasons)
```

### 3. Sistema de Backtesting Avançado

Crie arquivo `advanced_backtester.py`:

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

@dataclass
class TradeResult:
    """Resultado de um trade individual"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    volume: float
    type: str  # BUY/SELL
    profit: float
    pips: float
    duration_minutes: int
    exit_reason: str  # TP/SL/Manual

@dataclass
class BacktestResults:
    """Resultados completos do backtest"""
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    average_trade: float
    largest_win: float
    largest_loss: float
    average_win: float
    average_loss: float
    trades: List[TradeResult]

class AdvancedBacktester:
    """Sistema avançado de backtesting"""

    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity_curve = [initial_balance]
        self.open_positions = {}
        self.closed_trades = []
        self.trade_id_counter = 0

    async def run_backtest(self, strategy, historical_data: Dict, start_date: str, end_date: str) -> BacktestResults:
        """Executa backtest completo da estratégia"""

        try:
            logger.info(f"🧪 Iniciando backtest de {start_date} a {end_date}")

            # Resetar estado
            self.reset_backtest()

            # Processar dados barra por barra
            h1_bars = historical_data.get('H1', [])
            timestamps = [bar['time'] for bar in h1_bars]

            # Filtrar por período
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)

            filtered_bars = []
            for bar in h1_bars:
                bar_time = datetime.fromisoformat(bar['time'])
                if start_dt <= bar_time <= end_dt:
                    filtered_bars.append(bar)

            logger.info(f"📊 Processando {len(filtered_bars)} barras")

            # Simular trading barra por barra
            for i, bar in enumerate(filtered_bars):
                await self.process_bar(strategy, bar, historical_data, i)

                # Atualizar equity
                self.update_equity(bar)

                if i % 100 == 0:
                    progress = (i / len(filtered_bars)) * 100
                    logger.info(f"📈 Progresso: {progress:.1f}%")

            # Fechar posições abertas no final
            await self.close_all_positions(filtered_bars[-1])

            # Calcular resultados finais
            results = self.calculate_results()

            logger.info(f"✅ Backtest concluído")
            logger.info(f"📊 Total trades: {results.total_trades}")
            logger.info(f"💰 Lucro líquido: ${results.net_profit:.2f}")
            logger.info(f"📈 Win rate: {results.win_rate:.1f}%")

            return results

        except Exception as e:
            logger.error(f"❌ Erro no backtest: {e}")
            raise

    def reset_backtest(self):
        """Reseta estado do backtest"""
        self.current_balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.open_positions = {}
        self.closed_trades = []
        self.trade_id_counter = 0

    async def process_bar(self, strategy, bar: Dict, historical_data: Dict, bar_index: int):
        """Processa uma barra individual"""

        try:
            # Atualizar dados históricos com a barra atual
            current_data = self.update_historical_data(historical_data, bar_index)

            # Verificar posições abertas (SL/TP)
            await self.check_position_exits(bar)

            # Gerar sinal da estratégia
            state = await strategy.analyze_market(current_data)
            signal = await strategy.generate_signal(state)

            if signal and len(self.open_positions) < 2:  # Máximo 2 posições
                # Simular entrada
                await self.simulate_trade_entry(signal, bar)

        except Exception as e:
            logger.error(f"❌ Erro processando barra: {e}")

    def update_historical_data(self, historical_data: Dict, bar_index: int) -> Dict:
        """Atualiza dados históricos para análise"""
        updated_data = {}

        for tf, bars in historical_data.items():
            if tf == 'ticks':
                continue  # Pular ticks no backtest

            # Manter apenas barras até o índice atual
            if bar_index < len(bars):
                updated_data[tf] = bars[:bar_index + 1]

        return updated_data

    async def check_position_exits(self, bar: Dict):
        """Verifica se posições devem ser fechadas"""

        positions_to_close = []

        for trade_id, position in self.open_positions.items():
            current_price = bar['close']

            # Verificar take profit
            if position['type'] == 'BUY' and current_price >= position['take_profit']:
                positions_to_close.append((trade_id, 'TP', current_price))
            elif position['type'] == 'SELL' and current_price <= position['take_profit']:
                positions_to_close.append((trade_id, 'TP', current_price))

            # Verificar stop loss
            elif position['type'] == 'BUY' and current_price <= position['stop_loss']:
                positions_to_close.append((trade_id, 'SL', current_price))
            elif position['type'] == 'SELL' and current_price >= position['stop_loss']:
                positions_to_close.append((trade_id, 'SL', current_price))

        # Fechar posições identificadas
        for trade_id, reason, exit_price in positions_to_close:
            await self.close_position(trade_id, bar['time'], exit_price, reason)

    async def simulate_trade_entry(self, signal: 'Signal', bar: Dict):
        """Simula entrada em uma posição"""

        try:
            trade_id = self.trade_id_counter
            self.trade_id_counter += 1

            # Calcular volume (simplificado)
            risk_amount = self.current_balance * 0.01  # 1% de risco
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            volume = risk_amount / (sl_distance * 100)  # Simplificado

            position = {
                'id': trade_id,
                'type': signal.action,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'volume': volume,
                'entry_time': bar['time'],
                'signal': signal
            }

            self.open_positions[trade_id] = position

            logger.debug(f"📊 Entrada simulada: {signal.action} {volume:.3f} lotes")

        except Exception as e:
            logger.error(f"❌ Erro na simulação de entrada: {e}")

    async def close_position(self, trade_id: int, exit_time: str, exit_price: float, reason: str):
        """Fecha uma posição simulada"""

        if trade_id not in self.open_positions:
            return

        position = self.open_positions[trade_id]

        # Calcular lucro/perda
        if position['type'] == 'BUY':
            profit = (exit_price - position['entry_price']) * position['volume'] * 100
        else:  # SELL
            profit = (position['entry_price'] - exit_price) * position['volume'] * 100

        # Calcular pips
        if position['type'] == 'BUY':
            pips = (exit_price - position['entry_price']) * 100
        else:
            pips = (position['entry_price'] - exit_price) * 100

        # Calcular duração
        entry_dt = datetime.fromisoformat(position['entry_time'])
        exit_dt = datetime.fromisoformat(exit_time)
        duration = int((exit_dt - entry_dt).total_seconds() / 60)

        # Criar resultado do trade
        trade_result = TradeResult(
            entry_time=entry_dt,
            exit_time=exit_dt,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            volume=position['volume'],
            type=position['type'],
            profit=profit,
            pips=pips,
            duration_minutes=duration,
            exit_reason=reason
        )

        # Atualizar balanço
        self.current_balance += profit

        # Mover para trades fechados
        self.closed_trades.append(trade_result)
        del self.open_positions[trade_id]

        logger.debug(f"📊 Trade fechado: {reason} | Lucro: ${profit:.2f}")

    async def close_all_positions(self, last_bar: Dict):
        """Fecha todas as posições abertas no final do backtest"""

        for trade_id in list(self.open_positions.keys()):
            await self.close_position(trade_id, last_bar['time'], last_bar['close'], 'END_OF_BACKTEST')

    def update_equity(self, bar: Dict):
        """Atualiza curva de equity"""
        current_price = bar['close']
        floating_pnl = 0

        for position in self.open_positions.values():
            if position['type'] == 'BUY':
                floating_pnl += (current_price - position['entry_price']) * position['volume'] * 100
            else:
                floating_pnl += (position['entry_price'] - current_price) * position['volume'] * 100

        total_equity = self.current_balance + floating_pnl
        self.equity_curve.append(total_equity)

    def calculate_results(self) -> BacktestResults:
        """Calcula resultados finais do backtest"""

        if not self.closed_trades:
            return BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        # Métricas básicas
        total_trades = len(self.closed_trades)
        profitable_trades = len([t for t in self.closed_trades if t.profit > 0])
        losing_trades = total_trades - profitable_trades

        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

        # Profit/Loss
        profits = [t.profit for t in self.closed_trades if t.profit > 0]
        losses = [abs(t.profit) for t in self.closed_trades if t.profit < 0]

        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        net_profit = total_profit - total_loss

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        max_dd_percent = 0

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            drawdown_percent = (drawdown / peak) * 100 if peak > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_percent = drawdown_percent

        # Sharpe Ratio (simplificado)
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)

        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Trade statistics
        all_profits = [t.profit for t in self.closed_trades]
        average_trade = np.mean(all_profits) if all_profits else 0
        largest_win = max(all_profits) if all_profits else 0
        largest_loss = min(all_profits) if all_profits else 0

        average_win = np.mean(profits) if profits else 0
        average_loss = np.mean(losses) if losses else 0

        return BacktestResults(
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_dd_percent,
            sharpe_ratio=sharpe_ratio,
            average_trade=average_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_win=average_win,
            average_loss=average_loss,
            trades=self.closed_trades
        )

    def save_results(self, results: BacktestResults, filename: str = None):
        """Salva resultados do backtest em arquivo"""

        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Preparar dados para serialização
        data = asdict(results)
        data['trades'] = [asdict(trade) for trade in results.trades]
        data['equity_curve'] = self.equity_curve
        data['initial_balance'] = self.initial_balance

        # Converter datetime para string
        for trade in data['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"📄 Resultados salvos em: {filename}")
        return filename
```

### 4. Otimização de Parâmetros

Crie arquivo `strategy_optimizer.py`:

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from datetime import datetime

class StrategyOptimizer:
    """Otimizador de parâmetros de estratégia"""

    def __init__(self, strategy_class, historical_data: Dict):
        self.strategy_class = strategy_class
        self.historical_data = historical_data
        self.optimization_results = []

    def define_parameter_space(self) -> Dict[str, List]:
        """Define o espaço de parâmetros para otimização"""

        return {
            'risk_per_trade': [0.5, 0.75, 1.0, 1.25, 1.5],
            'rsi_period': [10, 14, 18],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'ema_fast': [7, 9, 11],
            'ema_slow': [19, 21, 23],
            'atr_multiplier': [1.2, 1.5, 1.8],
            'bb_period': [18, 20, 22],
            'bb_std': [1.8, 2.0, 2.2],
            'volume_threshold': [1.2, 1.5, 1.8]
        }

    def generate_parameter_combinations(self, param_space: Dict, max_combinations: int = 100) -> List[Dict]:
        """Gera combinações de parâmetros para teste"""

        keys = list(param_space.keys())
        values = list(param_space.values())

        # Gerar todas as combinações
        all_combinations = list(itertools.product(*values))

        # Se houver muitas combinações, amostrar aleatoriamente
        if len(all_combinations) > max_combinations:
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]

        # Converter para lista de dicionários
        combinations = []
        for combo in all_combinations:
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def calculate_fitness_score(self, results: 'BacktestResults') -> float:
        """Calcula score de fitness para combinação de parâmetros"""

        # Critérios e pesos
        criteria = {
            'net_profit': 0.25,
            'profit_factor': 0.20,
            'win_rate': 0.15,
            'sharpe_ratio': 0.20,
            'max_drawdown': -0.20,  # Negativo (menor é melhor)
        }

        # Normalizar métricas para 0-100
        scores = {}

        # Net profit (escala logarítmica)
        if results.net_profit > 0:
            scores['net_profit'] = min(100, np.log10(abs(results.net_profit) + 1) * 20)
        else:
            scores['net_profit'] = 0

        # Profit factor
        if results.profit_factor > 1:
            scores['profit_factor'] = min(100, results.profit_factor * 25)
        else:
            scores['profit_factor'] = 0

        # Win rate
        scores['win_rate'] = results.win_rate

        # Sharpe ratio
        scores['sharpe_ratio'] = min(100, max(0, results.sharpe_ratio * 25))

        # Max drawdown (invertido - menor é melhor)
        scores['max_drawdown'] = max(0, 100 - results.max_drawdown_percent * 2)

        # Calcular score ponderado
        total_score = sum(criteria[key] * scores[key] for key in criteria)
        return max(0, min(100, total_score))

    async def optimize_single_combination(self, params: Dict, start_date: str, end_date: str) -> Dict:
        """Otimiza uma única combinação de parâmetros"""

        try:
            # Criar estratégia com parâmetros personalizados
            strategy = self.strategy_class()

            # Aplicar parâmetros
            if 'risk_per_trade' in params:
                strategy.risk_per_trade = params['risk_per_trade']
            if 'rsi_period' in params:
                strategy.rsi_period = params['rsi_period']
            if 'ema_fast' in params:
                strategy.ema_fast = params['ema_fast']
            if 'ema_slow' in params:
                strategy.ema_slow = params['ema_slow']
            # ... outros parâmetros

            # Executar backtest
            backtester = AdvancedBacktester()
            results = await backtester.run_backtest(
                strategy, self.historical_data, start_date, end_date
            )

            # Calcular fitness score
            fitness = self.calculate_fitness_score(results)

            return {
                'parameters': params,
                'results': results,
                'fitness_score': fitness,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Erro na otimização: {e}")
            return {
                'parameters': params,
                'error': str(e),
                'fitness_score': 0
            }

    async def run_optimization(self, start_date: str, end_date: str, max_combinations: int = 50) -> List[Dict]:
        """Executa otimização completa"""

        logger.info("🔧 Iniciando otimização de parâmetros...")

        # Definir espaço de parâmetros
        param_space = self.define_parameter_space()

        # Gerar combinações
        combinations = self.generate_parameter_combinations(param_space, max_combinations)

        logger.info(f"📋 Testando {len(combinations)} combinações de parâmetros")

        # Executar otimização
        results = []
        for i, params in enumerate(combinations, 1):
            logger.info(f"🧪 Testando combinação {i}/{len(combinations)}")
            result = await self.optimize_single_combination(params, start_date, end_date)
            results.append(result)

        # Ordenar por fitness score
        results.sort(key=lambda x: x['fitness_score'], reverse=True)

        # Salvar resultados
        self.optimization_results = results
        await self.save_optimization_results()

        logger.info(f"✅ Otimização concluída")
        logger.info(f"🏆 Melhor score: {results[0]['fitness_score']:.2f}")

        return results

    async def save_optimization_results(self, filename: str = None):
        """Salva resultados da otimização"""

        if filename is None:
            filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Preparar dados para serialização
        data = []
        for result in self.optimization_results:
            if 'error' not in result:
                item = {
                    'parameters': result['parameters'],
                    'fitness_score': result['fitness_score'],
                    'net_profit': result['results'].net_profit,
                    'win_rate': result['results'].win_rate,
                    'profit_factor': result['results'].profit_factor,
                    'sharpe_ratio': result['results'].sharpe_ratio,
                    'max_drawdown': result['results'].max_drawdown_percent,
                    'total_trades': result['results'].total_trades
                }
            else:
                item = {
                    'parameters': result['parameters'],
                    'fitness_score': 0,
                    'error': result['error']
                }
            data.append(item)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"📄 Resultados da otimização salvos em: {filename}")
        return filename

    def get_best_parameters(self, top_n: int = 1) -> List[Dict]:
        """Retorna os melhores parâmetros encontrados"""

        if not self.optimization_results:
            return []

        valid_results = [r for r in self.optimization_results if 'error' not in r]
        return valid_results[:top_n]

    def analyze_parameter_sensitivity(self) -> Dict:
        """Analisa sensibilidade dos parâmetros"""

        if not self.optimization_results:
            return {}

        # Coletar todos os parâmetros válidos
        valid_results = [r for r in self.optimization_results if 'error' not in r]

        if not valid_results:
            return {}

        # Analisar cada parâmetro
        analysis = {}

        # Obter todos os nomes de parâmetros
        param_names = set()
        for result in valid_results:
            param_names.update(result['parameters'].keys())

        for param in param_names:
            param_values = []
            fitness_scores = []

            for result in valid_results:
                if param in result['parameters']:
                    param_values.append(result['parameters'][param])
                    fitness_scores.append(result['fitness_score'])

            if param_values:
                # Calcular correlação
                correlation = np.corrcoef(param_values, fitness_scores)[0, 1]

                analysis[param] = {
                    'correlation': correlation,
                    'importance': abs(correlation),
                    'best_value': param_values[np.argmax(fitness_scores)],
                    'range': [min(param_values), max(param_values)]
                }

        # Ordenar por importância
        analysis = dict(sorted(analysis.items(), key=lambda x: x[1]['importance'], reverse=True))

        return analysis
```

### 5. Exemplo de Uso Completo

Crie arquivo `run_advanced_strategy.py`:

```python
import asyncio
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

from ea_scalper_sdk import MT5Client
from advanced_scalping_strategy import AdvancedXAUUSDScalper
from advanced_backtester import AdvancedBacktester
from strategy_optimizer import StrategyOptimizer

async def main():
    """Execução completa do sistema avançado"""

    print("🚀 Sistema Avançado de Trading - EA_SCALPER_XAUUSD")
    print("=" * 60)

    # Carregar configuração
    load_dotenv()

    # Inicializar clientes
    mt5_client = MT5Client()

    # Conectar ao MT5
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')

    success = await mt5_client.connect(login, password, server)
    if not success:
        print("❌ Falha na conexão MT5")
        return

    try:
        # 1. Coletar dados históricos
        print("\n📊 Coletando dados históricos...")
        historical_data = await collect_historical_data(mt5_client)

        # 2. Definir período de backtest
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        print(f"📅 Período de teste: {start_date} a {end_date}")

        # 3. Otimizar parâmetros
        print("\n🔧 Otimizando parâmetros da estratégia...")
        optimizer = StrategyOptimizer(AdvancedXAUUSDScalper, historical_data)
        optimization_results = await optimizer.run_optimization(start_date, end_date, max_combinations=20)

        # 4. Obter melhores parâmetros
        best_params = optimizer.get_best_parameters(1)[0]['parameters']
        print(f"🏆 Melhores parâmetros encontrados:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        # 5. Executar backtest com melhores parâmetros
        print("\n🧪 Executando backtest final com melhores parâmetros...")
        strategy = AdvancedXAUUSDScalper()

        # Aplicar melhores parâmetros
        for param, value in best_params.items():
            if hasattr(strategy, param):
                setattr(strategy, param, value)

        backtester = AdvancedBacktester()
        final_results = await backtester.run_backtest(strategy, historical_data, start_date, end_date)

        # 6. Exibir resultados
        print(f"\n📊 RESULTADOS FINAIS:")
        print(f"   Total de trades: {final_results.total_trades}")
        print(f"   Win rate: {final_results.win_rate:.1f}%")
        print(f"   Lucro líquido: ${final_results.net_profit:.2f}")
        print(f"   Profit factor: {final_results.profit_factor:.2f}")
        print(f"   Sharpe ratio: {final_results.sharpe_ratio:.2f}")
        print(f"   Max drawdown: {final_results.max_drawdown_percent:.1f}%")

        # 7. Análise de sensibilidade
        print("\n📈 Análise de sensibilidade dos parâmetros:")
        sensitivity = optimizer.analyze_parameter_sensitivity()
        for param, analysis in list(sensitivity.items())[:5]:
            print(f"   {param}: importância {analysis['importance']:.3f}")

        # 8. Salvar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backtester.save_results(final_results, f"final_backtest_{timestamp}.json")

        # 9. Validar compliance FTMO
        ftmo_compliance = validate_ftmo_compliance(final_results)
        print(f"\n✅ Validação FTMO: {ftmo_compliance['status']}")
        if ftmo_compliance['status'] == 'APPROVED':
            print("🎉 Estratégia pronta para trading real!")
        else:
            print("⚠️ Estratégia precisa de ajustes para compliance FTMO")

    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await mt5_client.disconnect()

async def collect_historical_data(mt5_client: MT5Client) -> Dict:
    """Coleta dados históricos múltiplos timeframes"""

    timeframes = ["M5", "M15", "H1", "H4", "D1"]
    data = {}

    for tf in timeframes:
        bars = await mt5_client.get_bars("XAUUSD", tf, 1000)
        if bars:
            data[tf] = bars
            print(f"   ✅ {tf}: {len(bars)} barras")

    return data

def validate_ftmo_compliance(results: 'BacktestResults') -> Dict:
    """Valida compliance com regras FTMO"""

    checks = {
        'max_daily_loss': results.max_drawdown_percent <= 5.0,
        'max_total_loss': results.max_drawdown_percent <= 10.0,
        'min_profit_target': results.net_profit >= 1000,  # $1000 em $10k
        'min_trades': results.total_trades >= 10,
        'win_rate': results.win_rate >= 40,
        'profit_factor': results.profit_factor >= 1.2
    }

    passed = sum(checks.values())
    total = len(checks)

    if passed == total:
        status = "APPROVED"
    elif passed >= total * 0.8:
        status = "PARTIAL"
    else:
        status = "FAILED"

    return {
        'status': status,
        'score': (passed / total) * 100,
        'checks': checks,
        'details': {
            'max_drawdown': results.max_drawdown_percent,
            'total_profit': results.net_profit,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor
        }
    }

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusão

Neste tutorial avançado, você aprendeu:

1. **Framework robusto** para desenvolvimento de estratégias
2. **Análise técnica multi-timeframe** sofisticada
3. **Gestão de risco profissional** FTMO-compliant
4. **Sistema de backtesting completo** com métricas detalhadas
5. **Otimização de parâmetros** sistemática
6. **Validação de compliance** com regras FTMO

### Próximos Passos

1. **Customizar indicadores** para suas preferências
2. **Adicionar filtros fundamentais** (notícias, eventos)
3. **Implementar machine learning** para previsão
4. **Desenvolver dashboard** de monitoramento
5. **Configurar alertas** e notificações

### Melhores Práticas

- **Teste extensivamente** antes de usar capital real
- **Monitore performance** continuamente
- **Ajuste parâmetros** conforme necessário
- **Mantenha disciplina** na gestão de risco
- **Documente tudo** para análise futura

Seu sistema avançado está pronto para desenvolver estratégias sofisticadas e lucrativas!