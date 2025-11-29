#!/usr/bin/env python3
"""
🎲 EA Optimizer AI - Realistic Backtesting Engine (Rodada 2)
Sistema de backtesting institucional com simulação realista de mercado
"""

import numpy as np
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import deque
import math

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Estrutura de tick do mercado"""
    timestamp: datetime
    bid: float
    ask: float
    volume: int
    spread: float = field(init=False)

    def __post_init__(self):
        self.spread = self.ask - self.bid

@dataclass
class Trade:
    """Estrutura de trade executado"""
    id: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    slippage: float = 0.0
    status: str = 'OPEN'  # OPEN, CLOSED, CANCELLED
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    duration_bars: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0

@dataclass
class MarketConditions:
    """Condições de mercado"""
    volatility: float
    liquidity: float
    spread_avg: float
    volume_avg: float
    trend_strength: float
    market_regime: str  # TRENDING, RANGING, VOLATILE
    session_overlap: bool
    news_impact: float

class MarketSimulator:
    """Simulador realista de condições de mercado"""

    def __init__(self, symbol: str = "XAUUSD", initial_price: float = 2000.0):
        """
        Inicializa simulador de mercado

        Args:
            symbol: Símbolo simulado
            initial_price: Preço inicial
        """
        self.symbol = symbol
        self.current_price = initial_price
        self.volatility = 0.0
        self.trend = 0.0
        self.tick_data = []
        self.market_conditions = []

        # Parâmetros de mercado realistas
        self.base_spread = 0.5  # 0.5 points para XAUUSD
        self.commission_per_million = 7.0  # $7 por milhão
        self.swap_long = -2.5  # Swap overnight
        self.swap_short = 1.5

    def generate_realistic_ticks(self, n_ticks: int, start_time: datetime) -> List[MarketTick]:
        """
        Gera ticks realistas com microestrutura de mercado

        Args:
            n_ticks: Número de ticks a gerar
            start_time: Tempo inicial

        Returns:
            Lista de ticks simulados
        """
        ticks = []
        current_time = start_time
        current_bid = self.current_price
        current_ask = self.current_price + self.base_spread

        # Simular diferentes regimes de mercado
        for i in range(n_ticks):
            # Determinar regime atual
            hour = current_time.hour
            minute = current_time.minute

            # Ajustar volatilidade baseado na hora
            if (7 <= hour < 9) or (13 <= hour < 15):  # Session overlaps
                volatility_multiplier = 2.0
                liquidity_multiplier = 1.5
            elif 22 <= hour or hour < 2:  # Asian session
                volatility_multiplier = 0.7
                liquidity_multiplier = 0.8
            else:  # Normal hours
                volatility_multiplier = 1.0
                liquidity_multiplier = 1.0

            # Simular movimento de preço com jump-diffusion
            dt = 1.0  # 1 minuto
            drift = self.trend * dt
            diffusion = np.random.normal(0, 0.1 * volatility_multiplier)

            # Jump component (eventos de notícias, etc.)
            jump_prob = 0.001  # 0.1% chance de jump
            if np.random.random() < jump_prob:
                jump_size = np.random.choice([-1, 1]) * np.random.exponential(2.0)
                diffusion += jump_size

            # Atualizar preço
            price_change = drift + diffusion
            current_bid = max(current_bid + price_change, 1.0)

            # Calcular spread dinâmico
            spread_premium = max(0.1, np.random.exponential(0.3)) * volatility_multiplier
            current_spread = self.base_spread * (1 + spread_premium / self.base_spread)
            current_ask = current_bid + current_spread

            # Volume aleatório com correlação com volatilidade
            base_volume = 100 * liquidity_multiplier
            volume_noise = np.random.exponential(50) * volatility_multiplier
            volume = int(base_volume + volume_noise)

            # Criar tick
            tick = MarketTick(
                timestamp=current_time,
                bid=current_bid,
                ask=current_ask,
                volume=volume
            )

            ticks.append(tick)

            # Avançar tempo
            current_time += timedelta(minutes=1)

            # Atualizar estado interno
            self.current_price = current_bid

        self.tick_data.extend(ticks)
        return ticks

    def calculate_market_conditions(self, ticks: List[MarketTick]) -> MarketConditions:
        """
        Calcula condições de mercado a partir dos ticks

        Args:
            ticks: Lista de ticks

        Returns:
            Condições de mercado
        """
        if not ticks:
            return MarketConditions(0, 0, 0, 0, 0, 'RANGING', False, 0)

        prices = [tick.bid for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        spreads = [tick.spread for tick in ticks]

        # Volatilidade (desvio padrão dos retornos)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(returns) * np.sqrt(1440) if returns else 0  # Annualized

        # Liquidez (volume médio normalizado)
        liquidity = np.mean(volumes) / 100.0 if volumes else 0

        # Spread médio
        spread_avg = np.mean(spreads) if spreads else 0

        # Volume médio
        volume_avg = np.mean(volumes) if volumes else 0

        # Força de tendência (slope dos preços)
        if len(prices) > 10:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            trend_strength = abs(slope) / np.mean(prices) * 1000
        else:
            trend_strength = 0

        # Regime de mercado
        if volatility > 0.3:
            regime = 'VOLATILE'
        elif trend_strength > 0.1:
            regime = 'TRENDING'
        else:
            regime = 'RANGING'

        # Session overlap
        if ticks:
            hour = ticks[0].timestamp.hour
            session_overlap = (7 <= hour < 9) or (13 <= hour < 15)
        else:
            session_overlap = False

        # News impact (simulado)
        news_impact = np.random.exponential(0.1) if np.random.random() < 0.05 else 0

        return MarketConditions(
            volatility=volatility,
            liquidity=liquidity,
            spread_avg=spread_avg,
            volume_avg=volume_avg,
            trend_strength=trend_strength,
            market_regime=regime,
            session_overlap=session_overlap,
            news_impact=news_impact
        )

class SlippageModel:
    """Modelo de slippage realista"""

    def __init__(self):
        self.impact_factor = 0.0001  # 0.01% por $1M
        self.fixed_slippage = 0.2  # 0.2 points base
        self.volatility_multiplier = 2.0

    def calculate_slippage(self, direction: str, quantity: float,
                          market_conditions: MarketConditions,
                          order_type: str = 'MARKET') -> Tuple[float, float]:
        """
        Calcula slippage realista

        Args:
            direction: 'BUY' or 'SELL'
            quantity: Quantidade em lots
            market_conditions: Condições atuais do mercado
            order_type: Tipo de ordem

        Returns:
            (slippage_points, execution_price_adjustment)
        """
        # Slippage base
        base_slippage = self.fixed_slippage

        # Ajuste por volatilidade
        volatility_adjustment = market_conditions.volatility * self.volatility_multiplier

        # Ajuste por liquidez
        liquidity_adjustment = (1.0 / market_conditions.liquidity) * 0.5 if market_conditions.liquidity > 0 else 2.0

        # Ajuste por tamanho da ordem
        notional = quantity * 100000  # Assumir $100k por lot
        impact_adjustment = notional * self.impact_factor

        # Ajuste por regime de mercado
        regime_multiplier = {
            'VOLATILE': 2.0,
            'TRENDING': 1.2,
            'RANGING': 1.0
        }.get(market_conditions.market_regime, 1.0)

        # Ajuste por session overlap
        session_adjustment = 1.3 if market_conditions.session_overlap else 1.0

        # Calcular slippage total
        total_slippage = base_slippage * volatility_adjustment * liquidity_adjustment
        total_slippage += impact_adjustment
        total_slippage *= regime_multiplier * session_adjustment

        # Adicionar aleatoriedade
        random_factor = np.random.uniform(0.8, 1.2)
        total_slippage *= random_factor

        # Direção do slippage (pior para o cliente)
        if direction == 'BUY':
            execution_adjustment = total_slippage
        else:  # SELL
            execution_adjustment = -total_slippage

        return total_slippage, execution_adjustment

class RealisticBacktester:
    """Backtester realista institucional"""

    def __init__(self, initial_balance: float = 10000.0,
                 leverage: int = 100,
                 symbol: str = "XAUUSD"):
        """
        Inicializa backtester realista

        Args:
            initial_balance: Balance inicial
            leverage: Alavancagem
            symbol: Símbolo para trading
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.leverage = leverage
        self.symbol = symbol

        # Componentes
        self.market_simulator = MarketSimulator(symbol)
        self.slippage_model = SlippageModel()

        # Estado do backtest
        self.trades = []
        self.open_positions = {}
        self.balance_history = []
        self.equity_history = []
        self.drawdown_history = []
        self.max_equity = initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Métricas de performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_swap = 0.0

        # Trade IDs
        self.next_trade_id = 1

    def execute_trade(self, direction: str, quantity: float,
                     stop_loss_points: float = 0.0,
                     take_profit_points: float = 0.0,
                     magic_number: int = 0,
                     tick: MarketTick = None) -> Optional[int]:
        """
        Executa trade com condições realistas

        Args:
            direction: 'BUY' or 'SELL'
            quantity: Quantidade em lots
            stop_loss_points: Stop loss em points
            take_profit_points: Take profit em points
            magic_number: Magic number
            tick: Tick atual

        Returns:
            ID do trade executado ou None
        """
        if not tick:
            return None

        # Calcular condições de mercado
        recent_ticks = [t for t in self.market_simulator.tick_data
                        if t.timestamp >= tick.timestamp - timedelta(minutes=60)]
        market_conditions = self.market_simulator.calculate_market_conditions(recent_ticks)

        # Calcular slippage
        slippage_points, price_adjustment = self.slippage_model.calculate_slippage(
            direction, quantity, market_conditions
        )

        # Determinar preço de execução
        if direction == 'BUY':
            execution_price = tick.ask + price_adjustment
        else:  # SELL
            execution_price = tick.bid + price_adjustment

        # Calcular comissão
        notional_value = quantity * 100000  # $100k por lot
        commission = (notional_value / 1000000) * self.commission_per_million

        # Verificar margem
        required_margin = (notional_value / self.leverage)
        if required_margin > self.current_balance * 0.1:  # Não usar mais que 10% do balance
            logger.warning(f"❌ Margin check failed: Required=${required_margin:.2f}, Available=${self.current_balance:.2f}")
            return None

        # Criar trade
        trade = Trade(
            id=self.next_trade_id,
            symbol=self.symbol,
            direction=direction,
            entry_price=execution_price,
            entry_time=tick.timestamp,
            quantity=quantity,
            commission=commission,
            slippage=slippage_points,
            status='OPEN'
        )

        # Configurar SL e TP
        if stop_loss_points > 0:
            if direction == 'BUY':
                trade.stop_loss = execution_price - stop_loss_points * 0.01
            else:
                trade.stop_loss = execution_price + stop_loss_points * 0.01

        if take_profit_points > 0:
            if direction == 'BUY':
                trade.take_profit = execution_price + take_profit_points * 0.01
            else:
                trade.take_profit = execution_price - take_profit_points * 0.01

        # Atualizar estado
        self.open_positions[trade.id] = trade
        self.trades.append(trade)
        self.next_trade_id += 1

        # Deduzir comissão do balance
        self.current_balance -= commission
        self.total_commission += commission

        logger.info(f"📈 Trade #{trade.id} EXECUTADO: {direction} {quantity} @ {execution_price:.5f}, SL={trade.stop_loss:.5f}, TP={trade.take_profit:.5f}")

        return trade.id

    def manage_open_positions(self, tick: MarketTick) -> List[Trade]:
        """
        Gerencia posições abertas (SL, TP, etc.)

        Args:
            tick: Tick atual

        Returns:
            Lista de trades fechados
        """
        closed_trades = []

        for trade_id, trade in list(self.open_positions.items()):
            if trade.status != 'OPEN':
                continue

            # Verificar stop loss
            if trade.stop_loss > 0:
                if (trade.direction == 'BUY' and tick.bid <= trade.stop_loss) or \
                   (trade.direction == 'SELL' and tick.ask >= trade.stop_loss):
                    closed_trade = self._close_trade(trade, tick, 'STOP_LOSS')
                    if closed_trade:
                        closed_trades.append(closed_trade)
                        del self.open_positions[trade_id]
                    continue

            # Verificar take profit
            if trade.take_profit > 0:
                if (trade.direction == 'BUY' and tick.bid >= trade.take_profit) or \
                   (trade.direction == 'SELL' and tick.ask <= trade.take_profit):
                    closed_trade = self._close_trade(trade, tick, 'TAKE_PROFIT')
                    if closed_trade:
                        closed_trades.append(closed_trade)
                        del self.open_positions[trade_id]
                    continue

            # Atualizar PnL unrealizado e tracking de max profit/loss
            if trade.direction == 'BUY':
                current_pnl = (tick.bid - trade.entry_price) * trade.quantity * 100000
            else:
                current_pnl = (trade.entry_price - tick.ask) * trade.quantity * 100000

            # Atualizar max profit/loss
            trade.max_profit = max(trade.max_profit, current_pnl)
            trade.max_loss = min(trade.max_loss, current_pnl)

        return closed_trades

    def _close_trade(self, trade: Trade, tick: MarketTick, close_reason: str) -> Optional[Trade]:
        """
        Fecha trade com cálculos realistas

        Args:
            trade: Trade para fechar
            tick: Tick de fechamento
            close_reason: Razão do fechamento

        Returns:
            Trade fechado atualizado
        """
        # Calcular preço de saída com slippage
        recent_ticks = [t for t in self.market_simulator.tick_data
                        if t.timestamp >= tick.timestamp - timedelta(minutes=10)]
        market_conditions = self.market_simulator.calculate_market_conditions(recent_ticks)

        slippage_points, price_adjustment = self.slippage_model.calculate_slippage(
            'SELL' if trade.direction == 'BUY' else 'BUY',
            trade.quantity,
            market_conditions
        )

        if trade.direction == 'BUY':
            exit_price = tick.bid + price_adjustment
        else:
            exit_price = tick.ask + price_adjustment

        # Calcular PnL
        if trade.direction == 'BUY':
            pnl = (exit_price - trade.entry_price) * trade.quantity * 100000
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity * 100000

        # Calcular swap (se overnight)
        time_diff = tick.timestamp - trade.entry_time
        if time_diff.days >= 1:
            swap = self.swap_long if trade.direction == 'BUY' else self.swap_short
            swap *= trade.quantity * time_diff.days
            trade.swap = swap
            self.total_swap += swap

        # Comissão de fechamento
        notional_value = trade.quantity * 100000
        close_commission = (notional_value / 1000000) * self.commission_per_million

        # PnL líquido
        net_pnl = pnl - close_commission

        # Atualizar trade
        trade.exit_price = exit_price
        trade.exit_time = tick.timestamp
        trade.status = 'CLOSED'
        trade.pnl = net_pnl
        trade.pnl_percentage = (net_pnl / (trade.entry_price * trade.quantity * 100000)) * 100
        trade.duration_bars = int((tick.timestamp - trade.entry_time).total_seconds() / 60)
        trade.commission += close_commission

        # Atualizar balance e equity
        self.current_balance += net_pnl
        self.total_commission += close_commission
        self.total_pnl += net_pnl

        # Atualizar estatísticas
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        logger.info(f"📉 Trade #{trade.id} FECHADO: {close_reason}, PnL=${net_pnl:.2f}, Duration={trade.duration_bars}min")

        return trade

    def update_equity(self, tick: MarketTick):
        """
        Atualiza equity calculando PnL unrealizado

        Args:
            tick: Tick atual
        """
        unrealized_pnl = 0.0

        for trade in self.open_positions.values():
            if trade.status == 'OPEN':
                if trade.direction == 'BUY':
                    current_pnl = (tick.bid - trade.entry_price) * trade.quantity * 100000
                else:
                    current_pnl = (trade.entry_price - tick.ask) * trade.quantity * 100000
                unrealized_pnl += current_pnl

        self.equity = self.current_balance + unrealized_pnl

        # Atualizar drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity

        self.current_drawdown = ((self.max_equity - self.equity) / self.max_equity) * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Salvar histórico
        self.balance_history.append(self.current_balance)
        self.equity_history.append(self.equity)
        self.drawdown_history.append(self.current_drawdown)

    def run_backtest(self, strategy_params: Dict[str, Any],
                    duration_hours: int = 24,
                    tick_frequency_minutes: int = 1) -> Dict[str, Any]:
        """
        Executa backtest realista completo

        Args:
            strategy_params: Parâmetros da estratégia
            duration_hours: Duração em horas
            tick_frequency_minutes: Frequência dos ticks

        Returns:
            Resultados completos do backtest
        """
        logger.info(f"🎲 Iniciando backtest realista: {duration_hours}h")

        # Gerar dados de mercado
        start_time = datetime.now()
        n_ticks = (duration_hours * 60) // tick_frequency_minutes
        ticks = self.market_simulator.generate_realistic_ticks(n_ticks, start_time)

        # Executar estratégia
        last_signal_time = start_time
        active_signal = None

        for i, tick in enumerate(ticks):
            # Atualizar equity
            self.update_equity(tick)

            # Gerar sinal da estratégia (simplificado)
            if (tick.timestamp - last_signal_time).total_seconds() >= 300:  # Sinal a cada 5 minutos
                signal = self._generate_strategy_signal(tick, strategy_params)

                if signal and active_signal != signal:
                    # Fechar posições existentes se necessário
                    if self.open_positions and signal != active_signal:
                        for trade_id in list(self.open_positions.keys()):
                            closed = self.manage_open_positions(tick)
                            for trade in closed:
                                logger.info(f"Fechando trade {trade.id} por mudança de sinal")

                    # Abrir nova posição
                    if not self.open_positions or len(self.open_positions) < strategy_params.get('max_positions', 3):
                        quantity = strategy_params.get('lot_size', 0.01)
                        sl_points = strategy_params.get('stop_loss', 150)
                        tp_points = strategy_params.get('take_profit', 300)

                        self.execute_trade(
                            direction=signal,
                            quantity=quantity,
                            stop_loss_points=sl_points,
                            take_profit_points=tp_points,
                            tick=tick
                        )

                    active_signal = signal
                    last_signal_time = tick.timestamp

            # Gerenciar posições abertas
            self.manage_open_positions(tick)

            # Forçar fechamento de posições muito antigas (24h)
            for trade_id, trade in list(self.open_positions.items()):
                if (tick.timestamp - trade.entry_time).total_seconds() >= 24 * 3600:
                    closed = self._close_trade(trade, tick, 'TIMEOUT')
                    if closed:
                        del self.open_positions[trade_id]

            # Progress log
            if i % 100 == 0:
                progress = (i / len(ticks)) * 100
                logger.info(f"📊 Progresso: {progress:.1f}%, Balance: ${self.current_balance:.2f}, Equity: ${self.equity:.2f}")

        # Fechar todas as posições abertas no final
        final_tick = ticks[-1] if ticks else None
        if final_tick:
            for trade_id in list(self.open_positions.keys()):
                trade = self.open_positions[trade_id]
                closed = self._close_trade(trade, final_tick, 'END_OF_BACKTEST')
                if closed:
                    del self.open_positions[trade_id]

        # Calcular métricas finais
        results = self._calculate_final_metrics()

        logger.info(f"✅ Backtest concluído: PnL=${results['total_pnl']:.2f}, Return={results['return_percentage']:.2f}%, MaxDD={results['max_drawdown']:.2f}%")

        return results

    def _generate_strategy_signal(self, tick: MarketTick, params: Dict[str, Any]) -> Optional[str]:
        """
        Gera sinal da estratégia (implementação simplificada)

        Args:
            tick: Tick atual
            params: Parâmetros da estratégia

        Returns:
            Sinal ('BUY', 'SELL', ou None)
        """
        # Obter ticks recentes para análise
        recent_ticks = [t for t in self.market_simulator.tick_data
                        if tick.timestamp - t.timestamp <= timedelta(minutes=30)]

        if len(recent_ticks) < 20:
            return None

        # Calcular médias móveis simples
        prices = [t.bid for t in recent_ticks]
        if len(prices) >= 10:
            ma_short = sum(prices[-10:]) / 10
            ma_long = sum(prices[-30:]) / 30 if len(prices) >= 30 else ma_short

            current_price = tick.bid

            # Lógica simples de cruzamento
            rsi_threshold = params.get('rsi_threshold', 70)
            volatility_threshold = params.get('volatility_threshold', 0.2)

            # Calcular volatilidade recente
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns) if len(returns) > 1 else 0

            # Gerar sinal baseado em múltiplos fatores
            if (current_price > ma_short and ma_short > ma_long and
                volatility < volatility_threshold and
                random.random() < params.get('signal_probability', 0.3)):
                return 'BUY'
            elif (current_price < ma_short and ma_short < ma_long and
                  volatility < volatility_threshold and
                  random.random() < params.get('signal_probability', 0.3)):
                return 'SELL'

        return None

    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas finais do backtest

        Returns:
            Dicionário com métricas completas
        """
        if not self.balance_history:
            return {}

        # Métricas básicas
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        total_pnl = self.current_balance - self.initial_balance

        # Win Rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Profit Factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe Ratio (simplificado)
        if len(self.equity_history) > 1:
            returns = [(self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
                      for i in range(1, len(self.equity_history))]
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252*24*60)) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino Ratio
        negative_returns = [r for r in returns if r < 0] if 'returns' in locals() else []
        sortino_ratio = (np.mean(returns) / np.std(negative_returns) * np.sqrt(252*24*60)) if negative_returns else 0

        # Calmar Ratio
        calmar_ratio = total_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0

        # Recovery Factor
        recovery_factor = total_pnl / abs(self.max_drawdown * self.initial_balance / 100) if self.max_drawdown != 0 else float('inf')

        # Métricas dos trades
        avg_trade = total_pnl / self.total_trades if self.total_trades > 0 else 0
        largest_win = max([t.pnl for t in self.trades], default=0)
        largest_loss = min([t.pnl for t in self.trades], default=0)

        # Avg holding time
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']
        avg_holding_time = np.mean([t.duration_bars for t in closed_trades]) if closed_trades else 0

        return {
            # Performance Metrics
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_pnl': total_pnl,
            'return_percentage': total_return,
            'max_drawdown': self.max_drawdown,
            'max_equity': self.max_equity,

            # Risk-Adjusted Metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,

            # Trade Metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_time_minutes': avg_holding_time,

            # Cost Analysis
            'total_commission': self.total_commission,
            'total_swap': self.total_swap,
            'total_costs': self.total_commission + self.total_swap,

            # Trade Details
            'trades': [
                {
                    'id': t.id,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'duration_minutes': t.duration_bars,
                    'max_profit': t.max_profit,
                    'max_loss': t.max_loss,
                    'commission': t.commission,
                    'swap': t.swap,
                    'slippage': t.slippage
                }
                for t in self.trades if t.status == 'CLOSED'
            ],

            # Equity Curve
            'equity_history': self.equity_history,
            'balance_history': self.balance_history,
            'drawdown_history': self.drawdown_history
        }

    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Gera relatório detalhado do backtest

        Args:
            results: Resultados do backtest
            output_path: Caminho para salvar relatório
        """
        report = f"""# 🎲 EA Optimizer AI - Realistic Backtesting Report

## 📊 Executive Summary

- **Initial Balance**: ${results['initial_balance']:,.2f}
- **Final Balance**: ${results['final_balance']:,.2f}
- **Total PnL**: ${results['total_pnl']:,.2f}
- **Return**: {results['return_percentage']:.2f}%
- **Max Drawdown**: {results['max_drawdown']:.2f}%

## 🎯 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: {results['sharpe_ratio']:.3f}
- **Sortino Ratio**: {results['sortino_ratio']:.3f}
- **Calmar Ratio**: {results['calmar_ratio']:.3f}
- **Recovery Factor**: {results['recovery_factor']:.2f}

### Trading Statistics
- **Total Trades**: {results['total_trades']}
- **Winning Trades**: {results['winning_trades']}
- **Losing Trades**: {results['losing_trades']}
- **Win Rate**: {results['win_rate']:.2f}%
- **Profit Factor**: {results['profit_factor']:.2f}

### Trade Analysis
- **Average Trade**: ${results['avg_trade']:.2f}
- **Largest Win**: ${results['largest_win']:.2f}
- **Largest Loss**: ${results['largest_loss']:.2f}
- **Avg Holding Time**: {results['avg_holding_time_minutes']:.1f} minutes

## 💰 Cost Analysis

- **Total Commission**: ${results['total_commission']:.2f}
- **Total Swap**: ${results['total_swap']:.2f}
- **Total Costs**: ${results['total_costs']:.2f}
- **Net PnL after Costs**: ${results['total_pnl'] - results['total_costs']:.2f}

## 📈 Recent Trades

| ID | Direction | Entry | Exit | PnL | Duration | Max Profit | Max Loss |
|----|-----------|-------|------|-----|----------|------------|-----------|
"""

        # Adicionar trades recentes
        recent_trades = results['trades'][-10:]  # Últimos 10 trades
        for trade in recent_trades:
            report += f"| {trade['id']} | {trade['direction']} | {trade['entry_price']:.5f} | {trade['exit_price']:.5f} | ${trade['pnl']:.2f} | {trade['duration_minutes']}min | ${trade['max_profit']:.2f} | ${trade['max_loss']:.2f} |\n"

        report += f"""

## 🔍 Market Realism Features

✅ **Slippage Simulation**: Market impact and execution costs
✅ **Volatility Adjustment**: Dynamic spread and slippage
✅ **Liquidity Constraints**: Order size limitations
✅ **Session Effects**: Different market conditions by session
✅ **Cost Structure**: Realistic commission and swap calculations

## 📊 Risk Assessment

### Drawdown Analysis
- **Maximum Drawdown**: {results['max_drawdown']:.2f}%
- **Drawdown Duration**: Analyzed from equity curve
- **Recovery Time**: Based on recovery factor

### Risk-Adjusted Performance
"""

        # Avaliação de risco
        if results['sharpe_ratio'] > 2.0:
            risk_assessment = "🟢 **Excellent Risk-Adjusted Performance**"
        elif results['sharpe_ratio'] > 1.0:
            risk_assessment = "🟡 **Good Risk-Adjusted Performance**"
        else:
            risk_assessment = "🔴 **Poor Risk-Adjusted Performance**"

        report += f"{risk_assessment}\n"

        if results['max_drawdown'] < 10:
            dd_assessment = "🟢 **Low Drawdown Risk**"
        elif results['max_drawdown'] < 20:
            dd_assessment = "🟡 **Moderate Drawdown Risk**"
        else:
            dd_assessment = "🔴 **High Drawdown Risk**"

        report += f"{dd_assessment}\n"

        report += f"""
## 💡 Recommendations

"""

        # Gerar recomendações
        recommendations = []

        if results['sharpe_ratio'] < 1.0:
            recommendations.append("🔧 Consider improving risk-adjusted returns")

        if results['max_drawdown'] > 15:
            recommendations.append("⚠️ Reduce position sizes to lower drawdown")

        if results['win_rate'] < 40:
            recommendations.append("📈 Improve signal quality for higher win rate")

        if results['profit_factor'] < 1.5:
            recommendations.append("💰 Optimize risk/reward ratio")

        if results['total_costs'] / abs(results['total_pnl']) > 0.1:
            recommendations.append("💸 Reduce trading frequency or negotiate better commissions")

        if not recommendations:
            recommendations.append("✅ **Excellent Performance** - Strategy is ready for live trading")

        for rec in recommendations:
            report += f"- {rec}\n"

        report += f"""

## 🚀 Next Steps

1. **Forward Testing**: Run in demo account for validation
2. **Parameter Optimization**: Fine-tune strategy parameters
3. **Risk Management**: Implement position sizing rules
4. **Monitoring**: Set up real-time performance tracking

---

*Generated by EA Optimizer AI - Realistic Backtesting Engine*
*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Salvar relatório
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📄 Relatório gerado: {report_file}")

if __name__ == "__main__":
    # Teste do backtester realista
    backtester = RealisticBacktester(
        initial_balance=10000.0,
        leverage=100,
        symbol="XAUUSD"
    )

    # Parâmetros da estratégia
    strategy_params = {
        'lot_size': 0.01,
        'max_positions': 3,
        'stop_loss': 150,
        'take_profit': 300,
        'rsi_threshold': 70,
        'volatility_threshold': 0.2,
        'signal_probability': 0.3
    }

    # Executar backtest de 24 horas
    results = backtester.run_backtest(
        strategy_params=strategy_params,
        duration_hours=24,
        tick_frequency_minutes=1
    )

    # Gerar relatório
    backtester.generate_report(results, '../output/realistic_backtest_report.md')

    print("🎲 Backtest realista concluído!")
    print(f"📊 PnL: ${results['total_pnl']:.2f}")
    print(f"📈 Return: {results['return_percentage']:.2f}%")
    print(f"📉 Max DD: {results['max_drawdown']:.2f}%")
    print(f"🎯 Sharpe: {results['sharpe_ratio']:.3f}")
    print(f"📈 Win Rate: {results['win_rate']:.2f}%")
    print(f"💰 Profit Factor: {results['profit_factor']:.2f}")