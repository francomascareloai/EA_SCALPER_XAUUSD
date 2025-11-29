#!/usr/bin/env python3
"""
Exemplo 02: Bot de Trading Simples
==================================

Este exemplo demonstra um bot de trading básico que:
- Analisa mercado usando médias móveis
- Executa ordens baseadas em sinais simples
- Implementa gestão de risco básica
- Monitora posições abertas

AVISO: Este é um exemplo educacional. Não use em conta real sem testes adequados.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, time as dt_time

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client
from ea_scalper_sdk.exceptions import MT5Error

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleBot")

# Carregar variáveis de ambiente
load_dotenv()

class SimpleTradingBot:
    """Bot de trading simples para XAUUSD"""

    def __init__(self):
        self.mt5_client = None
        self.is_running = False
        self.symbol = "XAUUSD"
        self.magic_number = 12345
        self.max_positions = 1
        self.risk_percent = 1.0  # 1% de risco por trade
        self.trading_hours = (dt_time(8, 0), dt_time(20, 0))  # Horário de trading GMT

    async def initialize(self):
        """Inicializa o bot"""
        try:
            logger.info("🚀 Inicializando Simple Trading Bot...")

            # Conectar ao MT5
            self.mt5_client = MT5Client()
            login = int(os.getenv('MT5_LOGIN'))
            password = os.getenv('MT5_PASSWORD')
            server = os.getenv('MT5_SERVER')

            success = await self.mt5_client.connect(login, password, server)

            if not success:
                logger.error("❌ Falha na conexão com MT5")
                return False

            logger.info("✅ Conectado ao MetaTrader 5")

            # Verificar símbolo
            symbol_info = await self.mt5_client.get_symbol_info(self.symbol)
            if not symbol_info:
                logger.error(f"❌ Símbolo {self.symbol} não encontrado")
                return False

            logger.info(f"✅ Símbolo {self.symbol} verificado")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False

    def is_trading_time(self):
        """Verifica se está no horário de trading"""
        current_time = datetime.now().time()
        return self.trading_hours[0] <= current_time <= self.trading_hours[1]

    async def check_market_conditions(self):
        """Verifica condições básicas do mercado"""
        try:
            # Verificar spread
            symbol_info = await self.mt5_client.get_symbol_info(self.symbol)
            current_spread = symbol_info.get('spread', 0)

            if current_spread > 30:  # Spread muito alto
                logger.warning(f"⚠️ Spread muito alto: {current_spread} pontos")
                return False, f"Spread alto: {current_spread}"

            # Verificar se há barras suficientes
            bars = await self.mt5_client.get_bars(self.symbol, "H1", 50)
            if len(bars) < 50:
                logger.warning("⚠️ Dados históricos insuficientes")
                return False, "Dados insuficientes"

            return True, "Condições favoráveis"

        except Exception as e:
            logger.error(f"❌ Erro ao verificar condições: {e}")
            return False, "Erro na verificação"

    def calculate_indicators(self, bars):
        """Calcula indicadores técnicos simples"""
        if len(bars) < 50:
            return None

        closes = [bar['close'] for bar in bars]

        # Médias móveis
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50

        # RSI simplificado
        gains = []
        losses = []

        for i in range(1, len(closes[-14:])):
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
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        current_price = closes[-1]

        return {
            'price': current_price,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'trend': 'bullish' if sma_10 > sma_20 > sma_50 else 'bearish' if sma_10 < sma_20 < sma_50 else 'neutral'
        }

    def generate_signal(self, indicators):
        """Gera sinal de trading baseado nos indicadores"""
        if not indicators:
            return None

        price = indicators['price']
        sma_10 = indicators['sma_10']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        rsi = indicators['rsi']
        trend = indicators['trend']

        # Regras simples
        signal_strength = 0
        signal = "HOLD"

        # Sinal de compra
        if (trend == 'bullish' and
            price > sma_10 and
            sma_10 > sma_20 and
            30 < rsi < 70):

            signal = "BUY"
            signal_strength = 3

        elif (price > sma_20 and
              sma_10 > sma_20 and
              rsi < 30):

            signal = "BUY"
            signal_strength = 2

        # Sinal de venda
        elif (trend == 'bearish' and
              price < sma_10 and
              sma_10 < sma_20 and
              30 < rsi < 70):

            signal = "SELL"
            signal_strength = 3

        elif (price < sma_20 and
              sma_10 < sma_20 and
              rsi > 70):

            signal = "SELL"
            signal_strength = 2

        # Calcular níveis de SL/TP
        if signal != "HOLD":
            atr = self.calculate_atr(indicators['price'], 20)  # ATR de 20 períodos

            if signal == "BUY":
                stop_loss = price - (atr * 1.5)
                take_profit = price + (atr * 2.0)
            else:  # SELL
                stop_loss = price + (atr * 1.5)
                take_profit = price - (atr * 2.0)
        else:
            stop_loss = None
            take_profit = None

        return {
            'signal': signal,
            'strength': signal_strength,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rsi': rsi,
            'trend': trend
        }

    def calculate_atr(self, current_price, period=14):
        """Calcula ATR (Average True Range) simplificado"""
        # ATR simplificado como % do preço
        return current_price * 0.005  # 0.5% do preço como ATR padrão

    async def calculate_position_size(self):
        """Calcula tamanho da posição baseado no risco"""
        try:
            account_info = await self.mt5_client.get_account_info()
            balance = account_info.get('balance', 1000)

            # Risco em dinheiro
            risk_amount = balance * (self.risk_percent / 100)

            # Obter informação do símbolo para cálculo de valor do pip
            symbol_info = await self.mt5_client.get_symbol_info(self.symbol)
            tick_value = symbol_info.get('trade_tick_value', 10)

            # Calcular tamanho da posição (simplificado)
            # Assumindo 100 pips de stop loss para cálculo
            position_size = risk_amount / (100 * tick_value)

            # Limitar entre mínimo e máximo
            min_lot = symbol_info.get('volume_min', 0.01)
            max_lot = symbol_info.get('volume_max', 1.0)

            position_size = max(min_lot, min(position_size, max_lot))
            position_size = round(position_size, 2)

            return position_size

        except Exception as e:
            logger.error(f"❌ Erro ao calcular tamanho da posição: {e}")
            return 0.01  # Valor padrão seguro

    async def place_order(self, signal):
        """Executa ordem baseada no sinal"""
        try:
            if signal['signal'] == "HOLD":
                return None

            # Calcular tamanho da posição
            volume = await self.calculate_position_size()

            # Preparar ordem
            order_type = "MARKET_BUY" if signal['signal'] == "BUY" else "MARKET_SELL"

            order_data = {
                "symbol": self.symbol,
                "volume": volume,
                "order_type": order_type,
                "stop_loss": signal['stop_loss'],
                "take_profit": signal['take_profit'],
                "magic_number": self.magic_number,
                "comment": f"SimpleBot {signal['signal']}"
            }

            logger.info(f"📊 Executando {signal['signal']} {volume} lotes")
            logger.info(f"💰 Entry: ~{signal['price']:.2f}")
            logger.info(f"🛡️ SL: {signal['stop_loss']:.2f}")
            logger.info(f"🎯 TP: {signal['take_profit']:.2f}")

            # Executar ordem
            result = await self.mt5_client.place_order(order_data)

            if result['success']:
                logger.info(f"✅ Ordem executada: Ticket {result['order_ticket']}")
                logger.info(f"💰 Preço de execução: {result['execution_price']:.2f}")
                return result['order_ticket']
            else:
                logger.error(f"❌ Falha na ordem: {result['message']}")
                return None

        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem: {e}")
            return None

    async def manage_positions(self):
        """Gerencia posições abertas"""
        try:
            positions = await self.mt5_client.get_positions(self.symbol)

            if not positions:
                return

            logger.info(f"📊 Gerenciando {len(positions)} posição(ões)")

            for position in positions:
                profit = position['profit']
                current_price = position.get('current_price', position['open_price'])

                # Trailing stop simples
                if profit > 50:  # Se lucro > $50
                    new_sl = position['open_price'] + 20 if position['type'] == 'BUY' else position['open_price'] - 20

                    # Verificar se o novo SL é melhor
                    if position['type'] == 'BUY' and new_sl > position['stop_loss']:
                        await self.mt5_client.modify_position(position['ticket'], stop_loss=new_sl)
                        logger.info(f"📏 Trailing stop ajustado para {new_sl:.2f}")

                    elif position['type'] == 'SELL' and new_sl < position['stop_loss']:
                        await self.mt5_client.modify_position(position['ticket'], stop_loss=new_sl)
                        logger.info(f"📏 Trailing stop ajustado para {new_sl:.2f}")

                # Fechar posição se perda excessiva
                if profit < -100:  # Se perda > $100
                    logger.warning(f"⚠️ Fechando posição {position['ticket']} por perda excessiva")
                    close_result = await self.mt5_client.close_position(position['ticket'])
                    if close_result['success']:
                        logger.info(f"✅ Posição {position['ticket']} fechada")

        except Exception as e:
            logger.error(f"❌ Erro na gestão de posições: {e}")

    async def run(self):
        """Loop principal do bot"""
        self.is_running = True
        logger.info("🚀 Simple Trading Bot iniciado")

        try:
            while self.is_running:
                try:
                    # Verificar horário de trading
                    if not self.is_trading_time():
                        logger.info("⏰ Fora do horário de trading")
                        await asyncio.sleep(300)  # Aguardar 5 minutos
                        continue

                    # Verificar condições do mercado
                    can_trade, reason = await self.check_market_conditions()
                    if not can_trade:
                        logger.info(f"⚠️ {reason}")
                        await asyncio.sleep(60)
                        continue

                    # Verificar posições abertas
                    positions = await self.mt5_client.get_positions(self.symbol)
                    if len(positions) >= self.max_positions:
                        logger.info(f"📊 Número máximo de posições atingido ({self.max_positions})")
                        await self.manage_positions()
                        await asyncio.sleep(60)
                        continue

                    # Análise técnica
                    bars = await self.mt5_client.get_bars(self.symbol, "H1", 100)
                    indicators = self.calculate_indicators(bars)

                    if not indicators:
                        logger.warning("⚠️ Não foi possível calcular indicadores")
                        await asyncio.sleep(60)
                        continue

                    # Gerar sinal
                    signal = self.generate_signal(indicators)

                    logger.info(f"📊 Sinal: {signal['signal']} (Força: {signal['strength']})")
                    logger.info(f"💰 Preço: {signal['price']:.2f}")
                    logger.info(f"📈 RSI: {signal['rsi']:.1f}")
                    logger.info(f"📊 Tendência: {signal['trend']}")

                    # Executar ordem se sinal for forte
                    if signal['strength'] >= 2:
                        await self.place_order(signal)

                    # Gerenciar posições existentes
                    await self.manage_positions()

                    # Aguardar próximo ciclo
                    await asyncio.sleep(60)  # Verificar a cada minuto

                except Exception as e:
                    logger.error(f"❌ Erro no loop principal: {e}")
                    await asyncio.sleep(10)

        except KeyboardInterrupt:
            logger.info("🛑 Interrupção pelo usuário")
        finally:
            self.stop()

    def stop(self):
        """Para o bot"""
        self.is_running = False
        logger.info("🛑 Simple Trading Bot parado")

async def main():
    """Função principal"""
    print("🤖 Simple Trading Bot - Exemplo Educativo")
    print("=" * 50)
    print("⚠️ AVISO: Este é um exemplo educacional!")
    print("⚠️ Não use em conta real sem testes adequados!")
    print()

    bot = SimpleTradingBot()

    # Inicializar
    if not await bot.initialize():
        logger.error("❌ Falha na inicialização do bot")
        return

    try:
        # Executar bot
        await bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Encerrando bot...")
    finally:
        # Desconectar
        if bot.mt5_client:
            await bot.mt5_client.disconnect()
            logger.info("🔌 Desconectado do MetaTrader 5")

if __name__ == "__main__":
    asyncio.run(main())