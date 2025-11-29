#!/usr/bin/env python3
"""
Exemplo 03: Trading com Inteligência Artificial
===============================================

Este exemplo demonstra como integrar IA (LiteLLM) para:
- Análise avançada de mercado
- Geração de sinais de trading
- Otimização de estratégias
- Gestão de risco inteligente

Requisitos:
- LiteLLM proxy rodando em localhost:4000
- Chave de API OpenRouter configurada
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client, LLMClient
from ea_scalper_sdk.exceptions import MT5Error, LLMError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AITradingBot")

# Carregar variáveis de ambiente
load_dotenv()

class AIEnhancedTradingBot:
    """Bot de trading com análise de IA"""

    def __init__(self):
        self.mt5_client = None
        self.llm_client = None
        self.is_running = False
        self.symbol = "XAUUSD"
        self.magic_number = 54321
        self.confidence_threshold = 75  # Confiança mínima para executar trade
        self.market_analysis_cache = {}
        self.cache_duration = 300  # 5 minutos

    async def initialize(self):
        """Inicializa clientes e conexões"""
        try:
            logger.info("🚀 Inicializando AI Enhanced Trading Bot...")

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

            # Conectar ao LiteLLM
            self.llm_client = LLMClient(base_url="http://localhost:4000")

            # Testar conexão LLM
            models = await self.llm_client.list_models()
            available_models = [model['id'] for model in models.get('data', [])]
            logger.info(f"🤖 Modelos disponíveis: {available_models[:3]}...")

            logger.info("✅ Cliente LiteLLM inicializado")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False

    async def collect_market_data(self):
        """Coleta dados completos do mercado"""
        try:
            market_data = {}

            # Obter dados de múltiplos timeframes
            timeframes = {
                "M5": 200,
                "M15": 200,
                "H1": 100,
                "H4": 100,
                "D1": 30
            }

            for tf, count in timeframes.items():
                bars = await self.mt5_client.get_bars(self.symbol, tf, count)
                if bars:
                    market_data[tf] = {
                        'bars': bars,
                        'current_price': bars[-1]['close'],
                        'volume': sum(bar['volume'] for bar in bars[-20:]),
                        'volatility': self.calculate_volatility(bars)
                    }

            # Obter ticks recentes
            ticks = await self.mt5_client.get_ticks(self.symbol, 50)
            if ticks:
                market_data['ticks'] = {
                    'recent': ticks[-10:],
                    'spread': ticks[-1]['ask'] - ticks[-1]['bid'],
                    'momentum': self.calculate_tick_momentum(ticks)
                }

            # Obter informações da conta
            account_info = await self.mt5_client.get_account_info()
            market_data['account'] = account_info

            # Obter posições abertas
            positions = await self.mt5_client.get_positions(self.symbol)
            market_data['positions'] = positions

            # Calcular indicadores técnicos
            market_data['indicators'] = self.calculate_all_indicators(market_data)

            logger.info(f"📊 Dados coletados de {len(timeframes)} timeframes")
            return market_data

        except Exception as e:
            logger.error(f"❌ Erro na coleta de dados: {e}")
            return None

    def calculate_volatility(self, bars, period=20):
        """Calcula volatilidade como desvio padrão dos retornos"""
        if len(bars) < period:
            return 0

        closes = [bar['close'] for bar in bars[-period:]]
        returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]

        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)

        return (variance ** 0.5) * 100  # Em percentagem

    def calculate_tick_momentum(self, ticks):
        """Calcula momentum baseado em ticks"""
        if len(ticks) < 10:
            return 0

        price_changes = []
        for i in range(1, min(10, len(ticks))):
            change = (ticks[-i]['bid'] - ticks[-i-1]['bid']) / ticks[-i-1]['bid']
            price_changes.append(change)

        return sum(price_changes) / len(price_changes) * 100  # Em percentagem

    def calculate_all_indicators(self, market_data):
        """Calcula todos os indicadores técnicos"""
        indicators = {}

        # Para cada timeframe
        for tf, data in market_data.items():
            if tf == 'ticks' or tf == 'account' or tf == 'positions':
                continue

            if 'bars' in data and len(data['bars']) >= 50:
                bars = data['bars']
                closes = [bar['close'] for bar in bars]

                # Médias móveis
                indicators[tf] = {
                    'sma_20': sum(closes[-20:]) / 20,
                    'sma_50': sum(closes[-50:]) / 50,
                    'ema_12': self.calculate_ema(closes, 12),
                    'ema_26': self.calculate_ema(closes, 26),
                    'rsi': self.calculate_rsi(closes, 14),
                    'macd': self.calculate_macd(closes),
                    'bbands': self.calculate_bollinger_bands(closes, 20)
                }

        return indicators

    def calculate_ema(self, prices, period):
        """Calcula EMA (Exponential Moving Average)"""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcula RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50

        gains = []
        losses = []

        for i in range(1, len(prices[-period-1:])):
            change = prices[-i] - prices[-i-1]
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

    def calculate_macd(self, prices):
        """Calcula MACD"""
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26

        # Calcular signal line (EMA do MACD)
        if len(prices) >= 26 + 9:
            macd_values = []
            for i in range(26, len(prices)):
                ema_12_i = self.calculate_ema(prices[:i+1], 12)
                ema_26_i = self.calculate_ema(prices[:i+1], 26)
                macd_values.append(ema_12_i - ema_26_i)

            signal_line = self.calculate_ema(macd_values, 9)
            histogram = macd_line - signal_line
        else:
            signal_line = 0
            histogram = 0

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}

        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period

        variance = sum((price - middle) ** 2 for price in recent_prices) / period
        std = variance ** 0.5

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'std': std
        }

    def format_market_data_for_ai(self, market_data):
        """Formata dados do mercado para envio à IA"""
        try:
            # Dados atuais
            h1_data = market_data.get('H1', {})
            h4_data = market_data.get('H4', {})
            account_data = market_data.get('account', {})
            positions = market_data.get('positions', [])

            current_price = h1_data.get('current_price', 0)
            h1_indicators = market_data.get('indicators', {}).get('H1', {})
            h4_indicators = market_data.get('indicators', {}).get('H4', {})

            # Análise técnica
            h1_trend = self.determine_trend(h1_indicators)
            h4_trend = self.determine_trend(h4_indicators)

            # Formatar resumo para IA
            summary = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "current_price": current_price,
                "account_balance": account_data.get('balance', 0),
                "open_positions": len(positions),
                "timeframe_analysis": {
                    "H1": {
                        "trend": h1_trend,
                        "rsi": h1_indicators.get('rsi', 50),
                        "macd": h1_indicators.get('macd', {}),
                        "price_vs_sma": self.compare_price_to_mas(current_price, h1_indicators)
                    },
                    "H4": {
                        "trend": h4_trend,
                        "rsi": h4_indicators.get('rsi', 50),
                        "macd": h4_indicators.get('macd', {}),
                        "price_vs_sma": self.compare_price_to_mas(current_price, h4_indicators)
                    }
                },
                "market_conditions": {
                    "volatility": h1_data.get('volatility', 0),
                    "spread_pips": market_data.get('ticks', {}).get('spread', 0) * 100,
                    "volume_trend": "increasing" if h1_data.get('volume', 0) > 0 else "neutral"
                },
                "key_levels": self.find_key_levels(market_data),
                "risk_assessment": {
                    "daily_pnl": sum(pos.get('profit', 0) for pos in positions),
                    "max_positions_allowed": 2,
                    "current_risk_exposure": self.calculate_risk_exposure(positions, account_data)
                }
            }

            return summary

        except Exception as e:
            logger.error(f"❌ Erro ao formatar dados para IA: {e}")
            return None

    def determine_trend(self, indicators):
        """Determina tendência baseada nos indicadores"""
        try:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            ema_12 = indicators.get('ema_12', 0)
            ema_26 = indicators.get('ema_26', 0)

            if ema_12 > ema_26 > sma_20 > sma_50:
                return "strong_bullish"
            elif ema_12 > ema_26 and sma_20 > sma_50:
                return "bullish"
            elif ema_12 < ema_26 < sma_20 < sma_50:
                return "strong_bearish"
            elif ema_12 < ema_26 and sma_20 < sma_50:
                return "bearish"
            else:
                return "neutral"

        except:
            return "unknown"

    def compare_price_to_mas(self, price, indicators):
        """Compara preço com médias móveis"""
        try:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            ema_12 = indicators.get('ema_12', 0)

            if price > ema_12 > sma_20 > sma_50:
                return "above_all_averages"
            elif price > ema_12 > sma_20:
                return "above_short_averages"
            elif price < ema_12 < sma_20:
                return "below_averages"
            else:
                return "around_averages"

        except:
            return "unknown"

    def find_key_levels(self, market_data):
        """Encontra níveis chave de suporte e resistência"""
        try:
            h1_bars = market_data.get('H1', {}).get('bars', [])
            if len(h1_bars) < 50:
                return {"support": [], "resistance": []}

            highs = [bar['high'] for bar in h1_bars[-100:]]
            lows = [bar['low'] for bar in h1_bars[-100:]]

            # Encontrar topos e fundos significativos
            resistance_levels = []
            support_levels = []

            # Resistência: picos que se repetem
            for price in set(highs):
                if highs.count(price) >= 2:  # Pelo menos 2 topos no mesmo nível
                    resistance_levels.append(price)

            # Suporte: fundos que se repetem
            for price in set(lows):
                if lows.count(price) >= 2:  # Pelo menos 2 fundos no mesmo nível
                    support_levels.append(price)

            # Ordenar e limitar a 3 níveis cada
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels)[:3]

            return {
                "resistance": resistance_levels,
                "support": support_levels
            }

        except Exception as e:
            logger.error(f"❌ Erro ao encontrar níveis chave: {e}")
            return {"support": [], "resistance": []}

    def calculate_risk_exposure(self, positions, account_data):
        """Calcula exposição ao risco atual"""
        try:
            if not positions:
                return 0

            total_risk = 0
            balance = account_data.get('balance', 1000)

            for position in positions:
                # Calcular risco em valor absoluto
                sl_distance = abs(position['open_price'] - position['stop_loss'])
                position_risk = position['volume'] * sl_distance * 100  # Simplificado
                total_risk += position_risk

            return (total_risk / balance) * 100 if balance > 0 else 0

        except:
            return 0

    async def analyze_market_with_ai(self, market_data):
        """Usa IA para analisar mercado e gerar sinais"""
        try:
            # Verificar cache
            cache_key = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.market_analysis_cache:
                logger.info("📋 Usando análise em cache")
                return self.market_analysis_cache[cache_key]

            # Formatar dados para IA
            formatted_data = self.format_market_data_for_ai(market_data)
            if not formatted_data:
                return None

            # Criar prompt para IA
            prompt = f"""
            Como especialista em trading de XAUUSD, analise os seguintes dados e forneça recomendações:

            DADOS DE MERCADO:
            {json.dumps(formatted_data, indent=2)}

            CONSIDERE:
            1. Análise multi-timeframe (H1 e H4)
            2. Indicadores técnicos (RSI, MACD, Médias Móveis)
            3. Condições de mercado (volatilidade, spread)
            4. Níveis de suporte e resistência
            5. Gestão de risco FTMO-compliant (máx 5% perda diária)

            FORNEÇA ANÁLISE EM FORMATO JSON:
            {{
                "market_sentiment": "bullish/bearish/neutral",
                "signal_strength": 0-100,
                "recommended_action": "BUY/SELL/HOLD",
                "confidence_level": 0-100,
                "entry_price": preço_sugerido,
                "stop_loss": preço_sl,
                "take_profit": preço_tp,
                "risk_reward_ratio": 1.5,
                "reasoning": "explicação_detelhada",
                "risk_factors": ["fator1", "fator2"],
                "key_levels": {{
                    "support": [nível1, nível2],
                    "resistance": [nível1, nível2]
                }},
                "market_bias": "strong/moderate/weak",
                "volatility_expectation": "increasing/decreasing/stable"
            }}

            Seja conservador e priorize segurança. Apenas recomende trades com alta probabilidade de sucesso.
            """

            logger.info("🤖 Enviando dados para análise com IA...")

            # Chamar API LiteLLM
            response = await self.llm_client.chat_completion(
                model="deepseek-r1-free",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um analista técnico especializado em trading de XAUUSD com foco em gestão de risco FTMO-compliant. Seja analítico, conservador e forneça apenas recomendações de alta confiança."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )

            # Processar resposta
            ai_response = response['choices'][0]['message']['content']
            logger.info("🧠 Resposta da IA recebida")

            # Tentar parsear JSON
            try:
                # Extrair JSON da resposta
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    # Se não encontrar JSON, criar análise básica
                    analysis = {
                        "market_sentiment": "neutral",
                        "signal_strength": 50,
                        "recommended_action": "HOLD",
                        "confidence_level": 50,
                        "reasoning": ai_response[:500]  # Primeiros 500 caracteres
                    }

                # Adicionar timestamp
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['raw_response'] = ai_response

                # Salvar no cache
                self.market_analysis_cache[cache_key] = analysis

                # Limpar cache antigo
                self.cleanup_cache()

                logger.info(f"✅ Análise concluída: {analysis['recommended_action']} (Confiança: {analysis.get('confidence_level', 0)}%)")
                return analysis

            except json.JSONDecodeError as e:
                logger.error(f"❌ Erro ao parsear JSON da IA: {e}")
                return {
                    "market_sentiment": "neutral",
                    "recommended_action": "HOLD",
                    "confidence_level": 0,
                    "reasoning": "Erro no processamento da resposta da IA",
                    "raw_response": ai_response
                }

        except LLMError as e:
            logger.error(f"❌ Erro na API LiteLLM: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro na análise com IA: {e}")
            return None

    def cleanup_cache(self):
        """Remove análises antigas do cache"""
        current_time = datetime.now()
        keys_to_remove = []

        for key in self.market_analysis_cache.keys():
            try:
                # Extrair timestamp da chave
                time_str = key.replace('analysis_', '')
                analysis_time = datetime.strptime(time_str, '%Y%m%d_%H%M')

                # Remover se for mais antigo que cache_duration
                if (current_time - analysis_time).seconds > self.cache_duration:
                    keys_to_remove.append(key)

            except:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.market_analysis_cache[key]

        if keys_to_remove:
            logger.info(f"🗑️ Cache limpo: {len(keys_to_remove)} entradas removidas")

    async def execute_ai_trade(self, analysis):
        """Executa trade baseado na análise da IA"""
        try:
            if not analysis or analysis.get('recommended_action') == 'HOLD':
                logger.info("⏸️ Nenhum sinal de trading da IA")
                return False

            confidence = analysis.get('confidence_level', 0)
            if confidence < self.confidence_threshold:
                logger.info(f"⚠️ Confiança baixa ({confidence}%) - threshold é {self.confidence_threshold}%")
                return False

            # Verificar posições abertas
            positions = await self.mt5_client.get_positions(self.symbol)
            if len(positions) >= 2:  # Máximo 2 posições
                logger.info("📊 Número máximo de posições atingido")
                return False

            # Preparar ordem
            signal = analysis['recommended_action']
            entry_price = analysis.get('entry_price', 0)
            stop_loss = analysis.get('stop_loss', 0)
            take_profit = analysis.get('take_profit', 0)

            if not all([entry_price, stop_loss, take_profit]):
                logger.error("❌ Preços inválidos na análise da IA")
                return False

            # Calcular tamanho da posição (conservador)
            account_info = await self.mt5_client.get_account_info()
            balance = account_info.get('balance', 1000)
            risk_amount = balance * 0.005  # 0.5% de risco

            # Calcular posição baseada no stop loss
            sl_distance = abs(entry_price - stop_loss)
            symbol_info = await self.mt5_client.get_symbol_info(self.symbol)
            pip_value = symbol_info.get('trade_tick_value', 10) if symbol_info else 10

            position_size = risk_amount / (sl_distance * 100 * pip_value)
            position_size = max(0.01, min(position_size, 0.1))  # Entre 0.01 e 0.1 lotes
            position_size = round(position_size, 2)

            # Executar ordem
            order_type = "MARKET_BUY" if signal == "BUY" else "MARKET_SELL"

            order_data = {
                "symbol": self.symbol,
                "volume": position_size,
                "order_type": order_type,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "magic_number": self.magic_number,
                "comment": f"AI Bot {signal} (Conf: {confidence}%)"
            }

            logger.info(f"🤖 Executando trade da IA:")
            logger.info(f"   Sinal: {signal}")
            logger.info(f"   Confiança: {confidence}%")
            logger.info(f"   Volume: {position_size} lotes")
            logger.info(f"   Entry: ~{entry_price:.2f}")
            logger.info(f"   SL: {stop_loss:.2f}")
            logger.info(f"   TP: {take_profit:.2f}")
            logger.info(f"   R/R: 1:{analysis.get('risk_reward_ratio', 0)}")

            result = await self.mt5_client.place_order(order_data)

            if result['success']:
                logger.info(f"✅ Trade executado: Ticket {result['order_ticket']}")
                logger.info(f"💰 Preço real: {result['execution_price']:.2f}")
                return True
            else:
                logger.error(f"❌ Falha na execução: {result['message']}")
                return False

        except Exception as e:
            logger.error(f"❌ Erro na execução do trade: {e}")
            return False

    async def run_ai_analysis_cycle(self):
        """Executa ciclo completo de análise com IA"""
        try:
            # Coletar dados do mercado
            logger.info("📊 Coletando dados do mercado...")
            market_data = await self.collect_market_data()

            if not market_data:
                logger.error("❌ Falha na coleta de dados")
                return

            # Analisar com IA
            logger.info("🧠 Analisando mercado com IA...")
            analysis = await self.analyze_market_with_ai(market_data)

            if not analysis:
                logger.error("❌ Falha na análise com IA")
                return

            # Exibir resumo da análise
            logger.info("📋 Resumo da Análise da IA:")
            logger.info(f"   Sentimento: {analysis.get('market_sentiment', 'N/A')}")
            logger.info(f"   Sinal: {analysis.get('recommended_action', 'N/A')}")
            logger.info(f"   Confiança: {analysis.get('confidence_level', 0)}%")
            logger.info(f"   Força do Sinal: {analysis.get('signal_strength', 0)}")
            logger.info(f"   Viés de Mercado: {analysis.get('market_bias', 'N/A')}")
            logger.info(f"   R/R Ratio: 1:{analysis.get('risk_reward_ratio', 0)}")

            # Executar trade se apropriado
            if await self.execute_ai_trade(analysis):
                logger.info("🎉 Trade da IA executado com sucesso!")

            # Salvar análise para histórico
            await self.save_analysis_to_history(analysis, market_data)

        except Exception as e:
            logger.error(f"❌ Erro no ciclo de análise: {e}")

    async def save_analysis_to_history(self, analysis, market_data):
        """Salva análise no histórico"""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "market_snapshot": {
                    "price": market_data.get('H1', {}).get('current_price', 0),
                    "spread": market_data.get('ticks', {}).get('spread', 0),
                    "positions_count": len(market_data.get('positions', []))
                }
            }

            # Salvar em arquivo (simplificado)
            history_file = Path("ai_analysis_history.json")
            history = []

            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                except:
                    history = []

            history.append(history_entry)

            # Manter apenas últimas 100 entradas
            if len(history) > 100:
                history = history[-100:]

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Erro ao salvar análise: {e}")

    async def run(self):
        """Loop principal do bot"""
        self.is_running = True
        logger.info("🚀 AI Enhanced Trading Bot iniciado")

        try:
            while self.is_running:
                try:
                    # Executar ciclo de análise com IA
                    await self.run_ai_analysis_cycle()

                    # Aguardar próximo ciclo (15 minutos)
                    logger.info("⏰ Aguardando próximo ciclo de análise (15 minutos)...")
                    await asyncio.sleep(900)  # 15 minutos

                except KeyboardInterrupt:
                    logger.info("🛑 Interrupção pelo usuário")
                    break
                except Exception as e:
                    logger.error(f"❌ Erro no loop principal: {e}")
                    await asyncio.sleep(60)  # Aguardar 1 minuto antes de tentar novamente

        finally:
            self.stop()

    def stop(self):
        """Para o bot"""
        self.is_running = False
        logger.info("🛑 AI Enhanced Trading Bot parado")

async def main():
    """Função principal"""
    print("🤖 AI Enhanced Trading Bot")
    print("=" * 50)
    print("⚠️ AVISO: Bot com IA para fins educacionais")
    print("⚠️ Monitore cuidadosamente todas as operações")
    print()

    bot = AIEnhancedTradingBot()

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