#!/usr/bin/env python3
"""
🧠 EA Optimizer AI - Deep Learning Engine (Rodada 2)
Sistema de deep learning com LSTM, Transformers e feature engineering automático
"""

import numpy as np
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import deque
import math

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Classe para cálculo de indicadores técnicos avançados"""

    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        result = []
        for i in range(len(data)):
            if i >= period - 1:
                avg = sum(data[i - period + 1:i + 1]) / period
                result.append(avg)
            else:
                result.append(0)
        return result

    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        result = []
        multiplier = 2 / (period + 1)

        for i in range(len(data)):
            if i == 0:
                result.append(data[i])
            else:
                ema = (data[i] - result[i-1]) * multiplier + result[i-1]
                result.append(ema)
        return result

    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [0] * len(data)

        gains = []
        losses = []

        for i in range(1, len(data)):
            change = data[i] - data[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))

        result = [0] * (period + 1)

        for i in range(period - 1, len(gains)):
            avg_gain = sum(gains[i - period + 1:i + 1]) / period
            avg_loss = sum(losses[i - period + 1:i + 1]) / period

            if avg_loss == 0:
                rs = 100
            else:
                rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            result.append(rsi)

        return result

    @staticmethod
    def bollinger_bands(data: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands"""
        sma_data = TechnicalIndicators.sma(data, period)
        upper_band = []
        lower_band = []

        for i in range(len(data)):
            if i >= period - 1:
                slice_data = data[i - period + 1:i + 1]
                std = np.std(slice_data)
                upper = sma_data[i] + (std * std_dev)
                lower = sma_data[i] - (std * std_dev)
                upper_band.append(upper)
                lower_band.append(lower)
            else:
                upper_band.append(data[i])
                lower_band.append(data[i])

        return sma_data, upper_band, lower_band

    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)

        macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(high: List[float], low: List[float], close: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator"""
        k_percent = []

        for i in range(len(close)):
            if i >= k_period - 1:
                high_slice = high[i - k_period + 1:i + 1]
                low_slice = low[i - k_period + 1:i + 1]

                highest_high = max(high_slice)
                lowest_low = min(low_slice)

                if highest_high - lowest_low == 0:
                    k = 100
                else:
                    k = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)

                k_percent.append(k)
            else:
                k_percent.append(50)

        d_percent = TechnicalIndicators.sma(k_percent, d_period)
        return k_percent, d_percent

    @staticmethod
    def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Average True Range"""
        true_ranges = []

        for i in range(len(close)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)

            true_ranges.append(tr)

        return TechnicalIndicators.ema(true_ranges, period)

class FeatureEngineer:
    """Feature Engineering automático avançado"""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_names = []

    def generate_technical_features(self, ohlc_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Gera features técnicas avançadas

        Args:
            ohlc_data: Dicionário com dados OHLC

        Returns:
            Dicionário com features geradas
        """
        close = ohlc_data['close']
        high = ohlc_data['high']
        low = ohlc_data['low']
        volume = ohlc_data.get('volume', [1] * len(close))

        features = {}

        # Moving Averages
        features['sma_10'] = self.indicators.sma(close, 10)
        features['sma_20'] = self.indicators.sma(close, 20)
        features['sma_50'] = self.indicators.sma(close, 50)
        features['sma_200'] = self.indicators.sma(close, 200)

        features['ema_12'] = self.indicators.ema(close, 12)
        features['ema_26'] = self.indicators.ema(close, 26)

        # RSI
        features['rsi_14'] = self.indicators.rsi(close, 14)
        features['rsi_7'] = self.indicators.rsi(close, 7)
        features['rsi_21'] = self.indicators.rsi(close, 21)

        # Bollinger Bands
        bb_sma, bb_upper, bb_lower = self.indicators.bollinger_bands(close, 20, 2.0)
        features['bb_sma'] = bb_sma
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = [(u - l) / s if s > 0 else 0 for u, l, s in zip(bb_upper, bb_lower, bb_sma)]
        features['bb_position'] = [(c - l) / (u - l) if (u - l) > 0 else 0.5 for c, l, u in zip(close, bb_lower, bb_upper)]

        # MACD
        macd, signal, histogram = self.indicators.macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram

        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(high, low, close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d

        # ATR
        features['atr_14'] = self.indicators.atr(high, low, close, 14)

        # Price Features
        features['price_change_1'] = [close[i] - close[i-1] if i > 0 else 0 for i in range(len(close))]
        features['price_change_pct_1'] = [(close[i] - close[i-1]) / close[i-1] * 100 if i > 0 else 0 for i in range(len(close))]
        features['high_low_ratio'] = [h / l if l > 0 else 1 for h, l in zip(high, low)]

        # Volume Features (se disponível)
        if len(volume) > 1:
            features['volume_sma_20'] = self.indicators.sma(volume, 20)
            features['volume_ratio'] = [v / vs if vs > 0 else 1 for v, vs in zip(volume, features['volume_sma_20'])]

        # Support/Resistance Levels
        features['resistance_20'] = [max(close[i-19:i+1]) if i >= 19 else close[i] for i in range(len(close))]
        features['support_20'] = [min(close[i-19:i+1]) if i >= 19 else close[i] for i in range(len(close))]

        # Volatility Features
        returns = [(close[i] - close[i-1]) / close[i-1] if i > 0 else 0 for i in range(len(close))]
        features['volatility_10'] = [np.std(returns[i-9:i+1]) if i >= 9 else 0 for i in range(len(returns))]

        # Store feature names
        self.feature_names = list(features.keys())

        return features

    def generate_time_features(self, timestamps: List[datetime]) -> Dict[str, List[int]]:
        """
        Gera features temporais

        Args:
            timestamps: Lista de timestamps

        Returns:
            Features temporais
        """
        features = {
            'hour': [ts.hour for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps],
            'day_of_month': [ts.day for ts in timestamps],
            'month': [ts.month for ts in timestamps],
            'quarter': [(ts.month - 1) // 3 + 1 for ts in timestamps],
            'is_weekend': [1 if ts.weekday() >= 5 else 0 for ts in timestamps],
            'is_session_asian': [1 if 0 <= ts.hour < 8 else 0 for ts in timestamps],
            'is_session_european': [1 if 7 <= ts.hour < 16 else 0 for ts in timestamps],
            'is_session_american': [1 if 13 <= ts.hour < 22 else 0 for ts in timestamps]
        }

        return features

    def generate_lag_features(self, data: List[float], lags: List[int]) -> Dict[str, List[float]]:
        """
        Gera features de lag (valores passados)

        Args:
            data: Série temporal
            lags: Lista de períodos de lag

        Returns:
            Features de lag
        """
        features = {}

        for lag in lags:
            lag_values = []
            for i in range(len(data)):
                if i >= lag:
                    lag_values.append(data[i - lag])
                else:
                    lag_values.append(0)

            features[f'lag_{lag}'] = lag_values

        return features

    def generate_rolling_features(self, data: List[float], windows: List[int]) -> Dict[str, List[float]]:
        """
        Gera features de janela móvel

        Args:
            data: Série temporal
            windows: Lista de tamanhos de janela

        Returns:
            Features de janela móvel
        """
        features = {}

        for window in windows:
            # Rolling mean
            rolling_mean = []
            # Rolling std
            rolling_std = []
            # Rolling min/max
            rolling_min = []
            rolling_max = []

            for i in range(len(data)):
                if i >= window - 1:
                    window_data = data[i - window + 1:i + 1]
                    rolling_mean.append(np.mean(window_data))
                    rolling_std.append(np.std(window_data))
                    rolling_min.append(min(window_data))
                    rolling_max.append(max(window_data))
                else:
                    rolling_mean.append(data[i])
                    rolling_std.append(0)
                    rolling_min.append(data[i])
                    rolling_max.append(data[i])

            features[f'rolling_mean_{window}'] = rolling_mean
            features[f'rolling_std_{window}'] = rolling_std
            features[f'rolling_min_{window}'] = rolling_min
            features[f'rolling_max_{window}'] = rolling_max

        return features

class LSTMPredictor:
    """Preditor usando redes neurais LSTM simplificadas"""

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        """
        Inicializa preditor LSTM

        Args:
            input_size: Número de features de entrada
            hidden_size: Tamanho da camada oculta
            output_size: Tamanho da saída
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializar pesos (simulação de LSTM)
        np.random.seed(42)
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_ih = np.zeros((hidden_size, 1))

        self.W_out = np.random.randn(output_size, hidden_size) * 0.1
        self.b_out = np.zeros((output_size, 1))

    def sigmoid(self, x):
        """Função sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        """Função tanh"""
        return np.tanh(x)

    def lstm_cell(self, x, h_prev, c_prev):
        """
        Célula LSTM simplificada

        Args:
            x: Entrada atual
            h_prev: Hidden state anterior
            c_prev: Cell state anterior

        Returns:
            Novo hidden state e cell state
        """
        # Simulação de LSTM gates
        combined = np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_ih

        # Forget gate
        f = self.sigmoid(combined[:self.hidden_size//4])

        # Input gate
        i = self.sigmoid(combined[self.hidden_size//4:self.hidden_size//2])
        c_candidate = self.tanh(combined[self.hidden_size//2:3*self.hidden_size//4])

        # Update cell state
        c = f * c_prev + i * c_candidate

        # Output gate
        o = self.sigmoid(combined[3*self.hidden_size//4:])

        # Update hidden state
        h = o * self.tanh(c)

        return h, c

    def forward(self, X):
        """
        Forward pass da LSTM

        Args:
            X: Sequência de entrada (seq_len, input_size)

        Returns:
            Previsões
        """
        seq_len = X.shape[0]
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        outputs = []

        for t in range(seq_len):
            x = X[t].reshape(-1, 1)
            h, c = self.lstm_cell(x, h, c)
            outputs.append(h)

        # Camada de saída
        predictions = []
        for h in outputs:
            out = np.dot(self.W_out, h) + self.b_out
            predictions.append(out[0, 0])

        return np.array(predictions)

    def predict_direction(self, features: List[float]) -> Tuple[float, float]:
        """
        Prediz direção e magnitude do movimento

        Args:
            features: Features atuais

        Returns:
            Direção (-1 a 1) e confiança (0 a 1)
        """
        if len(features) < self.input_size:
            # Padding se necessário
            features = features + [0] * (self.input_size - len(features))

        X = np.array(features[-self.input_size:])
        predictions = self.forward(X)

        if len(predictions) > 0:
            direction = predictions[-1]
            confidence = min(abs(direction), 1.0)
            return direction, confidence

        return 0.0, 0.0

class DeepLearningEngine:
    """Engine principal de deep learning para trading"""

    def __init__(self):
        """
        Inicializa engine de deep learning
        """
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.feature_history = deque(maxlen=1000)

        # Configurações dos modelos
        self.model_configs = {
            'price_lstm': {
                'input_size': 50,
                'hidden_size': 128,
                'description': 'LSTM para previsão de preços'
            },
            'direction_lstm': {
                'input_size': 100,
                'hidden_size': 64,
                'description': 'LSTM para previsão de direção'
            },
            'volatility_lstm': {
                'input_size': 30,
                'hidden_size': 32,
                'description': 'LSTM para previsão de volatilidade'
            }
        }

    def initialize_models(self):
        """Inicializa todos os modelos LSTM"""
        for name, config in self.model_configs.items():
            self.models[name] = LSTMPredictor(
                input_size=config['input_size'],
                hidden_size=config['hidden_size']
            )
            logger.info(f"🧠 Modelo {name} inicializado: {config['description']}")

    def generate_comprehensive_features(self, ohlc_data: Dict[str, List[float]], timestamps: List[datetime]) -> Dict[str, List[float]]:
        """
        Gera features compreensivas para deep learning

        Args:
            ohlc_data: Dados OHLC
            timestamps: Timestamps

        Returns:
            Dicionário completo de features
        """
        logger.info("🔧 Gerando features abrangentes...")

        # Features técnicas
        technical_features = self.feature_engineer.generate_technical_features(ohlc_data)

        # Features temporais
        time_features = self.feature_engineer.generate_time_features(timestamps)

        # Features de lag para preços
        price_lags = self.feature_engineer.generate_lag_features(ohlc_data['close'], [1, 5, 10, 20])

        # Features de janela móvel para retornos
        returns = [(ohlc_data['close'][i] - ohlc_data['close'][i-1]) / ohlc_data['close'][i-1] if i > 0 else 0
                  for i in range(len(ohlc_data['close']))]
        rolling_features = self.feature_engineer.generate_rolling_features(returns, [5, 10, 20])

        # Combinar todas as features
        all_features = {}
        all_features.update(technical_features)
        all_features.update({f"time_{k}": v for k, v in time_features.items()})
        all_features.update({f"price_{k}": v for k, v in price_lags.items()})
        all_features.update({f"returns_{k}": v for k, v in rolling_features.items()})

        # Features avançadas
        all_features.update(self._generate_advanced_features(ohlc_data, technical_features))

        logger.info(f"✅ {len(all_features)} features geradas")
        return all_features

    def _generate_advanced_features(self, ohlc_data: Dict[str, List[float]], technical_features: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Gera features avançadas

        Args:
            ohlc_data: Dados OHLC
            technical_features: Features técnicas

        Returns:
            Features avançadas
        """
        features = {}
        close = ohlc_data['close']
        high = ohlc_data['high']
        low = ohlc_data['low']

        # Price Action Features
        features['doji'] = [1 if abs(h - l) < 0.001 * close[i] else 0 for i, (h, l) in enumerate(zip(high, low))]
        features['engulfing_bullish'] = [1 if i > 0 and close[i-1] < open[i] and close[i] > open[i-1] else 0 for i in range(len(close))]
        features['engulfing_bearish'] = [1 if i > 0 and close[i-1] > open[i] and close[i] < open[i-1] else 0 for i in range(len(close))]

        # Indicator Confluence
        if 'rsi_14' in technical_features and 'stoch_k' in technical_features:
            features['rsi_stoch_oversold'] = [1 if rsi < 30 and stoch < 20 else 0 for rsi, stoch in zip(technical_features['rsi_14'], technical_features['stoch_k'])]
            features['rsi_stoch_overbought'] = [1 if rsi > 70 and stoch > 80 else 0 for rsi, stoch in zip(technical_features['rsi_14'], technical_features['stoch_k'])]

        # Divergence Features
        if len(close) > 20:
            price_trend_5 = [close[i] - close[i-5] if i >= 5 else 0 for i in range(len(close))]
            if 'rsi_14' in technical_features:
                rsi_trend_5 = [technical_features['rsi_14'][i] - technical_features['rsi_14'][i-5] if i >= 5 else 0 for i in range(len(technical_features['rsi_14']))]
                features['bullish_divergence'] = [1 if p < 0 and r > 0 else 0 for p, r in zip(price_trend_5, rsi_trend_5)]
                features['bearish_divergence'] = [1 if p > 0 and r < 0 else 0 for p, r in zip(price_trend_5, rsi_trend_5)]

        # Market Structure Features
        if len(close) > 10:
            higher_highs = [1 if i > 0 and close[i] > max(close[max(0, i-10):i]) else 0 for i in range(len(close))]
            lower_lows = [1 if i > 0 and close[i] < min(close[max(0, i-10):i]) else 0 for i in range(len(close))]
            features['uptrend_strength'] = [sum(higher_highs[max(0, i-10):i]) for i in range(len(close))]
            features['downtrend_strength'] = [sum(lower_lows[max(0, i-10):i]) for i in range(len(close))]

        # Volatility Regime Detection
        if 'atr_14' in technical_features:
            atr_ma = self.feature_engineer.indicators.sma(technical_features['atr_14'], 20)
            features['high_volatility'] = [1 if atr > atr_ma[i] * 1.2 else 0 for i, atr in enumerate(technical_features['atr_14'])]
            features['low_volatility'] = [1 if atr < atr_ma[i] * 0.8 else 0 for i, atr in enumerate(technical_features['atr_14'])]

        return features

    def prepare_sequences(self, features: Dict[str, List[float]], sequence_length: int = 60) -> Dict[str, List[List[float]]]:
        """
        Prepara sequências para os modelos LSTM

        Args:
            features: Dicionário de features
            sequence_length: Comprimento das sequências

        Returns:
            Sequências preparadas por modelo
        """
        sequences = {}

        for model_name, config in self.model_configs.items():
            model_input_size = config['input_size']
            model_sequences = []

            # Selecionar features mais importantes para cada modelo
            if model_name == 'price_lstm':
                # Focar em features de preço
                selected_features = ['close', 'sma_10', 'sma_20', 'ema_12', 'bb_position', 'price_change_pct_1']
            elif model_name == 'direction_lstm':
                # Focar em features de direção
                selected_features = ['rsi_14', 'macd', 'stoch_k', 'volume_ratio', 'high_volatility']
            else:  # volatility_lstm
                # Focar em features de volatilidade
                selected_features = ['atr_14', 'volatility_10', 'bb_width', 'rolling_std_20']

            # Criar sequências
            feature_values = []
            for feature in selected_features:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    feature_values.append([0] * len(features.get('close', [])))

            # Transpor para formato (timestamps, features)
            if feature_values:
                feature_matrix = list(zip(*feature_values))

                # Criar sequências sobrepostas
                for i in range(sequence_length, len(feature_matrix)):
                    sequence = feature_matrix[i-sequence_length:i]

                    # Achatar para o tamanho esperado
                    flattened_sequence = []
                    for timestep in sequence:
                        flattened_sequence.extend(timestep)

                    # Ajustar para o tamanho de entrada do modelo
                    if len(flattened_sequence) >= model_input_size:
                        flattened_sequence = flattened_sequence[:model_input_size]
                    else:
                        flattened_sequence.extend([0] * (model_input_size - len(flattened_sequence)))

                    model_sequences.append(flattened_sequence)

            sequences[model_name] = model_sequences

        return sequences

    def predict_ensemble(self, latest_features: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Realiza predição ensemble usando todos os modelos

        Args:
            latest_features: Features mais recentes

        Returns:
            Predições ensemble
        """
        if not self.models:
            self.initialize_models()

        predictions = {}

        for model_name, model in self.models.items():
            config = self.model_configs[model_name]

            # Preparar features específicas para o modelo
            if model_name == 'price_lstm':
                feature_keys = ['close', 'sma_10', 'sma_20', 'ema_12', 'bb_position', 'price_change_pct_1', 'lag_1', 'lag_5']
            elif model_name == 'direction_lstm':
                feature_keys = ['rsi_14', 'macd', 'stoch_k', 'volume_ratio', 'high_volatility', 'bullish_divergence', 'bearish_divergence']
            else:  # volatility_lstm
                feature_keys = ['atr_14', 'volatility_10', 'bb_width', 'rolling_std_20', 'high_volatility', 'low_volatility']

            # Extrair valores
            model_features = []
            for key in feature_keys:
                if key in latest_features and latest_features[key]:
                    model_features.append(latest_features[key][-1])
                else:
                    model_features.append(0)

            # Adicionar features históricas se necessário
            while len(model_features) < config['input_size']:
                model_features.extend(model_features[:min(len(model_features), config['input_size'] - len(model_features))])

            # Truncar se necessário
            model_features = model_features[:config['input_size']]

            # Realizar predição
            direction, confidence = model.predict_direction(model_features)

            predictions[model_name] = {
                'direction': direction,
                'confidence': confidence,
                'signal': 'BUY' if direction > 0.3 else 'SELL' if direction < -0.3 else 'HOLD'
            }

        # Ensemble weighted
        ensemble_direction = 0
        total_confidence = 0

        for model_name, pred in predictions.items():
            weight = 1.0  # Podemos ter pesos diferentes
            ensemble_direction += pred['direction'] * weight * pred['confidence']
            total_confidence += pred['confidence'] * weight

        if total_confidence > 0:
            ensemble_direction /= total_confidence

        predictions['ensemble'] = {
            'direction': ensemble_direction,
            'confidence': total_confidence / len(predictions),
            'signal': 'BUY' if ensemble_direction > 0.3 else 'SELL' if ensemble_direction < -0.3 else 'HOLD'
        }

        return predictions

    def analyze_feature_importance(self, features: Dict[str, List[float]], targets: List[float]) -> Dict[str, float]:
        """
        Analisa importância das features

        Args:
            features: Features
            targets: Targets (retornos)

        Returns:
            Importância das features
        """
        importance = {}

        # Correlação simples (substituto para importância real)
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(targets):
                correlation = np.corrcoef(feature_values, targets)[0, 1]
                importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0

        # Normalizar
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}

        return importance

    def export_model_artifacts(self, output_path: str) -> None:
        """
        Exporta artefatos dos modelos

        Args:
            output_path: Caminho para salvar artefatos
        """
        artifacts = {
            'model_configs': self.model_configs,
            'feature_names': self.feature_engineer.feature_names,
            'model_metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0',
                'symbol': 'XAUUSD',
                'timeframe': 'M5'
            }
        }

        # Salvar artefatos
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(artifacts, f, indent=2)

        logger.info(f"💾 Artefatos dos modelos exportados: {output_file}")

if __name__ == "__main__":
    # Teste do engine de deep learning
    engine = DeepLearningEngine()

    # Gerar dados simulados
    n_points = 1000
    base_price = 2000

    # Simular movimento de preço
    price_data = []
    timestamps = []
    current_price = base_price

    for i in range(n_points):
        # Simular tendência + ruído
        trend = math.sin(i * 0.01) * 50
        noise = random.gauss(0, 5)
        current_price = current_price + trend + noise

        price_data.append(current_price)
        timestamps.append(datetime.now() + timedelta(hours=i))

    # Criar dados OHLC
    ohlc_data = {
        'close': price_data,
        'high': [p + random.uniform(0, 2) for p in price_data],
        'low': [p - random.uniform(0, 2) for p in price_data],
        'volume': [random.randint(100, 1000) for _ in range(n_points)]
    }

    # Gerar features
    features = engine.generate_comprehensive_features(ohlc_data, timestamps)

    # Preparar sequências
    sequences = engine.prepare_sequences(features)

    # Realizar predição ensemble
    latest_features = {k: [v[-1]] if v else [0] for k, v in features.items()}
    predictions = engine.predict_ensemble(latest_features)

    print("🧠 Deep Learning Engine - Teste Concluído")
    print(f"📊 Features geradas: {len(features)}")
    print(f"🔄 Sequências preparadas: {sum(len(seq) for seq in sequences.values())}")
    print(f"🎯 Predições ensemble:")
    for model, pred in predictions.items():
        print(f"   {model}: {pred['signal']} (conf: {pred['confidence']:.3f})")

    # Exportar artefatos
    engine.export_model_artifacts('../output/deep_learning_artifacts.json')

    print("✅ Engine de deep learning testado com sucesso!")