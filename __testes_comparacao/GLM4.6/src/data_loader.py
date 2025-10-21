#!/usr/bin/env python3
"""
📊 EA Optimizer AI - Data Loader Module
Carrega, limpa e processa dados de backtest para otimização
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestDataLoader:
    """Carregador e processador de dados de backtest"""

    def __init__(self, data_path: str):
        """
        Inicializa o carregador de dados

        Args:
            data_path: Caminho para o arquivo de dados (CSV ou JSON)
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None

    def load_data(self) -> pd.DataFrame:
        """
        Carrega dados de backtest de arquivos CSV ou JSON

        Returns:
            DataFrame com dados brutos
        """
        try:
            if self.data_path.suffix.lower() == '.csv':
                self.raw_data = self._load_csv()
            elif self.data_path.suffix.lower() == '.json':
                self.raw_data = self._load_json()
            else:
                raise ValueError(f"Formato de arquivo não suportado: {self.data_path.suffix}")

            logger.info(f"✅ Dados carregados: {len(self.raw_data)} registros")
            return self.raw_data

        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise

    def _load_csv(self) -> pd.DataFrame:
        """Carrega dados de arquivo CSV"""
        expected_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'profit', 'drawdown', 'winrate', 'sharpe_ratio', 'trades'
        ]

        df = pd.read_csv(self.data_path)

        # Validar colunas necessárias
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Colunas ausentes: {missing_cols}")

        # Converter timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def _load_json(self) -> pd.DataFrame:
        """Carrega dados de arquivo JSON"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Se for uma lista de trades
        if isinstance(data, list):
            df = pd.DataFrame(data)
        # Se for um dicionário com métricas
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Formato JSON não reconhecido")

        return df

    def clean_data(self) -> pd.DataFrame:
        """
        Limpa e valida os dados

        Returns:
            DataFrame limpo
        """
        if self.raw_data is None:
            raise ValueError("Carregue os dados primeiro com load_data()")

        df = self.raw_data.copy()

        # Remover valores nulos críticos
        critical_cols = ['profit', 'drawdown']
        df = df.dropna(subset=critical_cols)

        # Remover outliers extremos (mais de 5 desvios padrão)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['profit', 'drawdown']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df = df[np.abs(df[col] - mean_val) <= 5 * std_val]

        # Validar ranges lógicos
        if 'winrate' in df.columns:
            df = df[(df['winrate'] >= 0) & (df['winrate'] <= 100)]

        if 'drawdown' in df.columns:
            df = df[df['drawdown'] <= 100]  # Drawdown máximo de 100%

        self.processed_data = df
        logger.info(f"🧹 Dados limpos: {len(df)} registros (removidos {len(self.raw_data) - len(df)})")

        return df

    def engineer_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engenharia de features para o modelo de ML

        Returns:
            Tuple com features e targets
        """
        if self.processed_data is None:
            self.clean_data()

        df = self.processed_data.copy()

        # Features baseadas em parâmetros do EA
        feature_cols = []

        # Parâmetros técnicos
        tech_params = ['stop_loss', 'take_profit', 'atr_multiplier', 'risk_factor',
                      'ma_period', 'rsi_period', 'bb_std']

        # Parâmetros de sessão
        session_params = ['asian_session_start', 'asian_session_end',
                         'european_session_start', 'european_session_end',
                         'us_session_start', 'us_session_end']

        # Parâmetros de posicionamento
        position_params = ['lot_size', 'max_positions', 'pyramiding']

        all_params = tech_params + session_params + position_params

        # Adicionar features existentes
        for col in all_params:
            if col in df.columns:
                feature_cols.append(col)

        # Features derivadas
        if all(p in df.columns for p in ['stop_loss', 'take_profit']):
            df['risk_reward_ratio'] = df['take_profit'] / df['stop_loss']
            feature_cols.append('risk_reward_ratio')

        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            feature_cols.extend(['volume_ma', 'volume_ratio'])

        # Features de volatilidade
        if all(p in df.columns for p in ['high', 'low', 'close']):
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['true_range'].rolling(window=14).mean()
            df['volatility'] = df['atr'] / df['close']
            feature_cols.extend(['atr', 'volatility'])

        # Target: métrica composta de performance
        if 'profit' in df.columns and 'drawdown' in df.columns:
            # Profit Factor ajustado pelo drawdown
            df['profit_drawdown_ratio'] = np.where(
                df['drawdown'] > 0,
                df['profit'] / df['drawdown'],
                df['profit']
            )

            # Sharpe Ratio simulado se não existir
            if 'sharpe_ratio' not in df.columns:
                df['returns'] = df['profit'].pct_change()
                df['sharpe_simulated'] = df['returns'].mean() / df['returns'].std() if df['returns'].std() > 0 else 0
                df['sharpe_simulated'].fillna(0, inplace=True)

            # Target composto
            df['target_score'] = (
                df['profit_drawdown_ratio'] * 0.4 +
                df.get('sharpe_ratio', df.get('sharpe_simulated', 0)) * 0.3 +
                df.get('winrate', 0) / 100 * 0.3
            )

            target = df['target_score']
        else:
            target = df.get('profit', pd.Series(0, index=df.index))

        # Remover linhas com NaN nas features
        valid_features = [col for col in feature_cols if col in df.columns]
        features_df = df[valid_features].dropna()
        target = target.loc[features_df.index]

        self.features = features_df
        self.targets = target

        logger.info(f"🔧 Features criadas: {len(valid_features)} features, {len(features_df)} amostras")

        return features_df, target

    def get_feature_importance_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retorna dados prontos para análise de importância de features

        Returns:
            Tuple com features e targets
        """
        if self.features is None or self.targets is None:
            self.engineer_features()

        return self.features, self.targets

    def generate_summary_stats(self) -> Dict:
        """
        Gera estatísticas sumário dos dados

        Returns:
            Dicionário com estatísticas
        """
        if self.processed_data is None:
            self.clean_data()

        df = self.processed_data

        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else None,
                'end': df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else None
            },
            'performance_metrics': {
                'total_profit': df['profit'].sum() if 'profit' in df.columns else 0,
                'avg_profit': df['profit'].mean() if 'profit' in df.columns else 0,
                'max_drawdown': df['drawdown'].max() if 'drawdown' in df.columns else 0,
                'avg_drawdown': df['drawdown'].mean() if 'drawdown' in df.columns else 0,
                'winrate': df['winrate'].mean() if 'winrate' in df.columns else 0,
                'sharpe_ratio': df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0,
                'total_trades': df['trades'].sum() if 'trades' in df.columns else 0
            },
            'parameter_ranges': {}
        }

        # Adicionar ranges dos parâmetros
        param_cols = ['stop_loss', 'take_profit', 'atr_multiplier', 'risk_factor', 'lot_size']
        for col in param_cols:
            if col in df.columns:
                summary['parameter_ranges'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }

        return summary

def create_sample_data() -> None:
    """Cria dados de exemplo para teste"""
    np.random.seed(42)

    # Gerar dados simulados de backtest
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

    data = {
        'timestamp': dates,
        'stop_loss': np.random.uniform(50, 200, n_samples),
        'take_profit': np.random.uniform(100, 400, n_samples),
        'atr_multiplier': np.random.uniform(1.0, 3.0, n_samples),
        'risk_factor': np.random.uniform(0.5, 2.5, n_samples),
        'lot_size': np.random.uniform(0.01, 0.5, n_samples),
        'ma_period': np.random.randint(10, 50, n_samples),
        'rsi_period': np.random.randint(10, 30, n_samples),
        'open': np.random.uniform(1900, 2000, n_samples),
        'high': np.random.uniform(1950, 2050, n_samples),
        'low': np.random.uniform(1850, 1950, n_samples),
        'close': np.random.uniform(1900, 2000, n_samples),
        'volume': np.random.uniform(100, 1000, n_samples),
    }

    # Calcular métricas de performance baseadas nos parâmetros
    df = pd.DataFrame(data)

    # Simulação de profit baseada na relação risk/reward
    df['risk_reward'] = df['take_profit'] / df['stop_loss']
    df['profit'] = np.random.normal(100 * df['risk_reward'], 50, n_samples)

    # Drawdown inversamente proporcional ao risk/reward
    df['drawdown'] = np.random.uniform(5, 30, n_samples) / np.sqrt(df['risk_reward'])

    # Winrate baseada no ATR multiplier
    df['winrate'] = np.clip(50 + 10 * (2.0 - df['atr_multiplier']) + np.random.normal(0, 5, n_samples), 20, 80)

    # Sharpe ratio
    returns = df['profit'] / 1000  # Normalizar
    df['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Número de trades
    df['trades'] = np.random.poisson(20, n_samples)

    # Salvar dados de exemplo
    output_dir = Path(__file__).parent.parent / "data" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "sample_backtest.csv", index=False)
    logger.info(f"📊 Dados de exemplo criados: {output_dir / 'sample_backtest.csv'}")

if __name__ == "__main__":
    # Criar dados de exemplo
    create_sample_data()

    # Testar o carregador
    loader = BacktestDataLoader("data/input/sample_backtest.csv")
    loader.load_data()
    loader.clean_data()
    features, targets = loader.engineer_features()

    summary = loader.generate_summary_stats()
    print("📈 Estatísticas Resumo:")
    print(json.dumps(summary, indent=2, default=str))