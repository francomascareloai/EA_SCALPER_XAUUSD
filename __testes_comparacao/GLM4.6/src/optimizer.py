#!/usr/bin/env python3
"""
🤖 EA Optimizer AI - Optimization Engine
Módulo principal de otimização usando Optuna e Machine Learning
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json

from data_loader import BacktestDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EAOptimizer:
    """Otimizador de parâmetros de EA usando Optuna e Machine Learning"""

    def __init__(self,
                 data_path: str,
                 symbol: str = "XAUUSD",
                 timeframe: str = "M5",
                 optimization_metric: str = "profit_drawdown_ratio"):
        """
        Inicializa o otimizador

        Args:
            data_path: Caminho para os dados de backtest
            symbol: Símbolo trading (default: XAUUSD)
            timeframe: Timeframe para análise (default: M5)
            optimization_metric: Métrica principal de otimização
        """
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.optimization_metric = optimization_metric

        # Componentes
        self.data_loader = BacktestDataLoader(data_path)
        self.study = None
        self.best_params = None
        self.optimization_history = []
        self.ml_model = None
        self.scaler = StandardScaler()

        # Definição dos parâmetros otimizáveis
        self.param_space = self._define_parameter_space()

    def _define_parameter_space(self) -> Dict[str, Dict]:
        """
        Define o espaço de busca de parâmetros

        Returns:
            Dicionário com definições dos parâmetros
        """
        return {
            # Risk Management
            'stop_loss': {
                'type': 'int',
                'low': 50,
                'high': 300,
                'step': 10
            },
            'take_profit': {
                'type': 'int',
                'low': 100,
                'high': 600,
                'step': 10
            },
            'risk_factor': {
                'type': 'float',
                'low': 0.5,
                'high': 3.0,
                'step': 0.1
            },
            'atr_multiplier': {
                'type': 'float',
                'low': 0.8,
                'high': 3.0,
                'step': 0.1
            },

            # Technical Indicators
            'ma_period': {
                'type': 'int',
                'low': 5,
                'high': 50,
                'step': 5
            },
            'rsi_period': {
                'type': 'int',
                'low': 10,
                'high': 30,
                'step': 2
            },
            'rsi_oversold': {
                'type': 'int',
                'low': 20,
                'high': 35,
                'step': 1
            },
            'rsi_overbought': {
                'type': 'int',
                'low': 65,
                'high': 80,
                'step': 1
            },
            'bb_std': {
                'type': 'float',
                'low': 1.5,
                'high': 2.5,
                'step': 0.1
            },

            # Position Sizing
            'lot_size': {
                'type': 'float',
                'low': 0.01,
                'high': 0.5,
                'step': 0.01
            },
            'max_positions': {
                'type': 'int',
                'low': 1,
                'high': 5,
                'step': 1
            },

            # Trading Sessions
            'asian_session_start': {
                'type': 'int',
                'low': 0,
                'high': 4,
                'step': 1
            },
            'asian_session_end': {
                'type': 'int',
                'low': 6,
                'high': 9,
                'step': 1
            },
            'european_session_start': {
                'type': 'int',
                'low': 7,
                'high': 11,
                'step': 1
            },
            'european_session_end': {
                'type': 'int',
                'low': 16,
                'high': 19,
                'step': 1
            },
            'us_session_start': {
                'type': 'int',
                'low': 13,
                'high': 17,
                'step': 1
            },
            'us_session_end': {
                'type': 'int',
                'low': 21,
                'high': 23,
                'step': 1
            }
        }

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carrega e prepara dados para otimização

        Returns:
            Tuple com features e targets
        """
        logger.info("📊 Carregando e preparando dados...")

        # Carregar dados
        self.data_loader.load_data()
        self.data_loader.clean_data()
        features, targets = self.data_loader.engineer_features()

        logger.info(f"✅ Dados preparados: {len(features)} amostras, {len(features.columns)} features")
        return features, targets

    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para otimização com Optuna

        Args:
            trial: Trial do Optuna

        Returns:
            Score da otimização
        """
        # Gerar parâmetros para este trial
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config['step']
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config['step']
                )

        # Calcular score predito usando modelo ML
        if self.ml_model is not None:
            score = self._predict_score(params)
        else:
            # Fallback: usar heurística simples
            score = self._calculate_heuristic_score(params)

        # Adicionar penalidades para configurações inválidas
        score = self._apply_constraints_penalty(params, score)

        return score

    def _predict_score(self, params: Dict[str, float]) -> float:
        """
        Prediz score usando modelo de Machine Learning

        Args:
            params: Parâmetros do EA

        Returns:
            Score predito
        """
        # Preparar features para predição
        feature_vector = []
        feature_names = []

        for param_name in self.param_space.keys():
            if param_name in params:
                feature_vector.append(params[param_name])
                feature_names.append(param_name)

        # Adicionar features derivadas
        if 'stop_loss' in params and 'take_profit' in params:
            risk_reward = params['take_profit'] / params['stop_loss']
            feature_vector.append(risk_reward)
            feature_names.append('risk_reward_ratio')

        # Normalizar features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Predizer score
        score = self.ml_model.predict(feature_vector_scaled)[0]
        return float(score)

    def _calculate_heuristic_score(self, params: Dict[str, float]) -> float:
        """
        Calcula score baseado em heurísticas (fallback)

        Args:
            params: Parâmetros do EA

        Returns:
            Score heurístico
        """
        score = 0.0

        # Risk/Reward Ratio
        if 'stop_loss' in params and 'take_profit' in params:
            risk_reward = params['take_profit'] / params['stop_loss']
            score += min(risk_reward / 2.0, 3.0) * 25  # Máximo 75 pontos

        # Risk Factor (moderado é melhor)
        if 'risk_factor' in params:
            optimal_risk = 1.5
            risk_score = max(0, 1 - abs(params['risk_factor'] - optimal_risk) / optimal_risk)
            score += risk_score * 15  # Máximo 15 pontos

        # ATR Multiplier (valores moderados são melhores)
        if 'atr_multiplier' in params:
            optimal_atr = 1.5
            atr_score = max(0, 1 - abs(params['atr_multiplier'] - optimal_atr) / optimal_atr)
            score += atr_score * 10  # Máximo 10 pontos

        return score

    def _apply_constraints_penalty(self, params: Dict[str, float], score: float) -> float:
        """
        Aplica penalidades para violações de restrições

        Args:
            params: Parâmetros do EA
            score: Score original

        Returns:
            Score com penalidades aplicadas
        """
        penalty = 0.0

        # Stop Loss deve ser menor que Take Profit
        if 'stop_loss' in params and 'take_profit' in params:
            if params['stop_loss'] >= params['take_profit']:
                penalty += 50  # Penalidade grande

        # Risk Factor não deve ser muito alto
        if 'risk_factor' in params and params['risk_factor'] > 2.5:
            penalty += 20

        # Lot size não deve ser muito alto para XAUUSD
        if 'lot_size' in params and params['lot_size'] > 0.3:
            penalty += 15

        # Sessões não devem ter overlap impossível
        if all(key in params for key in ['asian_session_end', 'european_session_start']):
            if params['asian_session_end'] > params['european_session_start']:
                penalty += 10

        return max(0, score - penalty)

    def train_ml_model(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """
        Treina modelo de Machine Learning para predição de performance

        Args:
            features: Features de treinamento
            targets: Targets de treinamento
        """
        logger.info("🤖 Treinando modelo de Machine Learning...")

        # Split de dados
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Treinar modelo ensemble
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        best_model = None
        best_score = float('-inf')

        for name, model in models.items():
            # Treinar modelo
            model.fit(X_train_scaled, y_train)

            # Avaliar
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)

            logger.info(f"📈 Modelo {name}: R² = {r2:.4f}")

            if r2 > best_score:
                best_score = r2
                best_model = model

        self.ml_model = best_model
        logger.info(f"✅ Modelo treinado: {type(best_model).__name__} (R² = {best_score:.4f})")

    def optimize(self,
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 study_name: str = "ea_optimization") -> Dict[str, Any]:
        """
        Executa otimização de parâmetros

        Args:
            n_trials: Número de trials para otimização
            timeout: Timeout em segundos
            study_name: Nome do estudo

        Returns:
            Resultados da otimização
        """
        logger.info(f"🚀 Iniciando otimização: {n_trials} trials")

        # Carregar e preparar dados
        features, targets = self.load_and_prepare_data()

        # Treinar modelo ML
        self.train_ml_model(features, targets)

        # Criar estudo Optuna
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Callback para logging
        def callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(f"Trial {trial.number}: Best score = {study.best_value:.4f}")

        # Executar otimização
        self.study.optimize(
            lambda trial: self.objective_function(trial),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback]
        )

        # Armazenar melhores parâmetros
        self.best_params = self.study.best_params

        results = {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study_name': study_name,
            'optimization_history': [
                {
                    'trial_number': trial.number,
                    'score': trial.value,
                    'params': trial.params
                }
                for trial in self.study.trials
            ]
        }

        logger.info(f"✅ Otimização concluída: Best score = {self.study.best_value:.4f}")
        return results

    def optimize_with_cross_validation(self,
                                     n_trials: int = 100,
                                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Otimização com validação cruzada para robustez

        Args:
            n_trials: Número de trials
            cv_folds: Número de folds para CV

        Returns:
            Resultados com validação cruzada
        """
        logger.info(f"🔄 Iniciando otimização com CV ({cv_folds} folds)")

        # Carregar dados
        features, targets = self.load_and_prepare_data()

        # Otimização principal
        main_results = self.optimize(n_trials=n_trials)

        # Validação cruzada dos melhores parâmetros
        cv_scores = []
        for fold in range(cv_folds):
            # Split de dados para este fold
            X_train, X_val = train_test_split(features, test_size=0.2, random_state=fold)
            y_train, y_val = train_test_split(targets, test_size=0.2, random_state=fold)

            # Treinar modelo com fold específico
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = GradientBoostingRegressor(n_estimators=100, random_state=fold)
            model.fit(X_train_scaled, y_train)

            # Avaliar melhores parâmetros
            score = self._evaluate_params_with_model(self.best_params, model, scaler)
            cv_scores.append(score)

        cv_results = {
            'main_optimization': main_results,
            'cross_validation': {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_folds': cv_folds
            }
        }

        logger.info(f"✅ CV concluído: Mean CV score = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        return cv_results

    def _evaluate_params_with_model(self,
                                  params: Dict[str, float],
                                  model: Any,
                                  scaler: StandardScaler) -> float:
        """
        Avalia parâmetros usando modelo específico

        Args:
            params: Parâmetros para avaliar
            model: Modelo ML treinado
            scaler: Scaler ajustado

        Returns:
            Score avaliado
        """
        # Preparar features
        feature_vector = []
        for param_name in self.param_space.keys():
            if param_name in params:
                feature_vector.append(params[param_name])

        # Features derivadas
        if 'stop_loss' in params and 'take_profit' in params:
            risk_reward = params['take_profit'] / params['stop_loss']
            feature_vector.append(risk_reward)

        # Predizer
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)
        score = model.predict(feature_vector_scaled)[0]

        return float(score)

    def save_results(self, output_path: str) -> None:
        """
        Salva resultados da otimização

        Args:
            output_path: Caminho para salvar resultados
        """
        results = {
            'best_params': self.best_params,
            'best_score': self.study.best_value if self.study else None,
            'optimization_history': self.optimization_history,
            'parameter_space': self.param_space,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'optimization_metric': self.optimization_metric
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Resultados salvos: {output_file}")

    def load_results(self, input_path: str) -> None:
        """
        Carrega resultados de otimização anteriores

        Args:
            input_path: Caminho para carregar resultados
        """
        with open(input_path, 'r') as f:
            results = json.load(f)

        self.best_params = results['best_params']
        self.optimization_history = results.get('optimization_history', [])
        self.symbol = results['symbol']
        self.timeframe = results['timeframe']
        self.optimization_metric = results['optimization_metric']

        logger.info(f"📂 Resultados carregados: {input_path}")

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Gera relatório detalhado da otimização

        Returns:
            Relatório completo da otimização
        """
        if not self.study:
            raise ValueError("Execute otimização primeiro")

        # Análise dos trials
        trials_df = self.study.trials_dataframe()

        # Importância dos parâmetros
        param_importance = optuna.importance.get_param_importances(self.study)

        # Estatísticas dos melhores parâmetros
        best_trials = sorted(self.study.trials, key=lambda t: t.value, reverse=True)[:10]

        report = {
            'optimization_summary': {
                'total_trials': len(self.study.trials),
                'best_score': self.study.best_value,
                'best_params': self.best_params,
                'study_name': self.study.study_name
            },
            'parameter_analysis': {
                'importance': dict(param_importance),
                'best_trials_params': [
                    {
                        'rank': i + 1,
                        'score': trial.value,
                        'params': trial.params
                    }
                    for i, trial in enumerate(best_trials)
                ]
            },
            'convergence_analysis': {
                'trials_scores': [trial.value for trial in self.study.trials],
                'best_scores_so_far': list(self.study.trials_dataframe()['value'].expanding().max())
            }
        }

        return report

if __name__ == "__main__":
    # Teste do otimizador
    optimizer = EAOptimizer(
        data_path="../data/input/sample_backtest.csv",
        symbol="XAUUSD",
        timeframe="M5"
    )

    # Executar otimização
    results = optimizer.optimize(n_trials=50)

    # Gerar relatório
    report = optimizer.generate_optimization_report()
    print("📊 Relatório de Otimização:")
    print(json.dumps(report, indent=2, default=str))

    # Salvar resultados
    optimizer.save_results("../output/optimized_params.json")