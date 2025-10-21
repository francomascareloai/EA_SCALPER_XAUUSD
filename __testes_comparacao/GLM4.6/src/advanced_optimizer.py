#!/usr/bin/env python3
"""
🧠 EA Optimizer AI - Advanced Optimization Engine (Rodada 2)
Sistema de otimização multi-objetivo com algoritmos ensemble e deep learning
"""

import numpy as np
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import deque

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Resultado de otimização avançada"""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    pareto_rank: int
    crowding_distance: float
    diversity_score: float
    robustness_score: float

class AdvancedEAOptimizer:
    """Otimizador avançado com multi-objective optimization"""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M5"):
        """
        Inicializa o otimizador avançado

        Args:
            symbol: Símbolo de trading
            timeframe: Timeframe de análise
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.population = []
        self.pareto_front = []
        self.optimization_history = []
        self.ensemble_models = []

        # Definição avançada do espaço de parâmetros
        self.param_space = self._define_advanced_parameter_space()

        # Configurações de otimização
        self.population_size = 100
        self.max_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

        # Objetivos multi-dimensional
        self.objectives = [
            'profit_factor',      # Maximizar
            'sharpe_ratio',       # Maximizar
            'max_drawdown',       # Minimizar
            'win_rate',          # Maximizar
            'calmar_ratio',       # Maximizar
            'diversification',    # Maximizar
            'robustness',         # Maximizar
            'stability'           # Maximizar
        ]

    def _define_advanced_parameter_space(self) -> Dict[str, Dict]:
        """
        Define espaço avançado de parâmetros com mais granularidade

        Returns:
            Dicionário com definições detalhadas dos parâmetros
        """
        return {
            # Risk Management Avançado
            'dynamic_stop_loss': {
                'type': 'float',
                'low': 0.5,
                'high': 3.0,
                'step': 0.1,
                'description': 'Multiplicador dinâmico de SL baseado em volatilidade'
            },
            'adaptive_take_profit': {
                'type': 'float',
                'low': 1.0,
                'high': 5.0,
                'step': 0.1,
                'description': 'TP adaptativo baseado em momentum'
            },
            'volatility_target': {
                'type': 'float',
                'low': 0.05,
                'high': 0.30,
                'step': 0.01,
                'description': 'Target de volatilidade anualizada'
            },
            'risk_parity_weight': {
                'type': 'float',
                'low': 0.1,
                'high': 0.9,
                'step': 0.05,
                'description': 'Peso para risk parity'
            },

            # Machine Learning Parameters
            'ml_lookback_period': {
                'type': 'int',
                'low': 50,
                'high': 500,
                'step': 10,
                'description': 'Período de lookback para modelos ML'
            },
            'feature_importance_threshold': {
                'type': 'float',
                'low': 0.01,
                'high': 0.20,
                'step': 0.01,
                'description': 'Threshold para importância de features'
            },
            'ensemble_weights': {
                'type': 'dict',
                'description': 'Pesos para ensemble de modelos'
            },

            # Market Microstructure
            'spread_threshold': {
                'type': 'float',
                'low': 0.5,
                'high': 5.0,
                'step': 0.1,
                'description': 'Threshold máximo de spread para operar'
            },
            'liquidity_filter': {
                'type': 'float',
                'low': 0.1,
                'high': 1.0,
                'step': 0.05,
                'description': 'Filtro de liquidez mínima'
            },
            'market_impact_factor': {
                'type': 'float',
                'low': 0.001,
                'high': 0.05,
                'step': 0.001,
                'description': 'Fator de impacto de mercado'
            },

            # Advanced Technical Indicators
            'adaptive_ma_period': {
                'type': 'int',
                'low': 5,
                'high': 100,
                'step': 5,
                'description': 'Período de MA adaptativa'
            },
            'rsi_divergence_threshold': {
                'type': 'float',
                'low': 0.5,
                'high': 2.0,
                'step': 0.1,
                'description': 'Threshold para divergência RSI'
            },
            'bollinger_squeeze_threshold': {
                'type': 'float',
                'low': 0.8,
                'high': 1.5,
                'step': 0.05,
                'description': 'Threshold para squeeze de BB'
            },

            # Time and Volume Analysis
            'volume_profile_threshold': {
                'type': 'float',
                'low': 0.2,
                'high': 0.8,
                'step': 0.05,
                'description': 'Threshold para volume profile'
            },
            'time_decay_factor': {
                'type': 'float',
                'low': 0.9,
                'high': 0.99,
                'step': 0.01,
                'description': 'Fator de decaimento temporal'
            },
            'seasonal_adjustment': {
                'type': 'bool',
                'description': 'Ajuste sazonal ativado'
            },

            # Portfolio Management
            'correlation_threshold': {
                'type': 'float',
                'low': 0.1,
                'high': 0.9,
                'step': 0.05,
                'description': 'Threshold máximo de correlação'
            },
            'max_portfolio_exposure': {
                'type': 'float',
                'low': 0.1,
                'high': 1.0,
                'step': 0.05,
                'description': 'Exposição máxima do portfolio'
            },
            'rebalance_frequency': {
                'type': 'int',
                'low': 1,
                'high': 24,
                'step': 1,
                'description': 'Frequência de rebalanceamento (horas)'
            }
        }

    def initialize_population(self) -> List[Dict[str, Any]]:
        """
        Inicializa população para algoritmo genético

        Returns:
            Lista inicial de indivíduos
        """
        population = []

        for _ in range(self.population_size):
            individual = {}

            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'float':
                    individual[param_name] = random.uniform(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'int':
                    individual[param_name] = random.randint(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'bool':
                    individual[param_name] = random.choice([True, False])
                elif param_config['type'] == 'dict':
                    # Para ensemble weights
                    individual[param_name] = {
                        'lstm': random.uniform(0.2, 0.5),
                        'xgboost': random.uniform(0.2, 0.5),
                        'random_forest': random.uniform(0.1, 0.3)
                    }

            population.append(individual)

        return population

    def evaluate_multi_objectives(self, individual: Dict[str, Any]) -> Dict[str, float]:
        """
        Avalia múltiplos objetivos para um indivíduo

        Args:
            individual: Parâmetros do indivíduo

        Returns:
            Dicionário com valores dos objetivos
        """
        # Simulação avançada de performance
        objectives = {}

        # 1. Profit Factor (Maximizar)
        risk_reward = individual['adaptive_take_profit'] / individual['dynamic_stop_loss']
        base_profit_factor = 1.5 + (risk_reward - 1.0) * 0.5

        # Ajustes baseados em outros parâmetros
        if individual['volatility_target'] < 0.15:
            base_profit_factor += 0.3
        if individual['feature_importance_threshold'] < 0.05:
            base_profit_factor += 0.2

        objectives['profit_factor'] = max(0.5, min(5.0, base_profit_factor + random.uniform(-0.3, 0.3)))

        # 2. Sharpe Ratio (Maximizar)
        base_sharpe = 1.2 + individual['risk_parity_weight'] * 0.8

        if individual['ml_lookback_period'] > 200:
            base_sharpe += 0.3
        if individual['spread_threshold'] < 2.0:
            base_sharpe += 0.2

        objectives['sharpe_ratio'] = max(0.3, min(4.0, base_sharpe + random.uniform(-0.4, 0.4)))

        # 3. Maximum Drawdown (Minimizar)
        base_drawdown = 15.0 - individual['volatility_target'] * 30

        if individual['market_impact_factor'] < 0.01:
            base_drawdown -= 2.0
        if individual['liquidity_filter'] > 0.5:
            base_drawdown -= 1.5

        objectives['max_drawdown'] = max(2.0, min(25.0, base_drawdown + random.uniform(-3, 3)))

        # 4. Win Rate (Maximizar)
        base_winrate = 45 + individual['rsi_divergence_threshold'] * 10

        if individual['adaptive_ma_period'] > 20:
            base_winrate += 5
        if individual['bollinger_squeeze_threshold'] < 1.0:
            base_winrate += 3

        objectives['win_rate'] = max(25.0, min(85.0, base_winrate + random.uniform(-5, 5)))

        # 5. Calmar Ratio (Maximizar)
        profit = objectives['profit_factor'] * 1000  # Simulação
        calmar = profit / max(objectives['max_drawdown'], 1.0)
        objectives['calmar_ratio'] = max(0.2, min(8.0, calmar + random.uniform(-0.5, 0.5)))

        # 6. Diversification (Maximizar)
        base_diversification = individual['correlation_threshold'] * 50

        if individual['max_portfolio_exposure'] < 0.5:
            base_diversification += 15
        if individual['rebalance_frequency'] < 12:
            base_diversification += 10

        objectives['diversification'] = max(10.0, min(90.0, base_diversification + random.uniform(-8, 8)))

        # 7. Robustness (Maximizar)
        base_robustness = 40 + (1.0 - individual['time_decay_factor']) * 200

        if individual['seasonal_adjustment']:
            base_robustness += 15
        if individual['volume_profile_threshold'] > 0.4:
            base_robustness += 10

        objectives['robustness'] = max(20.0, min(95.0, base_robustness + random.uniform(-10, 10)))

        # 8. Stability (Maximizar)
        base_stability = 50 + individual['liquidity_filter'] * 30

        if individual['spread_threshold'] < 3.0:
            base_stability += 20
        if individual['market_impact_factor'] < 0.02:
            base_stability += 15

        objectives['stability'] = max(15.0, min(90.0, base_stability + random.uniform(-12, 12)))

        return objectives

    def fast_non_dominated_sort(self, population: List[OptimizationResult]) -> List[List[OptimizationResult]]:
        """
        Algoritmo Fast Non-Dominated Sorting (NSGA-II)

        Args:
            population: População de resultados

        Returns:
            Fronts de Pareto
        """
        fronts = [[]]

        # Calcular domination counts
        for i, p in enumerate(population):
            p.domination_count = 0
            p.dominated_solutions = []

            for j, q in enumerate(population):
                if i != j:
                    if self._dominates(p, q):
                        p.dominated_solutions.append(j)
                    elif self._dominates(q, p):
                        p.domination_count += 1

            if p.domination_count == 0:
                p.pareto_rank = 0
                fronts[0].append(p)

        # Construir fronts subsequentes
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q_index in p.dominated_solutions:
                    q = population[q_index]
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.pareto_rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remover último front vazio

    def _dominates(self, p: OptimizationResult, q: OptimizationResult) -> bool:
        """
        Verifica se p domina q nos objetivos

        Args:
            p: Primeiro resultado
            q: Segundo resultado

        Returns:
            True se p domina q
        """
        p_wins = 0
        q_wins = 0

        for obj in self.objectives:
            p_val = p.objectives[obj]
            q_val = q.objectives[obj]

            # Ajustar direção baseada no objetivo
            if obj in ['max_drawdown']:  # Objetivos a minimizar
                if p_val < q_val:
                    p_wins += 1
                elif q_val < p_val:
                    q_wins += 1
            else:  # Objetivos a maximizar
                if p_val > q_val:
                    p_wins += 1
                elif q_val > p_val:
                    q_wins += 1

        return p_wins > 0 and q_wins == 0

    def calculate_crowding_distance(self, front: List[OptimizationResult]) -> None:
        """
        Calcula distância de crowding para diversidade

        Args:
            front: Front de Pareto
        """
        if len(front) == 0:
            return

        # Inicializar crowding distance
        for p in front:
            p.crowding_distance = 0.0

        # Calcular para cada objetivo
        for obj in self.objectives:
            # Ordenar por objetivo
            front.sort(key=lambda x: x.objectives[obj])

            # Extremos recebem distância infinita
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calcular range do objetivo
            obj_min = front[0].objectives[obj]
            obj_max = front[-1].objectives[obj]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calcular crowding distance para pontos intermediários
            for i in range(1, len(front) - 1):
                distance = front[i + 1].objectives[obj] - front[i - 1].objectives[obj]
                front[i].crowding_distance += distance / obj_range

    def tournament_selection(self, population: List[OptimizationResult], tournament_size: int = 3) -> OptimizationResult:
        """
        Seleção por torneio

        Args:
            population: População atual
            tournament_size: Tamanho do torneio

        Returns:
            Indivíduo selecionado
        """
        tournament = random.sample(population, min(tournament_size, len(population)))

        # Ordenar por (pareto_rank, -crowding_distance)
        tournament.sort(key=lambda x: (x.pareto_rank, -x.crowding_distance))

        return tournament[0]

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Operação de crossover (recombinação)

        Args:
            parent1: Primeiro pai
            parent2: Segundo pai

        Returns:
            Dois filhos gerados
        """
        child1, child2 = {}, {}

        for param_name, param_config in self.param_space.items():
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]

            # Aplicar crossover para tipos complexos
            if param_config['type'] == 'dict' and param_name in parent1:
                for key in parent1[param_name]:
                    if random.random() < 0.5:
                        child1[param_name][key] = parent1[param_name][key]
                        child2[param_name][key] = parent2[param_name][key]
                    else:
                        child1[param_name][key] = parent2[param_name][key]
                        child2[param_name][key] = parent1[param_name][key]

        return child1, child2

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Operação de mutação

        Args:
            individual: Indivíduo a mutar

        Returns:
            Indivíduo mutado
        """
        mutated = individual.copy()

        for param_name, param_config in self.param_space.items():
            if random.random() < self.mutation_rate:
                if param_config['type'] == 'float':
                    # Gaussian mutation
                    current_value = individual[param_name]
                    mutation_strength = (param_config['high'] - param_config['low']) * 0.1
                    new_value = current_value + random.gauss(0, mutation_strength)
                    mutated[param_name] = max(param_config['low'], min(param_config['high'], new_value))

                elif param_config['type'] == 'int':
                    # Random reset mutation
                    mutated[param_name] = random.randint(param_config['low'], param_config['high'])

                elif param_config['type'] == 'bool':
                    mutated[param_name] = not individual[param_name]

                elif param_config['type'] == 'dict':
                    # Mutate dictionary values
                    for key in mutated[param_name]:
                        if random.random() < 0.3:
                            mutated[param_name][key] = random.uniform(0.1, 0.6)

                    # Normalize to sum to 1
                    total = sum(mutated[param_name].values())
                    for key in mutated[param_name]:
                        mutated[param_name][key] /= total

        return mutated

    def advanced_optimization_cycle(self) -> List[OptimizationResult]:
        """
        Executa ciclo completo de otimização avançada

        Returns:
            Front de Pareto final
        """
        logger.info("🧠 Iniciando otimização multi-objetivo avançada...")

        # Inicializar população
        population_params = self.initialize_population()
        population = []

        # Avaliar população inicial
        for params in population_params:
            objectives = self.evaluate_multi_objectives(params)

            result = OptimizationResult(
                parameters=params,
                objectives=objectives,
                pareto_rank=0,
                crowding_distance=0.0,
                diversity_score=0.0,
                robustness_score=0.0
            )
            population.append(result)

        logger.info(f"📊 População inicial: {len(population)} indivíduos")

        # Evolução por gerações
        for generation in range(self.max_generations):
            logger.info(f"🔄 Geração {generation + 1}/{self.max_generations}")

            # Fast Non-Dominated Sort
            fronts = self.fast_non_dominated_sort(population)

            # Calcular crowding distance
            for front in fronts:
                self.calculate_crowding_distance(front)

            # Criar nova população
            offspring = []

            while len(offspring) < self.population_size:
                # Seleção por torneio
                parent1_result = self.tournament_selection(population)
                parent2_result = self.tournament_selection(population)

                # Crossover
                child1_params, child2_params = self.crossover(
                    parent1_result.parameters,
                    parent2_result.parameters
                )

                # Mutação
                child1_params = self.mutate(child1_params)
                child2_params = self.mutate(child2_params)

                # Avaliar filhos
                child1_objectives = self.evaluate_multi_objectives(child1_params)
                child2_objectives = self.evaluate_multi_objectives(child2_params)

                child1 = OptimizationResult(
                    parameters=child1_params,
                    objectives=child1_objectives,
                    pareto_rank=0,
                    crowding_distance=0.0,
                    diversity_score=0.0,
                    robustness_score=0.0
                )

                child2 = OptimizationResult(
                    parameters=child2_params,
                    objectives=child2_objectives,
                    pareto_rank=0,
                    crowding_distance=0.0,
                    diversity_score=0.0,
                    robustness_score=0.0
                )

                offspring.extend([child1, child2])

            # Combinar pais e filhos
            combined_population = population + offspring

            # Selecionar próxima geração
            fronts = self.fast_non_dominated_sort(combined_population)

            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    self.calculate_crowding_distance(front)
                    next_population.extend(front)
                else:
                    # Preencher com melhores do último front
                    self.calculate_crowding_distance(front)
                    front.sort(key=lambda x: -x.crowding_distance)
                    remaining = self.population_size - len(next_population)
                    next_population.extend(front[:remaining])
                    break

            population = next_population

            # Calcular métricas da geração
            best_pareto = fronts[0] if fronts else []
            avg_fitness = np.mean([ind.objectives['sharpe_ratio'] for ind in best_pareto[:10]])

            logger.info(f"   📈 Best Pareto Size: {len(best_pareto)}")
            logger.info(f"   🎯 Avg Sharpe (Top 10): {avg_fitness:.3f}")

            # Salvar histórico
            self.optimization_history.append({
                'generation': generation + 1,
                'pareto_front_size': len(best_pareto),
                'best_sharpe': max([ind.objectives['sharpe_ratio'] for ind in best_pareto]) if best_pareto else 0,
                'avg_profit_factor': np.mean([ind.objectives['profit_factor'] for ind in best_pareto[:5]]) if best_pareto else 0,
                'best_diversity': max([ind.objectives['diversification'] for ind in best_pareto]) if best_pareto else 0
            })

        # Front final de Pareto
        final_fronts = self.fast_non_dominated_sort(population)
        self.calculate_crowding_distance(final_fronts[0])
        self.pareto_front = final_fronts[0]

        logger.info(f"✅ Otimização concluída: {len(self.pareto_front)} soluções não-dominadas")

        return self.pareto_front

    def select_best_solution(self, criteria: str = 'balanced') -> OptimizationResult:
        """
        Seleciona melhor solução do front de Pareto

        Args:
            criteria: Critério de seleção ('sharpe', 'profit', 'balanced', 'conservative')

        Returns:
            Melhor solução selecionada
        """
        if not self.pareto_front:
            raise ValueError("Execute otimização primeiro")

        if criteria == 'sharpe':
            # Maximizar Sharpe Ratio
            best = max(self.pareto_front, key=lambda x: x.objectives['sharpe_ratio'])

        elif criteria == 'profit':
            # Maximizar Profit Factor
            best = max(self.pareto_front, key=lambda x: x.objectives['profit_factor'])

        elif criteria == 'conservative':
            # Minimizar Drawdown com bom Sharpe
            conservative = [x for x in self.pareto_front if x.objectives['max_drawdown'] < 10]
            if conservative:
                best = max(conservative, key=lambda x: x.objectives['sharpe_ratio'])
            else:
                best = min(self.pareto_front, key=lambda x: x.objectives['max_drawdown'])

        else:  # balanced
            # Score balanceado
            def balanced_score(x):
                return (
                    x.objectives['sharpe_ratio'] * 0.3 +
                    x.objectives['profit_factor'] * 0.25 +
                    (25 - x.objectives['max_drawdown']) * 0.2 +
                    x.objectives['win_rate'] * 0.15 +
                    x.objectives['stability'] * 0.1
                )

            best = max(self.pareto_front, key=balanced_score)

        logger.info(f"🏆 Melhor solução selecionada (critério: {criteria})")
        logger.info(f"   📊 Sharpe: {best.objectives['sharpe_ratio']:.3f}")
        logger.info(f"   💰 Profit Factor: {best.objectives['profit_factor']:.3f}")
        logger.info(f"   📉 Max DD: {best.objectives['max_drawdown']:.2f}%")
        logger.info(f"   🎯 Win Rate: {best.objectives['win_rate']:.2f}%")

        return best

    def generate_ensemble_models(self, top_n: int = 5) -> None:
        """
        Gera ensemble de modelos a partir das melhores soluções

        Args:
            top_n: Número de melhores soluções para ensemble
        """
        if not self.pareto_front:
            raise ValueError("Execute otimização primeiro")

        # Selecionar top N soluções diversificadas
        sorted_by_diversity = sorted(
            self.pareto_front,
            key=lambda x: x.objectives['diversification'],
            reverse=True
        )

        diverse_solutions = sorted_by_diversity[:top_n]

        # Criar ensemble weights
        total_sharpe = sum(sol.objectives['sharpe_ratio'] for sol in diverse_solutions)

        self.ensemble_models = []
        for i, solution in enumerate(diverse_solutions):
            weight = solution.objectives['sharpe_ratio'] / total_sharpe

            ensemble_model = {
                'id': i,
                'parameters': solution.parameters,
                'objectives': solution.objectives,
                'weight': weight,
                'specialization': self._identify_specialization(solution)
            }

            self.ensemble_models.append(ensemble_model)

        logger.info(f"🤖 Ensemble criado com {len(self.ensemble_models)} modelos")

    def _identify_specialization(self, solution: OptimizationResult) -> str:
        """
        Identifica especialização da solução

        Args:
            solution: Solução para analisar

        Returns:
            Tipo de especialização
        """
        obj = solution.objectives

        if obj['profit_factor'] > 2.5:
            return "high_profit"
        elif obj['max_drawdown'] < 8:
            return "low_risk"
        elif obj['diversification'] > 70:
            return "diversified"
        elif obj['stability'] > 80:
            return "stable"
        elif obj['sharpe_ratio'] > 2.5:
            return "high_sharpe"
        else:
            return "balanced"

    def export_results(self, output_path: str) -> Dict[str, Any]:
        """
        Exporta resultados completos da otimização

        Args:
            output_path: Caminho para salvar resultados

        Returns:
            Resultados exportados
        """
        results = {
            'optimization_config': {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'objectives': self.objectives
            },
            'pareto_front': [
                {
                    'parameters': sol.parameters,
                    'objectives': sol.objectives,
                    'pareto_rank': sol.pareto_rank,
                    'crowding_distance': sol.crowding_distance
                }
                for sol in self.pareto_front
            ],
            'optimization_history': self.optimization_history,
            'ensemble_models': self.ensemble_models,
            'best_solutions': {
                'balanced': self.select_best_solution('balanced').objectives,
                'sharpe': self.select_best_solution('sharpe').objectives,
                'profit': self.select_best_solution('profit').objectives,
                'conservative': self.select_best_solution('conservative').objectives
            },
            'statistics': {
                'pareto_front_size': len(self.pareto_front),
                'avg_sharpe': np.mean([sol.objectives['sharpe_ratio'] for sol in self.pareto_front]),
                'avg_profit_factor': np.mean([sol.objectives['profit_factor'] for sol in self.pareto_front]),
                'avg_drawdown': np.mean([sol.objectives['max_drawdown'] for sol in self.pareto_front])
            }
        }

        # Salvar resultados
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Resultados exportados: {output_file}")
        return results

if __name__ == "__main__":
    # Teste do otimizador avançado
    optimizer = AdvancedEAOptimizer()

    # Executar otimização completa
    pareto_front = optimizer.advanced_optimization_cycle()

    # Selecionar melhor solução
    best_solution = optimizer.select_best_solution('balanced')

    # Gerar ensemble
    optimizer.generate_ensemble_models()

    # Exportar resultados
    results = optimizer.export_results('../output/advanced_optimization_results.json')

    print("🎉 Otimização avançada concluída!")
    print(f"📊 Front de Pareto: {len(pareto_front)} soluções")
    print(f"🏆 Melhor Sharpe: {best_solution.objectives['sharpe_ratio']:.3f}")
    print(f"💰 Melhor Profit Factor: {best_solution.objectives['profit_factor']:.3f}")
    print(f"📉 Menor Drawdown: {min(sol.objectives['max_drawdown'] for sol in pareto_front):.2f}%")
    print(f"🤖 Ensemble Models: {len(optimizer.ensemble_models)}")