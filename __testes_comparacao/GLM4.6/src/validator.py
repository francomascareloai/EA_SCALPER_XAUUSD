#!/usr/bin/env python3
"""
🔍 EA Optimizer AI - Validator Module
Validação e backtesting automático dos resultados de otimização
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EAValidator:
    """Validador de resultados de otimização de EA"""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M5"):
        """
        Inicializa o validador

        Args:
            symbol: Símbolo de trading
            timeframe: Timeframe para análise
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.validation_results = []

    def validate_optimization_results(self,
                                    optimization_results: Dict[str, Any],
                                    validation_method: str = "walk_forward") -> Dict[str, Any]:
        """
        Valida resultados da otimização usando múltiplos métodos

        Args:
            optimization_results: Resultados da otimização
            validation_method: Método de validação (walk_forward, cross_val, monte_carlo)

        Returns:
            Resultados da validação
        """
        logger.info(f"🔍 Validando otimização usando método: {validation_method}")

        best_params = optimization_results.get('best_params', {})
        best_score = optimization_results.get('best_score', 0)

        if validation_method == "walk_forward":
            validation_results = self._walk_forward_validation(best_params)
        elif validation_method == "cross_validation":
            validation_results = self._cross_validation(optimization_results)
        elif validation_method == "monte_carlo":
            validation_results = self._monte_carlo_validation(best_params)
        else:
            raise ValueError(f"Método de validação desconhecido: {validation_method}")

        # Adicionar métricas de validação
        validation_results.update({
            'original_score': best_score,
            'validation_method': validation_method,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'validation_timestamp': datetime.now().isoformat()
        })

        logger.info(f"✅ Validação concluída: Score validado = {validation_results.get('validated_score', 0):.4f}")
        return validation_results

    def _walk_forward_validation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validação Walk-Forward

        Args:
            params: Parâmetros do EA

        Returns:
            Resultados da validação walk-forward
        """
        logger.info("🚶 Executando validação Walk-Forward...")

        # Simular dados históricos (em implementação real, usar dados reais)
        n_periods = 252  # Um ano de dias de trading
        train_size = 180  # 6 meses para treinamento
        test_size = 30    # 1 mês para teste

        walk_forward_results = []

        for period in range(0, n_periods - train_size - test_size, test_size):
            # Dados de treinamento (período anterior)
            train_start = period
            train_end = period + train_size

            # Dados de teste (período seguinte)
            test_start = train_end
            test_end = test_start + test_size

            # Simular performance no período de teste
            test_performance = self._simulate_period_performance(
                params, test_start, test_end, "test"
            )

            walk_forward_results.append({
                'period_start': test_start,
                'period_end': test_end,
                'train_start': train_start,
                'train_end': train_end,
                'test_score': test_performance['score'],
                'test_profit': test_performance['profit'],
                'test_drawdown': test_performance['drawdown'],
                'test_winrate': test_performance['winrate'],
                'test_trades': test_performance['trades']
            })

        # Calcular estatísticas agregadas
        avg_score = np.mean([r['test_score'] for r in walk_forward_results])
        std_score = np.std([r['test_score'] for r in walk_forward_results])
        avg_profit = np.mean([r['test_profit'] for r in walk_forward_results])
        avg_drawdown = np.mean([r['test_drawdown'] for r in walk_forward_results])
        consistency = self._calculate_consistency(walk_forward_results)

        return {
            'validation_type': 'walk_forward',
            'validated_score': avg_score,
            'score_std': std_score,
            'consistency_score': consistency,
            'avg_profit': avg_profit,
            'avg_drawdown': avg_drawdown,
            'period_results': walk_forward_results,
            'total_periods': len(walk_forward_results),
            'robustness_score': self._calculate_robustness_score(walk_forward_results)
        }

    def _cross_validation(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validação Cruzada K-Fold

        Args:
            optimization_results: Resultados completos da otimização

        Returns:
            Resultados da validação cruzada
        """
        logger.info("🔄 Executando validação Cruzada...")

        trials = optimization_results.get('optimization_history', [])
        if len(trials) < 20:
            logger.warning("⚠️ Número insuficiente de trials para validação cruzada")
            return {'validation_type': 'cross_validation', 'error': 'Insufficient data'}

        # Extrair scores dos trials
        scores = [trial['score'] for trial in trials]

        # Validação cruzada 5-fold
        k_folds = 5
        fold_size = len(scores) // k_folds
        cv_scores = []

        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else len(scores)

            # Validação no fold
            fold_scores = scores[start_idx:end_idx]
            cv_scores.append({
                'fold': fold + 1,
                'mean_score': np.mean(fold_scores),
                'std_score': np.std(fold_scores),
                'min_score': np.min(fold_scores),
                'max_score': np.max(fold_scores),
                'samples': len(fold_scores)
            })

        # Estatísticas da validação cruzada
        overall_mean = np.mean([f['mean_score'] for f in cv_scores])
        overall_std = np.std([f['mean_score'] for f in cv_scores])

        return {
            'validation_type': 'cross_validation',
            'validated_score': overall_mean,
            'score_std': overall_std,
            'cv_folds': k_folds,
            'fold_results': cv_scores,
            'stability_score': 1.0 - (overall_std / overall_mean) if overall_mean > 0 else 0,
            'confidence_interval': [
                overall_mean - 1.96 * overall_std / np.sqrt(k_folds),
                overall_mean + 1.96 * overall_std / np.sqrt(k_folds)
            ]
        }

    def _monte_carlo_validation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validação Monte Carlo

        Args:
            params: Parâmetros do EA

        Returns:
            Resultados da validação Monte Carlo
        """
        logger.info("🎲 Executando validação Monte Carlo...")

        n_simulations = 1000
        simulation_results = []

        for sim in range(n_simulations):
            # Adicionar ruído aos parâmetros
            noisy_params = self._add_parameter_noise(params, noise_level=0.1)

            # Simular performance
            performance = self._simulate_period_performance(
                noisy_params, 0, 30, f"simulation_{sim}"
            )

            simulation_results.append(performance['score'])

        # Estatísticas das simulações
        mean_score = np.mean(simulation_results)
        std_score = np.std(simulation_results)
        percentile_5 = np.percentile(simulation_results, 5)
        percentile_95 = np.percentile(simulation_results, 95)

        # Calcular probabilidade de sucesso
        success_threshold = params.get('min_acceptable_score', 50)
        success_probability = np.mean([s >= success_threshold for s in simulation_results])

        return {
            'validation_type': 'monte_carlo',
            'validated_score': mean_score,
            'score_std': std_score,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'success_probability': success_probability,
            'n_simulations': n_simulations,
            'stability_score': 1.0 - (std_score / mean_score) if mean_score > 0 else 0,
            'simulation_results': simulation_results[:100]  # Salvar apenas as 100 primeiras para economizar espaço
        }

    def _simulate_period_performance(self,
                                   params: Dict[str, Any],
                                   start_period: int,
                                   end_period: int,
                                   period_name: str) -> Dict[str, float]:
        """
        Simula performance do EA em um período específico

        Args:
            params: Parâmetros do EA
            start_period: Período inicial
            end_period: Período final
            period_name: Nome do período

        Returns:
            Métricas de performance simuladas
        """
        # Duração do período
        period_length = end_period - start_period

        # Parâmetros de risco
        risk_reward = params.get('take_profit', 200) / max(params.get('stop_loss', 100), 1)
        risk_factor = params.get('risk_factor', 1.5)
        lot_size = params.get('lot_size', 0.01)

        # Simular base de performance
        base_score = 50.0  # Score base

        # Ajustar score baseado nos parâmetros
        score_adjustments = 0

        # Risk/Reward Ratio
        if risk_reward > 2.0:
            score_adjustments += 20
        elif risk_reward > 1.5:
            score_adjustments += 10
        elif risk_reward < 1.0:
            score_adjustments -= 20

        # Risk Factor (valores moderados são melhores)
        if 1.0 <= risk_factor <= 2.0:
            score_adjustments += 15
        elif risk_factor > 2.5:
            score_adjustments -= 10

        # ATR Multiplier
        atr_multiplier = params.get('atr_multiplier', 1.5)
        if 1.2 <= atr_multiplier <= 2.0:
            score_adjustments += 10

        # Adicionar variabilidade aleatória (simular condição de mercado)
        market_condition = np.random.normal(0, 15)
        noise = np.random.normal(0, 5)

        # Calcular métricas
        final_score = base_score + score_adjustments + market_condition + noise
        final_score = max(0, final_score)  # Score não pode ser negativo

        # Simular outras métricas baseadas no score
        profit = final_score * lot_size * period_length * 0.1
        drawdown = max(5, 30 - final_score * 0.2) + np.random.normal(0, 3)
        winrate = min(80, max(20, 40 + final_score * 0.3 + np.random.normal(0, 5)))
        trades = int(period_length * np.random.uniform(0.5, 2.0))

        return {
            'score': final_score,
            'profit': profit,
            'drawdown': max(0, drawdown),
            'winrate': winrate,
            'trades': trades,
            'period_name': period_name
        }

    def _add_parameter_noise(self, params: Dict[str, Any], noise_level: float) -> Dict[str, Any]:
        """
        Adiciona ruído aos parâmetros para simulação Monte Carlo

        Args:
            params: Parâmetros originais
            noise_level: Nível de ruído (0-1)

        Returns:
            Parâmetros com ruído adicionado
        """
        noisy_params = params.copy()

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Adicionar ruído gaussiano
                noise = np.random.normal(0, value * noise_level)
                noisy_value = value + noise

                # Manter dentro de limites razoáveis
                if 'stop_loss' in key or 'take_profit' in key:
                    noisy_value = max(10, noisy_value)
                elif 'risk_factor' in key or 'atr_multiplier' in key:
                    noisy_value = max(0.1, min(5.0, noisy_value))
                elif 'period' in key:
                    noisy_value = max(5, min(100, noisy_value))
                elif 'lot_size' in key:
                    noisy_value = max(0.01, min(1.0, noisy_value))

                noisy_params[key] = noisy_value

        return noisy_params

    def _calculate_consistency(self, results: List[Dict[str, Any]]) -> float:
        """
        Calcula consistência dos resultados across períodos

        Args:
            results: Lista de resultados por período

        Returns:
            Score de consistência (0-1)
        """
        if len(results) < 2:
            return 1.0

        scores = [r['test_score'] for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Consistência = 1 - (coeficiente de variação)
        consistency = 1.0 - (std_score / mean_score) if mean_score > 0 else 0
        return max(0, min(1.0, consistency))

    def _calculate_robustness_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calcula score de robustez da estratégia

        Args:
            results: Lista de resultados por período

        Returns:
            Score de robustez (0-100)
        """
        if not results:
            return 0

        scores = [r['test_score'] for r in results]
        profits = [r['test_profit'] for r in results]
        drawdowns = [r['test_drawdown'] for r in results]

        # Fatores de robustez
        avg_score = np.mean(scores)
        positive_periods = sum(1 for p in profits if p > 0) / len(profits)
        avg_drawdown = np.mean(drawdowns)
        score_consistency = self._calculate_consistency(results)

        # Score de robustez composto
        robustness = (
            avg_score * 0.3 +                    # Performance média
            positive_periods * 20 * 0.3 +        # Consistência de lucro
            max(0, 30 - avg_drawdown) * 0.2 +    # Controle de drawdown
            score_consistency * 20 * 0.2         # Consistência de score
        )

        return max(0, min(100, robustness))

    def generate_validation_report(self,
                                 validation_results: Dict[str, Any],
                                 output_path: str) -> str:
        """
        Gera relatório detalhado da validação

        Args:
            validation_results: Resultados da validação
            output_path: Caminho para salvar o relatório

        Returns:
            Caminho do relatório gerado
        """
        logger.info("📄 Gerando relatório de validação...")

        report_content = self._create_validation_report(validation_results)

        # Salvar relatório
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 Relatório de validação salvo: {report_file}")
        return str(report_file)

    def _create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Cria conteúdo do relatório de validação"""

        validation_type = validation_results.get('validation_type', 'unknown')
        validated_score = validation_results.get('validated_score', 0)
        original_score = validation_results.get('original_score', 0)

        report = f"""# 🔍 EA Optimizer AI - Relatório de Validação

## 📊 Sumário da Validação

- **Tipo de Validação**: {validation_type.replace('_', ' ').title()}
- **Data**: {validation_results.get('validation_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
- **Símbolo**: {validation_results.get('symbol', 'N/A')}
- **Timeframe**: {validation_results.get('timeframe', 'N/A')}

## 🎯 Resultados Principais

| Métrica | Valor | Avaliação |
|----------|-------|-----------|
| Score Original | {original_score:.4f | } | {'✅ Bom' if original_score > 70 else '⚠️ Médio' if original_score > 50 else '❌ Baixo'} |
| Score Validado | {validated_score:.4f | } | {'✅ Bom' if validated_score > 70 else '⚠️ Médio' if validated_score > 50 else '❌ Baixo'} |
| Diferença | {validated_score - original_score:+.4f | } | {'✅ Estável' if abs(validated_score - original_score) < 10 else '⚠️ Volátil'} |
"""

        # Adicionar seções específicas por tipo de validação
        if validation_type == 'walk_forward':
            report += self._create_walk_forward_section(validation_results)
        elif validation_type == 'cross_validation':
            report += self._create_cross_validation_section(validation_results)
        elif validation_type == 'monte_carlo':
            report += self._create_monte_carlo_section(validation_results)

        # Adicionar avaliação final
        report += self._create_final_assessment(validation_results)

        report += f"""
## 📋 Recomendações

{self._generate_recommendations(validation_results)}

---

*Relatório gerado automaticamente pelo EA Optimizer AI Validator*
"""

        return report

    def _create_walk_forward_section(self, results: Dict[str, Any]) -> str:
        """Cria seção específica para validação walk-forward"""
        avg_profit = results.get('avg_profit', 0)
        avg_drawdown = results.get('avg_drawdown', 0)
        robustness = results.get('robustness_score', 0)
        consistency = results.get('consistency_score', 0)
        total_periods = results.get('total_periods', 0)

        section = f"""
## 🚶 Análise Walk-Forward

### Métricas de Performance
- **Score Médio**: {results.get('validated_score', 0):.4f}
- **Desvio Padrão**: {results.get('score_std', 0):.4f}
- **Lucro Médio**: ${avg_profit:.2f}
- **Drawdown Médio**: {avg_drawdown:.2f}%
- **Total de Períodos**: {total_periods}

### Métricas de Robustez
- **Score de Robustez**: {robustness:.2f}/100 {'✅' if robustness > 70 else '⚠️' if robustness > 50 else '❌'}
- **Consistência**: {consistency:.2f}/1.0 {'✅' if consistency > 0.7 else '⚠️' if consistency > 0.5 else '❌'}

### Análise por Período
"""

        period_results = results.get('period_results', [])
        for i, period in enumerate(period_results[:5]):  # Mostrar apenas os 5 primeiros
            section += f"""
- **Período {i+1}**: Score={period['test_score']:.2f}, Lucro=${period['test_profit']:.2f}, DD={period['test_drawdown']:.2f}%
"""

        if len(period_results) > 5:
            section += f"- ... e mais {len(period_results) - 5} períodos\n"

        return section

    def _create_cross_validation_section(self, results: Dict[str, Any]) -> str:
        """Cria seção específica para validação cruzada"""
        cv_folds = results.get('cv_folds', 0)
        stability = results.get('stability_score', 0)
        confidence_interval = results.get('confidence_interval', [0, 0])

        section = f"""
## 🔄 Análise de Validação Cruzada

### Métricas Principais
- **Número de Folds**: {cv_folds}
- **Score Médio**: {results.get('validated_score', 0):.4f}
- **Desvio Padrão**: {results.get('score_std', 0):.4f}
- **Score de Estabilidade**: {stability:.2f}/1.0 {'✅' if stability > 0.7 else '⚠️' if stability > 0.5 else '❌'}

### Intervalo de Confiança (95%)
- **Limite Inferior**: {confidence_interval[0]:.4f}
- **Limite Superior**: {confidence_interval[1]:.4f}

### Resultados por Fold
"""

        fold_results = results.get('fold_results', [])
        for fold in fold_results:
            section += f"""
- **Fold {fold['fold']}**: Média={fold['mean_score']:.2f}, Std={fold['std_score']:.2f}, Min={fold['min_score']:.2f}, Max={fold['max_score']:.2f}
"""

        return section

    def _create_monte_carlo_section(self, results: Dict[str, Any]) -> str:
        """Cria seção específica para validação Monte Carlo"""
        n_simulations = results.get('n_simulations', 0)
        percentile_5 = results.get('percentile_5', 0)
        percentile_95 = results.get('percentile_95', 0)
        success_prob = results.get('success_probability', 0)
        stability = results.get('stability_score', 0)

        section = f"""
## 🎲 Análise Monte Carlo

### Estatísticas das Simulações
- **Número de Simulações**: {n_simulations}
- **Score Médio**: {results.get('validated_score', 0):.4f}
- **Desvio Padrão**: {results.get('score_std', 0):.4f}
- **Percentil 5%**: {percentile_5:.4f}
- **Percentil 95%**: {percentile_95:.4f}

### Métricas de Risco
- **Probabilidade de Sucesso**: {success_prob:.1%} {'✅' if success_prob > 0.7 else '⚠️' if success_prob > 0.5 else '❌'}
- **Score de Estabilidade**: {stability:.2f}/1.0 {'✅' if stability > 0.7 else '⚠️' if stability > 0.5 else '❌'}

### Análise de Cenários
"""

        if percentile_5 > 50:
            section += "- ✅ **Cenário Otimista**: Even no pior cenário (5%), performance é aceitável\n"
        elif percentile_5 > 30:
            section += "- ⚠️ **Cenário Moderado**: Pior cenário (5%) ainda pode ser aceitável\n"
        else:
            section += "- ❌ **Cenário Pessimista**: Pior cenário (5%) apresenta baixa performance\n"

        if success_prob > 0.8:
            section += "- ✅ **Alta Confiança**: Estratégia tem alta probabilidade de sucesso\n"
        elif success_prob > 0.6:
            section += "- ⚠️ **Confiança Moderada**: Estratégia tem probabilidade moderada de sucesso\n"
        else:
            section += "- ❌ **Baixa Confiança**: Estratégia tem baixa probabilidade de sucesso\n"

        return section

    def _create_final_assessment(self, validation_results: Dict[str, Any]) -> str:
        """Cria avaliação final dos resultados"""
        validated_score = validation_results.get('validated_score', 0)
        original_score = validation_results.get('original_score', 0)
        validation_type = validation_results.get('validation_type', '')

        # Determinar avaliação geral
        if validated_score > 70 and abs(validated_score - original_score) < 15:
            assessment = "✅ **Excelente**"
            assessment_detail = "Estratégia robusta e validada com alta confiança"
        elif validated_score > 50 and abs(validated_score - original_score) < 25:
            assessment = "⚠️ **Aceitável**"
            assessment_detail = "Estratégia razoável, mas com algumas limitações"
        else:
            assessment = "❌ **Precisa de Melhorias**"
            assessment_detail = "Estratégia apresenta problemas de robustez ou performance"

        return f"""
## 🏁 Avaliação Final

{assessment}

{assessment_detail}

### Status de Validação
- **Validação**: {'Aprovada' if validated_score > 50 else 'Reprovada'}
- **Recomendação**: {'Implementar em conta demo' if validated_score > 60 else 'Revisar parâmetros'}
- **Nível de Confiança**: {'Alto' if abs(validated_score - original_score) < 10 else 'Médio' if abs(validated_score - original_score) < 20 else 'Baixo'}
"""

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> str:
        """Gera recomendações baseadas nos resultados"""
        validated_score = validation_results.get('validated_score', 0)
        original_score = validation_results.get('original_score', 0)
        validation_type = validation_results.get('validation_type', '')

        recommendations = []

        if validated_score < 40:
            recommendations.append("❌ **Alta Prioridade**: Revisar completamente a estratégia. Score muito baixo.")
        elif validated_score < 60:
            recommendations.append("⚠️ **Média Prioridade**: Considerar reotimização com parâmetros diferentes.")
        else:
            recommendations.append("✅ **Baixa Prioridade**: Estratégia aceitável para testes em conta demo.")

        if abs(validated_score - original_score) > 20:
            recommendations.append("⚠️ **Atenção**: Grande diferença entre score otimizado e validado. Possível overfitting.")

        if validation_type == 'walk_forward':
            consistency = validation_results.get('consistency_score', 0)
            if consistency < 0.5:
                recommendations.append("⚠️ **Consistência**: Baixa consistência entre períodos. Revisar estabilidade da estratégia.")

        if validation_type == 'monte_carlo':
            success_prob = validation_results.get('success_probability', 0)
            if success_prob < 0.6:
                recommendations.append("⚠️ **Risco**: Baixa probabilidade de sucesso em simulações. Considerar estratégia mais conservadora.")

        # Recomendações gerais
        recommendations.extend([
            "📊 **Próximo Passo**: Executar backtest em conta demo por pelo menos 1 mês.",
            "🔍 **Monitoramento**: Acompanhar performance em diferentes condições de mercado.",
            "📈 **Ajustes**: Reotimizar parâmetros a cada 3-6 meses se necessário."
        ])

        return "\n".join(recommendations)

if __name__ == "__main__":
    # Teste do validador
    validator = EAValidator()

    # Resultados de exemplo
    sample_optimization_results = {
        'best_score': 75.5,
        'best_params': {
            'stop_loss': 120,
            'take_profit': 240,
            'risk_factor': 1.8,
            'atr_multiplier': 1.6
        },
        'optimization_history': [
            {'score': 65.2 + np.random.normal(0, 5)}
            for _ in range(50)
        ]
    }

    # Executar validação
    validation_results = validator.validate_optimization_results(
        sample_optimization_results,
        validation_method="walk_forward"
    )

    # Gerar relatório
    report_path = validator.generate_validation_report(
        validation_results,
        "../output/validation_report.md"
    )

    print(f"✅ Relatório de validação criado: {report_path}")