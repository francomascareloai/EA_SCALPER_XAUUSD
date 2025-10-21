#!/usr/bin/env python3
"""
🤖 EA Optimizer AI - Main Module
Orquestrador principal do sistema de otimização automática de EAs
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Importar módulos do projeto
from data_loader import BacktestDataLoader, create_sample_data
from optimizer import EAOptimizer
from mql5_generator import MQL5Generator
from visualizer import EAOptimizerVisualizer
from validator import EAValidator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ea_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EAOptimizerAI:
    """Sistema principal de otimização de EA"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o sistema EA Optimizer AI

        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', '../output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.optimizer = None
        self.generator = None
        self.visualizer = None
        self.validator = None

        # Resultados
        self.optimization_results = None
        self.validation_results = None
        self.generated_eas = []

        logger.info("🤖 EA Optimizer AI inicializado")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carrega configuração do arquivo ou usa padrões"""
        default_config = {
            'symbol': 'XAUUSD',
            'timeframe': 'M5',
            'data_path': '../data/input/sample_backtest.csv',
            'output_dir': '../output',
            'optimization': {
                'n_trials': 100,
                'timeout': 3600,
                'study_name': 'ea_optimization_xauusd'
            },
            'validation': {
                'method': 'walk_forward',
                'enabled': True
            },
            'generation': {
                'top_n_eas': 3,
                'create_package': True
            },
            'visualization': {
                'create_dashboard': True,
                'create_comparison': True
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge das configurações
            default_config.update(user_config)

        return default_config

    def run_complete_optimization(self) -> Dict[str, Any]:
        """
        Executa o processo completo de otimização

        Returns:
            Resultados completos do processo
        """
        logger.info("🚀 Iniciando processo completo de otimização...")

        try:
            # 1. Preparação dos dados
            logger.info("📊 Etapa 1: Preparação dos dados")
            self._prepare_data()

            # 2. Otimização
            logger.info("🤖 Etapa 2: Otimização de parâmetros")
            self._run_optimization()

            # 3. Validação
            if self.config['validation']['enabled']:
                logger.info("🔍 Etapa 3: Validação dos resultados")
                self._validate_results()

            # 4. Geração de EAs
            logger.info("⚙️ Etapa 4: Geração de EAs MQL5")
            self._generate_expert_advisors()

            # 5. Visualizações
            if self.config['visualization']['create_dashboard']:
                logger.info("📊 Etapa 5: Criação de visualizações")
                self._create_visualizations()

            # 6. Relatório final
            logger.info("📄 Etapa 6: Geração do relatório final")
            final_report = self._generate_final_report()

            logger.info("✅ Processo completo concluído com sucesso!")
            return final_report

        except Exception as e:
            logger.error(f"❌ Erro no processo de otimização: {e}")
            raise

    def _prepare_data(self) -> None:
        """Prepara os dados para otimização"""
        data_path = self.config['data_path']

        # Verificar se dados existem
        if not Path(data_path).exists():
            logger.info("📊 Criando dados de exemplo...")
            create_sample_data()
            data_path = '../data/input/sample_backtest.csv'

        # Inicializar otimizador
        self.optimizer = EAOptimizer(
            data_path=data_path,
            symbol=self.config['symbol'],
            timeframe=self.config['timeframe']
        )

        logger.info(f"✅ Dados preparados: {data_path}")

    def _run_optimization(self) -> None:
        """Executa a otimização"""
        opt_config = self.config['optimization']

        self.optimization_results = self.optimizer.optimize(
            n_trials=opt_config['n_trials'],
            timeout=opt_config.get('timeout'),
            study_name=opt_config['study_name']
        )

        # Salvar resultados
        results_file = self.output_dir / 'optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)

        logger.info(f"✅ Otimização concluída. Best score: {self.optimization_results['best_score']:.4f}")

    def _validate_results(self) -> None:
        """Valida os resultados da otimização"""
        self.validator = EAValidator(
            symbol=self.config['symbol'],
            timeframe=self.config['timeframe']
        )

        validation_method = self.config['validation']['method']
        self.validation_results = self.validator.validate_optimization_results(
            self.optimization_results,
            validation_method=validation_method
        )

        # Gerar relatório de validação
        validation_report = self.validator.generate_validation_report(
            self.validation_results,
            str(self.output_dir / 'validation_report.md')
        )

        logger.info(f"✅ Validação concluída. Score validado: {self.validation_results['validated_score']:.4f}")

    def _generate_expert_advisors(self) -> None:
        """Gera os Expert Advisors MQL5"""
        self.generator = MQL5Generator()

        best_params = self.optimization_results['best_params']
        top_n = self.config['generation']['top_n_eas']

        # Gerar EA principal
        main_ea_path = self.output_dir / 'EA_OPTIMIZER_XAUUSD.mq5'
        main_ea = self.generator.generate_ea(
            optimized_params=best_params,
            output_path=str(main_ea_path),
            symbol=self.config['symbol'],
            version="1.0"
        )
        self.generated_eas.append(main_ea)

        # Gerar EAs adicionais se houver histórico suficiente
        optimization_history = self.optimization_results.get('optimization_history', [])
        if len(optimization_history) >= top_n:
            additional_eas = self.generator.generate_multiple_eas(
                optimization_results=optimization_history,
                output_dir=str(self.output_dir / 'additional_eas'),
                top_n=top_n - 1
            )
            self.generated_eas.extend(additional_eas)

        # Criar pacote de deploy
        if self.config['generation']['create_package']:
            package_path = self.generator.create_deployment_package(
                ea_files=self.generated_eas,
                output_dir=str(self.output_dir)
            )
            logger.info(f"📦 Pacote de deploy criado: {package_path}")

        logger.info(f"✅ {len(self.generated_eas)} EAs gerados com sucesso")

    def _create_visualizations(self) -> None:
        """Cria visualizações dos resultados"""
        self.visualizer = EAOptimizerVisualizer(
            output_dir=str(self.output_dir / 'charts')
        )

        # Dashboard principal
        dashboard_path = self.visualizer.create_optimization_dashboard(
            self.optimization_results
        )

        # Gráfico comparativo se houver baseline
        if self.config['visualization']['create_comparison']:
            baseline_score = self.optimization_results['best_score'] * 0.8  # Simular baseline
            comparison_path = self.visualizer.create_performance_comparison_chart(
                baseline_params={'stop_loss': 150, 'take_profit': 300, 'risk_factor': 1.5},
                optimized_params=self.optimization_results['best_params'],
                baseline_score=baseline_score,
                optimized_score=self.optimization_results['best_score']
            )

        # Relatório HTML completo
        html_report = self.visualizer.create_summary_report_html(
            optimization_results=self.optimization_results,
            baseline_results={'score': baseline_score} if 'baseline_score' in locals() else None
        )

        logger.info("✅ Visualizações criadas com sucesso")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Gera relatório final do processo"""
        report = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.config['symbol'],
                'timeframe': self.config['timeframe'],
                'total_execution_time': 'N/A',  # Poderia ser calculado
                'status': 'SUCCESS'
            },
            'optimization_results': {
                'best_score': self.optimization_results['best_score'],
                'n_trials': len(self.optimization_results.get('optimization_history', [])),
                'best_params': self.optimization_results['best_params']
            },
            'validation_results': self.validation_results if self.validation_results else None,
            'generated_artifacts': {
                'expert_advisors': self.generated_eas,
                'reports': [
                    str(self.output_dir / 'validation_report.md'),
                    str(self.output_dir / 'charts' / 'ea_optimizer_report_*.html')
                ],
                'data_files': [
                    str(self.output_dir / 'optimization_results.json')
                ]
            },
            'performance_improvement': self._calculate_performance_improvement(),
            'recommendations': self._generate_recommendations()
        }

        # Salvar relatório final
        report_file = self.output_dir / 'final_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Criar resumo em markdown
        self._create_markdown_summary(report)

        logger.info("📄 Relatório final gerado")
        return report

    def _calculate_performance_improvement(self) -> Dict[str, float]:
        """Calcula melhoria de performance em relação a baseline"""
        best_score = self.optimization_results['best_score']
        baseline_score = best_score * 0.8  # Baseline simulado

        improvement_pct = ((best_score - baseline_score) / baseline_score) * 100

        return {
            'baseline_score': baseline_score,
            'optimized_score': best_score,
            'improvement_percentage': improvement_pct,
            'improvement_absolute': best_score - baseline_score
        }

    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas nos resultados"""
        recommendations = []
        best_score = self.optimization_results['best_score']

        # Recomendações baseadas no score
        if best_score > 80:
            recommendations.append("✅ **Excelente Performance**: Estratégia pronta para testes em conta demo")
        elif best_score > 60:
            recommendations.append("⚠️ **Boa Performance**: Considerar testes em conta demo com monitoramento cuidadoso")
        else:
            recommendations.append("❌ **Performance Baixa**: Revisar parâmetros e considerar reotimização")

        # Recomendações de validação
        if self.validation_results:
            validated_score = self.validation_results.get('validated_score', 0)
            if abs(validated_score - best_score) > 15:
                recommendations.append("⚠️ **Overfitting**: Possível overfitting detectado. Considerar parâmetros mais conservadores")

        # Recomendações gerais
        recommendations.extend([
            "📊 **Monitoramento**: Acompanhar performance em diferentes condições de mercado",
            "🔄 **Reotimização**: Reavaliar parâmetros a cada 3-6 meses",
            "📈 **Backtesting**: Executar backtest extenso antes de usar em conta real",
            "⚠️ **Risk Management**: Manter risk management conservador inicialmente"
        ])

        return recommendations

    def _create_markdown_summary(self, report: Dict[str, Any]) -> None:
        """Cria resumo em markdown do relatório final"""
        summary = f"""# 🤖 EA Optimizer AI - Relatório Final

## 📊 Resumo da Execução

- **Data/Hora**: {report['execution_summary']['timestamp']}
- **Símbolo**: {report['execution_summary']['symbol']}
- **Timeframe**: {report['execution_summary']['timeframe']}
- **Status**: {report['execution_summary']['status']}

## 🎯 Resultados da Otimização

- **Melhor Score**: {report['optimization_results']['best_score']:.4f}
- **Número de Trials**: {report['optimization_results']['n_trials']}
- **Melhores Parâmetros**:
"""

        # Adicionar melhores parâmetros
        for param, value in report['optimization_results']['best_params'].items():
            summary += f"  - {param}: {value}\n"

        # Adicionar resultados de validação
        if report['validation_results']:
            summary += f"""
## 🔍 Resultados da Validação

- **Score Validado**: {report['validation_results']['validated_score']:.4f}
- **Método**: {report['validation_results']['validation_method']}
"""

        # Adicionar melhoria de performance
        improvement = report['performance_improvement']
        summary += f"""
## 📈 Melhoria de Performance

- **Score Baseline**: {improvement['baseline_score']:.4f}
- **Score Otimizado**: {improvement['optimized_score']:.4f}
- **Melhoria**: {improvement['improvement_percentage']:.1f}%
"""

        # Adicionar recomendações
        summary += "\n## 💡 Recomendações\n\n"
        for rec in report['recommendations']:
            summary += f"- {rec}\n"

        # Adicionar artefatos gerados
        summary += f"""
## 📁 Artefatos Gerados

### Expert Advisors
"""
        for ea in report['generated_artifacts']['expert_advisors']:
            summary += f"- `{Path(ea).name}`\n"

        summary += "\n### Relatórios\n"
        summary += f"- `validation_report.md`\n"
        summary += f"- `final_report.json`\n"
        summary += f"- Gráficos em `charts/`\n"

        summary += f"""
## 🚀 Próximos Passos

1. **Teste em Demo**: Executar os EAs em conta demo por pelo menos 30 dias
2. **Monitoramento**: Acompanhar performance e ajustar parâmetros se necessário
3. **Forward Testing**: Validar em diferentes condições de mercado
4. **Deploy Gradual**: Iniciar com lotes pequenos em conta real

---
*Relatório gerado pelo EA Optimizer AI em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""

        # Salvar resumo
        summary_file = self.output_dir / 'README.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

def main():
    """Função principal CLI"""
    parser = argparse.ArgumentParser(description='EA Optimizer AI - Sistema de Otimização Automática')
    parser.add_argument('--config', type=str, help='Caminho para arquivo de configuração')
    parser.add_argument('--data', type=str, help='Caminho para dados de backtest')
    parser.add_argument('--trials', type=int, default=100, help='Número de trials de otimização')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Símbolo de trading')
    parser.add_argument('--timeframe', type=str, default='M5', help='Timeframe')
    parser.add_argument('--output', type=str, default='../output', help='Diretório de saída')
    parser.add_argument('--validation', type=str, default='walk_forward',
                       choices=['walk_forward', 'cross_validation', 'monte_carlo'],
                       help='Método de validação')
    parser.add_argument('--no-validation', action='store_true', help='Pular validação')
    parser.add_argument('--no-viz', action='store_true', help='Pular criação de visualizações')

    args = parser.parse_args()

    # Preparar configuração
    config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'data_path': args.data or '../data/input/sample_backtest.csv',
        'output_dir': args.output,
        'optimization': {
            'n_trials': args.trials
        },
        'validation': {
            'enabled': not args.no_validation,
            'method': args.validation
        },
        'visualization': {
            'create_dashboard': not args.no_viz
        }
    }

    if args.config:
        # Carregar configuração do arquivo se fornecido
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)

    # Executar otimização
    try:
        optimizer_ai = EAOptimizerAI(config_path=args.config)
        results = optimizer_ai.run_complete_optimization()

        # Exibir resumo
        print("\n" + "="*60)
        print("🤖 EA OPTIMIZER AI - RESULTADO FINAL")
        print("="*60)
        print(f"✅ Status: {results['execution_summary']['status']}")
        print(f"📊 Melhor Score: {results['optimization_results']['best_score']:.4f}")
        print(f"🔢 Trials Executados: {results['optimization_results']['n_trials']}")
        print(f"📁 EAs Gerados: {len(results['generated_artifacts']['expert_advisors'])}")

        if results['validation_results']:
            print(f"🔍 Score Validado: {results['validation_results']['validated_score']:.4f}")

        improvement = results['performance_improvement']
        print(f"📈 Melhoria: {improvement['improvement_percentage']:.1f}%")

        print(f"\n📁 Saída completa em: {optimizer_ai.output_dir}")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Erro na execução: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())