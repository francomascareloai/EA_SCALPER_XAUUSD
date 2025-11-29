#!/usr/bin/env python3
"""
Exemplo 04: Sistema de Backtesting Completo
===========================================

Este exemplo demonstra como implementar um sistema completo de backtesting:
- Configuração de parâmetros de teste
- Execução de backtests automatizados
- Análise detalhada de resultados
- Validação FTMO-compliant
- Otimização de parâmetros
- Geração de relatórios

Requisitos:
- MT5 Agent API disponível
- Dados históricos adequados
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from ea_scalper_sdk import MT5Client, AgentClient
from ea_scalper_sdk.exceptions import MT5Error

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BacktestSystem")

# Carregar variáveis de ambiente
load_dotenv()

class BacktestSystem:
    """Sistema completo de backtesting"""

    def __init__(self):
        self.mt5_client = None
        self.agent_client = None
        self.symbol = "XAUUSD"
        self.backtest_results = {}
        self.optimization_results = []

    async def initialize(self):
        """Inicializa clientes e conexões"""
        try:
            logger.info("🚀 Inicializando Sistema de Backtesting...")

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

            # Conectar ao Agent Management
            self.agent_client = AgentClient(base_url="http://localhost:8080")
            logger.info("✅ Agent Client inicializado")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False

    async def run_single_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um único backtest com configuração específica"""

        try:
            test_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            logger.info(f"🧪 Iniciando backtest {test_id}")
            logger.info(f"   Período: {config['period']}")
            logger.info(f"   Depósito: ${config['deposit']}")
            logger.info(f"   Alavancagem: 1:{config['leverage']}")

            # Preparar parâmetros para o agente de backtest
            backtest_params = {
                "expert_file": config.get('expert_file', "EA_XAUUSD_Scalper.ex5"),
                "symbol": self.symbol,
                "period": config['period'],
                "deposit": config['deposit'],
                "leverage": config['leverage'],
                "model": config.get('model', 4),  # Real ticks
                "spread": config.get('spread', 15),
                "optimization": False,
                "inputs": config.get('inputs', {})
            }

            # Executar backtest via agente
            task_result = await self.agent_client.execute_task(
                agent_name="backtest",
                task_parameters=backtest_params
            )

            if not task_result['success']:
                raise Exception(f"Falha ao iniciar backtest: {task_result['message']}")

            task_id = task_result['task_id']
            logger.info(f"📋 Backtest iniciado com ID: {task_id}")

            # Monitorar progresso
            result = await self.monitor_backtest_progress(task_id)

            if result:
                # Analisar resultados
                analysis = await self.analyze_backtest_results(result, config)
                self.backtest_results[test_id] = {
                    "config": config,
                    "results": result,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(f"✅ Backtest {test_id} concluído com sucesso")
                return self.backtest_results[test_id]
            else:
                logger.error(f"❌ Backtest {test_id} falhou")
                return None

        except Exception as e:
            logger.error(f"❌ Erro no backtest: {e}")
            return None

    async def monitor_backtest_progress(self, task_id: str, timeout: int = 1800) -> Optional[Dict]:
        """Monitora progresso do backtest"""

        start_time = datetime.now()
        logger.info("⏳ Monitorando progresso do backtest...")

        while True:
            try:
                # Verificar status
                status = await self.agent_client.get_task_status("backtest", task_id)

                if status['status'] == 'completed':
                    logger.info("✅ Backtest concluído")
                    # Obter resultados
                    results = await self.agent_client.get_task_results("backtest", task_id)
                    return results

                elif status['status'] == 'failed':
                    error_msg = status.get('error', 'Erro desconhecido')
                    logger.error(f"❌ Backtest falhou: {error_msg}")
                    return None

                elif status['status'] == 'running':
                    progress = status.get('progress', 0)
                    logger.info(f"📊 Progresso: {progress}%")

                    # Verificar timeout
                    elapsed = (datetime.now() - start_time).seconds
                    if elapsed > timeout:
                        logger.error(f"❌ Timeout do backtest ({timeout}s)")
                        return None

                    await asyncio.sleep(30)  # Verificar a cada 30 segundos

                else:
                    await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"❌ Erro no monitoramento: {e}")
                await asyncio.sleep(10)

    async def analyze_backtest_results(self, results: Dict, config: Dict) -> Dict[str, Any]:
        """Analisa detalhadamente os resultados do backtest"""

        try:
            logger.info("📊 Analisando resultados do backtest...")

            # Métricas básicas
            total_trades = results.get('total_trades', 0)
            profit_trades = results.get('profit_trades', 0)
            loss_trades = results.get('loss_trades', 0)
            win_rate = (profit_trades / total_trades * 100) if total_trades > 0 else 0
            profit_factor = results.get('profit_factor', 0)
            max_drawdown = results.get('max_drawdown', 0)
            daily_drawdown = results.get('daily_drawdown', 0)
            total_profit = results.get('total_profit', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)

            # Análise FTMO
            ftmo_analysis = await self.validate_ftmo_compliance(results, config)

            # Análise de consistência
            consistency_analysis = self.analyze_consistency(results)

            # Análise de risco
            risk_analysis = self.analyze_risk_metrics(results, config)

            # Análise de performance por mês
            monthly_analysis = self.analyze_monthly_performance(results)

            # Score geral
            overall_score = self.calculate_overall_score(
                win_rate, profit_factor, max_drawdown, ftmo_analysis
            )

            analysis = {
                "basic_metrics": {
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "daily_drawdown": daily_drawdown,
                    "total_profit": total_profit,
                    "sharpe_ratio": sharpe_ratio
                },
                "ftmo_compliance": ftmo_analysis,
                "consistency": consistency_analysis,
                "risk_metrics": risk_analysis,
                "monthly_performance": monthly_analysis,
                "overall_score": overall_score,
                "recommendations": self.generate_recommendations(
                    win_rate, profit_factor, max_drawdown, ftmo_analysis
                )
            }

            logger.info(f"📊 Análise concluída - Score: {overall_score}/100")
            return analysis

        except Exception as e:
            logger.error(f"❌ Erro na análise: {e}")
            return {"error": str(e)}

    async def validate_ftmo_compliance(self, results: Dict, config: Dict) -> Dict[str, Any]:
        """Valida compliance com regras FTMO"""

        try:
            logger.info("✅ Validando compliance FTMO...")

            # Critérios FTMO
            max_daily_loss = 5.0  # 5% perda diária máxima
            max_total_loss = 10.0  # 10% perda total máxima
            min_profit_target = 10.0  # 10% alvo de lucro
            min_trading_days = 10  # Mínimo 10 dias de trading

            daily_drawdown = results.get('daily_drawdown', 0)
            max_drawdown = results.get('max_drawdown', 0)
            total_profit = results.get('total_profit', 0)
            initial_deposit = config.get('deposit', 10000)
            profit_percentage = (total_profit / initial_deposit) * 100

            # Verificações
            checks = {
                "daily_loss_check": daily_drawdown <= max_daily_loss,
                "total_loss_check": max_drawdown <= max_total_loss,
                "profit_target_check": profit_percentage >= min_profit_target,
                "consistency_check": self.check_ftmo_consistency(results),
                "leverage_check": config.get('leverage', 100) <= 100,
                "scalping_check": self.validate_scalping_rules(results)
            }

            # Compliance geral
            all_passed = all(checks.values())
            checks['overall_compliance'] = all_passed

            # Nível de compliance
            if all_passed:
                compliance_level = "FULL"
            elif sum(checks.values()) >= 4:
                compliance_level = "PARTIAL"
            else:
                compliance_level = "FAILED"

            # Riscos específicos
            risks = []
            if daily_drawdown > 4.0:
                risks.append("Perda diária próxima ao limite")
            if max_drawdown > 8.0:
                risks.append("Drawdown máximo elevado")
            if profit_percentage < 8.0:
                risks.append("Alvo de lucro não atingido")

            return {
                "compliance_level": compliance_level,
                "checks": checks,
                "risks": risks,
                "score": sum(checks.values()) / len(checks) * 100,
                "recommendations": self.get_ftmo_recommendations(checks, risks)
            }

        except Exception as e:
            logger.error(f"❌ Erro na validação FTMO: {e}")
            return {"error": str(e)}

    def check_ftmo_consistency(self, results: Dict) -> bool:
        """Verifica consistência de lucros (regra FTMO)"""
        try:
            # Simplificado - em implementação real, analisaria lucros diários
            total_profit = results.get('total_profit', 0)
            profit_trades = results.get('profit_trades', 0)

            if profit_trades == 0:
                return False

            avg_profit = total_profit / profit_trades
            return avg_profit > 0  # Lucro médio positivo

        except:
            return False

    def validate_scalping_rules(self, results: Dict) -> bool:
        """Valida regras de scalping FTMO"""
        try:
            # Verificar se há muitas operações de curto prazo
            total_trades = results.get('total_trades', 0)
            profit_trades = results.get('profit_trades', 0)

            if total_trades > 500:  # Muitas operações
                return False

            # Verificar profit factor
            profit_factor = results.get('profit_factor', 0)
            if profit_factor < 1.2:  # Profit factor muito baixo
                return False

            return True

        except:
            return True  # Conservador

    def analyze_consistency(self, results: Dict) -> Dict[str, Any]:
        """Analisa consistência dos resultados"""

        try:
            # Análise de consistência de trades
            win_rate = (results.get('profit_trades', 0) / results.get('total_trades', 1)) * 100
            profit_factor = results.get('profit_factor', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)

            # Avaliação de consistência
            consistency_score = 0

            if win_rate >= 60:
                consistency_score += 30
            elif win_rate >= 50:
                consistency_score += 20
            elif win_rate >= 40:
                consistency_score += 10

            if profit_factor >= 1.5:
                consistency_score += 30
            elif profit_factor >= 1.3:
                consistency_score += 20
            elif profit_factor >= 1.1:
                consistency_score += 10

            if sharpe_ratio >= 1.5:
                consistency_score += 40
            elif sharpe_ratio >= 1.0:
                consistency_score += 25
            elif sharpe_ratio >= 0.5:
                consistency_score += 10

            return {
                "score": consistency_score,
                "level": "HIGH" if consistency_score >= 70 else "MEDIUM" if consistency_score >= 40 else "LOW",
                "win_rate_consistency": "STABLE" if 45 <= win_rate <= 65 else "UNSTABLE",
                "profit_consistency": "STABLE" if profit_factor >= 1.2 else "UNSTABLE",
                "risk_adjusted_return": sharpe_ratio
            }

        except Exception as e:
            logger.error(f"❌ Erro na análise de consistência: {e}")
            return {"error": str(e)}

    def analyze_risk_metrics(self, results: Dict, config: Dict) -> Dict[str, Any]:
        """Analisa métricas de risco"""

        try:
            max_dd = results.get('max_drawdown', 0)
            daily_dd = results.get('daily_drawdown', 0)
            total_profit = results.get('total_profit', 0)
            initial_deposit = config.get('deposit', 10000)

            # Métricas de risco
            recovery_factor = total_profit / max_dd if max_dd > 0 else 0
            risk_reward_ratio = total_profit / max_dd if max_dd > 0 else 0
            monthly_return = (total_profit / initial_deposit) * 100  # Simplificado

            # Classificação de risco
            risk_level = "LOW"
            if max_dd > 10:
                risk_level = "HIGH"
            elif max_dd > 5:
                risk_level = "MEDIUM"

            return {
                "max_drawdown": max_dd,
                "daily_drawdown": daily_dd,
                "recovery_factor": recovery_factor,
                "risk_reward_ratio": risk_reward_ratio,
                "monthly_return": monthly_return,
                "risk_level": risk_level,
                "var_95": self.calculate_var(results),  # Value at Risk 95%
                "expected_shortfall": self.calculate_expected_shortfall(results)
            }

        except Exception as e:
            logger.error(f"❌ Erro na análise de risco: {e}")
            return {"error": str(e)}

    def calculate_var(self, results: Dict) -> float:
        """Calcula Value at Risk (simplificado)"""
        try:
            max_dd = results.get('max_drawdown', 0)
            return max_dd * 1.2  # VaR como 120% do máximo drawdown
        except:
            return 0

    def calculate_expected_shortfall(self, results: Dict) -> float:
        """Calcula Expected Shortfall (simplificado)"""
        try:
            max_dd = results.get('max_drawdown', 0)
            return max_dd * 1.5  # ES como 150% do máximo drawdown
        except:
            return 0

    def analyze_monthly_performance(self, results: Dict) -> Dict[str, Any]:
        """Analisa performance mensal (simplificado)"""

        try:
            total_profit = results.get('total_profit', 0)
            total_trades = results.get('total_trades', 0)
            profit_trades = results.get('profit_trades', 0)

            # Simplificado - em implementação real usaria dados mensais
            months = 12  # Assumindo 1 ano
            monthly_profit = total_profit / months
            monthly_trades = total_trades / months
            monthly_win_rate = (profit_trades / total_trades) * 100 if total_trades > 0 else 0

            return {
                "months_analyzed": months,
                "avg_monthly_profit": monthly_profit,
                "avg_monthly_trades": monthly_trades,
                "avg_monthly_win_rate": monthly_win_rate,
                "profitable_months_estimate": int(months * 0.7) if monthly_profit > 0 else int(months * 0.3)
            }

        except Exception as e:
            logger.error(f"❌ Erro na análise mensal: {e}")
            return {"error": str(e)}

    def calculate_overall_score(self, win_rate: float, profit_factor: float,
                               max_drawdown: float, ftmo_analysis: Dict) -> int:
        """Calcula score geral da estratégia"""

        try:
            score = 0

            # Win rate (30 pontos)
            if win_rate >= 60:
                score += 30
            elif win_rate >= 50:
                score += 20
            elif win_rate >= 40:
                score += 10

            # Profit factor (25 pontos)
            if profit_factor >= 1.5:
                score += 25
            elif profit_factor >= 1.3:
                score += 15
            elif profit_factor >= 1.1:
                score += 5

            # Drawdown (25 pontos)
            if max_drawdown <= 5:
                score += 25
            elif max_drawdown <= 8:
                score += 15
            elif max_drawdown <= 12:
                score += 5

            # FTMO compliance (20 pontos)
            ftmo_score = ftmo_analysis.get('score', 0)
            score += int(ftmo_score * 0.2)

            return min(100, score)

        except:
            return 0

    def generate_recommendations(self, win_rate: float, profit_factor: float,
                                max_drawdown: float, ftmo_analysis: Dict) -> List[str]:
        """Gera recomendações baseadas nos resultados"""

        recommendations = []

        if win_rate < 40:
            recommendations.append("Considerar revisar critérios de entrada - win rate muito baixo")
        elif win_rate > 70:
            recommendations.append("Win rate muito alto - verificar se há overfitting")

        if profit_factor < 1.1:
            recommendations.append("Profit factor baixo - revisar gestão de risco e take profits")
        elif profit_factor > 2.5:
            recommendations.append("Profit factor excelente - validar com mais dados")

        if max_drawdown > 10:
            recommendations.append("Drawdown muito alto - reduzir tamanho das posições")
        elif max_drawdown < 3:
            recommendations.append("Drawdown muito baixo - possível aumento de retorno com mais risco")

        ftmo_level = ftmo_analysis.get('compliance_level', '')
        if ftmo_level == 'FAILED':
            recommendations.append("Estratégia não é FTMO-compliant - revisar parâmetros")
        elif ftmo_level == 'PARTIAL':
            recommendations.append("Estratégia parcialmente FTMO-compliant - ajustes necessários")

        if not recommendations:
            recommendations.append("Estratégia com bom equilíbrio de risco/retorno")

        return recommendations

    def get_ftmo_recommendations(self, checks: Dict, risks: List) -> List[str]:
        """Gera recomendações específicas FTMO"""

        recommendations = []

        if not checks.get('daily_loss_check', False):
            recommendations.append("Reduzir risco diário para atender limite FTMO (5%)")

        if not checks.get('total_loss_check', False):
            recommendations.append("Implementar stop de perda total em 10%")

        if not checks.get('profit_target_check', False):
            recommendations.append("Aumentar alvo de lucro para mínimo 10%")

        if not checks.get('consistency_check', False):
            recommendations.append("Melhorar consistência de lucros diários")

        if risks:
            recommendations.extend([f"Atenção: {risk}" for risk in risks])

        return recommendations

    async def run_optimization(self, base_config: Dict, optimization_params: Dict) -> List[Dict]:
        """Executa otimização de parâmetros"""

        try:
            logger.info("🔧 Iniciando otimização de parâmetros...")

            # Gerar combinações de parâmetros
            parameter_combinations = self.generate_parameter_combinations(optimization_params)

            logger.info(f"📋 {len(parameter_combinations)} combinações para testar")

            # Executar backtests para cada combinação
            results = []
            for i, params in enumerate(parameter_combinations, 1):
                logger.info(f"🧪 Testando combinação {i}/{len(parameter_combinations)}")

                # Mesclar com configuração base
                test_config = base_config.copy()
                test_config['inputs'].update(params)

                # Executar backtest
                result = await self.run_single_backtest(test_config)

                if result:
                    results.append({
                        'parameters': params,
                        'score': result['analysis']['overall_score'],
                        'win_rate': result['analysis']['basic_metrics']['win_rate'],
                        'profit_factor': result['analysis']['basic_metrics']['profit_factor'],
                        'max_drawdown': result['analysis']['basic_metrics']['max_drawdown'],
                        'ftmo_compliance': result['analysis']['ftmo_compliance']['compliance_level'],
                        'total_profit': result['analysis']['basic_metrics']['total_profit']
                    })

                # Pequena pausa entre testes
                await asyncio.sleep(2)

            # Ordenar resultados por score
            results.sort(key=lambda x: x['score'], reverse=True)

            # Salvar resultados da otimização
            self.optimization_results = results

            logger.info(f"✅ Otimização concluída - {len(results)} testes bem-sucedidos")

            return results

        except Exception as e:
            logger.error(f"❌ Erro na otimização: {e}")
            return []

    def generate_parameter_combinations(self, params: Dict) -> List[Dict]:
        """Gera todas as combinações de parâmetros"""

        try:
            import itertools

            keys = list(params.keys())
            values = list(params.values())

            combinations = []
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                combinations.append(param_dict)

            return combinations

        except Exception as e:
            logger.error(f"❌ Erro ao gerar combinações: {e}")
            return []

    async def generate_comprehensive_report(self, test_ids: List[str] = None) -> str:
        """Gera relatório completo dos backtests"""

        try:
            logger.info("📊 Gerando relatório completo...")

            if test_ids is None:
                test_ids = list(self.backtest_results.keys())

            report = []
            report.append("# RELATÓRIO COMPLETO DE BACKTESTING")
            report.append("=" * 60)
            report.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            report.append(f"Símbolo: {self.symbol}")
            report.append(f"Total de Testes: {len(test_ids)}")
            report.append("")

            # Resumo dos melhores resultados
            if self.optimization_results:
                report.append("## TOP 5 MELHORES CONFIGURAÇÕES")
                report.append("-" * 40)

                for i, result in enumerate(self.optimization_results[:5], 1):
                    report.append(f"{i}. Score: {result['score']}/100")
                    report.append(f"   Parâmetros: {result['parameters']}")
                    report.append(f"   Win Rate: {result['win_rate']:.1f}%")
                    report.append(f"   Profit Factor: {result['profit_factor']:.2f}")
                    report.append(f"   Max DD: {result['max_drawdown']:.1f}%")
                    report.append(f"   FTMO: {result['ftmo_compliance']}")
                    report.append(f"   Lucro: ${result['total_profit']:.2f}")
                    report.append("")

            # Análise detalhada dos principais testes
            report.append("## ANÁLISE DETALHADA")
            report.append("-" * 30)

            for test_id in test_ids[:3]:  # Top 3 testes
                result = self.backtest_results[test_id]
                analysis = result['analysis']

                report.append(f"### Teste: {test_id}")
                report.append(f"Config: {result['config']}")
                report.append("")

                # Métricas básicas
                metrics = analysis['basic_metrics']
                report.append("#### Métricas de Performance:")
                report.append(f"- Total de Trades: {metrics['total_trades']}")
                report.append(f"- Win Rate: {metrics['win_rate']:.1f}%")
                report.append(f"- Profit Factor: {metrics['profit_factor']:.2f}")
                report.append(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                report.append(f"- Lucro Total: ${metrics['total_profit']:.2f}")
                report.append(f"- Drawdown Máximo: {metrics['max_drawdown']:.1f}%")
                report.append(f"- Drawdown Diário: {metrics['daily_drawdown']:.1f}%")
                report.append("")

                # FTMO Compliance
                ftmo = analysis['ftmo_compliance']
                report.append("#### Compliance FTMO:")
                report.append(f"- Nível: {ftmo.get('compliance_level', 'N/A')}")
                report.append(f"- Score: {ftmo.get('score', 0):.1f}%")
                report.append("- Verificações:")
                for check, passed in ftmo.get('checks', {}).items():
                    if isinstance(passed, bool):
                        status = "✅" if passed else "❌"
                        report.append(f"  {status} {check}")
                report.append("")

                # Recomendações
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    report.append("#### Recomendações:")
                    for rec in recommendations:
                        report.append(f"- {rec}")
                    report.append("")

                report.append("---")

            # Conclusões
            report.append("## CONCLUSÕES")
            report.append("-" * 20)

            if self.optimization_results:
                best = self.optimization_results[0]
                report.append(f"Melhor configuração encontrada:")
                report.append(f"- Score: {best['score']}/100")
                report.append(f"- Win Rate: {best['win_rate']:.1f}%")
                report.append(f"- FTMO Compliance: {best['ftmo_compliance']}")

                if best['ftmo_compliance'] == 'FULL':
                    report.append("✅ ESTRATÉGIA PRONTA PARA TRADING REAL")
                elif best['ftmo_compliance'] == 'PARTIAL':
                    report.append("⚠️ ESTRATÉGIA PRECISA DE AJUSTES")
                else:
                    report.append("❌ ESTRATÉGIA NÃO É VIÁVEL ATUALMENTE")

            report_text = "\n".join(report)

            # Salvar relatório
            report_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report_text)

            logger.info(f"📄 Relatório salvo: {report_file}")
            return report_text

        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")
            return f"Erro ao gerar relatório: {str(e)}"

    async def export_results_to_csv(self, filename: str = None) -> str:
        """Exporta resultados para CSV"""

        try:
            if filename is None:
                filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Preparar dados
            data = []
            for test_id, result in self.backtest_results.items():
                config = result['config']
                analysis = result['analysis']
                metrics = analysis['basic_metrics']
                ftmo = analysis['ftmo_compliance']

                row = {
                    'test_id': test_id,
                    'timestamp': result['timestamp'],
                    'period': config.get('period', ''),
                    'deposit': config.get('deposit', 0),
                    'leverage': config.get('leverage', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_profit': metrics.get('total_profit', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'daily_drawdown': metrics.get('daily_drawdown', 0),
                    'overall_score': analysis.get('overall_score', 0),
                    'ftmo_compliance': ftmo.get('compliance_level', ''),
                    'ftmo_score': ftmo.get('score', 0)
                }

                # Adicionar parâmetros de entrada
                for key, value in config.get('inputs', {}).items():
                    row[f'param_{key}'] = value

                data.append(row)

            # Criar DataFrame e salvar
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)

            logger.info(f"📊 Resultados exportados para: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Erro ao exportar CSV: {e}")
            return None

async def main():
    """Função principal de demonstração"""
    print("🧪 Sistema de Backtesting Completo")
    print("=" * 50)

    backtest_system = BacktestSystem()

    # Inicializar
    if not await backtest_system.initialize():
        logger.error("❌ Falha na inicialização")
        return

    try:
        # Configuração base para backtest
        base_config = {
            'period': '2024.01.01-2024.12.31',
            'deposit': 10000,
            'leverage': 100,
            'model': 4,
            'spread': 15,
            'inputs': {
                'RiskPercent': 1.0,
                'MaxSpread': 20,
                'MagicNumber': 12345,
                'StopLossPips': 50,
                'TakeProfitPips': 100
            }
        }

        # Executar backtest único
        print("\n1️⃣ Executando backtest único...")
        single_result = await backtest_system.run_single_backtest(base_config)

        if single_result:
            analysis = single_result['analysis']
            print(f"✅ Backtest concluído!")
            print(f"📊 Score: {analysis['overall_score']}/100")
            print(f"📈 Win Rate: {analysis['basic_metrics']['win_rate']:.1f}%")
            print(f"💰 Lucro: ${analysis['basic_metrics']['total_profit']:.2f}")
            print(f"📉 Drawdown: {analysis['basic_metrics']['max_drawdown']:.1f}%")
            print(f"✅ FTMO: {analysis['ftmo_compliance']['compliance_level']}")

        # Executar otimização
        print("\n2️⃣ Executando otimização de parâmetros...")
        optimization_params = {
            'RiskPercent': [0.5, 1.0, 1.5],
            'StopLossPips': [30, 50, 70],
            'TakeProfitPips': [80, 100, 120],
            'MaxSpread': [15, 20, 25]
        }

        optimization_results = await backtest_system.run_optimization(base_config, optimization_params)

        if optimization_results:
            print(f"✅ Otimização concluída!")
            print(f"📊 {len(optimization_results)} combinações testadas")
            print(f"🏆 Melhor score: {optimization_results[0]['score']}/100")

            print("\nTop 3 melhores configurações:")
            for i, result in enumerate(optimization_results[:3], 1):
                print(f"{i}. Score: {result['score']}/100 - FTMO: {result['ftmo_compliance']}")
                print(f"   WR: {result['win_rate']:.1f}% - PF: {result['profit_factor']:.2f}")
                print(f"   DD: {result['max_drawdown']:.1f}% - Lucro: ${result['total_profit']:.2f}")

        # Gerar relatório completo
        print("\n3️⃣ Gerando relatório completo...")
        report = await backtest_system.generate_comprehensive_report()
        print("✅ Relatório gerado com sucesso!")

        # Exportar resultados
        print("\n4️⃣ Exportando resultados...")
        csv_file = await backtest_system.export_results_to_csv()
        if csv_file:
            print(f"✅ Resultados exportados para: {csv_file}")

    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro na execução: {e}")
    finally:
        # Limpeza
        if backtest_system.mt5_client:
            await backtest_system.mt5_client.disconnect()
        print("\n✅ Sistema de backtesting finalizado")

if __name__ == "__main__":
    asyncio.run(main())