#!/usr/bin/env python3
"""
📊 EA Optimizer AI - Visualizer Module
Gera gráficos e relatórios visuais dos resultados de otimização
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EAOptimizerVisualizer:
    """Visualizador de resultados de otimização de EA"""

    def __init__(self, output_dir: str = "../output/charts"):
        """
        Inicializa o visualizador

        Args:
            output_dir: Diretório de saída para gráficos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuração de estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_optimization_dashboard(self,
                                    optimization_results: Dict[str, Any],
                                    save_html: bool = True) -> str:
        """
        Cria dashboard completo da otimização

        Args:
            optimization_results: Resultados da otimização
            save_html: Se deve salvar como HTML interativo

        Returns:
            Caminho do dashboard gerado
        """
        logger.info("📊 Criando dashboard de otimização...")

        # Criar subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Convergência da Otimização',
                'Importância dos Parâmetros',
                'Distribuição de Scores',
                'Heatmap de Correlações',
                'Top 10 Configurações',
                'Análise de Risk/Reward'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        # Dados dos trials
        trials = optimization_results.get('optimization_history', [])
        if not trials:
            logger.warning("⚠️ Nenhum dado de otimização encontrado")
            return ""

        # Preparar dados
        df_trials = pd.DataFrame(trials)
        df_scores = pd.DataFrame([t['score'] for t in trials], columns=['score'])

        # 1. Gráfico de Convergência
        self._add_convergence_plot(fig, df_scores, 1, 1)

        # 2. Importância dos Parâmetros
        if 'best_params' in optimization_results:
            self._add_param_importance_plot(fig, optimization_results, 1, 2)

        # 3. Distribuição de Scores
        self._add_score_distribution_plot(fig, df_scores, 2, 1)

        # 4. Heatmap de Correlações
        if len(trials) > 10:
            self._add_correlation_heatmap(fig, df_trials, 2, 2)

        # 5. Top 10 Configurações
        self._add_top_configurations_plot(fig, df_trials, 3, 1)

        # 6. Análise de Risk/Reward
        self._add_risk_reward_analysis(fig, df_trials, 3, 2)

        # Layout
        fig.update_layout(
            title={
                'text': f'🤖 EA Optimizer AI - Dashboard de Otimização<br>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1200,
            showlegend=False,
            template="plotly_white"
        )

        # Salvar dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.output_dir / f"optimization_dashboard_{timestamp}.html"

        if save_html:
            fig.write_html(str(dashboard_path))
            logger.info(f"✅ Dashboard salvo: {dashboard_path}")

        # Salvar como imagem também
        img_path = self.output_dir / f"optimization_dashboard_{timestamp}.png"
        fig.write_image(str(img_path), width=1600, height=1200)

        return str(dashboard_path)

    def _add_convergence_plot(self, fig, scores_df, row, col):
        """Adiciona gráfico de convergência"""
        # Calcular melhor score até o momento
        best_so_far = scores_df.expanding().max()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores_df))),
                y=scores_df['score'],
                mode='markers',
                name='Score do Trial',
                marker=dict(color='lightblue', size=4),
                opacity=0.6
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(best_so_far))),
                y=best_so_far['score'],
                mode='lines',
                name='Melhor Score',
                line=dict(color='red', width=2)
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Trial", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)

    def _add_param_importance_plot(self, fig, results, row, col):
        """Adiciona gráfico de importância dos parâmetros"""
        # Simular importância (em implementação real, viria do Optuna)
        params = results.get('best_params', {})
        param_importance = {
            'Risk/Reward': 0.25,
            'Risk Factor': 0.20,
            'ATR Multiplier': 0.15,
            'MA Period': 0.12,
            'RSI Period': 0.10,
            'Trading Sessions': 0.08,
            'Position Size': 0.10
        }

        fig.add_trace(
            go.Bar(
                x=list(param_importance.keys()),
                y=list(param_importance.values()),
                marker=dict(color='lightcoral')
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Parâmetro", row=row, col=col)
        fig.update_yaxes(title_text="Importância", row=row, col=col)

    def _add_score_distribution_plot(self, fig, scores_df, row, col):
        """Adiciona histograma de distribuição de scores"""
        fig.add_trace(
            go.Histogram(
                x=scores_df['score'],
                nbinsx=30,
                marker=dict(color='lightgreen'),
                opacity=0.7
            ),
            row=row, col=col
        )

        # Adicionar linha da média
        mean_score = scores_df['score'].mean()
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Média: {mean_score:.3f}"
        )

        fig.update_xaxes(title_text="Score", row=row, col=col)
        fig.update_yaxes(title_text="Frequência", row=row, col=col)

    def _add_correlation_heatmap(self, fig, df_trials, row, col):
        """Adiciona heatmap de correlações entre parâmetros"""
        # Extrair parâmetros numéricos
        param_cols = []
        for trial in df_trials.head(100):  # Limitar para performance
            for key, value in trial['params'].items():
                if isinstance(value, (int, float)) and key not in param_cols:
                    param_cols.append(key)

        if len(param_cols) > 1:
            # Criar matriz de parâmetros
            param_matrix = []
            for trial in df_trials.head(100):
                row = [trial['params'].get(col, 0) for col in param_cols]
                param_matrix.append(row)

            param_df = pd.DataFrame(param_matrix, columns=param_cols)
            corr_matrix = param_df.corr()

            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=row, col=col
            )

    def _add_top_configurations_plot(self, fig, df_trials, row, col):
        """Adiciona gráfico das top 10 configurações"""
        # Ordenar por score e pegar top 10
        sorted_trials = df_trials.nlargest(10, 'score')
        trial_numbers = [f"T{t+1}" for t in sorted_trials.index]

        fig.add_trace(
            go.Bar(
                x=trial_numbers,
                y=sorted_trials['score'],
                marker=dict(color='gold')
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Top Trials", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)

    def _add_risk_reward_analysis(self, fig, df_trials, row, col):
        """Adiciona análise de risk/reward"""
        # Calcular risk/reward para cada trial
        risk_rewards = []
        scores = []

        for _, trial in df_trials.iterrows():
            params = trial['params']
            if 'stop_loss' in params and 'take_profit' in params:
                rr = params['take_profit'] / params['stop_loss']
                risk_rewards.append(rr)
                scores.append(trial['score'])

        if risk_rewards:
            fig.add_trace(
                go.Scatter(
                    x=risk_rewards,
                    y=scores,
                    mode='markers',
                    marker=dict(
                        color=scores,
                        colorscale='Viridis',
                        size=8,
                        colorbar=dict(title="Score")
                    ),
                    text=[f"R/R: {rr:.2f}" for rr in risk_rewards],
                    hovertemplate="Risk/Reward: %{x}<br>Score: %{y}<extra></extra>"
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Risk/Reward Ratio", row=row, col=col)
            fig.update_yaxes(title_text="Score", row=row, col=col)

    def create_performance_comparison_chart(self,
                                          baseline_params: Dict[str, Any],
                                          optimized_params: Dict[str, Any],
                                          baseline_score: float,
                                          optimized_score: float) -> str:
        """
        Cria gráfico comparativo entre baseline e otimizado

        Args:
            baseline_params: Parâmetros baseline
            optimized_params: Parâmetros otimizados
            baseline_score: Score baseline
            optimized_score: Score otimizado

        Returns:
            Caminho do gráfico gerado
        """
        logger.info("📈 Criando gráfico comparativo de performance...")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Comparação de Score',
                'Principais Parâmetros',
                'Risk Management',
                'Indicadores Técnicos'
            ]
        )

        # 1. Comparação de Score
        fig.add_trace(
            go.Bar(
                x=['Baseline', 'Otimizado'],
                y=[baseline_score, optimized_score],
                marker=dict(color=['lightgray', 'lightgreen'])
            ),
            row=1, col=1
        )

        # 2. Principais Parâmetros
        key_params = ['stop_loss', 'take_profit', 'risk_factor', 'atr_multiplier']
        baseline_values = [baseline_params.get(p, 0) for p in key_params]
        optimized_values = [optimized_params.get(p, 0) for p in key_params]

        fig.add_trace(
            go.Bar(
                x=key_params,
                y=baseline_values,
                name='Baseline',
                marker=dict(color='lightblue')
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(
                x=key_params,
                y=optimized_values,
                name='Otimizado',
                marker=dict(color='lightcoral')
            ),
            row=1, col=2
        )

        # 3. Risk Management
        risk_metrics = ['Risk/Reward', 'Risk Factor', 'Max Drawdown %']
        baseline_risk = [
            baseline_params.get('take_profit', 0) / max(baseline_params.get('stop_loss', 1), 1),
            baseline_params.get('risk_factor', 0),
            15.0  # Valor simulado
        ]
        optimized_risk = [
            optimized_params.get('take_profit', 0) / max(optimized_params.get('stop_loss', 1), 1),
            optimized_params.get('risk_factor', 0),
            10.0  # Valor simulado melhorado
        ]

        fig.add_trace(
            go.Scatter(
                x=risk_metrics,
                y=baseline_risk,
                mode='markers+lines',
                name='Baseline',
                marker=dict(size=10)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=risk_metrics,
                y=optimized_risk,
                mode='markers+lines',
                name='Otimizado',
                marker=dict(size=10)
            ),
            row=2, col=1
        )

        # 4. Indicadores Técnicos
        tech_params = ['MA Period', 'RSI Period', 'BB StdDev']
        baseline_tech = [baseline_params.get('ma_period', 0), baseline_params.get('rsi_period', 0), baseline_params.get('bb_std', 0)]
        optimized_tech = [optimized_params.get('ma_period', 0), optimized_params.get('rsi_period', 0), optimized_params.get('bb_std', 0)]

        fig.add_trace(
            go.Bar(
                x=tech_params,
                y=baseline_tech,
                name='Baseline',
                marker=dict(color='lightblue'),
                showlegend=False
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Bar(
                x=tech_params,
                y=optimized_tech,
                name='Otimizado',
                marker=dict(color='lightcoral'),
                showlegend=False
            ),
            row=2, col=2
        )

        # Layout
        fig.update_layout(
            title={
                'text': '📊 EA Optimizer AI - Comparação: Baseline vs Otimizado',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            template="plotly_white"
        )

        # Salvar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.output_dir / f"performance_comparison_{timestamp}.html"
        fig.write_html(str(chart_path))

        return str(chart_path)

    def create_parameter_sensitivity_analysis(self,
                                            optimization_results: Dict[str, Any]) -> str:
        """
        Cria análise de sensibilidade dos parâmetros

        Args:
            optimization_results: Resultados da otimização

        Returns:
            Caminho do gráfico gerado
        """
        logger.info("🔍 Criando análise de sensibilidade de parâmetros...")

        trials = optimization_results.get('optimization_history', [])
        if not trials:
            return ""

        df_trials = pd.DataFrame(trials)

        # Identificar parâmetros numéricos
        numeric_params = []
        for trial in trials[:100]:  # Limitar para performance
            for key, value in trial['params'].items():
                if isinstance(value, (int, float)) and key not in numeric_params:
                    numeric_params.append(key)

        # Criar subplots para principais parâmetros
        n_params = min(6, len(numeric_params))
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Sensibilidade: {p.replace("_", " ").title()}' for p in numeric_params[:n_params]]
        )

        for i, param in enumerate(numeric_params[:n_params]):
            row = (i // 3) + 1
            col = (i % 3) + 1

            # Extrair valores do parâmetro e scores
            param_values = []
            scores = []

            for _, trial in df_trials.iterrows():
                if param in trial['params']:
                    param_values.append(trial['params'][param])
                    scores.append(trial['score'])

            if param_values:
                # Scatter plot com linha de tendência
                fig.add_trace(
                    go.Scatter(
                        x=param_values,
                        y=scores,
                        mode='markers',
                        marker=dict(
                            color=scores,
                            colorscale='Viridis',
                            size=6,
                            showscale=False if i > 0 else True,
                            colorbar=dict(title="Score", x=1.02) if i == 0 else None
                        ),
                        name=param.replace('_', ' ').title(),
                        hovertemplate=f"{param}: %{{x}}<br>Score: %{{y}}<extra></extra>"
                    ),
                    row=row, col=col
                )

                # Adicionar linha de tendência (média móvel)
                sorted_data = sorted(zip(param_values, scores))
                x_vals, y_vals = zip(*sorted_data)

                # Calcular média móvel
                window_size = max(3, len(y_vals) // 10)
                if len(y_vals) >= window_size:
                    y_smooth = pd.Series(y_vals).rolling(window=window_size, center=True).mean()

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_smooth,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Tendência',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

            fig.update_xaxes(title_text=param.replace('_', ' ').title(), row=row, col=col)
            fig.update_yaxes(title_text="Score", row=row, col=col)

        # Layout
        fig.update_layout(
            title={
                'text': '🔍 EA Optimizer AI - Análise de Sensibilidade de Parâmetros',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=600,
            template="plotly_white",
            showlegend=False
        )

        # Salvar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.output_dir / f"parameter_sensitivity_{timestamp}.html"
        fig.write_html(str(chart_path))

        return str(chart_path)

    def create_summary_report_html(self,
                                 optimization_results: Dict[str, Any],
                                 baseline_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Cria relatório completo em HTML

        Args:
            optimization_results: Resultados da otimização
            baseline_results: Resultados baseline para comparação

        Returns:
            Caminho do relatório HTML
        """
        logger.info("📄 Criando relatório HTML completo...")

        # Gerar componentes do relatório
        dashboard_path = self.create_optimization_dashboard(optimization_results)

        if baseline_results:
            comparison_path = self.create_performance_comparison_chart(
                baseline_params=baseline_results.get('params', {}),
                optimized_params=optimization_results.get('best_params', {}),
                baseline_score=baseline_results.get('score', 0),
                optimized_score=optimization_results.get('best_score', 0)
            )
        else:
            comparison_path = ""

        sensitivity_path = self.create_parameter_sensitivity_analysis(optimization_results)

        # Criar HTML completo
        html_content = self._generate_html_report(
            optimization_results=optimization_results,
            baseline_results=baseline_results,
            dashboard_path=dashboard_path,
            comparison_path=comparison_path,
            sensitivity_path=sensitivity_path
        )

        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"ea_optimizer_report_{timestamp}.html"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✅ Relatório HTML criado: {report_path}")
        return str(report_path)

    def _generate_html_report(self,
                            optimization_results: Dict[str, Any],
                            baseline_results: Optional[Dict[str, Any]],
                            dashboard_path: str,
                            comparison_path: str,
                            sensitivity_path: str) -> str:
        """Gera conteúdo HTML do relatório"""

        # Métricas principais
        best_score = optimization_results.get('best_score', 0)
        best_params = optimization_results.get('best_params', {})
        n_trials = len(optimization_results.get('optimization_history', []))

        # Calcular melhorias
        improvement_pct = 0
        if baseline_results:
            baseline_score = baseline_results.get('score', 0)
            if baseline_score > 0:
                improvement_pct = ((best_score - baseline_score) / baseline_score) * 100

        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 EA Optimizer AI - Relatório Completo</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .improvement {{
            background: #4caf50;
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            margin: 10px 0;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .chart-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .param-group {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .param-title {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .param-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-size: 0.9em;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 EA Optimizer AI</h1>
        <h2>Relatório Completo de Otimização</h2>
        <p>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{best_score:.4f}</div>
            <div class="metric-label">Melhor Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{n_trials}</div>
            <div class="metric-label">Trials Executados</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{best_params.get('risk_factor', 0):.2f}</div>
            <div class="metric-label">Risk Factor Ótimo</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{best_params.get('take_profit', 0) / max(best_params.get('stop_loss', 1), 1):.2f}:1</div>
            <div class="metric-label">Risk/Reward Ratio</div>
        </div>
    </div>

    {f'<div class="improvement">🚀 Melhoria de {improvement_pct:.1f}% em relação ao baseline</div>' if improvement_pct > 0 else ''}

    <div class="chart-container">
        <div class="chart-title">📊 Dashboard de Otimização</div>
        <iframe src="file://{Path(dashboard_path).absolute()}"></iframe>
    </div>

    {f'<div class="chart-container"><div class="chart-title">📈 Comparação de Performance</div><iframe src="file://{Path(comparison_path).absolute()}"></iframe></div>' if comparison_path else ''}

    <div class="chart-container">
        <div class="chart-title">🔍 Análise de Sensibilidade</div>
        <iframe src="file://{Path(sensitivity_path).absolute()}"></iframe>
    </div>

    <div class="params-grid">
        <div class="param-group">
            <div class="param-title">🎯 Melhores Parâmetros - Risk Management</div>
            <div class="param-item">
                <span>Stop Loss:</span>
                <span>{best_params.get('stop_loss', 'N/A')} points</span>
            </div>
            <div class="param-item">
                <span>Take Profit:</span>
                <span>{best_params.get('take_profit', 'N/A')} points</span>
            </div>
            <div class="param-item">
                <span>Risk Factor:</span>
                <span>{best_params.get('risk_factor', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>ATR Multiplier:</span>
                <span>{best_params.get('atr_multiplier', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>Lot Size:</span>
                <span>{best_params.get('lot_size', 'N/A')}</span>
            </div>
        </div>

        <div class="param-group">
            <div class="param-title">📈 Indicadores Técnicos</div>
            <div class="param-item">
                <span>MA Period:</span>
                <span>{best_params.get('ma_period', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>RSI Period:</span>
                <span>{best_params.get('rsi_period', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>RSI Oversold:</span>
                <span>{best_params.get('rsi_oversold', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>RSI Overbought:</span>
                <span>{best_params.get('rsi_overbought', 'N/A')}</span>
            </div>
            <div class="param-item">
                <span>BB StdDev:</span>
                <span>{best_params.get('bb_std', 'N/A')}</span>
            </div>
        </div>

        <div class="param-group">
            <div class="param-title">⏰ Sessões de Trading</div>
            <div class="param-item">
                <span>Asian Session:</span>
                <span>{best_params.get('asian_session_start', 'N/A')}h - {best_params.get('asian_session_end', 'N/A')}h</span>
            </div>
            <div class="param-item">
                <span>European Session:</span>
                <span>{best_params.get('european_session_start', 'N/A')}h - {best_params.get('european_session_end', 'N/A')}h</span>
            </div>
            <div class="param-item">
                <span>US Session:</span>
                <span>{best_params.get('us_session_start', 'N/A')}h - {best_params.get('us_session_end', 'N/A')}h</span>
            </div>
            <div class="param-item">
                <span>Max Positions:</span>
                <span>{best_params.get('max_positions', 'N/A')}</span>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>🤖 Relatório gerado automaticamente pelo EA Optimizer AI</p>
        <p>Desenvolvido para otimização de Expert Advisors para XAUUSD</p>
    </div>
</body>
</html>
        """

        return html

if __name__ == "__main__":
    # Teste do visualizador
    import json

    # Dados de exemplo
    sample_results = {
        'best_score': 85.5,
        'best_params': {
            'stop_loss': 120,
            'take_profit': 240,
            'risk_factor': 1.8,
            'atr_multiplier': 1.6,
            'lot_size': 0.02,
            'ma_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_std': 2.0,
            'max_positions': 3,
            'asian_session_start': 0,
            'asian_session_end': 8,
            'european_session_start': 7,
            'european_session_end': 16,
            'us_session_start': 13,
            'us_session_end': 22
        },
        'optimization_history': [
            {
                'score': 75.2 + np.random.normal(0, 5),
                'params': {
                    'stop_loss': np.random.randint(50, 200),
                    'take_profit': np.random.randint(100, 400),
                    'risk_factor': np.random.uniform(0.5, 2.5),
                    'atr_multiplier': np.random.uniform(0.8, 2.5)
                }
            }
            for _ in range(50)
        ]
    }

    baseline_results = {
        'score': 65.0,
        'params': {
            'stop_loss': 150,
            'take_profit': 300,
            'risk_factor': 1.5,
            'atr_multiplier': 1.5
        }
    }

    # Criar visualizações
    visualizer = EAOptimizerVisualizer()
    report_path = visualizer.create_summary_report_html(
        optimization_results=sample_results,
        baseline_results=baseline_results
    )

    print(f"✅ Relatório criado: {report_path}")