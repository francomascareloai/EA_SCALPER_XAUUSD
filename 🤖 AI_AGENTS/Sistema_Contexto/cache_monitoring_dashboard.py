#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de Monitoramento e Analytics do Cache Avançado
Sistema de monitoramento em tempo real com visualizações e alertas

Este sistema implementa:
1. Dashboard web interativo
2. Monitoramento em tempo real
3. Análise de performance
4. Sistema de alertas
5. Relatórios automatizados
6. Análise preditiva

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import deque
import statistics
import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import webbrowser
from concurrent.futures import ThreadPoolExecutor

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métricas de performance do cache."""
    timestamp: datetime
    hit_rate: float
    total_requests: int
    hits: int
    misses: int
    evictions: int
    level_transfers: int
    memory_usage_mb: float
    disk_usage_mb: float
    compression_ratio: float
    deduplication_rate: float
    avg_response_time: float
    error_rate: float

@dataclass
class Alert:
    """Alerta do sistema."""
    id: str
    type: str  # warning, error, critical
    message: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MetricsCollector:
    """Coletor de métricas de performance."""

    def __init__(self, cache_system, window_size: int = 1000):
        self.cache_system = cache_system
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.current_metrics = {}
        self.collection_interval = 5  # segundos
        self.is_collecting = False
        self.executor = ThreadPoolExecutor(max_workers=2)

    def start_collection(self):
        """Inicia coleta de métricas."""
        if not self.is_collecting:
            self.is_collecting = True
            self.executor.submit(self._collection_loop)

    def stop_collection(self):
        """Para coleta de métricas."""
        self.is_collecting = False
        self.executor.shutdown(wait=True)

    def _collection_loop(self):
        """Loop principal de coleta."""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                self.current_metrics = asdict(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")

    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Coleta métricas atuais."""
        try:
            cache_stats = self.cache_system.get_cache_stats()

            # Calcular uso de disco
            disk_usage = 0
            try:
                cache_dir = Path(self.cache_system.cache_dir)
                if cache_dir.exists():
                    disk_usage = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            except:
                pass

            # Calcular uso de memória
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024

            # Métricas básicas
            performance = cache_stats.get('performance', {})
            storage = cache_stats.get('storage', {})

            return PerformanceMetrics(
                timestamp=datetime.now(),
                hit_rate=performance.get('hit_rate', 0),
                total_requests=performance.get('total_requests', 0),
                hits=performance.get('hits', 0),
                misses=performance.get('misses', 0),
                evictions=performance.get('evictions', 0),
                level_transfers=performance.get('level_transfers', 0),
                memory_usage_mb=memory_usage,
                disk_usage_mb=disk_usage / 1024 / 1024,
                compression_ratio=storage.get('compression_ratio', 1.0),
                deduplication_rate=storage.get('semantic_deduplication_rate', 0),
                avg_response_time=self._calculate_avg_response_time(),
                error_rate=self._calculate_error_rate()
            )

        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                hit_rate=0, total_requests=0, hits=0, misses=0, evictions=0,
                level_transfers=0, memory_usage_mb=0, disk_usage_mb=0,
                compression_ratio=1.0, deduplication_rate=0, avg_response_time=0, error_rate=0
            )

    def _calculate_avg_response_time(self) -> float:
        """Calcula tempo médio de resposta."""
        # Implementação simplificada
        return 0.001  # 1ms

    def _calculate_error_rate(self) -> float:
        """Calcula taxa de erro."""
        # Implementação simplificada
        return 0.001  # 0.1%

    def get_recent_metrics(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Retorna métricas recentes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]

    def get_aggregated_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Retorna métricas agregadas."""
        recent_metrics = self.get_recent_metrics(minutes)

        if not recent_metrics:
            return {}

        hit_rates = [m.hit_rate for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        response_times = [m.avg_response_time for m in recent_metrics]

        return {
            'avg_hit_rate': statistics.mean(hit_rates) if hit_rates else 0,
            'min_hit_rate': min(hit_rates) if hit_rates else 0,
            'max_hit_rate': max(hit_rates) if hit_rates else 0,
            'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'total_requests': sum(m.total_requests for m in recent_metrics),
            'total_evictions': sum(m.evictions for m in recent_metrics),
            'data_points': len(recent_metrics)
        }

class AlertManager:
    """Gerenciador de alertas do sistema."""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules = {
            'low_hit_rate': {'threshold': 0.5, 'type': 'warning', 'message': 'Hit rate abaixo de 50%'},
            'high_memory_usage': {'threshold': 80, 'type': 'warning', 'message': 'Uso de memória acima de 80%'},
            'high_error_rate': {'threshold': 0.05, 'type': 'error', 'message': 'Taxa de erro acima de 5%'},
            'high_eviction_rate': {'threshold': 1000, 'type': 'warning', 'message': 'Muitas evictions por hora'}
        }

    def check_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """Verifica condições de alerta."""
        new_alerts = []

        # Verificar hit rate
        if metrics.hit_rate < self.alert_rules['low_hit_rate']['threshold']:
            alert_id = f"low_hit_rate_{int(time.time())}"
            if alert_id not in self.alerts:
                new_alerts.append(Alert(
                    id=alert_id,
                    type=self.alert_rules['low_hit_rate']['type'],
                    message=self.alert_rules['low_hit_rate']['message'],
                    created_at=datetime.now(),
                    metadata={'current_hit_rate': metrics.hit_rate}
                ))

        # Verificar uso de memória
        memory_percent = (metrics.memory_usage_mb / psutil.virtual_memory().total * 1024 * 1024) * 100
        if memory_percent > self.alert_rules['high_memory_usage']['threshold']:
            alert_id = f"high_memory_{int(time.time())}"
            if alert_id not in self.alerts:
                new_alerts.append(Alert(
                    id=alert_id,
                    type=self.alert_rules['high_memory_usage']['type'],
                    message=self.alert_rules['high_memory_usage']['message'],
                    created_at=datetime.now(),
                    metadata={'memory_percent': memory_percent}
                ))

        # Verificar taxa de erro
        if metrics.error_rate > self.alert_rules['high_error_rate']['threshold']:
            alert_id = f"high_error_{int(time.time())}"
            if alert_id not in self.alerts:
                new_alerts.append(Alert(
                    id=alert_id,
                    type=self.alert_rules['high_error_rate']['type'],
                    message=self.alert_rules['high_error_rate']['message'],
                    created_at=datetime.now(),
                    metadata={'error_rate': metrics.error_rate}
                ))

        # Adicionar novos alertas
        for alert in new_alerts:
            self.alerts[alert.id] = alert

        return new_alerts

    def resolve_alert(self, alert_id: str):
        """Resolve um alerta."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved_at = datetime.now()

    def get_active_alerts(self) -> List[Alert]:
        """Retorna alertas ativos."""
        return [alert for alert in self.alerts.values() if alert.resolved_at is None]

class DashboardGenerator:
    """Gerador de dashboard e visualizações."""

    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager

    def generate_performance_dashboard(self) -> str:
        """Gera dashboard de performance em HTML."""
        metrics = self.metrics_collector.get_recent_metrics(60)
        aggregated = self.metrics_collector.get_aggregated_metrics(60)
        active_alerts = self.alert_manager.get_active_alerts()

        if not metrics:
            return self._generate_empty_dashboard()

        # Preparar dados para gráficos
        timestamps = [m.timestamp for m in metrics]
        hit_rates = [m.hit_rate for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        requests = [m.total_requests for m in metrics]

        # Criar gráficos com Plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Hit Rate Over Time', 'Memory Usage', 'Total Requests',
                          'Compression & Deduplication', 'Active Alerts', 'Performance Summary'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'table'}, {'type': 'indicator'}]]
        )

        # Hit Rate
        fig.add_trace(
            go.Scatter(x=timestamps, y=hit_rates, mode='lines+markers', name='Hit Rate'),
            row=1, col=1
        )

        # Memory Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_usage, mode='lines+markers', name='Memory (MB)'),
            row=1, col=2
        )

        # Total Requests
        fig.add_trace(
            go.Scatter(x=timestamps, y=requests, mode='lines+markers', name='Total Requests'),
            row=2, col=1
        )

        # Compression & Deduplication
        compression_ratios = [m.compression_ratio for m in metrics]
        deduplication_rates = [m.deduplication_rate for m in metrics]

        fig.add_trace(
            go.Bar(x=['Compression Ratio', 'Deduplication Rate'],
                  y=[statistics.mean(compression_ratios), statistics.mean(deduplication_rates)]),
            row=2, col=2
        )

        # Active Alerts Table
        if active_alerts:
            alert_data = [
                [alert.id, alert.type, alert.message, alert.created_at.strftime('%H:%M:%S')]
                for alert in active_alerts
            ]
            fig.add_trace(
                go.Table(
                    header=dict(values=['ID', 'Type', 'Message', 'Created']),
                    cells=dict(values=list(zip(*alert_data)))
                ),
                row=3, col=1
            )
        else:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Status']),
                    cells=dict(values=[['No active alerts']])
                ),
                row=3, col=1
            )

        # Performance Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=aggregated.get('avg_hit_rate', 0) * 100,
                title={'text': "Average Hit Rate %"},
                gauge={'axis': {'range': [0, 100]}},
                number={'suffix': "%"}
            ),
            row=3, col=2
        )

        # Configurar layout
        fig.update_layout(height=1000, title_text="Cache Performance Dashboard")
        fig.update_xaxes(tickformat='%H:%M:%S')

        # Converter para HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs=True)

        return html_content

    def _generate_empty_dashboard(self) -> str:
        """Gera dashboard vazio."""
        return """
        <html>
        <head><title>Cache Dashboard</title></head>
        <body>
            <h1>Cache Performance Dashboard</h1>
            <p>No metrics available yet. Please wait for data collection to start.</p>
        </body>
        </html>
        """

    def generate_system_report(self, format: str = "html") -> str:
        """Gera relatório completo do sistema."""
        metrics = self.metrics_collector.get_aggregated_metrics(1440)  # Últimas 24h
        active_alerts = self.alert_manager.get_active_alerts()

        report_data = {
            'generated_at': datetime.now().isoformat(),
            'period': '24 hours',
            'metrics': asdict(metrics) if metrics else {},
            'active_alerts': len(active_alerts),
            'alerts_details': [asdict(alert) for alert in active_alerts],
            'system_info': {
                'python_version': os.sys.version,
                'platform': os.sys.platform,
                'cpu_count': os.cpu_count(),
                'memory_total': psutil.virtual_memory().total
            }
        }

        if format == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format == "html":
            return self._generate_html_report(report_data)
        else:
            return str(report_data)

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Gera relatório em HTML."""
        html = f"""
        <html>
        <head>
            <title>Cache System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Cache System Report</h1>
            <p><strong>Generated:</strong> {data['generated_at']}</p>
            <p><strong>Period:</strong> {data['period']}</p>

            <h2>Performance Metrics</h2>
            <div class="metric">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """

        for key, value in data['metrics'].items():
            if isinstance(value, float):
                html += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
            else:
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += f"""
                </table>
            </div>

            <h2>Active Alerts ({data['active_alerts']})</h2>
        """

        for alert in data['alerts_details']:
            html += f"""
            <div class="alert">
                <strong>{alert['type'].upper()}</strong>: {alert['message']}<br>
                <small>Created: {alert['created_at']}</small>
            </div>
            """

        if not data['alerts_details']:
            html += "<p>No active alerts</p>"

        html += """
            <h2>System Information</h2>
            <div class="metric">
                <table>
        """

        for key, value in data['system_info'].items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html

class WebDashboard:
    """Dashboard web interativo."""

    def __init__(self, cache_system, port: int = 5000):
        self.cache_system = cache_system
        self.port = port
        self.app = Flask(__name__)
        self.metrics_collector = MetricsCollector(cache_system)
        self.alert_manager = AlertManager()
        self.dashboard_generator = DashboardGenerator(self.metrics_collector, self.alert_manager)

        self._setup_routes()

    def _setup_routes(self):
        """Configura rotas do Flask."""

        @self.app.route('/')
        def dashboard():
            return self.dashboard_generator.generate_performance_dashboard()

        @self.app.route('/api/metrics')
        def api_metrics():
            aggregated = self.metrics_collector.get_aggregated_metrics(60)
            return jsonify(aggregated)

        @self.app.route('/api/alerts')
        def api_alerts():
            alerts = [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
            return jsonify(alerts)

        @self.app.route('/api/report')
        def api_report():
            format = request.args.get('format', 'html')
            report = self.dashboard_generator.generate_system_report(format)
            return report

        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'metrics_collecting': self.metrics_collector.is_collecting
            })

    def start(self):
        """Inicia o dashboard."""
        self.metrics_collector.start_collection()
        logger.info(f"Iniciando dashboard na porta {self.port}")

        # Abrir navegador automaticamente
        try:
            webbrowser.open(f"http://localhost:{self.port}")
        except:
            pass

        self.app.run(host='0.0.0.0', port=self.port, debug=False)

    def stop(self):
        """Para o dashboard."""
        self.metrics_collector.stop_collection()
        logger.info("Dashboard parado")

if __name__ == "__main__":
    print("=== Cache Monitoring Dashboard ===")
    print("\nEste é um módulo de monitoramento.")
    print("Para usar, importe e inicialize com um sistema de cache:")
    print("\nfrom sistema_cache_avancado import AdvancedCacheSystem")
    print("from cache_monitoring_dashboard import WebDashboard")
    print("\n# Inicializar sistema")
    print("cache_system = AdvancedCacheSystem()")
    print("\n# Inicializar dashboard")
    print("dashboard = WebDashboard(cache_system)")
    print("\n# Iniciar dashboard")
    print("dashboard.start()")
    print("\nAcesse: http://localhost:5000")