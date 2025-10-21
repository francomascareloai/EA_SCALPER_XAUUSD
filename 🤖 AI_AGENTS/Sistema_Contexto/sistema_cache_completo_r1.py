#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Completo de Cache Avançado para R1
Integração completa de todos os componentes

Este sistema implementa:
1. Sistema de cache multi-nível completo
2. Integração com R1
3. Dashboard de monitoramento
4. Sistema de configuração
5. Benchmarking e testes
6. Migração e backup
7. API completa

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import logging
from flask import Flask, request, jsonify
import webbrowser

# Importações dos módulos do sistema
from sistema_cache_avancado import AdvancedCacheSystem
from sistema_contexto_expandido_2m import ContextManager
from integracao_cache_r1 import R1CacheIntegration
from cache_monitoring_dashboard import WebDashboard, MetricsCollector
from configuracao_cache import ConfigurationManager, ConfigProfile
from benchmark_cache import BenchmarkSuite
from cache_migration_system import MigrationManager, BackupManager

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteR1CacheSystem:
    """Sistema completo de cache para R1 com todos os componentes integrados."""

    def __init__(self,
                 cache_dir: str = "./cache/advanced",
                 config_dir: str = "./config",
                 enable_monitoring: bool = True,
                 enable_backup: bool = True):

        self.cache_dir = Path(cache_dir)
        self.config_dir = Path(config_dir)
        self.enable_monitoring = enable_monitoring
        self.enable_backup = enable_backup

        # Componentes do sistema
        self.config_manager = None
        self.cache_system = None
        self.r1_manager = None
        self.integration = None
        self.metrics_collector = None
        self.dashboard = None
        self.migration_manager = None
        self.backup_manager = None
        self.benchmark_suite = None

        # Estado do sistema
        self.system_status = "stopped"
        self.start_time = None

        # Lock para operações thread-safe
        self.system_lock = threading.RLock()

        logger.info("Sistema completo R1 Cache inicializado")

    def initialize_system(self):
        """Inicializa todos os componentes do sistema."""

        with self.system_lock:
            logger.info("Inicializando sistema completo...")

            try:
                # 1. Inicializar gerenciador de configuração
                self.config_manager = ConfigurationManager(str(self.config_dir))

                # Carregar configuração
                if self.config_manager.current_config is None:
                    self.config_manager.load_profile(ConfigProfile.PRODUCTION)

                config = self.config_manager.current_config

                # 2. Inicializar sistema de cache avançado
                self.cache_system = AdvancedCacheSystem(
                    cache_dir=config.cache_dir,
                    embedding_model=config.embedding_model,
                    max_memory_mb=config.max_memory_mb,
                    enable_semantic_dedup=config.enable_semantic_dedup
                )

                # 3. Inicializar gerenciador R1
                self.r1_manager = ContextManager(
                    base_url="http://localhost:4000",
                    model_name="deepseek-r1-free",
                    cache_dir=config.cache_dir,
                    max_context_tokens=163000,
                    target_context_tokens=2000000
                )

                # 4. Inicializar integração
                self.integration = R1CacheIntegration(
                    self.cache_system,
                    self.r1_manager
                )

                # 5. Inicializar monitoramento se habilitado
                if self.enable_monitoring:
                    self.metrics_collector = MetricsCollector(self.cache_system)
                    self.dashboard = WebDashboard(self.cache_system)

                # 6. Inicializar sistemas de backup e migração
                if self.enable_backup:
                    self.migration_manager = MigrationManager(config.cache_dir)
                    self.backup_manager = BackupManager(config.cache_dir)

                # 7. Inicializar suite de benchmark
                self.benchmark_suite = BenchmarkSuite(self.cache_system)

                logger.info("Sistema inicializado com sucesso")

            except Exception as e:
                logger.error(f"Erro na inicialização do sistema: {e}")
                raise

    def start_system(self):
        """Inicia o sistema completo."""

        with self.system_lock:
            if self.system_status == "running":
                logger.warning("Sistema já está em execução")
                return

            logger.info("Iniciando sistema completo...")

            try:
                # Iniciar coleta de métricas
                if self.metrics_collector:
                    self.metrics_collector.start_collection()

                # Iniciar dashboard
                if self.dashboard:
                    dashboard_thread = threading.Thread(target=self.dashboard.start)
                    dashboard_thread.daemon = True
                    dashboard_thread.start()

                self.system_status = "running"
                self.start_time = datetime.now()

                logger.info("Sistema iniciado com sucesso")

            except Exception as e:
                logger.error(f"Erro ao iniciar sistema: {e}")
                self.system_status = "error"

    def stop_system(self):
        """Para o sistema completo."""

        with self.system_lock:
            if self.system_status != "running":
                logger.warning("Sistema não está em execução")
                return

            logger.info("Parando sistema completo...")

            try:
                # Parar coleta de métricas
                if self.metrics_collector:
                    self.metrics_collector.stop_collection()

                # Parar dashboard
                if self.dashboard:
                    self.dashboard.stop()

                self.system_status = "stopped"

                logger.info("Sistema parado com sucesso")

            except Exception as e:
                logger.error(f"Erro ao parar sistema: {e}")
                self.system_status = "error"

    def chat_with_r1(self, query: str, **kwargs) -> Dict[str, Any]:
        """Interface principal para chat com R1 usando cache."""

        if self.system_status != "running":
            raise RuntimeError("Sistema não está em execução")

        if not self.integration:
            raise RuntimeError("Sistema não foi inicializado")

        return self.integration.chat_with_r1_cached(query, **kwargs)

    def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas do sistema."""

        stats = {
            'system_info': {
                'status': self.system_status,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
        }

        # Estatísticas do cache
        if self.cache_system:
            stats['cache_stats'] = self.cache_system.get_cache_stats()

        # Estatísticas da integração
        if self.integration:
            stats['integration_stats'] = self.integration.get_integration_stats()

        # Estatísticas de configuração
        if self.config_manager and self.config_manager.current_config:
            stats['config_summary'] = self.config_manager.get_configuration_summary()

        return stats

    def optimize_system(self):
        """Otimiza todo o sistema."""

        logger.info("Otimizando sistema completo...")

        try:
            # Otimizar cache
            if self.cache_system:
                self.cache_system.optimize_cache()

            # Otimizar configuração
            if self.config_manager and self.cache_system:
                self.config_manager.optimize_configuration(self.cache_system)

            # Otimizar integração
            if self.integration:
                self.integration.optimize_cache_for_r1()

            # Limpar cache antigo
            if self.cache_system:
                self.cache_system.cleanup_cache()

            logger.info("Otimização concluída")

        except Exception as e:
            logger.error(f"Erro na otimização: {e}")

    def backup_system(self, backup_type: str = "full") -> str:
        """Cria backup completo do sistema."""

        if not self.enable_backup or not self.backup_manager:
            raise RuntimeError("Sistema de backup não está habilitado")

        logger.info(f"Criando backup {backup_type}...")

        try:
            backup_info = self.backup_manager.create_backup(
                backup_type=backup_type,
                compression="gzip"
            )

            logger.info(f"Backup criado: {backup_info.id}")
            return backup_info.id

        except Exception as e:
            logger.error(f"Erro no backup: {e}")
            raise

    def run_benchmark(self, duration: int = 60) -> Dict[str, Any]:
        """Executa benchmark completo do sistema."""

        if not self.benchmark_suite:
            raise RuntimeError("Suite de benchmark não está inicializada")

        logger.info(f"Executando benchmark ({duration}s)...")

        try:
            results = self.benchmark_suite.run_comprehensive_benchmark(
                test_duration=duration,
                concurrency=4
            )

            logger.info("Benchmark concluído")
            return results

        except Exception as e:
            logger.error(f"Erro no benchmark: {e}")
            raise

    def update_configuration(self, updates: Dict[str, Any]):
        """Atualiza configuração do sistema."""

        if not self.config_manager:
            raise RuntimeError("Gerenciador de configuração não está inicializado")

        logger.info(f"Atualizando configuração: {updates}")

        try:
            self.config_manager.update_configuration(updates)

            # Reinicie componentes se necessário
            if 'cache_dir' in updates or 'max_memory_mb' in updates:
                logger.info("Reinicializando sistema de cache...")
                self.stop_system()
                self.initialize_system()
                self.start_system()

            logger.info("Configuração atualizada")

        except Exception as e:
            logger.error(f"Erro na atualização de configuração: {e}")
            raise

    def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Retorna informações sobre uma entrada específica do cache."""

        if not self.cache_system:
            return None

        # Procurar em todos os níveis
        for level_name, level_manager in self.cache_system.cache_levels.items():
            with level_manager.lock:
                if key in level_manager.entries:
                    entry = level_manager.entries[key]
                    return {
                        'key': entry.key,
                        'level': entry.level.value,
                        'created_at': entry.created_at.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'importance_score': entry.importance_score,
                        'compressed': entry.compressed,
                        'compression_ratio': entry.original_size / entry.compressed_size if entry.compressed and entry.compressed_size > 0 else 1.0,
                        'ttl': entry.ttl,
                        'expired': entry.is_expired()
                    }

        return None

    def create_api_server(self, port: int = 8080) -> Flask:
        """Cria servidor API para o sistema."""

        app = Flask(__name__)

        @app.route('/api/chat', methods=['POST'])
        def api_chat():
            """Endpoint para chat com R1."""
            try:
                data = request.get_json()
                query = data.get('query', '')
                kwargs = data.get('kwargs', {})

                result = self.chat_with_r1(query, **kwargs)

                return jsonify({
                    'success': True,
                    'data': result
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/api/stats', methods=['GET'])
        def api_stats():
            """Endpoint para estatísticas do sistema."""
            try:
                stats = self.get_system_stats()
                return jsonify({
                    'success': True,
                    'data': stats
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/api/cache', methods=['GET'])
        def api_cache_info():
            """Endpoint para informações do cache."""
            try:
                key = request.args.get('key')
                if key:
                    info = self.get_cache_info(key)
                    return jsonify({
                        'success': True,
                        'data': info
                    })
                else:
                    cache_stats = self.cache_system.get_cache_stats() if self.cache_system else {}
                    return jsonify({
                        'success': True,
                        'data': cache_stats
                    })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/api/optimize', methods=['POST'])
        def api_optimize():
            """Endpoint para otimização do sistema."""
            try:
                self.optimize_system()
                return jsonify({
                    'success': True,
                    'message': 'Sistema otimizado com sucesso'
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/api/backup', methods=['POST'])
        def api_backup():
            """Endpoint para backup do sistema."""
            try:
                backup_type = request.get_json().get('type', 'full')
                backup_id = self.backup_system(backup_type)
                return jsonify({
                    'success': True,
                    'backup_id': backup_id
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/api/benchmark', methods=['POST'])
        def api_benchmark():
            """Endpoint para executar benchmark."""
            try:
                data = request.get_json()
                duration = data.get('duration', 30)
                results = self.run_benchmark(duration)
                return jsonify({
                    'success': True,
                    'data': results
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint de health check."""
            return jsonify({
                'status': 'healthy' if self.system_status == 'running' else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            })

        return app

    def run_api_server(self, port: int = 8080):
        """Executa servidor API."""

        app = self.create_api_server(port)

        logger.info(f"Iniciando servidor API na porta {port}")

        try:
            # Abrir navegador automaticamente
            webbrowser.open(f"http://localhost:{port}")
        except:
            pass

        app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    print("=== Sistema Completo de Cache R1 ===")
    print("\nPara usar o sistema completo:")
    print("\n# Inicializar e iniciar sistema")
    print("system = CompleteR1CacheSystem()")
    print("system.initialize_system()")
    print("system.start_system()")
    print("\n# Usar chat com cache")
    print("result = system.chat_with_r1('Explain quantum computing')")
    print("print(result['response'])")
    print("\n# Ver estatísticas")
    print("stats = system.get_system_stats()")
    print("print(json.dumps(stats, indent=2))")
    print("\n# Executar benchmark")
    print("benchmark_results = system.run_benchmark(duration=30)")
    print("\n# Iniciar servidor API")
    print("system.run_api_server(port=8080)")
    print("\n=== Sistema Pronto ===")