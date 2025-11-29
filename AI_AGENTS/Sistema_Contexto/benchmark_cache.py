#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Benchmark e Testes do Cache Avançado
Testes de performance, carga e análise comparativa

Este sistema implementa:
1. Benchmarks de performance
2. Testes de carga e estresse
3. Análise comparativa
4. Profiling de código
5. Relatórios detalhados
6. Testes de concorrência

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import time
import json
import threading
import statistics
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import concurrent.futures
import psutil
import cProfile
import pstats
import io
from functools import wraps
import numpy as np
import hashlib
import random

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Resultado de um benchmark."""
    name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    operations: int
    operations_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_rate: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LoadTestResult:
    """Resultado de teste de carga."""
    name: str
    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    percentiles: Dict[str, float]
    memory_usage: Dict[str, float]
    cpu_usage: Dict[str, float]
    error_details: List[str] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []

class BenchmarkSuite:
    """Suite de benchmarks para o sistema de cache."""

    def __init__(self, cache_system):
        self.cache_system = cache_system
        self.results_dir = Path("./benchmark_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = cProfile.Profile()

    def run_comprehensive_benchmark(self,
                                  test_duration: int = 60,
                                  concurrency: int = 4) -> Dict[str, Any]:
        """Executa benchmark abrangente."""

        logger.info("Iniciando benchmark abrangente...")
        start_time = datetime.now()

        results = {}

        # Benchmark de operações básicas
        results['basic_operations'] = self._benchmark_basic_operations(test_duration)

        # Benchmark de cache hits/misses
        results['cache_performance'] = self._benchmark_cache_performance(test_duration)

        # Benchmark de compressão
        results['compression_performance'] = self._benchmark_compression_performance()

        # Benchmark de deduplicação
        results['deduplication_performance'] = self._benchmark_deduplication_performance()

        # Benchmark de concorrência
        results['concurrency_performance'] = self._benchmark_concurrency(concurrency, test_duration)

        # Benchmark de carga
        results['load_test'] = self._run_load_test(concurrency, test_duration * 10)

        # Análise de memória
        results['memory_analysis'] = self._analyze_memory_usage()

        # Profiling de performance
        results['performance_profile'] = self._profile_performance()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Resumo final
        summary = self._generate_benchmark_summary(results, total_duration)

        # Salvar resultados
        self._save_benchmark_results(results, summary)

        logger.info(f"Benchmark concluído em {total_duration:.1f} segundos")
        return summary

    def _benchmark_basic_operations(self, duration: int = 60) -> BenchmarkResult:
        """Benchmark de operações básicas (set/get)."""

        logger.info("Executando benchmark de operações básicas...")
        start_time = datetime.now()

        operations = 0
        response_times = []
        errors = 0

        test_data = [
            f"test_data_{i}_" + "x" * (i % 1000)
            for i in range(1000)
        ]

        end_time = start_time + timedelta(seconds=duration)

        while datetime.now() < end_time:
            try:
                op_start = time.time()

                # Operação SET
                key = f"bench_key_{operations % 1000}"
                value = test_data[operations % len(test_data)]
                self.cache_system.set_cache(key, value)

                # Operação GET
                retrieved = self.cache_system.get_cache(key)

                op_end = time.time()
                response_times.append((op_end - op_start) * 1000)  # ms

                operations += 1

            except Exception as e:
                errors += 1
                logger.error(f"Erro na operação {operations}: {e}")

        total_time = (datetime.now() - start_time).total_seconds()

        # Calcular estatísticas
        success_rate = (operations - errors) / operations if operations > 0 else 0
        error_rate = errors / operations if operations > 0 else 0

        if response_times:
            avg_response = statistics.mean(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max_response
            p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max_response
        else:
            avg_response = min_response = max_response = p50 = p95 = p99 = 0

        # Uso de recursos
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        cpu_usage = process.cpu_percent()

        return BenchmarkResult(
            name="basic_operations",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=total_time,
            operations=operations,
            operations_per_second=operations / total_time if total_time > 0 else 0,
            avg_response_time=avg_response,
            min_response_time=min_response,
            max_response_time=max_response,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success_rate=success_rate,
            error_rate=error_rate
        )

    def _benchmark_cache_performance(self, duration: int = 60) -> BenchmarkResult:
        """Benchmark de performance de cache (hit rate)."""

        logger.info("Executando benchmark de performance de cache...")
        start_time = datetime.now()

        # Preparar dados de teste
        test_keys = [f"perf_key_{i}" for i in range(100)]
        for key in test_keys:
            self.cache_system.set_cache(key, f"performance_test_data_{key}")

        operations = 0
        hits = 0
        response_times = []

        end_time = start_time + timedelta(seconds=duration)

        while datetime.now() < end_time:
            try:
                op_start = time.time()

                # Escolher chave aleatoriamente (algumas existem, outras não)
                if random.random() < 0.8:  # 80% hit rate esperado
                    key = random.choice(test_keys)
                else:
                    key = f"nonexistent_key_{operations}"

                result = self.cache_system.get_cache(key)
                if result is not None:
                    hits += 1

                op_end = time.time()
                response_times.append((op_end - op_start) * 1000)

                operations += 1

            except Exception as e:
                logger.error(f"Erro no benchmark de cache: {e}")

        total_time = (datetime.now() - start_time).total_seconds()
        hit_rate = hits / operations if operations > 0 else 0

        return BenchmarkResult(
            name="cache_performance",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=total_time,
            operations=operations,
            operations_per_second=operations / total_time if total_time > 0 else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            success_rate=1.0,
            error_rate=0.0,
            metadata={"hit_rate": hit_rate, "hits": hits}
        )

    def _benchmark_compression_performance(self) -> BenchmarkResult:
        """Benchmark de performance de compressão."""

        logger.info("Executando benchmark de compressão...")
        start_time = datetime.now()

        # Dados de teste de diferentes tamanhos
        test_data = {
            "small": "x" * 100,
            "medium": "x" * 10000,
            "large": "x" * 100000,
            "text": "Este é um texto de teste para compressão. " * 1000
        }

        total_operations = 0
        response_times = []

        for size_name, data in test_data.items():
            for i in range(10):  # 10 iterações por tamanho
                try:
                    op_start = time.time()

                    key = f"compression_test_{size_name}_{i}"
                    self.cache_system.set_cache(key, data, use_compression=True)
                    retrieved = self.cache_system.get_cache(key)

                    op_end = time.time()
                    response_times.append((op_end - op_start) * 1000)

                    total_operations += 1

                except Exception as e:
                    logger.error(f"Erro no teste de compressão: {e}")

        total_time = (datetime.now() - start_time).total_seconds()

        return BenchmarkResult(
            name="compression_performance",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=total_time,
            operations=total_operations,
            operations_per_second=total_operations / total_time if total_time > 0 else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            success_rate=1.0,
            error_rate=0.0
        )

    def _benchmark_deduplication_performance(self) -> BenchmarkResult:
        """Benchmark de performance de deduplicação semântica."""

        logger.info("Executando benchmark de deduplicação...")
        start_time = datetime.now()

        # Dados similares para teste de deduplicação
        base_text = "Este é um texto de exemplo para testar deduplicação semântica."
        test_data = [
            base_text,
            base_text + " Adicionando mais conteúdo.",
            "Este é um texto similar para deduplicação.",
            base_text.upper(),
            base_text.replace("exemplo", "teste")
        ]

        total_operations = 0
        response_times = []

        for i, data in enumerate(test_data):
            for j in range(5):  # 5 iterações por dado
                try:
                    op_start = time.time()

                    key = f"dedup_test_{i}_{j}"
                    self.cache_system.set_cache(key, data)
                    retrieved = self.cache_system.get_cache(key)

                    op_end = time.time()
                    response_times.append((op_end - op_start) * 1000)

                    total_operations += 1

                except Exception as e:
                    logger.error(f"Erro no teste de deduplicação: {e}")

        total_time = (datetime.now() - start_time).total_seconds()

        return BenchmarkResult(
            name="deduplication_performance",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=total_time,
            operations=total_operations,
            operations_per_second=total_operations / total_time if total_time > 0 else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            success_rate=1.0,
            error_rate=0.0
        )

    def _benchmark_concurrency(self, concurrency: int, duration: int) -> BenchmarkResult:
        """Benchmark de concorrência."""

        logger.info(f"Executando benchmark de concorrência ({concurrency} threads)...")
        start_time = datetime.now()

        results = []
        errors = []

        def worker_thread(thread_id: int):
            """Thread worker para teste de concorrência."""
            thread_operations = 0
            thread_response_times = []
            thread_errors = 0

            end_time = start_time + timedelta(seconds=duration)

            while datetime.now() < end_time:
                try:
                    op_start = time.time()

                    key = f"concurrency_test_{thread_id}_{thread_operations}"
                    value = f"Thread {thread_id} data {thread_operations}"
                    self.cache_system.set_cache(key, value)
                    retrieved = self.cache_system.get_cache(key)

                    op_end = time.time()
                    thread_response_times.append((op_end - op_start) * 1000)

                    thread_operations += 1

                except Exception as e:
                    thread_errors += 1
                    errors.append(str(e))

            return {
                'operations': thread_operations,
                'response_times': thread_response_times,
                'errors': thread_errors
            }

        # Executar threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(concurrency)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Erro na thread: {e}")

        # Agregar resultados
        total_operations = sum(r['operations'] for r in results)
        all_response_times = []
        for r in results:
            all_response_times.extend(r['response_times'])

        total_time = (datetime.now() - start_time).total_seconds()

        return BenchmarkResult(
            name="concurrency_performance",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=total_time,
            operations=total_operations,
            operations_per_second=total_operations / total_time if total_time > 0 else 0,
            avg_response_time=statistics.mean(all_response_times) if all_response_times else 0,
            min_response_time=min(all_response_times) if all_response_times else 0,
            max_response_time=max(all_response_times) if all_response_times else 0,
            p50_response_time=statistics.median(all_response_times) if all_response_times else 0,
            p95_response_time=statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) > 20 else 0,
            p99_response_time=statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) > 100 else 0,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            success_rate=1.0,
            error_rate=0.0,
            metadata={"concurrency_level": concurrency}
        )

    def _run_load_test(self, concurrency: int, total_requests: int) -> LoadTestResult:
        """Executa teste de carga."""

        logger.info(f"Executando teste de carga ({concurrency} threads, {total_requests} requests)...")
        start_time = datetime.now()

        successful_requests = 0
        failed_requests = 0
        response_times = []
        error_details = []

        def load_worker(thread_id: int, requests_per_thread: int):
            """Worker para teste de carga."""
            thread_success = 0
            thread_failed = 0
            thread_response_times = []
            thread_errors = []

            for i in range(requests_per_thread):
                try:
                    op_start = time.time()

                    key = f"load_test_{thread_id}_{i}"
                    value = f"Load test data {thread_id}_{i}" + "x" * (i % 100)
                    self.cache_system.set_cache(key, value)
                    retrieved = self.cache_system.get_cache(key)

                    op_end = time.time()
                    thread_response_times.append((op_end - op_start) * 1000)

                    thread_success += 1

                except Exception as e:
                    thread_failed += 1
                    thread_errors.append(f"Thread {thread_id}, req {i}: {str(e)}")

            return {
                'success': thread_success,
                'failed': thread_failed,
                'response_times': thread_response_times,
                'errors': thread_errors
            }

        # Calcular requests por thread
        requests_per_thread = total_requests // concurrency

        # Executar teste de carga
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(load_worker, i, requests_per_thread)
                for i in range(concurrency)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    successful_requests += result['success']
                    failed_requests += result['failed']
                    response_times.extend(result['response_times'])
                    error_details.extend(result['errors'])
                except Exception as e:
                    logger.error(f"Erro no worker de carga: {e}")

        total_time = (datetime.now() - start_time).total_seconds()

        # Calcular percentis
        percentiles = {}
        if response_times:
            percentiles = {
                'p50': statistics.median(response_times),
                'p95': statistics.quantiles(response_times, n=20)[18],
                'p99': statistics.quantiles(response_times, n=100)[98]
            }

        return LoadTestResult(
            name="load_test",
            concurrency_level=concurrency,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            percentiles=percentiles,
            memory_usage={
                'avg': psutil.Process().memory_info().rss / 1024 / 1024,
                'peak': psutil.virtual_memory().used / 1024 / 1024
            },
            cpu_usage={
                'avg': psutil.Process().cpu_percent(),
                'peak': psutil.cpu_percent(interval=1)
            },
            error_details=error_details
        )

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Análise detalhada de uso de memória."""

        process = psutil.Process()

        return {
            'rss_mb': process.memory_info().rss / 1024 / 1024,
            'vms_mb': process.memory_info().vms / 1024 / 1024,
            'shared_mb': process.memory_info().shared / 1024 / 1024,
            'text_mb': process.memory_info().text / 1024 / 1024,
            'data_mb': process.memory_info().data / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'system_memory': {
                'total': psutil.virtual_memory().total / 1024 / 1024,
                'available': psutil.virtual_memory().available / 1024 / 1024,
                'used': psutil.virtual_memory().used / 1024 / 1024,
                'percent': psutil.virtual_memory().percent
            }
        }

    def _profile_performance(self) -> Dict[str, Any]:
        """Profiling de performance do código."""

        # Executar profiling
        self.profiler.enable()

        # Executar operações de teste
        for i in range(1000):
            key = f"profile_test_{i}"
            self.cache_system.set_cache(key, f"test_data_{i}")
            self.cache_system.get_cache(key)

        self.profiler.disable()

        # Analisar resultados
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)

        return {
            'profile_output': s.getvalue(),
            'total_calls': ps.total_calls,
            'primitive_calls': ps.prim_calls
        }

    def _generate_benchmark_summary(self, results: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        """Gera resumo dos benchmarks."""

        summary = {
            'benchmark_info': {
                'total_duration': total_duration,
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'python_version': os.sys.version,
                    'platform': os.sys.platform,
                    'cpu_count': os.cpu_count(),
                    'memory_total': psutil.virtual_memory().total
                }
            },
            'results': {}
        }

        for test_name, result in results.items():
            if isinstance(result, BenchmarkResult):
                summary['results'][test_name] = {
                    'operations_per_second': result.operations_per_second,
                    'avg_response_time': result.avg_response_time,
                    'success_rate': result.success_rate,
                    'memory_usage_mb': result.memory_usage_mb,
                    'metadata': result.metadata
                }
            elif isinstance(result, LoadTestResult):
                summary['results'][test_name] = {
                    'requests_per_second': result.requests_per_second,
                    'avg_response_time': result.avg_response_time,
                    'success_rate': result.successful_requests / result.total_requests,
                    'concurrency_level': result.concurrency_level
                }

        return summary

    def _save_benchmark_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Salva resultados dos benchmarks."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': {k: asdict(v) for k, v in results.items()}
            }, f, indent=2, default=str)

        logger.info(f"Resultados salvos em: {results_file}")

if __name__ == "__main__":
    print("=== Sistema de Benchmark de Cache ===")
    print("\nEste é um módulo de benchmarking.")
    print("Para usar, importe e inicialize com um sistema de cache:")
    print("\nfrom sistema_cache_avancado import AdvancedCacheSystem")
    print("from benchmark_cache import BenchmarkSuite")
    print("\n# Inicializar sistema")
    print("cache_system = AdvancedCacheSystem()")
    print("\n# Inicializar benchmark")
    print("benchmark = BenchmarkSuite(cache_system)")
    print("\n# Executar benchmark abrangente")
    print("results = benchmark.run_comprehensive_benchmark(duration=30)")
    print("\n# Ver resultados")
    print("print(json.dumps(results, indent=2))")
    print("\n=== Demo Concluída ===")