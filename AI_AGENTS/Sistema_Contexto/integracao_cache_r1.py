#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Integração do Cache Avançado com R1
Integração completa entre sistema de cache e R1

Este sistema implementa:
1. Integração transparente com R1
2. Cache inteligente de prompts
3. Otimização de contexto
4. Backup e recovery integrado
5. Monitoramento em tempo real

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
import hashlib
import numpy as np

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class R1CacheIntegration:
    """Classe de integração entre sistema de cache avançado e R1."""

    def __init__(self, advanced_cache_system, r1_context_manager):
        self.cache_system = advanced_cache_system
        self.r1_manager = r1_context_manager
        self.cache_hits = 0
        self.cache_misses = 0
        self.prompt_cache_stats = {}
        self.context_optimization_stats = {}

        # Configurações de integração
        self.enable_prompt_caching = True
        self.enable_context_optimization = True
        self.enable_semantic_deduplication = True
        self.max_context_tokens = 163000
        self.target_expansion_factor = 10

        logger.info("Sistema de integração R1-Cache inicializado")

    def chat_with_r1_cached(self,
                          query: str,
                          system_prompt: Optional[str] = None,
                          use_cache: bool = True,
                          use_context: bool = True,
                          **kwargs) -> Dict[str, Any]:
        """Chat com R1 usando cache inteligente."""

        start_time = time.time()

        # Gerar hash da consulta para cache
        query_hash = self._generate_query_hash(query, system_prompt, kwargs)

        result = {
            'response': None,
            'cached': False,
            'context_expanded': False,
            'processing_time': 0,
            'cache_info': {},
            'performance_metrics': {}
        }

        # Verificar cache de prompts
        if use_cache and self.enable_prompt_caching:
            cached_response = self.cache_system.get_cache(f"prompt_{query_hash}")
            if cached_response:
                result['response'] = cached_response
                result['cached'] = True
                result['cache_info'] = {'source': 'prompt_cache', 'hash': query_hash}
                self.cache_hits += 1

                end_time = time.time()
                result['processing_time'] = end_time - start_time
                return result

        self.cache_misses += 1

        # Preparar contexto expandido se necessário
        expanded_context = ""
        if use_context and self.enable_context_optimization:
            try:
                expanded_context = self._prepare_expanded_context(query)
                result['context_expanded'] = len(expanded_context) > 0
            except Exception as e:
                logger.warning(f"Erro ao expandir contexto: {e}")

        # Executar chat com R1
        try:
            if expanded_context:
                # Combinar contexto expandido com query
                enhanced_query = f"""
{expanded_context}

---
Pergunta atual: {query}
""".strip()

                response = self.r1_manager.chat_with_expanded_context(
                    enhanced_query,
                    system_prompt=system_prompt,
                    **kwargs
                )
            else:
                response = self.r1_manager.chat_with_expanded_context(
                    query,
                    system_prompt=system_prompt,
                    **kwargs
                )

            result['response'] = response

            # Cache da resposta se apropriado
            if self.enable_prompt_caching and len(response) > 10:
                cache_key = f"prompt_{query_hash}"

                # Metadados para cache inteligente
                cache_metadata = {
                    'query_hash': query_hash,
                    'query_length': len(query),
                    'response_length': len(response),
                    'context_expanded': result['context_expanded'],
                    'system_prompt': bool(system_prompt),
                    'timestamp': datetime.now().isoformat()
                }

                self.cache_system.set_cache(
                    cache_key,
                    response,
                    ttl=kwargs.get('ttl', 3600),  # 1 hora padrão
                    importance=self._calculate_response_importance(query, response),
                    use_compression=True,
                    metadata=cache_metadata
                )

        except Exception as e:
            logger.error(f"Erro no chat R1: {e}")
            result['response'] = f"Erro: {str(e)}"

        end_time = time.time()
        result['processing_time'] = end_time - start_time

        # Métricas de performance
        result['performance_metrics'] = {
            'cache_hit_rate': self._calculate_hit_rate(),
            'context_expansion_ratio': len(expanded_context) / len(query) if expanded_context else 1.0,
            'processing_time_ms': result['processing_time'] * 1000
        }

        return result

    def _generate_query_hash(self, query: str, system_prompt: Optional[str], kwargs: Dict[str, Any]) -> str:
        """Gera hash único para a consulta."""
        content = f"{query}|{system_prompt or ''}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _prepare_expanded_context(self, query: str) -> str:
        """Prepara contexto expandido usando o sistema de cache."""

        # Buscar contexto relevante no cache
        relevant_chunks = self.cache_system.search_relevant_context(
            query, max_chunks=20, min_similarity=0.3
        )

        if not relevant_chunks:
            return ""

        # Construir contexto expandido
        context_parts = []
        current_tokens = 0
        target_tokens = int(self.max_context_tokens * 0.7)  # 70% do limite

        for chunk in relevant_chunks:
            if current_tokens + chunk.token_count <= target_tokens:
                context_parts.append(f"[CONTEXTO] {chunk.content}")
                current_tokens += chunk.token_count
            else:
                break

        return "\n\n".join(context_parts)

    def _calculate_response_importance(self, query: str, response: str) -> float:
        """Calcula importância da resposta para cache."""

        factors = {
            'query_length': min(len(query) / 1000, 1.0),
            'response_length': min(len(response) / 2000, 1.0),
            'response_complexity': min(response.count(' ') / 1000, 1.0),
            'has_code': 0.3 if '```' in response or 'def ' in response else 0.0,
            'has_list': 0.2 if any(char in response for char in ['-', '*', '1.', '2.']) else 0.0
        }

        return sum(factors.values()) / len(factors)

    def _calculate_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_integration_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da integração."""

        cache_stats = self.cache_system.get_cache_stats()

        return {
            'cache_performance': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self._calculate_hit_rate(),
                'total_requests': self.cache_hits + self.cache_misses
            },
            'r1_performance': {
                'context_expansions': self.context_optimization_stats.get('expansions', 0),
                'avg_expansion_time': self.context_optimization_stats.get('avg_time', 0)
            },
            'system_stats': cache_stats
        }

    def optimize_cache_for_r1(self):
        """Otimiza cache especificamente para uso com R1."""

        logger.info("Otimizando cache para R1...")

        # Limpar cache antigo de prompts
        self.cache_system.cleanup_cache(max_age_days=7)

        # Otimizar configurações para R1
        optimization_updates = {
            'enable_semantic_dedup': True,
            'semantic_similarity_threshold': 0.8,
            'prefetch_enabled': True,
            'predictive_caching': True,
            'max_embedding_cache_size': 20000
        }

        # Aplicar otimizações
        for key, value in optimization_updates.items():
            if hasattr(self.cache_system, key):
                setattr(self.cache_system, key, value)

        # Executar otimização do cache
        self.cache_system.optimize_cache()

        logger.info("Otimização concluída")

    def preload_common_contexts(self, contexts: List[str]):
        """Pre-carrega contextos comuns no cache."""

        logger.info(f"Pre-carregando {len(contexts)} contextos...")

        for context in contexts:
            try:
                self.cache_system.add_context(context)
            except Exception as e:
                logger.warning(f"Erro ao pre-carregar contexto: {e}")

        logger.info("Pre-carregamento concluído")

    def export_r1_cache_data(self, export_path: str):
        """Exporta dados de cache específicos do R1."""

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'integration_stats': self.get_integration_stats(),
            'prompt_cache_entries': {},
            'context_cache_entries': {}
        }

        # Coletar entradas de cache de prompts
        try:
            for level_manager in self.cache_system.cache_levels.values():
                with level_manager.lock:
                    for key, entry in level_manager.entries.items():
                        if key.startswith('prompt_'):
                            export_data['prompt_cache_entries'][key] = {
                                'content': entry.value if isinstance(entry.value, str) else str(entry.value),
                                'created_at': entry.created_at.isoformat(),
                                'access_count': entry.access_count,
                                'importance': entry.importance_score
                            }
                        elif key.startswith('context_'):
                            export_data['context_cache_entries'][key] = {
                                'content': entry.value if isinstance(entry.value, str) else str(entry.value),
                                'created_at': entry.created_at.isoformat(),
                                'access_count': entry.access_count,
                                'importance': entry.importance_score
                            }
        except Exception as e:
            logger.error(f"Erro ao exportar dados de cache: {e}")

        # Salvar arquivo de exportação
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Dados de cache exportados para: {export_path}")

if __name__ == "__main__":
    print("=== Sistema de Integração R1-Cache ===")
    print("\nEste é um módulo de integração.")
    print("Para usar, importe e inicialize com sistemas de cache e R1:")
    print("\nfrom sistema_cache_avancado import AdvancedCacheSystem")
    print("from sistema_contexto_expandido_2m import ContextManager")
    print("from integracao_cache_r1 import R1CacheIntegration")
    print("\n# Inicializar sistemas")
    print("cache_system = AdvancedCacheSystem()")
    print("r1_manager = ContextManager()")
    print("\n# Inicializar integração")
    print("integration = R1CacheIntegration(cache_system, r1_manager)")
    print("\n# Usar chat com cache")
    print("result = integration.chat_with_r1_cached('Explain quantum computing')")
    print("print(result['response'])")
    print("\n=== Demo Concluída ===")