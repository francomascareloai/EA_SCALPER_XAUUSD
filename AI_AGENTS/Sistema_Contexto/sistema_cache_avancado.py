#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Cache Avançado com Deduplicação Semântica
Multi-Level Caching e Gerenciamento Inteligente de Cache

Este sistema implementa uma arquitetura de cache avançada com:
1. Multi-level caching (L1-L4)
2. Deduplicação semântica usando embeddings
3. Estratégias inteligentes de eviction (LRU, LFU, Adaptive)
4. Compressão automática com seleção de algoritmo
5. Monitoramento e analytics em tempo real
6. Integração com sistema R1

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import json
import hashlib
import pickle
import gzip
import lzma
import zlib
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import heapq
import psutil
import statistics
from enum import Enum

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Níveis de cache disponíveis."""
    L1_MEMORY = "l1_memory"        # Cache em RAM ultra-rápido
    L2_FAST_DISK = "l2_fast_disk"  # SSD cache persistente
    L3_SLOW_DISK = "l3_slow_disk"  # HDD cache archival
    L4_ARCHIVE = "l4_archive"      # Armazenamento comprimido

class EvictionStrategy(Enum):
    """Estratégias de remoção de cache."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    ADAPTIVE = "adaptive"          # Adaptativo baseado em padrões
    HYBRID = "hybrid"              # Combinação de LRU + LFU

class CompressionAlgorithm(Enum):
    """Algoritmos de compressão disponíveis."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"
    AUTO = "auto"                  # Seleção automática

@dataclass
class CacheEntry:
    """Entrada de cache com metadados avançados."""
    key: str
    value: Any
    level: CacheLevel
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    importance_score: float = 0.5
    embedding: Optional[np.ndarray] = None
    compressed: bool = False
    compression_algo: CompressionAlgorithm = CompressionAlgorithm.NONE
    original_size: int = 0
    compressed_size: int = 0
    semantic_hash: Optional[str] = None
    cluster_id: Optional[str] = None
    access_pattern: List[datetime] = None

    def __post_init__(self):
        if self.access_pattern is None:
            self.access_pattern = []
        self.original_size = len(pickle.dumps(self.value)) if self.value else 0

    def is_expired(self) -> bool:
        """Verifica se a entrada expirou."""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)

    def update_access(self):
        """Atualiza estatísticas de acesso."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.access_pattern.append(self.last_accessed)
        # Manter apenas os últimos 100 acessos
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]

class SemanticDeduplicator:
    """Sistema de deduplicação semântica usando embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.content_hashes: Dict[str, str] = {}
        self.semantic_clusters: Dict[str, List[str]] = defaultdict(list)

    def generate_semantic_hash(self, content: str) -> Tuple[str, np.ndarray]:
        """Gera hash semântico e embedding para conteúdo."""
        embedding = self.model.encode(content)
        # Normalizar embedding para usar como hash
        normalized = embedding / np.linalg.norm(embedding)
        semantic_hash = hashlib.md5(normalized.tobytes()).hexdigest()[:16]
        return semantic_hash, embedding

    def find_similar_content(self, content: str) -> List[Tuple[str, float]]:
        """Encontra conteúdo similar baseado em embeddings."""
        if not content.strip():
            return []

        semantic_hash, embedding = self.generate_semantic_hash(content)
        similar_items = []

        for existing_hash, existing_embedding in self.content_hashes.items():
            if existing_hash == semantic_hash:
                continue

            try:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    existing_embedding.reshape(1, -1)
                )[0][0]

                if similarity >= self.threshold:
                    similar_items.append((existing_hash, similarity))
            except:
                continue

        return sorted(similar_items, key=lambda x: x[1], reverse=True)

    def should_deduplicate(self, content: str) -> Tuple[bool, Optional[str]]:
        """Decide se deve deduplicar o conteúdo."""
        similar = self.find_similar_content(content)
        if similar:
            best_match = similar[0]
            if best_match[1] >= self.threshold:
                return True, best_match[0]
        return False, None

class CompressionManager:
    """Gerenciador de compressão com seleção automática de algoritmo."""

    def __init__(self):
        self.algorithms = {
            CompressionAlgorithm.GZIP: self._compress_gzip,
            CompressionAlgorithm.LZMA: self._compress_lzma,
            CompressionAlgorithm.ZLIB: self._compress_zlib
        }

    def _compress_gzip(self, data: bytes) -> bytes:
        """Compressão usando GZIP."""
        return gzip.compress(data)

    def _compress_lzma(self, data: bytes) -> bytes:
        """Compressão usando LZMA."""
        return lzma.compress(data)

    def _compress_zlib(self, data: bytes) -> bytes:
        """Compressão usando ZLIB."""
        return zlib.compress(data)

    def compress(self, data: Any, algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO) -> Tuple[bytes, CompressionAlgorithm]:
        """Comprime dados com algoritmo selecionado."""
        if algorithm == CompressionAlgorithm.NONE:
            return pickle.dumps(data), algorithm

        raw_data = pickle.dumps(data)

        if algorithm == CompressionAlgorithm.AUTO:
            # Testar todos os algoritmos e escolher o melhor
            results = {}
            for algo_name, compress_func in self.algorithms.items():
                try:
                    compressed = compress_func(raw_data)
                    ratio = len(raw_data) / len(compressed)
                    results[algo_name] = (compressed, ratio)
                except:
                    continue

            if results:
                best_algo = max(results.keys(), key=lambda x: results[x][1])
                return results[best_algo][0], best_algo

        if algorithm in self.algorithms:
            return self.algorithms[algorithm](raw_data), algorithm

        return raw_data, CompressionAlgorithm.NONE

    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> Any:
        """Descomprime dados."""
        if algorithm == CompressionAlgorithm.NONE:
            return pickle.loads(data)

        decompress_funcs = {
            CompressionAlgorithm.GZIP: lambda x: gzip.decompress(x),
            CompressionAlgorithm.LZMA: lambda x: lzma.decompress(x),
            CompressionAlgorithm.ZLIB: lambda x: zlib.decompress(x)
        }

        if algorithm in decompress_funcs:
            raw_data = decompress_funcs[algorithm](data)
            return pickle.loads(raw_data)

        return pickle.loads(data)

class CacheLevelManager:
    """Gerenciador de nível específico de cache."""

    def __init__(self, level: CacheLevel, max_size: int, strategy: EvictionStrategy):
        self.level = level
        self.max_size = max_size
        self.strategy = strategy
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_order: OrderedDict[str, datetime] = OrderedDict()
        self.frequency: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Recupera entrada do cache."""
        with self.lock:
            if key in self.entries:
                entry = self.entries[key]
                if not entry.is_expired():
                    entry.update_access()
                    self._update_access_patterns(key, entry)
                    return entry
                else:
                    del self.entries[key]
            return None

    def put(self, entry: CacheEntry) -> bool:
        """Adiciona entrada ao cache."""
        with self.lock:
            if len(self.entries) >= self.max_size:
                self._evict_entries()

            self.entries[entry.key] = entry
            self._update_access_patterns(entry.key, entry)
            return True

    def _update_access_patterns(self, key: str, entry: CacheEntry):
        """Atualiza padrões de acesso baseado na estratégia."""
        if self.strategy == EvictionStrategy.LRU:
            self.access_order[key] = entry.last_accessed
        elif self.strategy == EvictionStrategy.LFU:
            self.frequency[key] = entry.access_count

    def _evict_entries(self):
        """Remove entradas baseado na estratégia de eviction."""
        if self.strategy == EvictionStrategy.LRU:
            # Remover menos recentemente usado
            lru_key = min(self.access_order.keys(), key=lambda x: self.access_order[x])
            if lru_key in self.entries:
                del self.entries[lru_key]
                del self.access_order[lru_key]

        elif self.strategy == EvictionStrategy.LFU:
            # Remover menos frequentemente usado
            lfu_key = min(self.frequency.keys(), key=lambda x: self.frequency[x])
            if lfu_key in self.entries:
                del self.entries[lfu_key]
                del self.frequency[lfu_key]

        elif self.strategy == EvictionStrategy.ADAPTIVE:
            # Estratégia adaptativa baseada em múltiplos fatores
            scores = {}
            for key, entry in self.entries.items():
                time_score = (datetime.now() - entry.last_accessed).total_seconds()
                freq_score = 1.0 / (entry.access_count + 1)
                importance_score = 1.0 - entry.importance_score
                scores[key] = time_score * 0.4 + freq_score * 0.4 + importance_score * 0.2

            if scores:
                victim_key = max(scores.keys(), key=lambda x: scores[x])
                if victim_key in self.entries:
                    del self.entries[victim_key]

        # Remover entradas expiradas
        expired_keys = [k for k, v in self.entries.items() if v.is_expired()]
        for key in expired_keys:
            del self.entries[key]

class AdvancedCacheSystem:
    """Sistema de cache avançado com múltiplos níveis e deduplicação semântica."""

    def __init__(self,
                 cache_dir: str = "./cache/advanced",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_memory_mb: int = 512,
                 enable_semantic_dedup: bool = True):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.semantic_dedup = SemanticDeduplicator(embedding_model) if enable_semantic_dedup else None
        self.compression_manager = CompressionManager()

        # Configurar níveis de cache
        self.cache_levels = {
            CacheLevel.L1_MEMORY: CacheLevelManager(
                CacheLevel.L1_MEMORY,
                max_size=max_memory_mb * 1024 * 1024,  # bytes
                strategy=EvictionStrategy.LRU
            ),
            CacheLevel.L2_FAST_DISK: CacheLevelManager(
                CacheLevel.L2_FAST_DISK,
                max_size=10000,  # entradas
                strategy=EvictionStrategy.LFU
            )
        }

        # Estatísticas e monitoramento
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'deduplications': 0,
            'level_transfers': 0
        }

        # Cache de embeddings e hashes semânticos
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.semantic_cache: Dict[str, str] = {}

        # Lock para operações thread-safe
        self.lock = threading.RLock()

    def set_cache(self,
                  key: str,
                  value: Any,
                  ttl: Optional[int] = None,
                  importance: float = 0.5,
                  use_compression: bool = True,
                  force_level: Optional[CacheLevel] = None) -> str:
        """Adiciona item ao cache com deduplicação semântica."""

        with self.lock:
            # Verificar deduplicação semântica
            content_str = str(value)
            should_dedup = False
            duplicate_key = None

            if self.semantic_dedup and content_str:
                should_dedup, duplicate_key = self.semantic_dedup.should_deduplicate(content_str)
                if should_dedup:
                    self.stats['deduplications'] += 1
                    logger.info(f"Conteúdo duplicado detectado: {key} -> {duplicate_key}")
                    return duplicate_key

            # Gerar embedding e hash semântico
            semantic_hash = None
            embedding = None
            if self.semantic_dedup and content_str:
                semantic_hash, embedding = self.semantic_dedup.generate_semantic_hash(content_str)
                self.embedding_cache[semantic_hash] = embedding

            # Comprimir se necessário
            compressed_value = value
            compression_algo = CompressionAlgorithm.NONE
            compressed_size = 0

            if use_compression:
                compressed_data, compression_algo = self.compression_manager.compress(value)
                compressed_value = compressed_data
                compressed_size = len(compressed_data)
                if compression_algo != CompressionAlgorithm.NONE:
                    self.stats['compressions'] += 1

            # Criar entrada de cache
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                level=force_level or CacheLevel.L1_MEMORY,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl,
                importance_score=importance,
                embedding=embedding,
                compressed=compression_algo != CompressionAlgorithm.NONE,
                compression_algo=compression_algo,
                compressed_size=compressed_size,
                semantic_hash=semantic_hash
            )

            # Adicionar aos níveis de cache
            success = self.cache_levels[CacheLevel.L1_MEMORY].put(entry)

            if success and CacheLevel.L2_FAST_DISK in self.cache_levels:
                # Também salvar no disco para persistência
                l2_entry = CacheEntry(**asdict(entry))
                l2_entry.level = CacheLevel.L2_FAST_DISK
                self.cache_levels[CacheLevel.L2_FAST_DISK].put(l2_entry)
                self._save_to_disk(l2_entry)

            # Atualizar cache semântico
            if semantic_hash:
                self.semantic_cache[semantic_hash] = key

            logger.info(f"Cache SET: {key} (compressão: {compression_algo.value})")
            return key

    def get_cache(self, key: str, include_expired: bool = False) -> Optional[Any]:
        """Recupera item do cache com busca multi-nível."""

        with self.lock:
            # Buscar nos níveis de cache
            for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_FAST_DISK]:
                if level in self.cache_levels:
                    entry = self.cache_levels[level].get(key)
                    if entry:
                        self.stats['hits'] += 1

                        # Descomprimir se necessário
                        if entry.compressed:
                            value = self.compression_manager.decompress(
                                entry.value, entry.compression_algo
                            )
                        else:
                            value = entry.value

                        # Promover para L1 se veio de L2
                        if level == CacheLevel.L2_FAST_DISK:
                            self._promote_to_l1(entry)
                            self.stats['level_transfers'] += 1

                        logger.info(f"Cache HIT: {key} (nível: {level.value})")
                        return value

            # Se não encontrou, tentar buscar do disco
            entry = self._load_from_disk(key)
            if entry:
                self.stats['hits'] += 1
                value = self.compression_manager.decompress(
                    entry.value, entry.compression_algo
                ) if entry.compressed else entry.value

                # Adicionar de volta ao L1
                self._promote_to_l1(entry)
                logger.info(f"Cache HIT: {key} (disco)")
                return value

            self.stats['misses'] += 1
            logger.info(f"Cache MISS: {key}")
            return None

    def _promote_to_l1(self, entry: CacheEntry):
        """Promove entrada para cache L1."""
        l1_entry = CacheEntry(**asdict(entry))
        l1_entry.level = CacheLevel.L1_MEMORY
        self.cache_levels[CacheLevel.L1_MEMORY].put(l1_entry)

    def _save_to_disk(self, entry: CacheEntry):
        """Salva entrada no disco."""
        try:
            disk_path = self.cache_dir / f"{entry.key}.cache"
            with open(disk_path, 'wb') as f:
                pickle.dump(asdict(entry), f)
        except Exception as e:
            logger.error(f"Erro ao salvar no disco: {e}")

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Carrega entrada do disco."""
        try:
            disk_path = self.cache_dir / f"{key}.cache"
            if disk_path.exists():
                with open(disk_path, 'rb') as f:
                    data = pickle.load(f)
                    return CacheEntry(**data)
        except Exception as e:
            logger.error(f"Erro ao carregar do disco: {e}")
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas do cache."""

        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            # Estatísticas por nível
            level_stats = {}
            for level, manager in self.cache_levels.items():
                with manager.lock:
                    level_stats[level.value] = {
                        'entries': len(manager.entries),
                        'max_size': manager.max_size,
                        'utilization': len(manager.entries) / manager.max_size if manager.max_size > 0 else 0
                    }

            # Estatísticas de compressão
            total_compressed = sum(1 for level in self.cache_levels.values()
                                 for entry in level.entries.values() if entry.compressed)

            return {
                'performance': {
                    'total_requests': total_requests,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'hit_rate': hit_rate,
                    'evictions': self.stats['evictions'],
                    'level_transfers': self.stats['level_transfers']
                },
                'storage': {
                    'total_entries': sum(len(m.entries) for m in self.cache_levels.values()),
                    'compressed_entries': total_compressed,
                    'compression_ratio': self._calculate_compression_ratio(),
                    'semantic_deduplication_rate': self.stats['deduplications'] / max(self.stats['hits'] + self.stats['misses'], 1)
                },
                'levels': level_stats,
                'memory_usage': self._get_memory_usage()
            }

    def _calculate_compression_ratio(self) -> float:
        """Calcula taxa de compressão média."""
        ratios = []
        for level_manager in self.cache_levels.values():
            with level_manager.lock:
                for entry in level_manager.entries.values():
                    if entry.compressed and entry.original_size > 0:
                        ratio = entry.original_size / entry.compressed_size
                        ratios.append(ratio)

        return statistics.mean(ratios) if ratios else 1.0

    def _get_memory_usage(self) -> Dict[str, float]:
        """Retorna uso de memória."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

    def cleanup_cache(self, max_age_days: int = 30, min_access_count: int = 1):
        """Limpa cache removendo entradas antigas e pouco acessadas."""

        with self.lock:
            total_cleaned = 0

            for level_manager in self.cache_levels.values():
                with level_manager.lock:
                    to_remove = []

                    for key, entry in level_manager.entries.items():
                        age_days = (datetime.now() - entry.created_at).days

                        if (age_days > max_age_days and entry.access_count < min_access_count) or entry.is_expired():
                            to_remove.append(key)

                    for key in to_remove:
                        del level_manager.entries[key]
                        total_cleaned += 1
                        self.stats['evictions'] += 1

            logger.info(f"Cache cleanup: {total_cleaned} entradas removidas")
            return total_cleaned

    def optimize_cache(self):
        """Otimiza cache baseado em padrões de uso."""

        with self.lock:
            # Análise de padrões de acesso
            access_patterns = defaultdict(list)

            for level_manager in self.cache_levels.values():
                with level_manager.lock:
                    for entry in level_manager.entries.values():
                        if len(entry.access_pattern) >= 2:
                            intervals = []
                            for i in range(1, len(entry.access_pattern)):
                                interval = (entry.access_pattern[i] - entry.access_pattern[i-1]).total_seconds()
                                intervals.append(interval)
                            if intervals:
                                access_patterns[entry.key] = intervals

            # Identificar entradas com padrões previsíveis
            predictable_entries = []
            for key, intervals in access_patterns.items():
                if len(intervals) >= 3:
                    # Calcular variância dos intervalos
                    variance = statistics.variance(intervals) if len(intervals) > 1 else 0
                    if variance < 3600:  # Menos de 1 hora de variação
                        predictable_entries.append(key)

            logger.info(f"Cache otimizado: {len(predictable_entries)} entradas com padrões previsíveis")
            return len(predictable_entries)

if __name__ == "__main__":
    # Demo do sistema avançado
    print("=== Sistema de Cache Avançado ===")

    cache_system = AdvancedCacheSystem(
        max_memory_mb=256,
        enable_semantic_dedup=True
    )

    # Teste básico
    print("\n1. Teste de cache básico:")
    cache_system.set_cache("teste1", "Este é um valor de teste", ttl=3600)
    result = cache_system.get_cache("teste1")
    print(f"   Resultado: {result}")

    # Teste de deduplicação semântica
    print("\n2. Teste de deduplicação semântica:")
    cache_system.set_cache("teste2", "Este é um valor de teste muito similar", ttl=3600)
    cache_system.set_cache("teste3", "Este é um valor de teste muito similar ao anterior", ttl=3600)

    # Estatísticas
    print("\n3. Estatísticas do sistema:")
    stats = cache_system.get_cache_stats()
    print(f"   Hit Rate: {stats['performance']['hit_rate']:.1%}")
    print(f"   Entradas totais: {stats['storage']['total_entries']}")
    print(f"   Taxa de deduplicação: {stats['storage']['semantic_deduplication_rate']:.1%}")

    print("\n=== Demo Concluída ===")