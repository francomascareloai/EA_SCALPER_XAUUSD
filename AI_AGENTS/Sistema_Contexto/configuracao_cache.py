#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Configuração e Gerenciamento do Cache Avançado
Configurações dinâmicas, perfis pré-definidos e otimização automática

Este sistema implementa:
1. Perfis de configuração pré-definidos
2. Configuração dinâmica em tempo de execução
3. Otimização automática de parâmetros
4. Validação de configurações
5. Backup e restauração de configurações
6. Interface de configuração via API

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import json
import yaml
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum
import statistics
from concurrent.futures import ThreadPoolExecutor

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigProfile(Enum):
    """Perfis de configuração pré-definidos."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    MEMORY_OPTIMIZED = "memory_optimized"
    STORAGE_OPTIMIZED = "storage_optimized"
    ENTERPRISE = "enterprise"

@dataclass
class CacheConfiguration:
    """Configuração completa do sistema de cache."""

    # Configurações básicas
    cache_dir: str = "./cache/advanced"
    max_memory_mb: int = 512
    enable_semantic_dedup: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

    # Configurações de níveis
    l1_memory_max_entries: int = 10000
    l2_fast_disk_max_entries: int = 50000
    l3_slow_disk_max_entries: int = 200000
    l4_archive_max_entries: int = 1000000

    # Estratégias de eviction
    l1_eviction_strategy: str = "lru"
    l2_eviction_strategy: str = "lfu"
    l3_eviction_strategy: str = "adaptive"

    # Configurações de compressão
    enable_compression: bool = True
    default_compression: str = "auto"
    compression_threshold_bytes: int = 1024

    # Configurações de TTL
    default_ttl: Optional[int] = None
    max_ttl: int = 86400  # 24 horas
    cleanup_interval: int = 3600  # 1 hora

    # Configurações de backup
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 horas
    backup_retention_days: int = 30
    backup_compression: str = "gzip"

    # Configurações de monitoramento
    monitoring_enabled: bool = True
    metrics_collection_interval: int = 5
    dashboard_port: int = 5000
    alert_enabled: bool = True

    # Configurações avançadas
    semantic_similarity_threshold: float = 0.85
    max_embedding_cache_size: int = 10000
    prefetch_enabled: bool = True
    predictive_caching: bool = True
    distributed_sync_enabled: bool = False

    # Configurações de otimização
    auto_optimize_enabled: bool = True
    optimization_interval: int = 3600
    performance_target_hit_rate: float = 0.8
    memory_target_percent: float = 75.0

    # Metadados
    profile: ConfigProfile = ConfigProfile.PRODUCTION
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "2.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ConfigurationManager:
    """Gerenciador de configurações do sistema de cache."""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[CacheConfiguration] = None
        self.config_history: List[CacheConfiguration] = []
        self.config_lock = threading.RLock()

        # Perfis pré-definidos
        self.predefined_profiles = self._create_predefined_profiles()

        # Carregar configuração existente
        self.load_configuration()

    def _create_predefined_profiles(self) -> Dict[ConfigProfile, CacheConfiguration]:
        """Cria perfis de configuração pré-definidos."""

        profiles = {}

        # Development Profile
        profiles[ConfigProfile.DEVELOPMENT] = CacheConfiguration(
            cache_dir="./cache/dev",
            max_memory_mb=256,
            l1_memory_max_entries=5000,
            enable_semantic_dedup=True,
            monitoring_enabled=True,
            dashboard_port=5001,
            profile=ConfigProfile.DEVELOPMENT,
            metadata={"description": "Perfil de desenvolvimento com configurações relaxadas"}
        )

        # Production Profile
        profiles[ConfigProfile.PRODUCTION] = CacheConfiguration(
            cache_dir="./cache/production",
            max_memory_mb=2048,
            l1_memory_max_entries=20000,
            l2_fast_disk_max_entries=100000,
            backup_enabled=True,
            monitoring_enabled=True,
            alert_enabled=True,
            auto_optimize_enabled=True,
            profile=ConfigProfile.PRODUCTION,
            metadata={"description": "Perfil de produção com alta disponibilidade"}
        )

        # High Performance Profile
        profiles[ConfigProfile.HIGH_PERFORMANCE] = CacheConfiguration(
            cache_dir="./cache/high_perf",
            max_memory_mb=4096,
            l1_memory_max_entries=50000,
            l2_fast_disk_max_entries=200000,
            enable_compression=False,
            semantic_similarity_threshold=0.9,
            prefetch_enabled=True,
            predictive_caching=True,
            profile=ConfigProfile.HIGH_PERFORMANCE,
            metadata={"description": "Perfil de alta performance com foco em velocidade"}
        )

        # Memory Optimized Profile
        profiles[ConfigProfile.MEMORY_OPTIMIZED] = CacheConfiguration(
            cache_dir="./cache/memory_opt",
            max_memory_mb=128,
            l1_memory_max_entries=2000,
            enable_semantic_dedup=False,
            enable_compression=True,
            default_compression="lzma",
            compression_threshold_bytes=512,
            profile=ConfigProfile.MEMORY_OPTIMIZED,
            metadata={"description": "Perfil otimizado para uso mínimo de memória"}
        )

        # Storage Optimized Profile
        profiles[ConfigProfile.STORAGE_OPTIMIZED] = CacheConfiguration(
            cache_dir="./cache/storage_opt",
            max_memory_mb=1024,
            l3_slow_disk_max_entries=500000,
            l4_archive_max_entries=2000000,
            enable_compression=True,
            backup_compression="lzma",
            cleanup_interval=1800,
            profile=ConfigProfile.STORAGE_OPTIMIZED,
            metadata={"description": "Perfil otimizado para armazenamento em disco"}
        )

        # Enterprise Profile
        profiles[ConfigProfile.ENTERPRISE] = CacheConfiguration(
            cache_dir="./cache/enterprise",
            max_memory_mb=8192,
            l1_memory_max_entries=100000,
            l2_fast_disk_max_entries=500000,
            l3_slow_disk_max_entries=1000000,
            l4_archive_max_entries=5000000,
            backup_enabled=True,
            backup_interval=21600,  # 6 horas
            backup_retention_days=90,
            monitoring_enabled=True,
            distributed_sync_enabled=True,
            auto_optimize_enabled=True,
            optimization_interval=1800,
            profile=ConfigProfile.ENTERPRISE,
            metadata={"description": "Perfil enterprise com recursos completos"}
        )

        return profiles

    def load_profile(self, profile: ConfigProfile) -> CacheConfiguration:
        """Carrega um perfil de configuração pré-definido."""
        with self.config_lock:
            if profile in self.predefined_profiles:
                self.current_config = self.predefined_profiles[profile].__class__(
                    **asdict(self.predefined_profiles[profile])
                )
                self.current_config.updated_at = datetime.now()
                logger.info(f"Perfil carregado: {profile.value}")
                return self.current_config
            else:
                raise ValueError(f"Perfil não encontrado: {profile}")

    def create_custom_profile(self, name: str, base_profile: ConfigProfile,
                            overrides: Dict[str, Any]) -> CacheConfiguration:
        """Cria um perfil customizado baseado em um perfil existente."""
        with self.config_lock:
            if base_profile not in self.predefined_profiles:
                raise ValueError(f"Perfil base não encontrado: {base_profile}")

            # Criar cópia do perfil base
            base_config = self.predefined_profiles[base_profile]
            custom_config = CacheConfiguration(**asdict(base_config))

            # Aplicar overrides
            for key, value in overrides.items():
                if hasattr(custom_config, key):
                    setattr(custom_config, key, value)
                else:
                    logger.warning(f"Propriedade desconhecida ignorada: {key}")

            custom_config.profile = ConfigProfile(name)
            custom_config.updated_at = datetime.now()
            custom_config.metadata["custom"] = True
            custom_config.metadata["base_profile"] = base_profile.value

            # Adicionar aos perfis disponíveis
            self.predefined_profiles[ConfigProfile(name)] = custom_config

            logger.info(f"Perfil customizado criado: {name}")
            return custom_config

    def update_configuration(self, updates: Dict[str, Any]) -> CacheConfiguration:
        """Atualiza configuração atual com novos valores."""
        with self.config_lock:
            if not self.current_config:
                raise ValueError("Nenhuma configuração carregada")

            # Salvar estado anterior no histórico
            self.config_history.append(CacheConfiguration(**asdict(self.current_config)))

            # Aplicar atualizações
            for key, value in updates.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                else:
                    logger.warning(f"Propriedade desconhecida ignorada: {key}")

            self.current_config.updated_at = datetime.now()

            # Salvar configuração
            self.save_configuration()

            logger.info(f"Configuração atualizada: {len(updates)} propriedades")
            return self.current_config

    def validate_configuration(self, config: Optional[CacheConfiguration] = None) -> Tuple[bool, List[str]]:
        """Valida uma configuração."""
        if config is None:
            config = self.current_config

        if not config:
            return False, ["Nenhuma configuração para validar"]

        errors = []

        # Validações básicas
        if config.max_memory_mb <= 0:
            errors.append("max_memory_mb deve ser maior que 0")

        if config.l1_memory_max_entries <= 0:
            errors.append("l1_memory_max_entries deve ser maior que 0")

        if not (0 <= config.semantic_similarity_threshold <= 1):
            errors.append("semantic_similarity_threshold deve estar entre 0 e 1")

        if config.compression_threshold_bytes < 0:
            errors.append("compression_threshold_bytes deve ser >= 0")

        if config.max_ttl <= 0:
            errors.append("max_ttl deve ser maior que 0")

        # Validações de caminhos
        try:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Diretório de cache inválido: {e}")

        # Validações de estratégias
        valid_strategies = ["lru", "lfu", "adaptive", "hybrid"]
        if config.l1_eviction_strategy not in valid_strategies:
            errors.append(f"l1_eviction_strategy deve ser uma das: {valid_strategies}")

        # Validações de compressão
        valid_compressions = ["none", "gzip", "lzma", "zlib", "auto"]
        if config.default_compression not in valid_compressions:
            errors.append(f"default_compression deve ser uma das: {valid_compressions}")

        return len(errors) == 0, errors

    def optimize_configuration(self, cache_system) -> CacheConfiguration:
        """Otimiza configuração baseado no uso atual."""
        with self.config_lock:
            if not self.current_config or not cache_system:
                return self.current_config

            logger.info("Iniciando otimização de configuração...")

            # Obter estatísticas atuais
            stats = cache_system.get_cache_stats()
            performance = stats.get('performance', {})
            storage = stats.get('storage', {})

            updates = {}

            # Otimizar hit rate
            current_hit_rate = performance.get('hit_rate', 0)
            if current_hit_rate < self.current_config.performance_target_hit_rate:
                # Aumentar cache de memória se possível
                if self.current_config.max_memory_mb < 4096:
                    updates['max_memory_mb'] = min(self.current_config.max_memory_mb * 1.5, 4096)

                # Ajustar estratégia de eviction
                if current_hit_rate < 0.5:
                    updates['l1_eviction_strategy'] = 'adaptive'

            # Otimizar uso de memória
            memory_usage = stats.get('memory_usage', {}).get('percent', 0)
            if memory_usage > self.current_config.memory_target_percent:
                updates['enable_compression'] = True
                updates['default_compression'] = 'auto'

            # Otimizar compressão
            compression_ratio = storage.get('compression_ratio', 1.0)
            if compression_ratio < 1.5 and self.current_config.enable_compression:
                # Experimentar algoritmos de compressão melhores
                if self.current_config.default_compression == 'gzip':
                    updates['default_compression'] = 'lzma'

            # Otimizar deduplicação
            deduplication_rate = storage.get('semantic_deduplication_rate', 0)
            if deduplication_rate > 0.1:  # Se há deduplicação significativa
                updates['semantic_similarity_threshold'] = min(
                    self.current_config.semantic_similarity_threshold + 0.05, 0.95
                )

            # Aplicar otimizações
            if updates:
                self.update_configuration(updates)
                logger.info(f"Configuração otimizada: {updates}")
            else:
                logger.info("Nenhuma otimização necessária")

            return self.current_config

    def save_configuration(self, filename: Optional[str] = None) -> str:
        """Salva configuração atual em arquivo."""
        with self.config_lock:
            if not self.current_config:
                raise ValueError("Nenhuma configuração para salvar")

            if not filename:
                filename = f"config_{self.current_config.profile.value}_{int(datetime.now().timestamp())}.json"

            config_path = self.config_dir / filename

            # Preparar dados para salvamento
            config_data = asdict(self.current_config)
            config_data['profile'] = self.current_config.profile.value
            config_data['created_at'] = self.current_config.created_at.isoformat()
            config_data['updated_at'] = self.current_config.updated_at.isoformat()

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuração salva: {config_path}")
            return str(config_path)

    def load_configuration(self, filename: Optional[str] = None) -> CacheConfiguration:
        """Carrega configuração de arquivo."""
        with self.config_lock:
            if not filename:
                # Procurar arquivo de configuração mais recente
                config_files = list(self.config_dir.glob("config_*.json"))
                if config_files:
                    filename = str(max(config_files, key=lambda x: x.stat().st_mtime))

            if not filename or not Path(filename).exists():
                # Carregar perfil de produção como padrão
                return self.load_profile(ConfigProfile.PRODUCTION)

            config_path = Path(filename)

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Reconstruir configuração
                config_data['profile'] = ConfigProfile(config_data['profile'])
                config_data['created_at'] = datetime.fromisoformat(config_data['created_at'])
                config_data['updated_at'] = datetime.fromisoformat(config_data['updated_at'])

                self.current_config = CacheConfiguration(**config_data)
                logger.info(f"Configuração carregada: {config_path}")

                return self.current_config

            except Exception as e:
                logger.error(f"Erro ao carregar configuração: {e}")
                return self.load_profile(ConfigProfile.PRODUCTION)

    def export_configuration(self, format: str = "json") -> str:
        """Exporta configuração atual."""
        with self.config_lock:
            if not self.current_config:
                raise ValueError("Nenhuma configuração para exportar")

            config_data = asdict(self.current_config)
            config_data['profile'] = self.current_config.profile.value
            config_data['exported_at'] = datetime.now().isoformat()

            if format == "json":
                return json.dumps(config_data, indent=2, ensure_ascii=False)
            elif format == "yaml":
                return yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Formato não suportado: {format}")

    def import_configuration(self, config_data: str, format: str = "json") -> CacheConfiguration:
        """Importa configuração."""
        with self.config_lock:
            if format == "json":
                data = json.loads(config_data)
            elif format == "yaml":
                data = yaml.safe_load(config_data)
            else:
                raise ValueError(f"Formato não suportado: {format}")

            # Salvar configuração anterior
            if self.current_config:
                self.config_history.append(CacheConfiguration(**asdict(self.current_config)))

            # Criar nova configuração
            data['profile'] = ConfigProfile(data['profile'])
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.now()

            self.current_config = CacheConfiguration(**data)

            # Salvar configuração
            self.save_configuration()

            logger.info("Configuração importada com sucesso")
            return self.current_config

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Retorna resumo da configuração atual."""
        with self.config_lock:
            if not self.current_config:
                return {}

            return {
                'profile': self.current_config.profile.value,
                'version': self.current_config.version,
                'memory_mb': self.current_config.max_memory_mb,
                'cache_dir': self.current_config.cache_dir,
                'compression': self.current_config.enable_compression,
                'deduplication': self.current_config.enable_semantic_dedup,
                'monitoring': self.current_config.monitoring_enabled,
                'backup': self.current_config.backup_enabled,
                'optimization': self.current_config.auto_optimize_enabled,
                'last_updated': self.current_config.updated_at.isoformat(),
                'metadata': self.current_config.metadata
            }

if __name__ == "__main__":
    print("=== Sistema de Configuração de Cache ===")

    # Demo do sistema de configuração
    config_manager = ConfigurationManager()

    print("\n1. Perfis disponíveis:")
    for profile in ConfigProfile:
        description = config_manager.predefined_profiles[profile].metadata.get('description', 'Sem descrição')
        print(f"   {profile.value}: {description}")

    print("\n2. Carregando perfil de produção:")
    config = config_manager.load_profile(ConfigProfile.PRODUCTION)
    print(f"   Memória: {config.max_memory_mb} MB")
    print(f"   Compressão: {config.enable_compression}")
    print(f"   Deduplicação: {config.enable_semantic_dedup}")

    print("\n3. Criando perfil customizado:")
    custom_config = config_manager.create_custom_profile(
        "custom_dev",
        ConfigProfile.DEVELOPMENT,
        {"max_memory_mb": 128, "enable_compression": True}
    )
    print(f"   Perfil customizado criado: {custom_config.profile}")

    print("\n4. Exportando configuração:")
    exported = config_manager.export_configuration("json")
    print(f"   Configuração exportada ({len(exported)} caracteres)")

    print("\n=== Demo Concluída ===")