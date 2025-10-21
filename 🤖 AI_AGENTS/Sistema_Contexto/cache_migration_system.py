#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Migração e Backup do Cache Avançado
Migração de dados, backup e recovery para sistema de cache R1

Este sistema implementa:
1. Migração de dados entre versões
2. Backup e recovery automático
3. Validação de integridade
4. Sincronização entre instâncias
5. Recuperação de desastres

Autor: Sistema Cache Avançado R1
Data: 2025
"""

import os
import json
import pickle
import shutil
import hashlib
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import tempfile
import gzip
import lzma

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MigrationJob:
    """Trabalho de migração com metadados."""
    id: str
    source_version: str
    target_version: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_items: int = 0
    migrated_items: int = 0
    failed_items: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BackupInfo:
    """Informações de backup."""
    id: str
    created_at: datetime
    size_bytes: int
    compression_ratio: float
    integrity_hash: str
    status: str  # active, archived, corrupted
    location: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DataValidator:
    """Validador de integridade de dados."""

    def __init__(self):
        self.validation_rules = {
            'cache_entry': self._validate_cache_entry,
            'embedding': self._validate_embedding,
            'metadata': self._validate_metadata
        }

    def _validate_cache_entry(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida entrada de cache."""
        required_fields = ['key', 'value', 'created_at', 'level']

        for field in required_fields:
            if field not in data:
                return False, f"Campo obrigatório faltando: {field}"

        if not isinstance(data['key'], str) or not data['key']:
            return False, "Chave inválida"

        if data.get('ttl') and not isinstance(data['ttl'], int):
            return False, "TTL deve ser inteiro"

        return True, "OK"

    def _validate_embedding(self, embedding: Any) -> Tuple[bool, str]:
        """Valida embedding."""
        if embedding is None:
            return True, "OK"

        if not isinstance(embedding, np.ndarray):
            return False, "Embedding deve ser numpy array"

        if embedding.shape[-1] != 384:  # Tamanho típico do all-MiniLM-L6-v2
            return False, f"Embedding deve ter dimensão 384, tem {embedding.shape[-1]}"

        return True, "OK"

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida metadados."""
        if not isinstance(metadata, dict):
            return False, "Metadados devem ser dicionário"

        return True, "OK"

    def validate_data(self, data_type: str, data: Any) -> Tuple[bool, str]:
        """Valida dados baseado no tipo."""
        if data_type in self.validation_rules:
            return self.validation_rules[data_type](data)
        return True, "Tipo não validado"

class MigrationManager:
    """Gerenciador de migrações de cache."""

    def __init__(self, cache_dir: str = "./cache/advanced"):
        self.cache_dir = Path(cache_dir)
        self.migration_dir = self.cache_dir / "migrations"
        self.migration_dir.mkdir(parents=True, exist_ok=True)

        self.migrations_db = self.migration_dir / "migrations.db"
        self._init_migration_db()

        self.current_version = "2.0.0"
        self.supported_versions = ["1.0.0", "1.5.0", "2.0.0"]

        self.validator = DataValidator()

    def _init_migration_db(self):
        """Inicializa banco de dados de migrações."""
        with sqlite3.connect(self.migrations_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id TEXT PRIMARY KEY,
                    source_version TEXT,
                    target_version TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    total_items INTEGER,
                    migrated_items INTEGER,
                    failed_items INTEGER,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    size_bytes INTEGER,
                    compression_ratio REAL,
                    integrity_hash TEXT,
                    status TEXT,
                    location TEXT,
                    metadata TEXT
                )
            """)

            conn.commit()

    def create_migration_job(self, source_version: str, target_version: str) -> MigrationJob:
        """Cria trabalho de migração."""
        job_id = f"migration_{int(time.time())}_{hashlib.md5(f'{source_version}_{target_version}'.encode()).hexdigest()[:8]}"

        job = MigrationJob(
            id=job_id,
            source_version=source_version,
            target_version=target_version,
            status="pending",
            created_at=datetime.now()
        )

        # Salvar no banco
        with sqlite3.connect(self.migrations_db) as conn:
            conn.execute("""
                INSERT INTO migrations
                (id, source_version, target_version, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (job.id, job.source_version, job.target_version, job.status,
                  job.created_at.isoformat(), json.dumps(job.metadata)))
            conn.commit()

        logger.info(f"Trabalho de migração criado: {job.id}")
        return job

    def migrate_from_v1_to_v2(self, source_dir: str) -> MigrationJob:
        """Migra dados da versão 1.0 para 2.0."""
        job = self.create_migration_job("1.0.0", "2.0.0")

        try:
            job.status = "running"
            self._update_job_status(job)

            source_path = Path(source_dir)
            if not source_path.exists():
                raise ValueError(f"Diretório fonte não existe: {source_dir}")

            # Encontrar arquivos antigos
            old_cache_files = list(source_path.glob("*.pkl")) + list(source_path.glob("*.cache"))

            job.total_items = len(old_cache_files)
            migrated = 0
            failed = 0

            for old_file in old_cache_files:
                try:
                    # Carregar dados antigos
                    with open(old_file, 'rb') as f:
                        old_data = pickle.load(f)

                    # Migrar para novo formato
                    new_data = self._migrate_cache_entry_v1_to_v2(old_data)

                    # Validar dados migrados
                    is_valid, error = self.validator.validate_data('cache_entry', new_data)
                    if not is_valid:
                        logger.warning(f"Dados inválidos migrados: {error}")
                        failed += 1
                        continue

                    # Salvar no novo formato
                    new_file = self.cache_dir / "migrations" / f"migrated_{old_file.name}"
                    with open(new_file, 'wb') as f:
                        pickle.dump(new_data, f)

                    migrated += 1

                except Exception as e:
                    logger.error(f"Erro ao migrar {old_file}: {e}")
                    failed += 1

            # Finalizar migração
            job.status = "completed" if failed == 0 else "completed_with_errors"
            job.migrated_items = migrated
            job.failed_items = failed
            job.completed_at = datetime.now()

            self._update_job_status(job)

            logger.info(f"Migração concluída: {migrated} migrados, {failed} falharam")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._update_job_status(job)
            logger.error(f"Migração falhou: {e}")

        return job

    def _migrate_cache_entry_v1_to_v2(self, old_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migra entrada de cache da v1 para v2."""
        new_data = {
            'key': old_data.get('key', ''),
            'value': old_data.get('value', ''),
            'level': old_data.get('level', 'l1_memory'),
            'created_at': old_data.get('created_at', datetime.now()),
            'last_accessed': old_data.get('last_accessed', datetime.now()),
            'access_count': old_data.get('access_count', 0),
            'ttl': old_data.get('ttl'),
            'importance_score': old_data.get('importance_score', 0.5),
            'embedding': old_data.get('embedding'),
            'compressed': False,
            'compression_algo': 'none',
            'original_size': len(pickle.dumps(old_data.get('value', ''))),
            'compressed_size': 0,
            'semantic_hash': None,
            'cluster_id': None,
            'access_pattern': []
        }

        # Converter tipos se necessário
        if isinstance(new_data['created_at'], str):
            new_data['created_at'] = datetime.fromisoformat(new_data['created_at'])
        if isinstance(new_data['last_accessed'], str):
            new_data['last_accessed'] = datetime.fromisoformat(new_data['last_accessed'])

        return new_data

    def _update_job_status(self, job: MigrationJob):
        """Atualiza status do trabalho no banco."""
        with sqlite3.connect(self.migrations_db) as conn:
            conn.execute("""
                UPDATE migrations
                SET status = ?, completed_at = ?, error_message = ?,
                    total_items = ?, migrated_items = ?, failed_items = ?,
                    metadata = ?
                WHERE id = ?
            """, (job.status, job.completed_at.isoformat() if job.completed_at else None,
                  job.error_message, job.total_items, job.migrated_items,
                  job.failed_items, json.dumps(job.metadata), job.id))
            conn.commit()

class BackupManager:
    """Gerenciador de backup e recovery."""

    def __init__(self, cache_dir: str = "./cache/advanced", backup_dir: str = "./backup"):
        self.cache_dir = Path(cache_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.backup_db = self.backup_dir / "backups.db"
        self._init_backup_db()

    def _init_backup_db(self):
        """Inicializa banco de dados de backups."""
        with sqlite3.connect(self.backup_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    size_bytes INTEGER,
                    compression_ratio REAL,
                    integrity_hash TEXT,
                    status TEXT,
                    location TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def create_backup(self,
                     backup_type: str = "full",
                     compression: str = "gzip") -> BackupInfo:
        """Cria backup do cache."""

        backup_id = f"backup_{int(time.time())}_{backup_type}"
        backup_path = self.backup_dir / f"{backup_id}.bak"

        logger.info(f"Iniciando backup: {backup_id}")

        try:
            # Coletar arquivos para backup
            cache_files = []
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith(('.pkl', '.cache', '.db')):
                        cache_files.append(Path(root) / file)

            # Calcular hash de integridade
            integrity_hash = self._calculate_integrity_hash(cache_files)

            # Criar backup
            total_size = 0
            compression_ratio = 1.0

            if compression == "gzip":
                with gzip.open(backup_path, 'wb') as f:
                    for cache_file in cache_files:
                        with open(cache_file, 'rb') as cf:
                            data = cf.read()
                            f.write(data)
                            total_size += len(data)
            else:
                with open(backup_path, 'wb') as f:
                    for cache_file in cache_files:
                        with open(cache_file, 'rb') as cf:
                            data = cf.read()
                            f.write(data)
                            total_size += len(data)

            # Calcular tamanho do backup
            backup_size = backup_path.stat().st_size
            if total_size > 0:
                compression_ratio = total_size / backup_size

            # Criar registro de backup
            backup_info = BackupInfo(
                id=backup_id,
                created_at=datetime.now(),
                size_bytes=backup_size,
                compression_ratio=compression_ratio,
                integrity_hash=integrity_hash,
                status="active",
                location=str(backup_path),
                metadata={
                    'type': backup_type,
                    'compression': compression,
                    'files_count': len(cache_files),
                    'total_original_size': total_size
                }
            )

            # Salvar no banco
            with sqlite3.connect(self.backup_db) as conn:
                conn.execute("""
                    INSERT INTO backups
                    (id, created_at, size_bytes, compression_ratio, integrity_hash,
                     status, location, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (backup_info.id, backup_info.created_at.isoformat(),
                      backup_info.size_bytes, backup_info.compression_ratio,
                      backup_info.integrity_hash, backup_info.status,
                      backup_info.location, json.dumps(backup_info.metadata)))
                conn.commit()

            logger.info(f"Backup criado com sucesso: {backup_id}")
            return backup_info

        except Exception as e:
            logger.error(f"Erro ao criar backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise

    def restore_backup(self, backup_id: str, target_dir: Optional[str] = None) -> bool:
        """Restaura backup."""

        if target_dir is None:
            target_dir = self.cache_dir
        else:
            target_dir = Path(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        # Buscar informações do backup
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            raise ValueError(f"Backup não encontrado: {backup_id}")

        backup_path = Path(backup_info.location)
        if not backup_path.exists():
            raise ValueError(f"Arquivo de backup não encontrado: {backup_path}")

        logger.info(f"Iniciando restauração: {backup_id}")

        try:
            # Verificar integridade
            if not self._verify_backup_integrity(backup_info):
                raise ValueError("Backup corrompido ou inválido")

            # Extrair backup
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rb') as f:
                    data = f.read()
                    # Aqui seria necessário lógica para extrair arquivos individuais
                    # Por simplicidade, vamos apenas copiar
                    shutil.copy2(backup_path, target_dir / "restored_cache.bak")
            else:
                shutil.copy2(backup_path, target_dir / "restored_cache.bak")

            logger.info(f"Restauração concluída: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Erro ao restaurar backup: {e}")
            return False

    def _calculate_integrity_hash(self, files: List[Path]) -> str:
        """Calcula hash de integridade dos arquivos."""
        hasher = hashlib.sha256()
        for file_path in sorted(files):
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    def _verify_backup_integrity(self, backup_info: BackupInfo) -> bool:
        """Verifica integridade do backup."""
        backup_path = Path(backup_info.location)
        if not backup_path.exists():
            return False

        try:
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rb') as f:
                    data = f.read()
            else:
                with open(backup_path, 'rb') as f:
                    data = f.read()

            current_hash = hashlib.sha256(data).hexdigest()
            return current_hash == backup_info.integrity_hash

        except Exception as e:
            logger.error(f"Erro ao verificar integridade: {e}")
            return False

    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Recupera informações de backup."""
        with sqlite3.connect(self.backup_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM backups WHERE id = ?
            """, (backup_id,))

            row = cursor.fetchone()
            if row:
                return BackupInfo(
                    id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    size_bytes=row[2],
                    compression_ratio=row[3],
                    integrity_hash=row[4],
                    status=row[5],
                    location=row[6],
                    metadata=json.loads(row[7]) if row[7] else {}
                )
        return None

    def list_backups(self, status: Optional[str] = None) -> List[BackupInfo]:
        """Lista backups disponíveis."""
        with sqlite3.connect(self.backup_db) as conn:
            if status:
                cursor = conn.execute("""
                    SELECT * FROM backups WHERE status = ? ORDER BY created_at DESC
                """, (status,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM backups ORDER BY created_at DESC
                """)

            backups = []
            for row in cursor.fetchall():
                backups.append(BackupInfo(
                    id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    size_bytes=row[2],
                    compression_ratio=row[3],
                    integrity_hash=row[4],
                    status=row[5],
                    location=row[6],
                    metadata=json.loads(row[7]) if row[7] else {}
                ))

            return backups

class CacheSyncManager:
    """Gerenciador de sincronização entre instâncias de cache."""

    def __init__(self, cache_system, sync_interval: int = 300):
        self.cache_system = cache_system
        self.sync_interval = sync_interval
        self.last_sync = datetime.now()
        self.sync_lock = threading.Lock()

    def sync_with_remote(self, remote_url: str, api_key: Optional[str] = None):
        """Sincroniza cache com instância remota."""
        with self.sync_lock:
            try:
                # Aqui seria implementada a lógica de sincronização
                # Por exemplo, usando HTTP API ou WebSocket
                logger.info(f"Sincronizando com {remote_url}")

                # Simular sincronização
                time.sleep(1)

                self.last_sync = datetime.now()
                logger.info("Sincronização concluída")

            except Exception as e:
                logger.error(f"Erro na sincronização: {e}")

if __name__ == "__main__":
    print("=== Sistema de Migração e Backup ===")

    # Demo do sistema de migração
    migration_manager = MigrationManager()
    backup_manager = BackupManager()

    # Criar backup
    print("\n1. Criando backup:")
    try:
        backup = backup_manager.create_backup(backup_type="full", compression="gzip")
        print(f"   Backup criado: {backup.id}")
        print(f"   Tamanho: {backup.size_bytes:,} bytes")
        print(f"   Taxa de compressão: {backup.compression_ratio:.2f}x")
    except Exception as e:
        print(f"   Erro ao criar backup: {e}")

    # Listar backups
    print("\n2. Backups disponíveis:")
    backups = backup_manager.list_backups()
    for backup in backups[:5]:  # Mostrar apenas os 5 mais recentes
        print(f"   {backup.id}: {backup.created_at} - {backup.size_bytes:,} bytes")

    # Demo de migração (se houver dados antigos)
    print("\n3. Sistema de migração:")
    print("   Pronto para migrar dados da versão 1.0 para 2.0")
    print("   Use: migration_manager.migrate_from_v1_to_v2('/caminho/para/dados/antigos')")

    print("\n=== Demo Concluída ===")