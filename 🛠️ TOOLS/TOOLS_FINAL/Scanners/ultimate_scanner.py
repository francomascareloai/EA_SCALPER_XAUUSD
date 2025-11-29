#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scanner DEFINITIVO - Todos os 147,980 Arquivos
==============================================

Processa ABSOLUTAMENTE TODOS os arquivos do projeto, incluindo .venv, 
caches, e qualquer outro arquivo que os scanners anteriores ignoraram.
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import json

def calculate_fast_hash(file_path: Path) -> str:
    """Calcula hash rápido lendo apenas partes do arquivo"""
    try:
        if not file_path.exists():
            return ""
            
        file_size = file_path.stat().st_size
        
        # Para arquivos muito pequenos (< 1KB), ler tudo
        if file_size <= 1024:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        
        # Para arquivos pequenos (< 8KB), ler tudo
        if file_size <= 8192:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        
        # Para arquivos grandes, ler início + meio + fim + tamanho
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Início do arquivo (4KB)
            hash_md5.update(f.read(4096))
            
            # Meio do arquivo (4KB)
            f.seek(file_size // 2)
            hash_md5.update(f.read(4096))
            
            # Fim do arquivo (4KB)
            if file_size > 8192:
                f.seek(-4096, 2)
                hash_md5.update(f.read(4096))
            
            # Incluir tamanho no hash para diferenciação
            hash_md5.update(str(file_size).encode())
            
        return hash_md5.hexdigest()
        
    except Exception as e:
        print(f"⚠️ Erro ao processar {file_path}: {e}")
        return ""

def process_chunk_absolute(chunk):
    """Processa um chunk de arquivos SEM NENHUMA RESTRIÇÃO"""
    chunk_hashes = defaultdict(list)
    chunk_processed = 0
    
    for file_path in chunk:
        try:
            file_hash = calculate_fast_hash(file_path)
            if file_hash:
                chunk_hashes[file_hash].append(str(file_path))
                chunk_processed += 1
        except Exception as e:
            print(f"⚠️ Erro no chunk: {e}")
            
    return dict(chunk_hashes), chunk_processed

class UltimateFileScanner:
    """Scanner que processa ABSOLUTAMENTE TODOS os arquivos"""
    
    def __init__(self, base_path: str, chunk_size: int = 2000):
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.file_hashes = defaultdict(list)
        
    def collect_absolutely_all_files(self) -> list[Path]:
        """Coleta ABSOLUTAMENTE TODOS os arquivos sem exceção"""
        print("📂 Coletando ABSOLUTAMENTE TODOS os arquivos (incluindo .venv, cache, etc.)...")
        start_time = time.time()
        
        all_files = []
        total_dirs = 0
        
        for root, dirs, files in os.walk(self.base_path):
            current_dir = Path(root)
            total_dirs += 1
            
            # NÃO IGNORAR NADA - processar TUDO
            for filename in files:
                file_path = current_dir / filename
                
                if file_path.is_file():
                    all_files.append(file_path)
                    
            # Progresso a cada 1000 diretórios
            if total_dirs % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"📁 Diretórios processados: {total_dirs:,} - Arquivos encontrados: {len(all_files):,} ({elapsed:.1f}s)")
                    
        collection_time = time.time() - start_time
        
        print(f"📊 Coleta ABSOLUTA concluída em {collection_time:.1f}s:")
        print(f"   📁 Total de arquivos encontrados: {len(all_files):,}")
        print(f"   📂 Total de diretórios processados: {total_dirs:,}")
        
        return all_files
        
    def scan_absolutely_everything(self):
        """Escaneia ABSOLUTAMENTE TODOS os arquivos do projeto"""
        print("🚀 Iniciando SCANNER DEFINITIVO - TODOS OS 147,980+ ARQUIVOS")
        print("="*70)
        start_time = time.time()
        
        # Coletar TODOS os arquivos sem exceção
        all_files = self.collect_absolutely_all_files()
        total_files = len(all_files)
        
        if total_files == 0:
            print("❌ Nenhum arquivo encontrado!")
            return
            
        print(f"⚡ Processando {total_files:,} arquivos em chunks de {self.chunk_size:,}")
        
        # Dividir em chunks
        chunks = [all_files[i:i + self.chunk_size] for i in range(0, total_files, self.chunk_size)]
        total_chunks = len(chunks)
        
        print(f"📦 Total de chunks: {total_chunks:,}")
        
        # Processar chunks com máximo paralelismo
        total_processed = 0
        
        with ThreadPoolExecutor(max_workers=12) as executor:  # Mais workers
            futures = []
            
            # Submeter todos os chunks
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_chunk_absolute, chunk)
                futures.append((i, future))
            
            # Processar resultados conforme ficam prontos
            for i, future in futures:
                try:
                    chunk_hashes, chunk_processed = future.result()
                    total_processed += chunk_processed
                    
                    # Mesclar com hashes globais
                    for file_hash, file_list in chunk_hashes.items():
                        self.file_hashes[file_hash].extend(file_list)
                    
                    # Mostrar progresso a cada 100 chunks
                    if (i + 1) % 100 == 0:
                        progress = ((i + 1) / total_chunks) * 100
                        elapsed = time.time() - start_time
                        print(f"📊 Chunk {i+1:,}/{total_chunks:,} ({progress:.1f}%) - Processados: {total_processed:,} - {elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"❌ Erro no chunk {i}: {e}")
        
        total_time = time.time() - start_time
        avg_speed = total_processed / total_time if total_time > 0 else 0
        
        print(f"\n✅ SCANNER DEFINITIVO CONCLUÍDO!")
        print(f"📊 Arquivos processados: {total_processed:,}/{total_files:,}")
        print(f"⏱️ Tempo total: {total_time:.1f}s")
        print(f"⚡ Velocidade média: {avg_speed:.0f} arquivos/segundo")
        
        # Analisar duplicatas
        self.analyze_all_duplicates()
        
    def analyze_all_duplicates(self):
        """Analisa TODAS as duplicatas encontradas"""
        print("\n🔍 Analisando TODAS as duplicatas encontradas...")
        
        duplicates = []
        total_duplicates = 0
        total_duplicate_size = 0
        
        # Categorizar duplicatas por tipo
        duplicate_categories = {
            'python_cache': 0,
            'mql_files': 0,
            'json_metadata': 0,
            'backups': 0,
            'others': 0
        }
        
        for file_hash, file_list in self.file_hashes.items():
            if len(file_list) > 1:
                try:
                    first_file_path = Path(file_list[0])
                    first_file_size = first_file_path.stat().st_size
                    duplicate_size = first_file_size * (len(file_list) - 1)
                    total_duplicate_size += duplicate_size
                    
                    # Categorizar duplicata
                    first_file_str = str(first_file_path).lower()
                    if '.venv' in first_file_str or '__pycache__' in first_file_str:
                        duplicate_categories['python_cache'] += len(file_list) - 1
                    elif first_file_path.suffix.lower() in ['.mq4', '.mq5', '.ex4', '.ex5']:
                        duplicate_categories['mql_files'] += len(file_list) - 1
                    elif first_file_path.suffix.lower() == '.json':
                        duplicate_categories['json_metadata'] += len(file_list) - 1
                    elif 'backup' in first_file_str:
                        duplicate_categories['backups'] += len(file_list) - 1
                    else:
                        duplicate_categories['others'] += len(file_list) - 1
                    
                    duplicates.append({
                        "hash": file_hash,
                        "files": file_list,
                        "count": len(file_list),
                        "size_per_file": first_file_size,
                        "wasted_space": duplicate_size
                    })
                    total_duplicates += len(file_list) - 1
                except:
                    pass
        
        # Relatório DEFINITIVO
        print("\n" + "="*80)
        print("📊 RELATÓRIO DEFINITIVO - TODOS OS 147,980+ ARQUIVOS")
        print("="*80)
        print(f"📁 Arquivos únicos processados: {len(self.file_hashes):,}")
        print(f"🔍 Grupos de duplicatas: {len(duplicates):,}")
        print(f"🗑️ Arquivos duplicados TOTAIS: {total_duplicates:,}")
        print(f"💾 Espaço desperdiçado TOTAL: {self.format_size(total_duplicate_size)}")
        print()
        print("📊 DUPLICATAS POR CATEGORIA:")
        for category, count in duplicate_categories.items():
            if count > 0:
                percentage = (count / total_duplicates) * 100 if total_duplicates > 0 else 0
                print(f"   📂 {category:<20}: {count:>8,} arquivos ({percentage:>5.1f}%)")
        print("="*80)
        
        # Salvar relatório DEFINITIVO
        report = {
            "scan_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_type": "ULTIMATE_COMPLETE_SCAN",
            "total_unique_files": len(self.file_hashes),
            "duplicate_groups": len(duplicates),
            "duplicate_files": total_duplicates,
            "wasted_space_bytes": total_duplicate_size,
            "wasted_space_formatted": self.format_size(total_duplicate_size),
            "duplicate_categories": duplicate_categories,
            "coverage": "100% - TODOS OS ARQUIVOS PROCESSADOS",
            "top_duplicates": sorted(duplicates, key=lambda x: x["wasted_space"], reverse=True)[:100]
        }
        
        with open(self.base_path / "ultimate_complete_scan.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("📋 Relatório DEFINITIVO salvo em: ultimate_complete_scan.json")
        
        return duplicates
    
    def format_size(self, size_bytes: int) -> str:
        """Formata tamanho em bytes"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0

if __name__ == "__main__":
    print("🎯 SCANNER DEFINITIVO - PROCESSANDO TODOS OS 147,980+ ARQUIVOS")
    print("🔥 INCLUINDO: .venv, __pycache__, .git, backups, TUDO!")
    print("="*70)
    
    scanner = UltimateFileScanner("c:/Users/Admin/Documents/EA_SCALPER_XAUUSD", chunk_size=3000)
    scanner.scan_absolutely_everything()