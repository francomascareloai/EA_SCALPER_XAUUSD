#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigador Completo de Arquivos
=================================

Analisa TODOS os 144k+ arquivos para identificar onde estão os arquivos
que nossos scanners anteriores não processaram.
"""

import os
from pathlib import Path
from collections import defaultdict
import time

class CompleteFileInvestigator:
    """Investigador que encontra TODOS os arquivos do projeto"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.all_files = []
        self.directory_stats = defaultdict(int)
        self.extension_stats = defaultdict(int)
        self.large_directories = []
        
    def scan_all_files_no_filter(self):
        """Escaneia TODOS os arquivos sem qualquer filtro"""
        print("🔍 Investigando TODOS os arquivos sem filtros...")
        start_time = time.time()
        
        total_files = 0
        total_size = 0
        
        # Escanear TUDO sem exceção
        for root, dirs, files in os.walk(self.base_path):
            current_dir = Path(root)
            file_count_in_dir = len(files)
            
            # Contabilizar diretório
            self.directory_stats[str(current_dir)] = file_count_in_dir
            
            if file_count_in_dir > 1000:  # Diretórios com muitos arquivos
                self.large_directories.append((str(current_dir), file_count_in_dir))
            
            for filename in files:
                file_path = current_dir / filename
                try:
                    if file_path.is_file():
                        self.all_files.append(str(file_path))
                        total_files += 1
                        
                        # Estatísticas de extensão
                        ext = file_path.suffix.lower()
                        self.extension_stats[ext] += 1
                        
                        # Tamanho
                        try:
                            total_size += file_path.stat().st_size
                        except:
                            pass
                            
                except Exception as e:
                    print(f"⚠️ Erro ao processar {file_path}: {e}")
                    
            # Progresso a cada 10k arquivos
            if total_files % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"📊 Processados: {total_files:,} arquivos ({elapsed:.1f}s)")
        
        scan_time = time.time() - start_time
        
        print(f"\n✅ Investigação completa!")
        print(f"📁 Total de arquivos encontrados: {total_files:,}")
        print(f"💾 Tamanho total: {self.format_size(total_size)}")
        print(f"⏱️ Tempo: {scan_time:.1f} segundos")
        
        return total_files
    
    def analyze_large_directories(self):
        """Analisa diretórios com muitos arquivos"""
        print("\n📊 DIRETÓRIOS COM MAIS ARQUIVOS:")
        print("="*80)
        
        # Ordenar por quantidade de arquivos
        self.large_directories.sort(key=lambda x: x[1], reverse=True)
        
        for i, (dir_path, file_count) in enumerate(self.large_directories[:20]):
            # Encurtar caminho para exibição
            short_path = dir_path.replace(str(self.base_path), "...")
            if len(short_path) > 60:
                short_path = short_path[:57] + "..."
            print(f"{i+1:2d}. {short_path:<60} {file_count:>6,} arquivos")
    
    def analyze_extensions(self):
        """Analisa tipos de arquivo por extensão"""
        print("\n📊 TIPOS DE ARQUIVO MAIS COMUNS:")
        print("="*50)
        
        # Ordenar por quantidade
        sorted_extensions = sorted(self.extension_stats.items(), key=lambda x: x[1], reverse=True)
        
        for i, (ext, count) in enumerate(sorted_extensions[:20]):
            ext_display = ext if ext else "(sem extensão)"
            print(f"{i+1:2d}. {ext_display:<15} {count:>8,} arquivos")
    
    def find_hidden_directories(self):
        """Procura por diretórios que podem ter sido ignorados"""
        print("\n🔍 PROCURANDO DIRETÓRIOS OCULTOS OU IGNORADOS:")
        print("="*60)
        
        suspicious_dirs = []
        
        for dir_path, file_count in self.directory_stats.items():
            dir_name = Path(dir_path).name.lower()
            
            # Procurar por diretórios que podem ter sido ignorados
            if any(pattern in dir_name for pattern in [
                '.venv', '.git', '__pycache__', 'node_modules',
                '.pytest_cache', '.vscode', '.idea', 'temp',
                'cache', 'logs'
            ]):
                suspicious_dirs.append((dir_path, file_count, "Diretório sistema"))
            elif file_count > 5000:
                suspicious_dirs.append((dir_path, file_count, "Diretório muito grande"))
        
        if suspicious_dirs:
            for dir_path, file_count, reason in suspicious_dirs[:10]:
                short_path = dir_path.replace(str(self.base_path), "...")
                print(f"📁 {short_path}")
                print(f"   Arquivos: {file_count:,} - Motivo: {reason}")
        else:
            print("✅ Nenhum diretório suspeito encontrado")
    
    def compare_with_previous_scans(self):
        """Compara com nossos scans anteriores"""
        print("\n📊 COMPARAÇÃO COM SCANS ANTERIORES:")
        print("="*50)
        print(f"🔍 Investigação atual: {len(self.all_files):,} arquivos")
        print(f"📊 Scanner rápido anterior: ~52,793 arquivos")
        print(f"📊 Scanner completo anterior: ~82,508 arquivos")
        print(f"❓ Diferença descoberta: {len(self.all_files) - 52793:,} arquivos")
        
        # Calcular percentual de cobertura anterior
        if len(self.all_files) > 0:
            coverage_fast = (52793 / len(self.all_files)) * 100
            coverage_complete = (82508 / len(self.all_files)) * 100
            print(f"📈 Cobertura scanner rápido: {coverage_fast:.1f}%")
            print(f"📈 Cobertura scanner completo: {coverage_complete:.1f}%")
    
    def save_complete_file_list(self):
        """Salva lista completa de arquivos para análise"""
        output_file = self.base_path / "complete_file_investigation.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"INVESTIGAÇÃO COMPLETA DE ARQUIVOS\n")
            f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de arquivos: {len(self.all_files):,}\n")
            f.write("="*60 + "\n\n")
            
            f.write("DIRETÓRIOS COM MAIS ARQUIVOS:\n")
            f.write("-"*40 + "\n")
            for dir_path, file_count in sorted(self.directory_stats.items(), key=lambda x: x[1], reverse=True)[:50]:
                f.write(f"{file_count:6,} arquivos - {dir_path}\n")
            
            f.write(f"\n\nTODOS OS ARQUIVOS ENCONTRADOS:\n")
            f.write("-"*40 + "\n")
            for file_path in self.all_files:
                f.write(f"{file_path}\n")
        
        print(f"📋 Lista completa salva em: {output_file}")
    
    def format_size(self, size_bytes: int) -> str:
        """Formata tamanho em bytes"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def run_complete_investigation(self):
        """Executa investigação completa"""
        print("🕵️ INICIANDO INVESTIGAÇÃO COMPLETA DE ARQUIVOS")
        print("="*60)
        
        # 1. Escanear tudo
        total_files = self.scan_all_files_no_filter()
        
        # 2. Analisar diretórios grandes
        self.analyze_large_directories()
        
        # 3. Analisar extensões
        self.analyze_extensions()
        
        # 4. Procurar diretórios ocultos
        self.find_hidden_directories()
        
        # 5. Comparar com scans anteriores
        self.compare_with_previous_scans()
        
        # 6. Salvar lista completa
        self.save_complete_file_list()
        
        print(f"\n🎯 INVESTIGAÇÃO CONCLUÍDA!")
        print(f"📁 Total encontrado: {total_files:,} arquivos")
        
        return total_files

if __name__ == "__main__":
    investigator = CompleteFileInvestigator("c:/Users/Admin/Documents/EA_SCALPER_XAUUSD")
    investigator.run_complete_investigation()