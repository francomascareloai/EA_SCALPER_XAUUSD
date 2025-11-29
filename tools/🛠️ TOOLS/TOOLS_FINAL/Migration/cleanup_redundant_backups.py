#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limpeza de Backups Redundantes
==============================

Remove múltiplos diretórios de backup redundantes, mantendo apenas
o backup mais recente e essencial.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class BackupCleaner:
    """Remove backups redundantes do projeto"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.removed_dirs = []
        self.space_freed = 0
        
    def calculate_dir_size(self, dir_path: Path) -> int:
        """Calcula o tamanho total de um diretório"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    try:
                        total_size += filepath.stat().st_size
                    except:
                        pass
        except:
            pass
        return total_size
    
    def remove_redundant_backups(self):
        """Remove diretórios de backup redundantes"""
        print("🗑️ INICIANDO LIMPEZA DE BACKUPS REDUNDANTES")
        print("="*50)
        
        # Diretórios de backup para remover (mantendo apenas LIMPEZA_FINAL_COMPLETA)
        backup_dirs_to_remove = [
            "BACKUP_MIGRATION",
            "BACKUP_SEGURANCA", 
            "Backups",
            "ADVANCED_CLEANUP",
            "FINAL_CLEANUP",
            "REMOVED_DUPLICATES_SMART"
        ]
        
        total_size_before = 0
        
        for backup_dir_name in backup_dirs_to_remove:
            backup_dir = self.base_path / backup_dir_name
            
            if backup_dir.exists() and backup_dir.is_dir():
                # Calcular tamanho antes da remoção
                dir_size = self.calculate_dir_size(backup_dir)
                total_size_before += dir_size
                
                print(f"📁 Removendo: {backup_dir_name}")
                print(f"   💾 Tamanho: {self.format_size(dir_size)}")
                
                try:
                    # Remover o diretório completamente
                    shutil.rmtree(str(backup_dir))
                    
                    self.removed_dirs.append({
                        "name": backup_dir_name,
                        "size_bytes": dir_size,
                        "size_formatted": self.format_size(dir_size),
                        "removed_at": datetime.now().isoformat()
                    })
                    
                    self.space_freed += dir_size
                    print(f"   ✅ Removido com sucesso!")
                    
                except Exception as e:
                    print(f"   ❌ Erro ao remover {backup_dir_name}: {e}")
            else:
                print(f"📁 {backup_dir_name}: Não encontrado ou já removido")
        
        # Relatório final
        self.generate_cleanup_report()
    
    def generate_cleanup_report(self):
        """Gera relatório da limpeza de backups"""
        print(f"\n" + "="*60)
        print("📊 RELATÓRIO DA LIMPEZA DE BACKUPS")
        print("="*60)
        print(f"🗑️ Diretórios removidos: {len(self.removed_dirs)}")
        print(f"💾 Espaço liberado: {self.format_size(self.space_freed)}")
        print(f"📁 Backup mantido: LIMPEZA_FINAL_COMPLETA/ (mais recente)")
        
        if self.removed_dirs:
            print(f"\n📋 DETALHES DOS BACKUPS REMOVIDOS:")
            for removed in self.removed_dirs:
                print(f"   📂 {removed['name']:<25}: {removed['size_formatted']:>10}")
        
        # Verificar espaço atual
        remaining_backup = self.base_path / "LIMPEZA_FINAL_COMPLETA"
        if remaining_backup.exists():
            remaining_size = self.calculate_dir_size(remaining_backup)
            print(f"\n📁 Backup restante:")
            print(f"   📂 LIMPEZA_FINAL_COMPLETA: {self.format_size(remaining_size)}")
        
        print("="*60)
        print("✅ LIMPEZA CONCLUÍDA - Backups redundantes removidos!")
        print("📌 Mantido apenas o backup essencial mais recente")
        print("="*60)
    
    def format_size(self, size_bytes: int) -> str:
        """Formata tamanho em bytes"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

if __name__ == "__main__":
    base_path = "c:/Users/Admin/Documents/EA_SCALPER_XAUUSD"
    
    cleaner = BackupCleaner(base_path)
    cleaner.remove_redundant_backups()