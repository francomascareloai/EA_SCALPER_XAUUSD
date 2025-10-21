#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Instalação Automática - Sistema de Contexto Expandido

Este script automatiza a instalação e configuração de todas as dependências
necessárias para o sistema de contexto expandido de 2M tokens.

Autor: Assistente AI
Data: 2025
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path
from typing import List, Tuple, Optional

class InstaladorSistemaContexto:
    """
    Classe responsável pela instalação automática do sistema.
    """
    
    def __init__(self):
        self.sistema_os = platform.system().lower()
        self.python_version = sys.version_info
        self.diretorio_base = Path(__file__).parent
        self.logs = []
        
    def log(self, mensagem: str, tipo: str = "INFO"):
        """
        Adiciona uma mensagem ao log.
        
        Args:
            mensagem: Mensagem a ser logada
            tipo: Tipo da mensagem (INFO, WARN, ERROR, SUCCESS)
        """
        timestamp = time.strftime("%H:%M:%S")
        emoji_map = {
            "INFO": "ℹ️",
            "WARN": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        emoji = emoji_map.get(tipo, "📝")
        log_entry = f"[{timestamp}] {emoji} {mensagem}"
        self.logs.append(log_entry)
        print(log_entry)
        
    def verificar_python(self) -> bool:
        """
        Verifica se a versão do Python é compatível.
        
        Returns:
            True se a versão for compatível, False caso contrário
        """
        self.log(f"Verificando versão do Python: {sys.version}")
        
        if self.python_version < (3, 8):
            self.log("Python 3.8+ é necessário para este sistema", "ERROR")
            return False
            
        if self.python_version >= (3, 12):
            self.log("Python 3.12+ detectado - algumas dependências podem precisar de ajustes", "WARN")
            
        self.log(f"Versão do Python {self.python_version.major}.{self.python_version.minor} é compatível", "SUCCESS")
        return True
        
    def verificar_pip(self) -> bool:
        """
        Verifica se o pip está disponível e atualizado.
        
        Returns:
            True se o pip estiver disponível, False caso contrário
        """
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, check=True)
            self.log(f"Pip disponível: {result.stdout.strip()}")
            
            # Tentar atualizar o pip
            self.log("Atualizando pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         capture_output=True, check=True)
            self.log("Pip atualizado com sucesso", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Erro ao verificar/atualizar pip: {e}", "ERROR")
            return False
            
    def instalar_dependencias(self) -> bool:
        """
        Instala todas as dependências necessárias.
        
        Returns:
            True se a instalação foi bem-sucedida, False caso contrário
        """
        requirements_file = self.diretorio_base / "requirements.txt"
        
        if not requirements_file.exists():
            self.log(f"Arquivo requirements.txt não encontrado em {requirements_file}", "ERROR")
            return False
            
        self.log("Instalando dependências do requirements.txt...")
        
        try:
            # Instalar dependências básicas primeiro
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            
            if self.sistema_os == "windows":
                # No Windows, pode ser necessário usar --user em alguns casos
                self.log("Sistema Windows detectado")
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log("Dependências básicas instaladas com sucesso", "SUCCESS")
            
            # Instalar dependências específicas que podem falhar
            dependencias_especiais = [
                "sentence-transformers",
                "scikit-learn",
                "torch",  # Pode ser necessário para sentence-transformers
            ]
            
            for dep in dependencias_especiais:
                try:
                    self.log(f"Verificando instalação de {dep}...")
                    subprocess.run([sys.executable, "-c", f"import {dep.replace('-', '_')}"], 
                                 capture_output=True, check=True)
                    self.log(f"{dep} já está instalado", "SUCCESS")
                except subprocess.CalledProcessError:
                    self.log(f"Instalando {dep}...")
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                 capture_output=True, check=True)
                    self.log(f"{dep} instalado com sucesso", "SUCCESS")
                    
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Erro ao instalar dependências: {e}", "ERROR")
            if e.stdout:
                self.log(f"STDOUT: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"STDERR: {e.stderr}", "ERROR")
            return False
            
    def criar_estrutura_diretorios(self) -> bool:
        """
        Cria a estrutura de diretórios necessária.
        
        Returns:
            True se a criação foi bem-sucedida, False caso contrário
        """
        diretorios = [
            "cache",
            "cache/litellm_cache",
            "cache_contexto_2m",
            "logs",
            "dados",
            "resultados"
        ]
        
        self.log("Criando estrutura de diretórios...")
        
        try:
            for diretorio in diretorios:
                caminho = self.diretorio_base / diretorio
                caminho.mkdir(parents=True, exist_ok=True)
                self.log(f"Diretório criado: {caminho}")
                
            self.log("Estrutura de diretórios criada com sucesso", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Erro ao criar diretórios: {e}", "ERROR")
            return False
            
    def verificar_arquivo_env(self) -> bool:
        """
        Verifica e cria o arquivo .env se necessário.
        
        Returns:
            True se o arquivo .env está configurado, False caso contrário
        """
        env_file = self.diretorio_base / ".env"
        env_example = self.diretorio_base / ".env.example"
        
        if env_file.exists():
            self.log("Arquivo .env já existe")
            
            # Verificar se tem a chave do OpenRouter
            with open(env_file, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                
            if "OPENROUTER_API_KEY" in conteudo and "sk-or-v1-" in conteudo:
                self.log("Chave do OpenRouter encontrada no .env", "SUCCESS")
                return True
            else:
                self.log("Chave do OpenRouter não encontrada ou inválida no .env", "WARN")
                return False
                
        elif env_example.exists():
            self.log("Copiando .env.example para .env...")
            
            try:
                with open(env_example, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                    
                with open(env_file, 'w', encoding='utf-8') as f:
                    f.write(conteudo)
                    
                self.log("Arquivo .env criado a partir do .env.example", "SUCCESS")
                self.log("IMPORTANTE: Configure sua OPENROUTER_API_KEY no arquivo .env", "WARN")
                return False
                
            except Exception as e:
                self.log(f"Erro ao criar .env: {e}", "ERROR")
                return False
        else:
            self.log("Nem .env nem .env.example encontrados", "WARN")
            self.log("Criando .env básico...")
            
            try:
                conteudo_env = """
# Configuração do OpenRouter
OPENROUTER_API_KEY=sua-chave-aqui

# Configuração do LiteLLM
LITELLM_LOG=INFO
LITELLM_DROP_PARAMS=true

# Cache
CACHE_TYPE=disk
CACHE_DIR=./cache

# Configurações do sistema de contexto
MAX_TOKENS_MODELO=150000
MAX_CHUNKS_PARALELOS=3
TAMANHO_CHUNK_DEFAULT=8000
SOBREPOSICAO_CHUNK=500
"""
                
                with open(env_file, 'w', encoding='utf-8') as f:
                    f.write(conteudo_env.strip())
                    
                self.log("Arquivo .env básico criado", "SUCCESS")
                self.log("IMPORTANTE: Configure sua OPENROUTER_API_KEY no arquivo .env", "WARN")
                return False
                
            except Exception as e:
                self.log(f"Erro ao criar .env básico: {e}", "ERROR")
                return False
                
    def testar_importacoes(self) -> bool:
        """
        Testa se todas as importações necessárias funcionam.
        
        Returns:
            True se todas as importações funcionam, False caso contrário
        """
        self.log("Testando importações...")
        
        modulos_teste = [
            ("litellm", "LiteLLM"),
            ("openai", "OpenAI"),
            ("sentence_transformers", "Sentence Transformers"),
            ("sklearn", "Scikit-learn"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("requests", "Requests"),
            ("yaml", "PyYAML"),
            ("dotenv", "Python-dotenv"),
            ("diskcache", "DiskCache"),
            ("tqdm", "TQDM"),
            ("rich", "Rich")
        ]
        
        falhas = []
        
        for modulo, nome in modulos_teste:
            try:
                __import__(modulo)
                self.log(f"✓ {nome} importado com sucesso")
            except ImportError as e:
                self.log(f"✗ Falha ao importar {nome}: {e}", "ERROR")
                falhas.append(nome)
                
        if falhas:
            self.log(f"Falhas de importação: {', '.join(falhas)}", "ERROR")
            return False
        else:
            self.log("Todas as importações testadas com sucesso", "SUCCESS")
            return True
            
    def executar_teste_basico(self) -> bool:
        """
        Executa um teste básico do sistema.
        
        Returns:
            True se o teste passou, False caso contrário
        """
        self.log("Executando teste básico do sistema...")
        
        try:
            # Tentar importar o sistema de contexto expandido
            sys.path.append(str(self.diretorio_base))
            
            try:
                from sistema_contexto_expandido_2m import ContextManager
                self.log("Sistema de contexto expandido importado com sucesso", "SUCCESS")
            except ImportError as e:
                self.log(f"Erro ao importar sistema de contexto: {e}", "ERROR")
                return False
                
            # Teste básico de inicialização
             sistema = ContextManager(
                 base_url="http://localhost:4000",
                 model_name="deepseek-r1-free",
                 cache_dir=str(self.diretorio_base / "cache_contexto_2m")
             )
            
            self.log("Sistema inicializado com sucesso", "SUCCESS")
            
            # Teste de chunking
            texto_teste = "Este é um teste básico do sistema de chunking. " * 100
            chunks = sistema.dividir_em_chunks(texto_teste, tamanho_chunk=200)
            
            if len(chunks) > 0:
                self.log(f"Teste de chunking passou: {len(chunks)} chunks criados", "SUCCESS")
                return True
            else:
                self.log("Teste de chunking falhou: nenhum chunk criado", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Erro durante teste básico: {e}", "ERROR")
            return False
            
    def salvar_log_instalacao(self) -> None:
        """
        Salva o log da instalação em um arquivo.
        """
        log_file = self.diretorio_base / "logs" / f"instalacao_{int(time.time())}.log"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.logs))
            self.log(f"Log da instalação salvo em: {log_file}", "SUCCESS")
        except Exception as e:
            self.log(f"Erro ao salvar log: {e}", "ERROR")
            
    def executar_instalacao_completa(self) -> bool:
        """
        Executa todo o processo de instalação.
        
        Returns:
            True se a instalação foi bem-sucedida, False caso contrário
        """
        self.log("🚀 Iniciando instalação do Sistema de Contexto Expandido")
        self.log("=" * 60)
        
        etapas = [
            ("Verificação do Python", self.verificar_python),
            ("Verificação do Pip", self.verificar_pip),
            ("Criação de diretórios", self.criar_estrutura_diretorios),
            ("Instalação de dependências", self.instalar_dependencias),
            ("Configuração do .env", self.verificar_arquivo_env),
            ("Teste de importações", self.testar_importacoes),
            ("Teste básico do sistema", self.executar_teste_basico)
        ]
        
        sucesso_total = True
        
        for nome_etapa, funcao_etapa in etapas:
            self.log(f"\n📋 Executando: {nome_etapa}")
            
            try:
                sucesso = funcao_etapa()
                if sucesso:
                    self.log(f"✅ {nome_etapa} concluída com sucesso")
                else:
                    self.log(f"❌ {nome_etapa} falhou")
                    sucesso_total = False
            except Exception as e:
                self.log(f"❌ Erro em {nome_etapa}: {e}", "ERROR")
                sucesso_total = False
                
        # Salvar log
        self.salvar_log_instalacao()
        
        # Relatório final
        self.log("\n" + "=" * 60)
        if sucesso_total:
            self.log("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!", "SUCCESS")
            self.log("\n📋 PRÓXIMOS PASSOS:")
            self.log("   1. Configure sua OPENROUTER_API_KEY no arquivo .env")
            self.log("   2. Execute: python exemplo_uso_contexto_2m.py")
            self.log("   3. Teste o sistema com seus próprios documentos")
        else:
            self.log("❌ INSTALAÇÃO CONCLUÍDA COM PROBLEMAS", "ERROR")
            self.log("\n🔧 AÇÕES RECOMENDADAS:")
            self.log("   1. Verifique os logs de erro acima")
            self.log("   2. Instale manualmente as dependências que falharam")
            self.log("   3. Execute novamente este script")
            
        return sucesso_total

def main():
    """
    Função principal do instalador.
    """
    print("Sistema de Contexto Expandido - Instalador Automático")
    print("Versão: 1.0")
    print("Suporte para processamento de até 2 milhões de tokens")
    print()
    
    instalador = InstaladorSistemaContexto()
    
    try:
        sucesso = instalador.executar_instalacao_completa()
        sys.exit(0 if sucesso else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Instalação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro crítico durante a instalação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()