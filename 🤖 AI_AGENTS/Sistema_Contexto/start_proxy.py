#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para iniciar o Proxy LiteLLM automaticamente
Este script resolve o problema de rate limit e URL inválida
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Inicia o proxy LiteLLM diretamente"""
    print("🚀 Iniciando Proxy LiteLLM para Roo Code...")
    print("=" * 50)
    
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Verifica se a chave da API está configurada
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ OPENROUTER_API_KEY não configurada no arquivo .env")
        print("💡 Obtenha sua chave em: https://openrouter.ai/settings/integrations")
        return
    
    # Cria diretório de cache
    cache_dir = Path("cache/litellm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Diretório de cache: {cache_dir}")
    
    # Verifica se o arquivo de configuração existe
    config_file = Path("litellm_config.yaml")
    if not config_file.exists():
        print(f"❌ Arquivo de configuração não encontrado: {config_file}")
        return
    
    print("✅ Configuração encontrada")
    print("\n" + "="*60)
    print("📋 CONFIGURAÇÃO PARA ROO CODE:")
    print("="*60)
    print("Base URL: http://localhost:4000")
    print("API Key: qualquer_string")
    print("Modelo: deepseek-r1-free")
    print("="*60)
    print("\n🔄 Iniciando proxy na porta 4000...")
    print("💡 Pressione Ctrl+C para parar")
    print("-" * 50)
    
    # Comando para iniciar o proxy
    cmd = [
        "litellm",
        "--config", "litellm_config.yaml",
        "--port", "4000",
        "--host", "0.0.0.0"
    ]
    
    try:
        # Inicia o proxy
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Proxy parado pelo usuário")
    except FileNotFoundError:
        print("❌ Comando 'litellm' não encontrado")
        print("💡 Execute: pip install 'litellm[proxy,caching]'")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao iniciar proxy: {e}")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()