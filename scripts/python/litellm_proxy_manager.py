"""
LiteLLM Proxy Server Starter
Servidor intermediário para OpenRouter com prompt caching
"""
import os
import subprocess
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class LiteLLMProxyManager:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 4000
        self.proxy_url = f"http://{self.host}:{self.port}"
        self.config_file = "litellm_config.yaml"
        self.master_key = "sk-litellm-proxy-key-12345"
        
    def check_requirements(self):
        """
        Verifica se tudo está configurado
        """
        print("🔍 Verificando configurações...")
        
        # Verificar API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("❌ OPENROUTER_API_KEY não encontrada no .env")
            return False
        
        print(f"✅ API Key configurada: {api_key[:15]}...")
        
        # Verificar arquivo de configuração
        if not os.path.exists(self.config_file):
            print(f"❌ Arquivo {self.config_file} não encontrado")
            return False
        
        print(f"✅ Arquivo de configuração: {self.config_file}")
        return True
    
    def start_proxy(self):
        """
        Inicia o servidor LiteLLM proxy
        """
        if not self.check_requirements():
            return False
        
        print(f"\n🚀 Iniciando LiteLLM Proxy...")
        print(f"📡 Host: {self.host}:{self.port}")
        print(f"🔑 Master Key: {self.master_key}")
        print(f"📋 Config: {self.config_file}")
        
        try:
            # Comando para iniciar o proxy
            cmd = [
                sys.executable, "-m", "litellm", "--config", self.config_file,
                "--host", self.host,
                "--port", str(self.port),
                "--detailed_debug"
            ]
            
            print(f"💻 Comando: {' '.join(cmd)}")
            print("\n" + "="*60)
            print("🎯 LITELLM PROXY INICIANDO...")
            print("="*60)
            print("Para parar: Ctrl+C")
            print(f"URL do Proxy: {self.proxy_url}")
            print(f"UI Admin: {self.proxy_url}/ui")
            print("="*60)
            
            # Iniciar processo
            process = subprocess.Popen(cmd, cwd=os.getcwd())
            return process
            
        except Exception as e:
            print(f"❌ Erro ao iniciar proxy: {e}")
            return None
    
    def test_proxy(self):
        """
        Testa se o proxy está funcionando
        """
        print(f"\n🧪 Testando proxy em {self.proxy_url}...")
        
        try:
            # Esperar um pouco para o proxy iniciar
            time.sleep(5)
            
            # Testar health check
            response = requests.get(f"{self.proxy_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Proxy está rodando!")
                return True
            else:
                print(f"⚠️ Proxy respondeu com status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao testar proxy: {e}")
            return False
    
    def get_proxy_info(self):
        """
        Informações para conectar no proxy
        """
        info = f"""
🎯 INFORMAÇÕES DO PROXY LITELLM:

📡 URL Base: {self.proxy_url}
🔑 Master Key: {self.master_key}

🤖 MODELOS DISPONÍVEIS:
- qwen-coder (Qwen 3 Coder Free)
- deepseek-r1 (DeepSeek R1 Free)

🔌 PARA ROO CODE - Use estas configurações:
- Base URL: {self.proxy_url}/v1
- API Key: {self.master_key}
- Modelo: qwen-coder ou deepseek-r1

📊 Interface Admin: {self.proxy_url}/ui
- Usuário: admin
- Senha: trading123

💾 PROMPT CACHING: Ativo (TTL: 3600s/1800s)
⚡ RATE LIMITING: 10 RPM / 1000 TPM
"""
        return info

def main():
    """
    Gerenciador principal do proxy
    """
    manager = LiteLLMProxyManager()
    
    print("🎯 LITELLM PROXY MANAGER")
    print("="*50)
    
    # Mostrar informações
    print(manager.get_proxy_info())
    
    print("\n📝 OPÇÕES:")
    print("1. Iniciar Proxy")
    print("2. Testar Proxy") 
    print("3. Mostrar Info")
    print("4. Sair")
    
    while True:
        choice = input("\n🔢 Escolha uma opção (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 INICIANDO PROXY...")
            process = manager.start_proxy()
            if process:
                try:
                    process.wait()  # Esperar até ser interrompido
                except KeyboardInterrupt:
                    print("\n🛑 Proxy interrompido pelo usuário")
                    process.terminate()
            break
            
        elif choice == "2":
            manager.test_proxy()
            
        elif choice == "3":
            print(manager.get_proxy_info())
            
        elif choice == "4":
            print("👋 Saindo...")
            break
            
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()
