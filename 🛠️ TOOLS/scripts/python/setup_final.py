"""
Setup Final - OpenRouter + Trading Agent
Configuração completa com prompt caching
"""
import os

def create_env_file():
    """
    Cria arquivo .env com template
    """
    env_content = """# OpenRouter API Key - Obtenha em: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your_actual_api_key_here

# Configurações opcionais
OPENROUTER_APP_NAME="Trading Agent Organizer"
OPENROUTER_SITE_URL="https://github.com/your_repo"

# Modelos preferidos
DEFAULT_MODEL="anthropic/claude-3-5-sonnet"
BACKUP_MODEL="openai/gpt-4o"

# Cache settings
PROMPT_CACHE_TTL=3600
RESPONSE_CACHE_TTL=1800"""

    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ Arquivo .env criado!")
    print("📝 EDITE o arquivo .env e coloque sua API key real")

def check_dependencies():
    """
    Verifica dependências instaladas
    """
    try:
        import httpx
        print("✅ httpx:", httpx.__version__)
    except ImportError:
        print("❌ httpx não instalado")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv: OK")
    except ImportError:
        print("❌ python-dotenv não instalado")
        return False
    
    try:
        import requests
        print("✅ requests:", requests.__version__)
    except ImportError:
        print("❌ requests não instalado")
        return False
        
    return True

def create_test_script():
    """
    Cria script de teste rápido
    """
    test_content = '''"""
Teste rápido do Trading Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

def quick_test():
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key or api_key == 'sk-or-v1-your_actual_api_key_here':
        print("❌ Configure sua API key no arquivo .env primeiro!")
        print("🔗 Obtenha em: https://openrouter.ai/keys")
        return False
    
    print("✅ API Key configurada!")
    print(f"🔑 Key: {api_key[:15]}...")
    
    # Importar e testar agent
    try:
        from trading_agent_simple import TradingAgentSimple
        agent = TradingAgentSimple()
        print("✅ Trading Agent inicializado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao inicializar agent: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTE RÁPIDO - TRADING AGENT")
    print("="*40)
    quick_test()
'''
    
    with open('test_agent.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Script de teste criado: test_agent.py")

def main():
    """
    Setup principal
    """
    print("🚀 SETUP FINAL - OPENROUTER + TRADING AGENT")
    print("="*60)
    
    # Verificar dependências
    print("\n1️⃣ VERIFICANDO DEPENDÊNCIAS:")
    if not check_dependencies():
        print("\n❌ Instale dependências faltantes:")
        print("pip install httpx python-dotenv requests")
        return
    
    # Criar arquivos de configuração
    print("\n2️⃣ CRIANDO ARQUIVOS DE CONFIGURAÇÃO:")
    
    if not os.path.exists('.env'):
        create_env_file()
    else:
        print("✅ Arquivo .env já existe")
    
    create_test_script()
    
    print("\n3️⃣ PRÓXIMOS PASSOS:")
    print("🔗 1. Vá para: https://openrouter.ai/keys")
    print("🔑 2. Crie uma conta e obtenha sua API key")
    print("📝 3. Edite o arquivo .env e cole sua API key")
    print("🧪 4. Execute: python test_agent.py")
    print("🤖 5. Execute: python trading_agent_simple.py")
    
    print("\n✅ SETUP COMPLETO!")
    print("💾 Prompt caching implementado")
    print("🎯 Pronto para organizar códigos trading!")

if __name__ == "__main__":
    main()
