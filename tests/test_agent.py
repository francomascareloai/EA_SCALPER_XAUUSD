"""
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
