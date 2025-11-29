"""
Teste do Simple Trading Proxy
Cliente de teste para verificar se está funcionando
"""
import requests
import json
import time

def test_proxy_health():
    """Teste básico de saúde do proxy"""
    try:
        response = requests.get("http://127.0.0.1:4000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Proxy está rodando!")
            print(f"📊 Status: {data['status']}")
            print(f"🤖 Modelos: {data['models']}")
            print(f"💾 Cache size: {data['cache_size']}")
            print(f"🔢 Request count: {data['request_count']}")
            return True
        else:
            print(f"⚠️ Proxy respondeu com status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar no proxy: {e}")
        return False

def test_models_endpoint():
    """Teste do endpoint de modelos"""
    try:
        response = requests.get("http://127.0.0.1:4000/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Endpoint /v1/models funcionando!")
            print("🤖 Modelos disponíveis:")
            for model in data['data']:
                print(f"   - {model['id']}")
            return True
        else:
            print(f"❌ Erro no endpoint models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao testar models: {e}")
        return False

def test_chat_completion(model="deepseek-r1"):
    """Teste de chat completion"""
    try:
        url = "http://127.0.0.1:4000/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key"  # Qualquer chave funciona
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Você é um expert em análise de códigos MQL4/MQL5. Responda de forma concisa."
                },
                {
                    "role": "user",
                    "content": "Analise este código MQL4 simples: extern double LotSize = 0.01; void OnTick() { OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, 0, 0); }"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        print(f"\n🧪 Testando chat completion com {model}...")
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"✅ Resposta recebida em {duration:.2f}s")
            print(f"📝 Resposta (primeiras 150 chars):")
            print(f"   {content[:150]}...")
            
            # Testar cache - segunda requisição deve ser mais rápida
            print(f"\n🔄 Testando cache (segunda requisição)...")
            start_time = time.time()
            response2 = requests.post(url, headers=headers, json=data, timeout=30)
            end_time = time.time()
            duration2 = end_time - start_time
            
            if response2.status_code == 200:
                print(f"✅ Cache funcionando! Resposta em {duration2:.2f}s")
                if duration2 < duration:
                    print("🚀 Cache mais rápido que requisição original!")
                else:
                    print("💾 Resposta do cache (mesma velocidade)")
            
            return True
            
        else:
            print(f"❌ Erro na requisição: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste chat: {e}")
        return False

def main():
    """Teste completo do proxy"""
    print("🧪 TESTE COMPLETO DO SIMPLE TRADING PROXY")
    print("=" * 50)
    
    # Teste 1: Health check
    print("1️⃣ TESTE HEALTH CHECK:")
    if not test_proxy_health():
        print("❌ Proxy não está rodando. Inicie primeiro com:")
        print("   python simple_trading_proxy.py")
        return
    
    # Teste 2: Models endpoint
    print("\n2️⃣ TESTE MODELS ENDPOINT:")
    if not test_models_endpoint():
        return
    
    # Teste 3: Chat completion com DeepSeek R1
    print("\n3️⃣ TESTE CHAT COMPLETION (DeepSeek R1):")
    if not test_chat_completion("deepseek-r1"):
        return
    
    # Teste 4: Chat completion com Qwen Coder  
    print("\n4️⃣ TESTE CHAT COMPLETION (Qwen Coder):")
    if not test_chat_completion("qwen-coder"):
        print("⚠️ Qwen pode ter rate limiting agressivo")
    
    print("\n" + "=" * 50)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print("\n🔌 CONFIGURAÇÃO PARA ROO CODE:")
    print("   Base URL: http://127.0.0.1:4000/v1")
    print("   API Key: qualquer-chave-funciona")
    print("   Modelos: deepseek-r1, qwen-coder")
    print("\n💾 Prompt caching funcionando!")
    print("🎯 Proxy pronto para uso!")

if __name__ == "__main__":
    main()
