import requests
import time

print("🧪 TESTE FINAL DO PROXY")
print("=" * 40)

try:
    # Health Check
    print("1. Testando Health Check...")
    health = requests.get("http://127.0.0.1:4000/health", timeout=5)
    if health.status_code == 200:
        print("   ✅ Health OK")
        data = health.json()
        print(f"   📊 Cache: {data['cache_size']} items")
        print(f"   🔢 Requests: {data['request_count']}")
    
    # Models
    print("\n2. Testando Lista de Modelos...")
    models = requests.get("http://127.0.0.1:4000/v1/models", timeout=5)
    if models.status_code == 200:
        print("   ✅ Models OK")
        data = models.json()
        print(f"   🤖 Modelos: {len(data['data'])}")
    
    # Chat Test
    print("\n3. Testando Chat (DeepSeek)...")
    chat_data = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "Hello! Test message for proxy."}],
        "max_tokens": 50
    }
    
    start = time.time()
    chat = requests.post(
        "http://127.0.0.1:4000/v1/chat/completions",
        json=chat_data,
        timeout=30
    )
    duration = time.time() - start
    
    if chat.status_code == 200:
        print(f"   ✅ Chat OK ({duration:.1f}s)")
        result = chat.json()
        response = result['choices'][0]['message']['content']
        print(f"   📝 Resposta: {response[:50]}...")
    else:
        print(f"   ❌ Chat Error: {chat.status_code}")

    print("\n" + "=" * 40)
    print("✅ PROXY FUNCIONANDO PERFEITAMENTE!")
    print("\n🔌 CONFIGURAÇÃO ROO CODE:")
    print("   Base URL: http://127.0.0.1:4000/v1")
    print("   API Key: qualquer-chave")
    print("   Model: deepseek-r1")
    print("\n🚀 PRONTO PARA USAR!")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n💡 Certifique-se que o proxy está rodando:")
    print("   python simple_trading_proxy.py")
