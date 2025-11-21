import requests
import time

print("🧪 TESTANDO PROXY COM IP REAL")
print("=" * 50)

# IPs para testar
ips = [
    "http://192.168.7.8:4000",      # IP real da máquina
    "http://127.0.0.1:4000",        # Localhost
]

for ip in ips:
    print(f"\n🔍 Testando: {ip}")
    
    try:
        # Health Check
        health = requests.get(f"{ip}/health", timeout=5)
        if health.status_code == 200:
            print(f"   ✅ Health OK")
            
            # Test Chat
            chat_data = {
                "model": "deepseek-r1",
                "messages": [{"role": "user", "content": "Hello from test!"}],
                "max_tokens": 30
            }
            
            chat = requests.post(
                f"{ip}/v1/chat/completions",
                json=chat_data,
                timeout=30
            )
            
            if chat.status_code == 200:
                print(f"   ✅ Chat OK")
                print(f"   🔌 Base URL: {ip}/v1")
            else:
                print(f"   ❌ Chat Error: {chat.status_code}")
        else:
            print(f"   ❌ Health Error: {health.status_code}")
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")

print("\n" + "=" * 50)
print("🎯 CONFIGURAÇÕES PARA ROO CODE:")
print("\n📊 OPÇÃO 1 (IP da máquina):")
print("   Base URL: http://192.168.7.8:4000/v1")
print("\n📊 OPÇÃO 2 (Localhost):")  
print("   Base URL: http://127.0.0.1:4000/v1")
print("\n🔑 Outras configurações:")
print("   API Key: qualquer-chave")
print("   Model: deepseek-r1")
print("\n🚀 Use a opção que funcionar melhor!")
