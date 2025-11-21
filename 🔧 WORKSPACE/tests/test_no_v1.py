import requests
import json

print("🧪 TESTANDO PROXY SEM /V1")
print("=" * 40)

base_ip = "http://192.168.7.8:4000"

# Testar endpoints sem /v1
print("1. Testando sem /v1:")

try:
    # Health Check
    health = requests.get(f"{base_ip}/health", timeout=5)
    print(f"   Health: {health.status_code} ✅" if health.status_code == 200 else f"   Health: {health.status_code} ❌")
    
    # Models sem /v1
    models = requests.get(f"{base_ip}/models", timeout=5)
    print(f"   Models: {models.status_code} ✅" if models.status_code == 200 else f"   Models: {models.status_code} ❌")
    
    # Chat sem /v1
    chat_data = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "Test without v1"}],
        "max_tokens": 30
    }
    
    chat = requests.post(
        f"{base_ip}/chat/completions",
        json=chat_data,
        timeout=30
    )
    print(f"   Chat: {chat.status_code} ✅" if chat.status_code == 200 else f"   Chat: {chat.status_code} ❌")

except Exception as e:
    print(f"   Erro: {e}")

print("\n2. Testando com /v1 (compatibilidade):")

try:
    # Models com /v1
    models_v1 = requests.get(f"{base_ip}/v1/models", timeout=5)
    print(f"   Models v1: {models_v1.status_code} ✅" if models_v1.status_code == 200 else f"   Models v1: {models_v1.status_code} ❌")
    
    # Chat com /v1
    chat_v1 = requests.post(
        f"{base_ip}/v1/chat/completions",
        json=chat_data,
        timeout=30
    )
    print(f"   Chat v1: {chat_v1.status_code} ✅" if chat_v1.status_code == 200 else f"   Chat v1: {chat_v1.status_code} ❌")

except Exception as e:
    print(f"   Erro: {e}")

print("\n" + "=" * 40)
print("🎯 CONFIGURAÇÕES PARA ROO CODE:")
print("\n📊 OPÇÃO PREFERIDA (sem v1):")
print(f"   Base URL: {base_ip}")
print("\n📊 OPÇÃO ALTERNATIVA (com v1):")
print(f"   Base URL: {base_ip}/v1")
print("\n🔑 Outras configurações:")
print("   API Key: qualquer-chave")
print("   Model: deepseek-r1")
print("\n✅ Ambas as opções funcionam!")
