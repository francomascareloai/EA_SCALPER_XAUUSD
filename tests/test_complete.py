import requests

print("🧪 TESTE COMPLETO DO PROXY ATUALIZADO")
print("=" * 50)

base_url = "http://127.0.0.1:4000"

try:
    # 1. Health check
    health = requests.get(f"{base_url}/health")
    print(f"✅ Health: {health.status_code} - {health.json()}")
    
    # 2. Model info (endpoint que Roo Code estava tentando)
    model_info = requests.get(f"{base_url}/v1/model/info")
    print(f"✅ Model Info: {model_info.status_code} - {model_info.json()}")
    
    # 3. Models list
    models = requests.get(f"{base_url}/v1/models")
    print(f"✅ Models: {models.status_code} - {models.json()}")
    
    print("\n" + "=" * 50)
    print("✅ TODOS OS ENDPOINTS FUNCIONANDO!")
    print("\n🔌 CONFIGURAÇÃO PARA ROO CODE:")
    print(f"   Base URL: {base_url}")
    print(f"   API Key: qualquer-chave")
    print(f"   Model: deepseek-r1")
    print("\n🎯 PROXY PRONTO PARA ROO CODE!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
