#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Simples do Proxy LiteLLM
Verifica se o proxy está funcionando e identifica problemas
"""

import requests
import json

# Configurações
BASE_URL = "http://localhost:4000"

def test_proxy_health():
    """Testa se o proxy está funcionando"""
    print("🔍 Testando saúde do proxy...")
    
    try:
        # Teste de health check
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✅ Proxy está online")
        
        # Teste de modelos disponíveis
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        print(f"Models endpoint: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"📋 Modelos disponíveis: {len(models.get('data', []))}")
            for model in models.get('data', [])[:3]:
                print(f"  - {model.get('id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Erro no health check: {e}")
        return False
    
    return True

def test_simple_completion():
    """Testa uma completion simples"""
    print("\n🤖 Testando completion simples...")
    
    # Teste sem autenticação
    payload = {
        "model": "deepseek-r1-free",
        "messages": [
            {"role": "user", "content": "Hello! Just say 'Hi' back."}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            message = data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
            print(f"✅ Resposta: {message}")
            return True
        else:
            print(f"❌ Erro: {response.status_code}")
            print(f"Resposta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erro na completion: {e}")
        return False

def test_with_auth():
    """Testa com diferentes tipos de autenticação"""
    print("\n🔐 Testando com autenticação...")
    
    auth_headers = [
        {},  # Sem auth
        {"Authorization": "Bearer test"},  # Bearer token
        {"Authorization": "Bearer sk-test"},  # Bearer com sk-
        {"X-API-Key": "test"},  # API Key header
    ]
    
    payload = {
        "model": "deepseek-r1-free",
        "messages": [
            {"role": "user", "content": "Test"}
        ],
        "max_tokens": 5
    }
    
    for i, headers in enumerate(auth_headers):
        print(f"\nTeste {i+1}: {headers if headers else 'Sem autenticação'}")
        
        try:
            full_headers = {"Content-Type": "application/json"}
            full_headers.update(headers)
            
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=full_headers,
                json=payload,
                timeout=15
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"  ✅ Sucesso!")
                return True
            elif response.status_code == 401:
                print(f"  🔒 Não autorizado")
            else:
                print(f"  ❌ Erro: {response.text[:100]}")
                
        except Exception as e:
            print(f"  ❌ Erro: {e}")
    
    return False

def main():
    """Função principal"""
    print("🚀 TESTE SIMPLES DO PROXY LITELLM")
    print("=" * 50)
    print(f"🌐 URL: {BASE_URL}")
    
    # Teste 1: Health check
    health_ok = test_proxy_health()
    
    if not health_ok:
        print("\n❌ Proxy não está respondendo corretamente")
        return
    
    # Teste 2: Completion simples
    completion_ok = test_simple_completion()
    
    if not completion_ok:
        # Teste 3: Diferentes tipos de auth
        auth_ok = test_with_auth()
        
        if not auth_ok:
            print("\n❌ Nenhum método de autenticação funcionou")
            print("\n💡 Possíveis soluções:")
            print("  1. Verificar configuração do LiteLLM")
            print("  2. Verificar se as chaves de API estão corretas")
            print("  3. Verificar se o proxy requer autenticação")
            print("  4. Verificar logs do proxy para mais detalhes")
        else:
            print("\n✅ Autenticação funcionando!")
    else:
        print("\n✅ Proxy funcionando perfeitamente!")

if __name__ == "__main__":
    main()