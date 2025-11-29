#!/usr/bin/env python3
"""
Teste direto com OpenRouter API para verificar se a chave está funcionando
"""

import requests
import json
import time
from datetime import datetime

# Configurações
OPENROUTER_API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"

def test_openrouter_direct():
    """Testa diretamente a API do OpenRouter"""
    print("🔗 Testando OpenRouter API diretamente...")
    print("="*50)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "LiteLLM Local Test"
    }
    
    # Teste 1: Listar modelos
    print("\n📋 Teste 1: Listando modelos disponíveis...")
    try:
        response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=30)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"  ✅ {len(models.get('data', []))} modelos encontrados")
            
            # Procurar pelos modelos que queremos
            deepseek_models = [m for m in models.get('data', []) if 'deepseek' in m.get('id', '').lower()]
            gpt_models = [m for m in models.get('data', []) if 'gpt-3.5' in m.get('id', '').lower()]
            
            print(f"  🧠 DeepSeek models: {len(deepseek_models)}")
            for model in deepseek_models[:3]:  # Mostrar apenas os primeiros 3
                print(f"    - {model.get('id')}")
                
            print(f"  🤖 GPT-3.5 models: {len(gpt_models)}")
            for model in gpt_models[:3]:  # Mostrar apenas os primeiros 3
                print(f"    - {model.get('id')}")
        else:
            print(f"  ❌ Erro: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Erro na requisição: {e}")
    
    # Teste 2: Chat completion simples
    print("\n💬 Teste 2: Chat completion simples...")
    
    models_to_test = [
        "deepseek/deepseek-r1-0528:free",
        "openai/gpt-3.5-turbo:free"
    ]
    
    for model in models_to_test:
        print(f"\n🤖 Testando modelo: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Olá! Responda apenas 'Funcionando!' se você conseguir me ouvir."
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            print(f"  Status: {response.status_code}")
            print(f"  Tempo: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"  ✅ Resposta: {content.strip()}")
                
                # Informações de uso
                usage = data.get('usage', {})
                print(f"  📊 Tokens: {usage.get('total_tokens', 0)} (input: {usage.get('prompt_tokens', 0)}, output: {usage.get('completion_tokens', 0)})")
                
            else:
                print(f"  ❌ Erro: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Erro na requisição: {e}")
    
    # Teste 3: Teste de contexto grande
    print("\n📏 Teste 3: Teste de contexto grande (DeepSeek)...")
    
    # Criar um texto longo
    long_text = "Este é um teste de contexto longo. " * 1000  # ~35,000 caracteres
    
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {
                "role": "user",
                "content": f"Analise este texto e me diga quantas vezes a palavra 'teste' aparece: {long_text}"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        end_time = time.time()
        
        print(f"  Status: {response.status_code}")
        print(f"  Tempo: {end_time - start_time:.2f}s")
        print(f"  Tamanho do texto: {len(long_text):,} caracteres")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"  ✅ Resposta: {content.strip()}")
            
            # Informações de uso
            usage = data.get('usage', {})
            print(f"  📊 Tokens: {usage.get('total_tokens', 0)} (input: {usage.get('prompt_tokens', 0)}, output: {usage.get('completion_tokens', 0)})")
            
        else:
            print(f"  ❌ Erro: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Erro na requisição: {e}")
    
    print("\n" + "="*50)
    print(f"🕐 Teste concluído em {datetime.now().strftime('%H:%M:%S')}")
    print("="*50)

if __name__ == "__main__":
    test_openrouter_direct()