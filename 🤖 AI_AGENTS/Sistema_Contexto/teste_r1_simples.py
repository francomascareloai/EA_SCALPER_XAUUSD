#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Simples do Modelo R1 (DeepSeek) sem Cache

Este script testa o modelo R1 gratuito do DeepSeek
através do OpenRouter sem cache para verificar se está funcionando.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

try:
    import litellm
    from litellm import completion
except ImportError:
    print("❌ LiteLLM não encontrado. Instale com: pip install litellm")
    sys.exit(1)

def test_r1_model():
    """Testar o modelo R1 sem cache"""
    print("🚀 TESTE SIMPLES DO MODELO R1 (SEM CACHE)")
    print("=" * 50)
    
    # Verificar API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY não encontrada no arquivo .env")
        return False
        
    print(f"✅ API Key configurada: {api_key[:20]}...")
    
    # Preparar mensagem de teste
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente especializado em trading e análise de mercado."
        },
        {
            "role": "user",
            "content": "Explique em 3 frases o que são Order Blocks no trading ICT/SMC."
        }
    ]
    
    try:
        print("\n🔄 Fazendo request para o modelo R1...")
        start_time = time.time()
        
        # Fazer request direto sem cache
        response = completion(
            model="openrouter/deepseek/deepseek-r1-0528:free",
            messages=messages,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=500,
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "DeepSeek R1 - Teste Simples"
            }
        )
        
        response_time = time.time() - start_time
        
        # Exibir resultado
        print(f"\n✅ SUCESSO!")
        print(f"⏱️  Tempo de resposta: {response_time:.3f}s")
        print(f"🤖 Modelo: deepseek-r1-0528:free")
        
        if hasattr(response, 'usage'):
            print(f"🔢 Tokens usados: {response.usage.total_tokens}")
        
        print("\n💬 RESPOSTA DO R1:")
        print("─" * 60)
        print(response.choices[0].message.content)
        print("─" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao fazer request: {e}")
        print("\n🔧 Possíveis causas:")
        print("1. API key inválida")
        print("2. Modelo não disponível")
        print("3. Problema de conectividade")
        print("4. Rate limit atingido")
        return False

def test_multiple_models():
    """Testar múltiplos modelos para comparação"""
    print("\n🔄 TESTE DE MÚLTIPLOS MODELOS")
    print("=" * 50)
    
    models = [
        ("openrouter/deepseek/deepseek-r1-0528:free", "DeepSeek R1 (Reasoning)"),
        ("openrouter/openai/gpt-3.5-turbo", "GPT-3.5 Turbo (Free)"),
        ("openrouter/meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (Free)")
    ]
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    messages = [
        {
            "role": "user",
            "content": "Em uma frase, explique o que é um Fair Value Gap (FVG) no trading."
        }
    ]
    
    for model_id, model_name in models:
        try:
            print(f"\n🤖 Testando: {model_name}")
            start_time = time.time()
            
            response = completion(
                model=model_id,
                messages=messages,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=200,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Teste Comparativo"
                }
            )
            
            response_time = time.time() - start_time
            
            print(f"✅ Sucesso - {response_time:.3f}s")
            print(f"💬 {response.choices[0].message.content[:100]}...")
            
        except Exception as e:
            print(f"❌ Falhou: {str(e)[:100]}...")

def interactive_chat():
    """Chat interativo simples com R1"""
    print("\n💬 CHAT INTERATIVO COM R1")
    print("=" * 50)
    print("Digite suas perguntas (digite 'sair' para terminar)\n")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    conversation = [
        {
            "role": "system",
            "content": "Você é um assistente especializado em trading, análise de mercado e programação. Seja conciso e direto."
        }
    ]
    
    while True:
        user_input = input("🤔 Sua pergunta: ").strip()
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
            
        if not user_input:
            continue
            
        conversation.append({"role": "user", "content": user_input})
        
        try:
            print("🔄 Pensando...")
            
            response = completion(
                model="openrouter/deepseek/deepseek-r1-0528:free",
                messages=conversation,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=1000,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Chat R1"
                }
            )
            
            assistant_response = response.choices[0].message.content
            print(f"\n🤖 R1: {assistant_response}\n")
            
            # Adicionar resposta à conversa
            conversation.append({
                "role": "assistant", 
                "content": assistant_response
            })
            
        except Exception as e:
            print(f"❌ Erro: {e}\n")

def main():
    """Função principal"""
    print("🚀 DEEPSEEK R1 - TESTE SIMPLES (SEM CACHE)")
    print("=" * 60)
    
    # Menu de opções
    while True:
        print("\n📋 MENU DE OPÇÕES:")
        print("1. 🧪 Teste Básico do R1")
        print("2. 🔄 Comparar Múltiplos Modelos")
        print("3. 💬 Chat Interativo")
        print("4. 🚪 Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == '1':
            success = test_r1_model()
            if success:
                print("\n🎉 Modelo R1 está funcionando perfeitamente!")
            else:
                print("\n⚠️ Verifique a configuração e tente novamente.")
                
        elif choice == '2':
            test_multiple_models()
            
        elif choice == '3':
            interactive_chat()
            
        elif choice == '4':
            print("👋 Até logo!")
            break
            
        else:
            print("❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()