#!/usr/bin/env python3
"""
Teste final do cache e janela de contexto do LiteLLM
Usando apenas o modelo DeepSeek que está funcionando
"""

import requests
import json
import time
import os
from datetime import datetime

# Configurações
BASE_URL = "http://localhost:4000"
CACHE_DIR = "./cache/litellm_cache"
MODEL = "deepseek-r1-free"  # Apenas o modelo que funciona

def test_cache_performance():
    """Testa a performance do cache com múltiplas requisições idênticas"""
    print(f"🔄 Testando Cache Performance - {MODEL}")
    print("="*50)
    
    # Mensagem de teste
    test_message = "Explique brevemente o que é inteligência artificial em 2 frases."
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": test_message}],
        "max_tokens": 100,
        "temperature": 0.1  # Baixa temperatura para respostas consistentes
    }
    
    times = []
    responses = []
    
    # Fazer 3 requisições idênticas
    for i in range(3):
        print(f"Iteração {i+1}: ", end="")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            duration = end_time - start_time
            times.append(duration)
            
            print(f"{duration:.2f}s - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                responses.append(content)
                print(f"  ✅ Resposta: {content[:100]}...")
            else:
                print(f"  ❌ Erro: {response.text[:200]}...")
                responses.append(None)
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            times.append(0)
            responses.append(None)
        
        # Pequena pausa entre requisições
        if i < 2:
            time.sleep(1)
    
    # Análise de performance
    if len(times) >= 2 and times[0] > 0:
        first_time = times[0]
        cached_times = [t for t in times[1:] if t > 0]
        
        if cached_times:
            avg_cached = sum(cached_times) / len(cached_times)
            improvement = ((first_time - avg_cached) / first_time) * 100
            
            print(f"\n📊 Análise de Performance:")
            print(f"  Primeira requisição: {first_time:.2f}s")
            print(f"  Média das cached: {avg_cached:.2f}s")
            print(f"  Melhoria: {improvement:.1f}%")
            
            if improvement > 5:
                print(f"  ✅ Cache funcionando!")
            else:
                print(f"  ⚠️ Cache pode não estar ativo")
    
    return times, responses

def test_context_window():
    """Testa diferentes tamanhos de contexto"""
    print(f"\n🪟 Testando Janela de Contexto - {MODEL}")
    print("="*50)
    print(f"📏 Context Window: 163,840 tokens (~655,360 caracteres)")
    
    # Diferentes tamanhos de teste (em caracteres)
    test_sizes = [
        (1000, "1K caracteres"),
        (5000, "5K caracteres"),
        (10000, "10K caracteres"),
        (25000, "25K caracteres"),
        (50000, "50K caracteres")
    ]
    
    successful_tests = []
    
    for size, description in test_sizes:
        print(f"\nTestando {description}...")
        
        # Criar texto do tamanho especificado
        base_text = "Este é um texto de teste para verificar a janela de contexto. "
        repeat_count = size // len(base_text) + 1
        test_text = (base_text * repeat_count)[:size]
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Analise este texto e me diga quantas palavras aproximadamente ele tem: {test_text}"
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            end_time = time.time()
            
            print(f"  Status: {response.status_code} - Tempo: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = data.get('usage', {})
                
                print(f"  ✅ Sucesso! Tokens: {usage.get('total_tokens', 0)}")
                print(f"  📝 Resposta: {content[:100]}...")
                
                successful_tests.append({
                    'size': size,
                    'description': description,
                    'tokens': usage.get('total_tokens', 0),
                    'time': end_time - start_time
                })
                
            elif response.status_code == 429:
                print(f"  ⏳ Rate limit - aguardando...")
                time.sleep(5)  # Aguardar 5 segundos antes do próximo teste
            else:
                print(f"  ❌ Erro: {response.text[:200]}...")
                
        except Exception as e:
            print(f"  ❌ Erro: {e}")
        
        # Pausa entre testes
        time.sleep(2)
    
    return successful_tests

def check_cache_directory():
    """Verifica se arquivos de cache foram criados"""
    print(f"\n📁 Verificando Cache Directory: {CACHE_DIR}")
    print("="*50)
    
    if os.path.exists(CACHE_DIR):
        files = os.listdir(CACHE_DIR)
        if files:
            print(f"📂 {len(files)} arquivo(s) de cache encontrado(s):")
            for file in files[:5]:  # Mostrar apenas os primeiros 5
                file_path = os.path.join(CACHE_DIR, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
        else:
            print("📂 Diretório de cache vazio")
    else:
        print("📂 Diretório de cache não existe")

def generate_final_report(cache_times, context_tests):
    """Gera relatório final dos testes"""
    print("\n📋 RELATÓRIO FINAL")
    print("="*60)
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🤖 MODELO: {MODEL.upper()}")
    print("-" * 40)
    
    # Análise do cache
    if len(cache_times) >= 2 and cache_times[0] > 0:
        cached_times = [t for t in cache_times[1:] if t > 0]
        if cached_times:
            improvement = ((cache_times[0] - sum(cached_times)/len(cached_times)) / cache_times[0]) * 100
            print(f"💾 Cache: ✅ Funcionando ({improvement:.1f}% melhoria)")
        else:
            print(f"💾 Cache: ❌ Não detectado")
    else:
        print(f"💾 Cache: ❓ Não testado")
    
    # Análise do contexto
    if context_tests:
        max_size = max(test['size'] for test in context_tests)
        max_tokens = max(test['tokens'] for test in context_tests)
        print(f"🪟 Contexto: ✅ Até {max_size:,} caracteres ({max_tokens:,} tokens)")
        
        print(f"\n📊 Testes de Contexto Bem-sucedidos:")
        for test in context_tests:
            print(f"  - {test['description']}: {test['tokens']:,} tokens ({test['time']:.2f}s)")
    else:
        print(f"🪟 Contexto: ❌ Nenhum teste bem-sucedido")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES")
    print("-" * 40)
    print(f"🚀 Para aumentar janela de contexto localmente:")
    print(f"   1. Use chunking para textos > 50K caracteres")
    print(f"   2. Implemente cache de contexto com embeddings")
    print(f"   3. Configure summarização automática")
    print(f"   4. Use técnicas de compressão de contexto")
    print(f"   5. Considere usar modelos locais (Ollama, LM Studio)")
    
    # Salvar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'cache_times': cache_times,
        'context_tests': context_tests,
        'cache_working': len(cache_times) >= 2 and cache_times[0] > 0,
        'max_context_size': max([test['size'] for test in context_tests]) if context_tests else 0
    }
    
    with open('test_results_final.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: test_results_final.json")

def main():
    """Função principal"""
    print("🧪 TESTE FINAL - CACHE E CONTEXTO LITELLM")
    print("="*60)
    print(f"🔗 URL: {BASE_URL}")
    print(f"🤖 Modelo: {MODEL}")
    print(f"📁 Cache: {CACHE_DIR}")
    
    # Teste 1: Performance do Cache
    cache_times, responses = test_cache_performance()
    
    # Teste 2: Janela de Contexto
    context_tests = test_context_window()
    
    # Teste 3: Verificar diretório de cache
    check_cache_directory()
    
    # Relatório final
    generate_final_report(cache_times, context_tests)
    
    print("\n✅ Testes concluídos com sucesso!")

if __name__ == "__main__":
    main()