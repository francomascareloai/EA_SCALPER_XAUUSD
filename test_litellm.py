"""
Script de teste para LiteLLM
Execute: python test_litellm.py
"""

import os
import sys

def test_installation():
    """Verificar instalação do LiteLLM"""
    print("\n🔍 Verificando instalação...")
    try:
        import litellm
        print(f"   ✅ LiteLLM versão: {litellm.__version__}")
        return True
    except ImportError:
        print("   ❌ LiteLLM não encontrado!")
        return False

def test_claude():
    """Testar Claude/Anthropic"""
    import litellm
    
    print("\n🤖 Testando Claude (Anthropic)...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("   ⚠️ ANTHROPIC_API_KEY não configurada")
        print("   💡 Configure com: set ANTHROPIC_API_KEY=sua_chave")
        return False
    
    print(f"   🔑 API Key encontrada: {api_key[:10]}...")
    
    try:
        response = litellm.completion(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Diga apenas: Olá, estou funcionando!"}],
            max_tokens=50
        )
        result = response.choices[0].message.content
        print(f"   ✅ Resposta: {result}")
        return True
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False

def test_openai_compatible():
    """Testar usando formato OpenAI"""
    import litellm
    
    print("\n🔄 Testando formato OpenAI compatível...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("   ⚠️ Pulando - sem API key")
        return False
    
    try:
        # LiteLLM converte automaticamente
        response = litellm.completion(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": "Quanto é 2+2? Responda só o número."}
            ],
            max_tokens=10
        )
        result = response.choices[0].message.content
        print(f"   ✅ Resposta: {result}")
        return True
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False

def test_streaming():
    """Testar streaming"""
    import litellm
    
    print("\n📡 Testando streaming...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("   ⚠️ Pulando - sem API key")
        return False
    
    try:
        print("   Resposta: ", end="", flush=True)
        response = litellm.completion(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Conte de 1 a 5 separado por vírgula."}],
            max_tokens=50,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n   ✅ Streaming funcionando!")
        return True
    except Exception as e:
        print(f"\n   ❌ Erro: {e}")
        return False

def show_available_models():
    """Mostrar modelos disponíveis"""
    print("\n📋 Modelos Claude disponíveis no LiteLLM:")
    models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514", 
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]
    for m in models:
        print(f"   • {m}")

def main():
    print("=" * 55)
    print("   🧪 TESTE DO LITELLM - EA_SCALPER_XAUUSD")
    print("=" * 55)
    
    if not test_installation():
        print("\n❌ Instale com: pip install litellm")
        sys.exit(1)
    
    show_available_models()
    
    results = {
        "Claude": test_claude(),
        "OpenAI Format": test_openai_compatible(),
        "Streaming": test_streaming(),
    }
    
    print("\n" + "=" * 55)
    print("   📊 RESUMO DOS TESTES")
    print("=" * 55)
    
    for name, passed in results.items():
        status = "✅ OK" if passed else "❌ FALHOU"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 Todos os testes passaram!" if all_passed else "⚠️ Alguns testes falharam"))
    print("=" * 55)

if __name__ == "__main__":
    main()
