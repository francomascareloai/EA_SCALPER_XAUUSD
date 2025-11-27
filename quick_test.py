"""
Teste rápido - Execute: python quick_test.py
"""
import os
import litellm

# Definir API key diretamente (ou use variável de ambiente)
# os.environ["ANTHROPIC_API_KEY"] = "sua_chave_aqui"

print("🔄 Testando conexão...")

try:
    resp = litellm.completion(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Diga: teste OK"}],
        max_tokens=20
    )
    print(f"✅ Funcionando! Resposta: {resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n💡 Verifique se ANTHROPIC_API_KEY está configurada:")
    print("   set ANTHROPIC_API_KEY=sk-ant-sua-chave")
