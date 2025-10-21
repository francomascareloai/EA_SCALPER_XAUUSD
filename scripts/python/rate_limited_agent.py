"""
Dual Proxy com Rate Limiting
Sistema de controle de requisições para evitar 429
"""
import os
import json
import httpx
from dotenv import load_dotenv
import hashlib
import time
import random

load_dotenv()

class RateLimitedDualProxy:
    """
    Trading Agent com rate limiting para evitar 429
    """
    
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY não encontrada no arquivo .env")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/trading-organizer",
            "X-Title": "Rate Limited Trading Agent"
        }
        
        # Modelos gratuitos
        self.qwen_model = "qwen/qwen3-coder:free"
        self.r1_model = "deepseek/deepseek-r1-0528:free"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0  # 2 segundos entre requisições
        self.request_count = 0
        
        # Cache
        self._cache = {}
        
        print("✅ Rate Limited Dual Proxy inicializado")
        print(f"🔑 API Key: {self.api_key[:15]}...")
        print(f"🤖 Qwen Model: {self.qwen_model}")
        print(f"🧠 R1 Model: {self.r1_model}")
        print(f"⏱️ Rate Limit: {self.min_delay}s entre requisições")
    
    def _wait_for_rate_limit(self):
        """
        Aguarda o tempo necessário para não exceder rate limit
        """
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            print(f"⏳ Aguardando {sleep_time:.1f}s (rate limiting)...")
            time.sleep(sleep_time)
    
    def _make_safe_request(self, model, messages, max_tokens=1000, temperature=0.1):
        """
        Faz requisição com rate limiting e retry
        """
        self._wait_for_rate_limit()
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.request_count += 1
                print(f"🔄 Requisição #{self.request_count} para {model} (tentativa {attempt + 1})")
                
                with httpx.Client() as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=data,
                        timeout=45.0
                    )
                    
                    self.last_request_time = time.time()
                    
                    if response.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"⚠️ Rate limit hit, aguardando {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    result = response.json()
                    print("✅ Requisição bem-sucedida")
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"❌ Erro tentativa {attempt + 1}: {e}")
                    print(f"🔄 Tentando novamente em {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {"error": f"❌ Falha após {max_retries} tentativas: {e}"}
        
        return {"error": "❌ Todas as tentativas falharam"}
    
    def quick_code_analysis(self, code_snippet, filename=""):
        """
        Análise rápida com Qwen 3 Coder
        """
        cache_key = f"qwen_{hashlib.md5(code_snippet[:100].encode()).hexdigest()[:8]}"
        
        if cache_key in self._cache:
            print(f"💾 Cache hit: {cache_key}")
            return self._cache[cache_key]
        
        messages = [
            {"role": "system", "content": """
Você é um expert em códigos de trading MQL4/MQL5.
Faça análise RÁPIDA e CONCISA.

FORMATO RESPOSTA:
1. TIPO: [EA/Indicator/Script]
2. ESTRATÉGIA: [Nome da estratégia]
3. QUALIDADE: [Score 1-10]
4. PRINCIPAIS RECURSOS: [Lista pontos principais]
5. ISSUES: [Problemas encontrados]

Seja DIRETO e TÉCNICO.
"""},
            {"role": "user", "content": f"""
Analise este código trading:

ARQUIVO: {filename}
```
{code_snippet[:1000]}  
```

Análise rápida seguindo o formato.
"""}
        ]
        
        response = self._make_safe_request(
            self.qwen_model,
            messages,
            max_tokens=500,
            temperature=0.1
        )
        
        if "error" in response:
            return response["error"]
        
        content = response["choices"][0]["message"]["content"]
        self._cache[cache_key] = content
        return content
    
    def quick_ftmo_check(self, ea_code, filename=""):
        """
        Verificação FTMO rápida com R1
        """
        cache_key = f"r1_{hashlib.md5(ea_code[:100].encode()).hexdigest()[:8]}"
        
        if cache_key in self._cache:
            print(f"💾 Cache hit: {cache_key}")
            return self._cache[cache_key]
        
        messages = [
            {"role": "system", "content": """
Analise RAPIDAMENTE se o EA é FTMO compliant.

REGRAS FTMO:
- Max 5% perda diária
- Max 10% perda total
- Stop Loss obrigatório
- Sem martingale
- Risk management adequado

RESPONDA:
✅ FTMO APPROVED - se 100% conforme
⚠️ FTMO RISK - se tem riscos
❌ FTMO REJECTED - se não conforme

Seja CONCISO e DIRETO.
"""},
            {"role": "user", "content": f"""
FTMO check rápido:

ARQUIVO: {filename}
```
{ea_code[:800]}
```

Resultado: APPROVED/RISK/REJECTED com motivo breve.
"""}
        ]
        
        response = self._make_safe_request(
            self.r1_model,
            messages,
            max_tokens=300,
            temperature=0.05
        )
        
        if "error" in response:
            return response["error"]
        
        content = response["choices"][0]["message"]["content"]
        self._cache[cache_key] = content
        return content
    
    def test_both_models(self):
        """
        Teste básico dos dois modelos
        """
        print("\n🧪 TESTE BÁSICO DOS MODELOS")
        print("="*50)
        
        # Código de teste simples
        test_code = """
extern double LotSize = 0.01;
extern int StopLoss = 50;

void OnTick() {
    if (OrdersTotal() == 0) {
        OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, 
                 Ask-StopLoss*Point, Ask+50*Point);
    }
}
"""
        
        # Teste 1: Qwen 3 Coder
        print("\n1️⃣ TESTANDO QWEN 3 CODER:")
        print("-" * 30)
        qwen_result = self.quick_code_analysis(test_code, "test_ea.mq4")
        print(qwen_result)
        
        # Teste 2: DeepSeek R1
        print("\n2️⃣ TESTANDO DEEPSEEK R1:")
        print("-" * 30)
        r1_result = self.quick_ftmo_check(test_code, "test_ea.mq4")
        print(r1_result)
        
        # Stats
        print("\n📊 ESTATÍSTICAS:")
        print(f"💾 Cache entries: {len(self._cache)}")
        print(f"🔢 Total requests: {self.request_count}")
        
        return {"qwen": qwen_result, "r1": r1_result}

def main():
    """
    Teste com rate limiting
    """
    try:
        agent = RateLimitedDualProxy()
        print("\n" + "="*60)
        print("🚀 TESTE RATE LIMITED DUAL PROXY")
        print("="*60)
        
        # Executar teste
        results = agent.test_both_models()
        
        print("\n✅ TESTE CONCLUÍDO!")
        print("\nRESULTADOS RESUMIDOS:")
        print("="*40)
        print(f"🤖 Qwen 3 Coder: {'✅ OK' if 'error' not in results['qwen'] else '❌ Erro'}")
        print(f"🧠 DeepSeek R1: {'✅ OK' if 'error' not in results['r1'] else '❌ Erro'}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n📝 Verifique:")
        print("1. API key válida no .env")
        print("2. Conexão com internet")
        print("3. Limites de rate da OpenRouter")

if __name__ == "__main__":
    main()
