"""
Trading Agent com Dual Proxy System
Qwen 3 Coder (Free) + DeepSeek R1 (Free)
Sistema otimizado com prompt caching
"""
import os
import json
import httpx
from dotenv import load_dotenv
import hashlib
import time

load_dotenv()

class DualProxyTradingAgent:
    """
    Trading Agent com dois proxies principais:
    1. Qwen 3 Coder - Para análise de código
    2. DeepSeek R1 - Para compliance e organização
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
            "X-Title": "Trading Agent Dual Proxy"
        }
        
        # Configuração dos dois proxies
        self.proxies = {
            "code_analyzer": {
                "model": "qwen/qwen3-coder:free",  # Qwen 3 Coder Free
                "description": "Análise de código, debugging, estrutura",
                "max_tokens": 2000,
                "temperature": 0.1
            },
            "compliance_organizer": {
                "model": "deepseek/deepseek-r1-0528:free",  # R1 Free  
                "description": "FTMO compliance, organização, estratégia",
                "max_tokens": 1500,
                "temperature": 0.05
            }
        }
        
        # Cache de prompts em memória
        self._prompt_cache = {}
        
        print("✅ Dual Proxy Trading Agent inicializado")
        print(f"🔑 API Key: {self.api_key[:15]}...")
        print("🤖 Proxy 1: Qwen 3 Coder Free (Análise de código)")
        print("🧠 Proxy 2: DeepSeek R1 Free (Compliance & Organização)")
        print(f"📊 Modelo Código: {self.proxies['code_analyzer']['model']}")
        print(f"🛡️ Modelo FTMO: {self.proxies['compliance_organizer']['model']}")
    
    def _make_request(self, proxy_type, messages, cache_key=None):
        """
        Faz requisição para o proxy especificado
        """
        if proxy_type not in self.proxies:
            return {"error": f"❌ Proxy '{proxy_type}' não encontrado"}
        
        # Verificar cache
        if cache_key and cache_key in self._prompt_cache:
            print(f"💾 Cache hit: {cache_key} ({proxy_type})")
            return self._prompt_cache[cache_key]
        
        proxy_config = self.proxies[proxy_type]
        
        data = {
            "model": proxy_config["model"],
            "messages": messages,
            "max_tokens": proxy_config["max_tokens"],
            "temperature": proxy_config["temperature"]
        }
        
        try:
            print(f"🔄 Usando {proxy_config['description']} ({proxy_config['model']})")
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Salvar no cache
                if cache_key:
                    self._prompt_cache[cache_key] = result
                    print(f"💾 Salvo no cache: {cache_key}")
                
                return result
                
        except Exception as e:
            return {"error": f"❌ Erro na requisição {proxy_type}: {e}"}
    
    def analyze_code_structure(self, code_content, filename=""):
        """
        Análise de código usando Qwen 3 Coder
        """
        cache_key = f"qwen_code_{hashlib.md5(code_content.encode()).hexdigest()[:10]}"
        
        system_prompt = """
Você é um EXPERT em análise de código de trading (MQL4/MQL5/Pine Script).
Analise a estrutura, lógica e qualidade do código fornecido.

FOQUE EM:
- Tipo de código (EA/Indicator/Script)
- Estratégia de trading identificada
- Estrutura e organização do código
- Qualidade técnica (1-10)
- Possíveis bugs ou melhorias
- Compatibilidade com diferentes brokers

Seja técnico e preciso na análise.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
ARQUIVO: {filename}

CÓDIGO:
```
{code_content}
```

ANÁLISE SOLICITADA:
1. 🔍 TIPO: EA/Indicator/Script/Library
2. 📈 ESTRATÉGIA: Qual estratégia está implementada
3. 🏗️ ESTRUTURA: Qualidade da organização do código
4. ⚙️ FUNCIONALIDADES: Principais recursos identificados
5. 🐛 ISSUES: Problemas ou pontos de atenção
6. 📊 QUALIDADE: Score 1-10 com justificativa
7. 🔧 MELHORIAS: Sugestões técnicas
8. 🏷️ TAGS: Para categorização

Responda de forma estruturada e técnica.
"""}
        ]
        
        response = self._make_request("code_analyzer", messages, cache_key)
        
        if "error" in response:
            return response["error"]
        
        return response["choices"][0]["message"]["content"]
    
    def ftmo_compliance_analysis(self, ea_code, filename=""):
        """
        Análise FTMO usando DeepSeek R1
        """
        cache_key = f"r1_ftmo_{hashlib.md5(ea_code.encode()).hexdigest()[:10]}"
        
        system_prompt = """
Você é um ESPECIALISTA em regras FTMO e compliance para prop firms.
Sua análise deve ser RIGOROSA e TÉCNICA.

REGRAS FTMO CRÍTICAS:
- Daily Loss Limit: Máximo 5% de perda por dia
- Max Loss Limit: Máximo 10% de perda total
- Daily Profit Target: Máximo 5% de lucro por dia
- Risk per Trade: 0.5-1% máximo
- Minimum Trading Days: Mínimo 10 dias
- NO MARTINGALE: Proibido multiplicar lotes
- NO GRID sem SL: Grid só com stop loss
- NO HIGH FREQUENCY: Evitar scalping agressivo
- NO NEWS TRADING: Durante eventos de alto impacto

Analise o código e seja SEVERO na avaliação.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
ANÁLISE FTMO COMPLIANCE - {filename}

CÓDIGO EA:
```
{ea_code}
```

VERIFICAÇÃO RIGOROSA:
1. ✅/❌ RISK MANAGEMENT: Controle de risco por trade
2. ✅/❌ DAILY LOSS CONTROL: Proteção perda diária 5%
3. ✅/❌ MAX LOSS CONTROL: Proteção perda total 10%
4. ✅/❌ PROFIT LIMITS: Controle lucro diário 5%
5. ✅/❌ NO MARTINGALE: Sem multiplicação de lotes
6. ✅/❌ STOP LOSS MANDATORY: SL em todas posições
7. ✅/❌ POSITION SIZE: Tamanho adequado das posições
8. ✅/❌ DRAWDOWN PROTECTION: Proteção contra DD
9. ✅/❌ TIME RESTRICTIONS: Controles de horário
10. ✅/❌ NEWS FILTER: Filtro de notícias

RESULTADO FINAL:
- ✅ FTMO APPROVED (se 100% conforme)
- ⚠️ FTMO RISK (se tem riscos menores)
- ❌ FTMO REJECTED (se não conforme)

JUSTIFICATIVA: Explique cada ponto detalhadamente.
CORREÇÕES: Se necessário, liste o que deve ser corrigido.

Seja RIGOROSO na análise!
"""}
        ]
        
        response = self._make_request("compliance_organizer", messages, cache_key)
        
        if "error" in response:
            return response["error"]
        
        return response["choices"][0]["message"]["content"]
    
    def organize_trading_files(self, file_list):
        """
        Organização de arquivos usando DeepSeek R1
        """
        cache_key = f"r1_organize_{hashlib.md5(str(file_list).encode()).hexdigest()[:10]}"
        
        system_prompt = """
Você é um ORGANIZADOR MASTER de códigos de trading.
Crie uma estrutura PROFISSIONAL e ESCALÁVEL.

ESTRUTURA OBRIGATÓRIA:
```
PROJETO_TRADING_ELITE/
├── 📁 01_EA_FTMO_READY/
│   ├── XAUUSD_Scalpers/
│   ├── EURUSD_Trend/
│   └── Multi_Symbol/
├── 📁 02_INDICATORS_SMC/
│   ├── Order_Blocks/
│   ├── Volume_Analysis/
│   └── Market_Structure/
├── 📁 03_SCRIPTS_TOOLS/
│   ├── Risk_Management/
│   ├── Analysis_Tools/
│   └── Utilities/
└── 📁 04_DOCUMENTATION/
    ├── INDEX_MASTER.md
    ├── FTMO_READY_LIST.md
    └── QUALITY_RANKINGS.md
```

NOMENCLATURA: [TIPO]_[ESTRATEGIA]_v[VER]_[MERCADO]_[ESPECIAL].[EXT]

PRIORIDADES:
1. FTMO Compliance (máximo)
2. XAUUSD specialists
3. SMC/ICT concepts
4. Risk management
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
ORGANIZAÇÃO DE ARQUIVOS TRADING:

LISTA DE ARQUIVOS:
{chr(10).join([f"- {file}" for file in file_list])}

EXECUTE:
1. 🏗️ ESTRUTURA: Crie hierarquia completa de pastas
2. 🏷️ RENOMEAÇÃO: Aplique nomenclatura rigorosa
3. 📁 CATEGORIZAÇÃO: Organize por estratégia/tipo
4. 🎯 PRIORIZAÇÃO: FTMO ready primeiro
5. 📊 RANKING: Ordene por qualidade/importância
6. 📋 DOCUMENTAÇÃO: Crie entries para INDEX
7. 🏆 ELITE SELECTION: Destaque os melhores

RESULTADO:
- Estrutura de pastas completa
- Lista renomeada com paths
- INDEX entries formatados
- Ranking de qualidade
- Recomendações de uso

Seja DETALHADO e PROFISSIONAL!
"""}
        ]
        
        response = self._make_request("compliance_organizer", messages, cache_key)
        
        if "error" in response:
            return response["error"]
        
        return response["choices"][0]["message"]["content"]
    
    def dual_analysis(self, code_content, filename=""):
        """
        Análise completa usando ambos os proxies
        """
        print("🔄 Iniciando análise dual proxy...")
        
        # 1. Análise técnica com Qwen
        print("\n1️⃣ ANÁLISE TÉCNICA (Qwen 3 Coder):")
        code_analysis = self.analyze_code_structure(code_content, filename)
        
        # 2. Análise FTMO com R1
        print("\n2️⃣ ANÁLISE FTMO (DeepSeek R1):")
        ftmo_analysis = self.ftmo_compliance_analysis(code_content, filename)
        
        return {
            "technical_analysis": code_analysis,
            "ftmo_compliance": ftmo_analysis
        }
    
    def get_proxy_stats(self):
        """
        Estatísticas dos proxies
        """
        stats = {
            "cache_size": len(self._prompt_cache),
            "cache_keys": list(self._prompt_cache.keys()),
            "proxies_configured": len(self.proxies)
        }
        return stats

def main():
    """
    Teste dos dois proxies
    """
    try:
        agent = DualProxyTradingAgent()
        print("\n" + "="*70)
        print("🎯 TESTE DUAL PROXY - QWEN 3 CODER + DEEPSEEK R1")
        print("="*70)
        
        # Código de exemplo para teste
        sample_ea = """
//+------------------------------------------------------------------+
//| EA FTMO Scalper XAUUSD                                          |
//+------------------------------------------------------------------+
extern double LotSize = 0.01;
extern int StopLoss = 50;
extern int TakeProfit = 100;
extern double MaxDailyLoss = 0.05;
extern double MaxDailyProfit = 0.05;

void OnTick() {
    // Daily DD Protection
    if (AccountEquity() / AccountBalance() <= 0.95) {
        CloseAllPositions();
        return;
    }
    
    // Daily Profit Protection  
    if (AccountEquity() / AccountBalance() >= 1.05) {
        CloseAllPositions();
        return;
    }
    
    if (OrdersTotal() == 0 && IsTradeTime()) {
        OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, 
                 Ask-StopLoss*Point, Ask+TakeProfit*Point);
    }
}

bool IsTradeTime() {
    int hour = Hour();
    return (hour >= 8 && hour <= 16); // London + NY
}

void CloseAllPositions() {
    for(int i = OrdersTotal()-1; i >= 0; i--) {
        if(OrderSelect(i, SELECT_BY_POS)) {
            OrderClose(OrderTicket(), OrderLots(), OrderClosePrice(), 3);
        }
    }
}
"""
        
        print("🧪 Testando análise dual...")
        
        # Teste 1: Análise técnica (Qwen)
        print("\n" + "="*50)
        print("🔍 TESTE 1: ANÁLISE TÉCNICA (Qwen 3 Coder)")
        print("="*50)
        tech_result = agent.analyze_code_structure(sample_ea, "EA_FTMO_Scalper_XAUUSD.mq4")
        print(tech_result[:500] + "..." if len(tech_result) > 500 else tech_result)
        
        # Teste 2: Análise FTMO (R1)
        print("\n" + "="*50)
        print("🛡️ TESTE 2: ANÁLISE FTMO (DeepSeek R1)")
        print("="*50)
        ftmo_result = agent.ftmo_compliance_analysis(sample_ea, "EA_FTMO_Scalper_XAUUSD.mq4")
        print(ftmo_result[:500] + "..." if len(ftmo_result) > 500 else ftmo_result)
        
        # Teste 3: Organização de arquivos (R1)
        print("\n" + "="*50)
        print("📁 TESTE 3: ORGANIZAÇÃO (DeepSeek R1)")
        print("="*50)
        files_to_organize = [
            "scalper_xauusd.mq4",
            "trend_follower_eurusd.mq5", 
            "volume_indicator.mq4",
            "risk_calculator.mq5"
        ]
        org_result = agent.organize_trading_files(files_to_organize)
        print(org_result[:500] + "..." if len(org_result) > 500 else org_result)
        
        # Estatísticas
        print("\n" + "="*50)
        print("📊 ESTATÍSTICAS")
        print("="*50)
        stats = agent.get_proxy_stats()
        print(f"💾 Cache entries: {stats['cache_size']}")
        print(f"🤖 Proxies ativos: {stats['proxies_configured']}")
        print(f"🔑 Cache keys: {stats['cache_keys']}")
        
        print("\n✅ TESTE DUAL PROXY CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n📝 Verifique:")
        print("1. API key no arquivo .env")
        print("2. Conexão com internet")
        print("3. Créditos OpenRouter")

if __name__ == "__main__":
    main()
