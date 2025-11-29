"""
Trading Agent com LiteLLM e OpenRouter
Exemplo de uso com prompt caching otimizado
"""

import litellm
import os
from dotenv import load_dotenv

load_dotenv()

class TradingAgent:
    def __init__(self):
        """
        Inicializa o agente de trading com LiteLLM
        """
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY não encontrada no arquivo .env")
        
        # Configurar LiteLLM
        litellm.api_key = self.api_key
        litellm.api_base = "https://openrouter.ai/api/v1"
        
        # Prompt base que será cacheado
        self.system_prompt = """
🤖 PROMPT ESPECIALIZADO: AGENTE ORGANIZADOR DE CÓDIGOS TRADING

Você é um ORGANIZADOR EXPERT em códigos de trading (MQL4/MQL5/Pine Script). 
Sua missão é criar e manter uma estrutura de arquivos LIMPA, LÓGICA e ESCALÁVEL.

FOCO PRINCIPAL:
- FTMO compliance (máxima prioridade)
- XAUUSD specialists (prioridade alta)  
- SMC/Order Blocks (prioridade alta)
- Risk management (prioridade alta)

ESTRUTURA OBRIGATÓRIA:
PROJETO_TRADING_COMPLETO/
├── 📁 EA_FTMO_XAUUSD_ELITE/
├── 📁 CODIGO_FONTE_LIBRARY/
└── 📄 MASTER_INDEX.md

NOMENCLATURA RIGOROSA: [TIPO]_[NOME]v[VERSAO][ESPECIFICO].[EXT]
Exemplos: EA_OrderBlocks_v2.1_XAUUSD_FTMO.mq5

Sempre mantenha documentação completa no INDEX com:
- Estratégia, mercado, timeframe
- Compliance FTMO (✅/❌)
- Status de teste
- Tags para busca
"""
    
    def analyze_code(self, code_content, filename=""):
        """
        Analisa código de trading e sugere organização
        """
        try:
            response = litellm.completion(
                model="openrouter/anthropic/claude-3-5-sonnet",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
Analise este código de trading e forneça:

CÓDIGO: {filename}
```
{code_content}
```

ANÁLISE SOLICITADA:
1. Tipo (EA/Indicator/Script)
2. Estratégia identificada
3. Mercado compatível
4. FTMO compliance (✅/❌)
5. Nome padronizado sugerido
6. Pasta de destino
7. Tags para INDEX
8. Status de qualidade (1-5)

Responda no formato estruturado do Agente Organizador.
"""}
                ],
                temperature=0.1,
                max_tokens=1000,
                # Ativar cache para o system prompt
                cache={"ttl": 3600}  # Cache por 1 hora
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erro na análise: {e}"
    
    def organize_files(self, file_list):
        """
        Organiza lista de arquivos de trading
        """
        try:
            response = litellm.completion(
                model="openrouter/anthropic/claude-3-5-sonnet",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
Organize estes arquivos de trading:

ARQUIVOS:
{chr(10).join(file_list)}

TAREFAS:
1. Renomear conforme padrão
2. Categorizar por estratégia  
3. Definir pasta de destino
4. Criar entries para INDEX
5. Priorizar FTMO compliance

Forneça estrutura completa de organização.
"""}
                ],
                temperature=0.1,
                max_tokens=1500,
                cache={"ttl": 3600}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erro na organização: {e}"
    
    def create_ftmo_analysis(self, ea_code):
        """
        Análise específica de compliance FTMO
        """
        try:
            response = litellm.completion(
                model="openrouter/anthropic/claude-3-5-sonnet",
                messages=[
                    {"role": "system", "content": self.system_prompt + """

REGRAS FTMO CRÍTICAS:
- Max 5% loss em 1 dia
- Max 10% loss total
- Max 5% profit por dia
- Risk per trade: 0.5-1%
- No martingale
- No grid sem stop loss
- Mínimo 10 dias trading
"""},
                    {"role": "user", "content": f"""
Analise este EA para FTMO compliance:

```
{ea_code}
```

VERIFICAR:
1. Risk management (✅/❌)
2. Daily drawdown control (✅/❌) 
3. Max drawdown control (✅/❌)
4. Profit limits (✅/❌)
5. Martingale detection (✅/❌)
6. Stop loss obrigatório (✅/❌)

RESULTADO: ✅ FTMO READY ou ❌ NÃO COMPLIANCE
"""}
                ],
                temperature=0.1,
                max_tokens=800,
                cache={"ttl": 1800}  # Cache por 30 minutos
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erro na análise FTMO: {e}"

def main():
    """
    Exemplo de uso do Trading Agent
    """
    try:
        agent = TradingAgent()
        print("✅ Trading Agent inicializado com sucesso!")
        print("🔗 Conectado ao OpenRouter via LiteLLM")
        print("💾 Prompt caching ativado")
        print("\n" + "="*50)
        
        # Exemplo de uso
        sample_code = """
//+------------------------------------------------------------------+
//| Expert Advisor Example                                           |
//+------------------------------------------------------------------+
extern double LotSize = 0.01;
extern int StopLoss = 50;
extern int TakeProfit = 100;

void OnTick() {
    if (OrdersTotal() == 0) {
        OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, Ask-StopLoss*Point, Ask+TakeProfit*Point);
    }
}
"""
        
        print("🔍 Analisando código exemplo...")
        result = agent.analyze_code(sample_code, "EA_Example.mq4")
        print("\n📋 RESULTADO DA ANÁLISE:")
        print(result)
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n📝 Para configurar:")
        print("1. Obtenha API key em: https://openrouter.ai/keys")
        print("2. Crie .env com: OPENROUTER_API_KEY=sua_chave")

if __name__ == "__main__":
    main()
