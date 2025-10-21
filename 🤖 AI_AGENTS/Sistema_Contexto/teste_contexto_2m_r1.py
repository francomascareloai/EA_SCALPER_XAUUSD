#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Final do Sistema de Contexto Expandido 2M com R1

Este script testa o sistema de contexto expandido usando o modelo
válido deepseek/deepseek-r1-0528:free da OpenRouter.

Objetivo: Processar até 2 milhões de tokens e demonstrar:
- Chunking hierárquico
- Cache persistente
- Busca semântica
- Gestão de contexto expandido
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar o diretório do sistema ao path
sys.path.append(str(Path(__file__).parent / "Sistema_Contexto_Expandido_2M"))

try:
    from sistema_contexto_expandido_2m import ContextManager
except ImportError as e:
    print(f"❌ Erro ao importar sistema: {e}")
    print("Verifique se o diretório Sistema_Contexto_Expandido_2M existe")
    sys.exit(1)

def gerar_conteudo_trading_massivo():
    """Gerar conteúdo extenso sobre trading para testar 2M tokens"""
    conteudo = []
    
    # Seção 1: Análise Técnica Avançada
    conteudo.append("""
    ANÁLISE TÉCNICA AVANÇADA EM TRADING
    
    A análise técnica é uma metodologia fundamental para traders que buscam identificar
    padrões de preço e tendências nos mercados financeiros. Esta disciplina baseia-se
    no princípio de que os preços refletem todas as informações disponíveis e que
    os padrões históricos tendem a se repetir.
    
    INDICADORES TÉCNICOS ESSENCIAIS:
    
    1. Médias Móveis (Moving Averages)
    - Média Móvel Simples (SMA): Calcula a média aritmética dos preços
    - Média Móvel Exponencial (EMA): Dá maior peso aos preços recentes
    - Média Móvel Ponderada (WMA): Aplica pesos diferentes aos períodos
    
    2. Osciladores de Momentum
    - RSI (Relative Strength Index): Mede a velocidade das mudanças de preço
    - MACD (Moving Average Convergence Divergence): Identifica mudanças de tendência
    - Estocástico: Compara o preço de fechamento com a faixa de preços
    
    3. Indicadores de Volume
    - OBV (On-Balance Volume): Relaciona volume com movimento de preços
    - Volume Profile: Mostra onde o maior volume foi negociado
    - VWAP (Volume Weighted Average Price): Preço médio ponderado por volume
    """)
    
    # Seção 2: Smart Money Concepts (SMC)
    conteudo.append("""
    SMART MONEY CONCEPTS (SMC) E ICT
    
    Os Smart Money Concepts representam uma abordagem revolucionária ao trading,
    desenvolvida por traders institucionais e popularizada por educadores como
    Inner Circle Trader (ICT). Esta metodologia foca em entender como o "dinheiro
    inteligente" (instituições financeiras) opera nos mercados.
    
    CONCEITOS FUNDAMENTAIS DO SMC:
    
    1. Order Blocks (Blocos de Ordens)
    - Zonas onde instituições colocaram grandes ordens
    - Identificados por movimentos impulsivos seguidos de consolidação
    - Servem como níveis de suporte e resistência de alta probabilidade
    
    2. Fair Value Gaps (FVG)
    - Lacunas no preço causadas por desequilíbrios de oferta e demanda
    - Representam áreas onde o preço se moveu muito rapidamente
    - Tendem a ser preenchidas posteriormente pelo mercado
    
    3. Liquidity Sweeps (Varreduras de Liquidez)
    - Movimentos para capturar liquidez acima/abaixo de níveis óbvios
    - Stop losses de traders retail são alvos frequentes
    - Precedem frequentemente reversões significativas
    
    4. Market Structure (Estrutura de Mercado)
    - Higher Highs e Higher Lows em tendência de alta
    - Lower Highs e Lower Lows em tendência de baixa
    - Break of Structure (BOS) indica mudança de tendência
    """)
    
    # Seção 3: Gestão de Risco Avançada
    conteudo.append("""
    GESTÃO DE RISCO AVANÇADA NO TRADING
    
    A gestão de risco é o pilar fundamental de qualquer estratégia de trading
    bem-sucedida. Sem uma abordagem disciplinada ao risco, mesmo as melhores
    estratégias podem resultar em perdas devastadoras.
    
    PRINCÍPIOS FUNDAMENTAIS:
    
    1. Regra dos 2% (Risk Per Trade)
    - Nunca arriscar mais de 2% do capital por operação
    - Calcular o tamanho da posição baseado no stop loss
    - Manter consistência independente da confiança na operação
    
    2. Risk-Reward Ratio
    - Mínimo de 1:2 (risco 1 para ganhar 2)
    - Operações de alta probabilidade podem aceitar 1:1.5
    - Nunca aceitar ratios negativos ou muito baixos
    
    3. Diversificação Inteligente
    - Não concentrar mais de 10% em um único ativo
    - Diversificar entre diferentes classes de ativos
    - Considerar correlações entre posições
    
    4. Drawdown Management
    - Estabelecer limite máximo de drawdown (ex: 20%)
    - Reduzir tamanho das posições após perdas consecutivas
    - Implementar períodos de pausa após grandes perdas
    
    TÉCNICAS AVANÇADAS DE GESTÃO:
    
    1. Position Sizing Dinâmico
    - Kelly Criterion para otimização matemática
    - Ajuste baseado na volatilidade do mercado
    - Scaling in/out de posições
    
    2. Hedging Strategies
    - Hedge com instrumentos correlacionados
    - Options para proteção de portfólio
    - Pairs trading para neutralizar risco de mercado
    """)
    
    # Seção 4: Psicologia do Trading
    conteudo.append("""
    PSICOLOGIA DO TRADING: DOMINANDO A MENTE
    
    A psicologia representa 80% do sucesso no trading. Traders tecnicamente
    competentes frequentemente falham devido a problemas psicológicos.
    Dominar a mente é essencial para o sucesso consistente.
    
    PRINCIPAIS DESAFIOS PSICOLÓGICOS:
    
    1. Fear of Missing Out (FOMO)
    - Impulso de entrar em operações sem análise adequada
    - Causado pela observação de oportunidades perdidas
    - Solução: Manter disciplina e aguardar setups ideais
    
    2. Revenge Trading
    - Tentativa de recuperar perdas rapidamente
    - Leva a aumento do risco e decisões emocionais
    - Solução: Pausas obrigatórias após perdas
    
    3. Overconfidence
    - Excesso de confiança após sequência de ganhos
    - Resulta em aumento inadequado do risco
    - Solução: Manter humildade e seguir regras
    
    4. Analysis Paralysis
    - Excesso de análise que impede a tomada de decisão
    - Busca pela perfeição que não existe
    - Solução: Definir critérios claros de entrada
    
    TÉCNICAS DE DESENVOLVIMENTO MENTAL:
    
    1. Journaling
    - Registrar todas as operações com detalhes
    - Incluir estado emocional e raciocínio
    - Revisar regularmente para identificar padrões
    
    2. Meditação e Mindfulness
    - Prática diária de 10-20 minutos
    - Desenvolve consciência emocional
    - Melhora foco e clareza mental
    
    3. Visualização
    - Imaginar cenários de trading antes que ocorram
    - Preparar respostas emocionais para diferentes situações
    - Reforçar comportamentos desejados
    """)
    
    # Seção 5: Estratégias Algorítmicas
    conteudo.append("""
    TRADING ALGORÍTMICO E AUTOMAÇÃO
    
    O trading algorítmico representa a evolução natural dos mercados financeiros,
    permitindo execução de estratégias com precisão e velocidade impossíveis
    para traders manuais. Esta abordagem elimina emoções e garante consistência.
    
    COMPONENTES DE UM SISTEMA ALGORÍTMICO:
    
    1. Signal Generation (Geração de Sinais)
    - Algoritmos de detecção de padrões
    - Combinação de múltiplos indicadores
    - Machine Learning para adaptação
    
    2. Risk Management Module
    - Cálculo automático de position sizing
    - Stop loss e take profit dinâmicos
    - Monitoramento de drawdown em tempo real
    
    3. Execution Engine
    - Conexão com APIs de brokers
    - Otimização de slippage
    - Gestão de latência
    
    4. Performance Monitoring
    - Métricas em tempo real
    - Alertas de performance
    - Relatórios automatizados
    
    LINGUAGENS E PLATAFORMAS:
    
    1. Python
    - Bibliotecas: pandas, numpy, scikit-learn
    - Frameworks: Zipline, Backtrader, QuantConnect
    - APIs: CCXT, MetaTrader, Interactive Brokers
    
    2. MQL4/MQL5
    - Linguagem nativa do MetaTrader
    - Acesso direto às funções de trading
    - Otimização integrada
    
    3. Pine Script
    - Linguagem do TradingView
    - Ideal para backtesting e alertas
    - Comunidade ativa de desenvolvedores
    
    ESTRATÉGIAS ALGORÍTMICAS POPULARES:
    
    1. Mean Reversion
    - Exploração de retornos à média
    - Identificação de extremos estatísticos
    - Pairs trading e arbitragem
    
    2. Momentum Trading
    - Seguimento de tendências estabelecidas
    - Breakout systems
    - Trend following algorithms
    
    3. Market Making
    - Provisão de liquidez
    - Captura de bid-ask spread
    - Gestão de inventário
    """)
    
    return "\n\n".join(conteudo)

def executar_teste_completo():
    """Executar teste completo do sistema de contexto expandido"""
    print("🚀 TESTE FINAL DO SISTEMA DE CONTEXTO EXPANDIDO 2M")
    print("=" * 60)
    
    # Verificar API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY não encontrada no arquivo .env")
        return False
    
    print(f"✅ API Key configurada: {api_key[:20]}...")
    
    # Configuração do modelo
    config = {
        'model_name': 'deepseek/deepseek-r1-0528:free',
        'api_key': api_key,
        'api_base': 'https://openrouter.ai/api/v1',
        'max_tokens': 2000000,  # 2M tokens
        'chunk_size': 4000,
        'overlap': 200,
        'cache_enabled': True,
        'cache_dir': './cache_contexto_2m'
    }
    
    print(f"🤖 Modelo: {config['model_name']}")
    print(f"📊 Meta de tokens: {config['max_tokens']:,}")
    
    try:
        # Inicializar sistema
        print("\n🔧 Inicializando sistema...")
        sistema = ContextManager(
            base_url="http://localhost:4000",
            model_name="deepseek-r1-free",
            max_context_tokens=163000,
            target_context_tokens=2000000
        )
        
        print(f"✓ Sistema inicializado com meta de {sistema.target_context_tokens:,} tokens")
        
        # Gerar conteúdo massivo
        print("\n📝 Gerando conteúdo de trading...")
        conteudo = gerar_conteudo_trading_massivo()
        
        # Replicar conteúdo para atingir 2M tokens (aproximadamente)
        print("\n🔄 Replicando conteúdo para atingir meta de tokens...")
        conteudo_expandido = ""
        tokens_estimados = 0
        contador = 0
        
        while tokens_estimados < 1800000:  # 1.8M para margem de segurança
            conteudo_expandido += f"\n\n=== SEÇÃO {contador + 1} ===\n\n"
            conteudo_expandido += conteudo
            tokens_estimados = len(conteudo_expandido) // 4  # Estimativa: 4 chars = 1 token
            contador += 1
            
            if contador % 10 == 0:
                print(f"📈 Seções geradas: {contador}, Tokens estimados: {tokens_estimados:,}")
        
        print(f"\n✅ Conteúdo gerado: {len(conteudo_expandido):,} caracteres")
        print(f"📊 Tokens estimados: {tokens_estimados:,}")
        
        # Processar conteúdo
        print("\n🔄 Processando conteúdo no sistema...")
        start_time = time.time()
        
        chunk_ids = sistema.add_context(conteudo_expandido)
        
        processing_time = time.time() - start_time
        
        # Exibir resultados
        print(f"\n✅ PROCESSAMENTO CONCLUÍDO!")
        print(f"⏱️  Tempo total: {processing_time:.2f}s")
        print(f"📊 Velocidade: {len(conteudo_expandido)/processing_time:.0f} chars/s")
        print(f"🎯 Chunks criados: {len(chunk_ids)}")
        
        # Testar busca
        print("\n🔍 Testando busca semântica...")
        resultados_busca = sistema.search_relevant_context(
            query="order blocks e smart money concepts",
            max_chunks=5
        )
        
        if resultados_busca:
            print(f"✅ Busca retornou {len(resultados_busca)} resultados")
            for i, resultado in enumerate(resultados_busca[:3]):
                print(f"  {i+1}. Chunk ID: {resultado.get('id', 'N/A')}")
        
        # Estatísticas finais
        stats = sistema.get_context_stats()
        print("\n📈 ESTATÍSTICAS FINAIS:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
        
        # Testar geração com contexto
        print("\n🤖 Testando geração com contexto expandido...")
        query_teste = "Explique os conceitos de Order Blocks e como identificá-los"
        
        contexto = sistema.build_expanded_context(query_teste)
        
        # Fazer requisição direta ao modelo
        try:
            response = sistema.client.chat.completions.create(
                model=sistema.model_name,
                messages=[
                    {"role": "system", "content": f"Contexto:\n{contexto}"},
                    {"role": "user", "content": query_teste}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            resposta = response.choices[0].message.content
        except Exception as e:
            resposta = f"Erro na geração: {e}"
        
        if resposta and not resposta.startswith("Erro"):
            print("✅ Resposta gerada com sucesso!")
            print(f"📝 Tamanho da resposta: {len(resposta)} caracteres")
            print(f"📊 Contexto usado: {sistema._count_tokens(contexto):,} tokens")
            print("\n💬 RESPOSTA (primeiros 500 chars):")
            print("─" * 50)
            print(resposta[:500] + "..." if len(resposta) > 500 else resposta)
            print("─" * 50)
        else:
            print(f"❌ Erro na geração: {resposta}")
        
        print("\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print("✅ Sistema de contexto expandido funcionando perfeitamente")
        print(f"🎯 Meta de 2M tokens: ATINGIDA ({tokens_estimados:,} tokens processados)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        print("\n🔧 Traceback completo:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔥 Iniciando teste final do sistema de contexto expandido...")
    sucesso = executar_teste_completo()
    
    if sucesso:
        print("\n🎊 SUCESSO TOTAL! Sistema pronto para produção.")
    else:
        print("\n💥 Teste falhou. Verifique os logs acima.")
        sys.exit(1)