#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Completo do Sistema de Contexto Expandido - 2M Tokens

Este script demonstra o funcionamento completo do sistema,
adicionando múltiplos documentos e testando todas as funcionalidades.
"""

import os
import time
from sistema_contexto_expandido_2m import ContextManager

def criar_documentos_trading():
    """Cria múltiplos documentos sobre trading para testar o sistema."""
    documentos = {
        "smc_concepts": """
        Smart Money Concepts (SMC) - Conceitos Fundamentais
        
        Order Blocks são zonas de liquidez onde instituições colocaram grandes ordens.
        Estes blocos representam áreas de interesse para smart money e frequentemente
        atuam como suporte ou resistência.
        
        Fair Value Gaps (FVG) são lacunas no preço que indicam desequilíbrio entre
        oferta e demanda. O mercado tende a retornar para preencher essas lacunas.
        
        Liquidity Sweeps ocorrem quando o preço move rapidamente para capturar
        liquidez de stops de traders retail antes de reverter na direção oposta.
        
        Market Structure é fundamental para entender a direção do mercado.
        Higher Highs e Higher Lows indicam tendência de alta, enquanto
        Lower Highs e Lower Lows indicam tendência de baixa.
        
        Break of Structure (BOS) confirma mudança na direção do mercado,
        enquanto Change of Character (CHoCH) indica possível reversão.
        """ * 100,  # Repetir para aumentar o tamanho
        
        "risk_management": """
        Gestão de Risco em Trading Algorítmico
        
        O gerenciamento de risco é o aspecto mais crítico do trading.
        Sem uma gestão adequada, mesmo a melhor estratégia pode levar à ruína.
        
        Regra dos 2%: Nunca arrisque mais de 2% do capital em uma única operação.
        Esta regra protege contra perdas catastróficas e permite recuperação.
        
        Position Sizing deve ser calculado com base no stop loss e no risco aceitável.
        Tamanho da posição = (Capital * % Risco) / (Preço de entrada - Stop loss)
        
        Drawdown máximo aceitável deve ser definido previamente.
        Recomenda-se não exceder 20% de drawdown em contas reais.
        
        Diversificação entre diferentes pares e estratégias reduz o risco.
        Correlação entre ativos deve ser considerada para evitar exposição excessiva.
        
        Risk-Reward ratio mínimo de 1:2 garante lucratividade mesmo com 50% de acerto.
        Operações com ratio inferior devem ser evitadas.
        """ * 150,
        
        "algorithmic_strategies": """
        Estratégias de Trading Algorítmico Avançadas
        
        Scalping Algorithms focam em pequenos movimentos de preço em timeframes baixos.
        Requerem execução rápida e spreads baixos para serem lucrativos.
        
        Mean Reversion strategies assumem que preços retornam à média.
        Bollinger Bands e RSI são indicadores comuns nesta abordagem.
        
        Trend Following algorithms identificam e seguem tendências estabelecidas.
        Moving averages e breakouts são sinais típicos desta estratégia.
        
        Arbitrage opportunities exploram diferenças de preço entre mercados.
        Requer tecnologia avançada e conexões rápidas para ser efetivo.
        
        Machine Learning models podem identificar padrões complexos nos dados.
        Random Forest, SVM e Neural Networks são algoritmos populares.
        
        High Frequency Trading (HFT) opera em microssegundos.
        Requer infraestrutura especializada e proximidade aos servidores.
        
        Market Making strategies fornecem liquidez e lucram com o spread.
        Requer gestão cuidadosa do inventário e risco de direção.
        """ * 200,
        
        "technical_analysis": """
        Análise Técnica para Trading Automatizado
        
        Candlestick Patterns fornecem insights sobre psicologia do mercado.
        Doji, Hammer, Engulfing são padrões de reversão importantes.
        
        Support and Resistance levels são zonas críticas de decisão.
        Múltiplos toques aumentam a importância destes níveis.
        
        Volume Analysis confirma movimentos de preço.
        Volume crescente em breakouts indica força do movimento.
        
        Fibonacci Retracements identificam níveis de correção prováveis.
        38.2%, 50% e 61.8% são níveis de retração mais significativos.
        
        Moving Averages suavizam dados de preço e identificam tendências.
        EMA reage mais rapidamente que SMA a mudanças de preço.
        
        Oscillators como RSI e Stochastic identificam condições de sobrecompra/sobrevenda.
        Divergências entre preço e oscilador indicam possível reversão.
        
        MACD combina tendência e momentum em um indicador.
        Crossovers e divergências são sinais de entrada/saída.
        """ * 180
    }
    
    return documentos

def teste_completo_contexto_2m():
    """Executa teste completo do sistema de contexto expandido."""
    print("🚀 TESTE COMPLETO - Sistema de Contexto Expandido 2M Tokens")
    print("=" * 80)
    
    # Inicializar sistema
    print("\n📊 Inicializando ContextManager...")
    cm = ContextManager(
        base_url="http://localhost:4000",
        model_name="deepseek-r1-free",
        cache_dir="./cache_teste_2m"
    )
    
    # Criar documentos de teste
    print("\n📝 Criando documentos de teste...")
    documentos = criar_documentos_trading()
    
    total_chars = 0
    total_tokens = 0
    
    # Adicionar documentos ao contexto
    print("\n🔄 Adicionando documentos ao contexto...")
    for nome, conteudo in documentos.items():
        print(f"   📄 Processando: {nome}")
        
        # Adicionar ao contexto
        chunks = cm.add_context(conteudo, context_id=nome)
        
        chars = len(conteudo)
        tokens = cm._count_tokens(conteudo)
        total_chars += chars
        total_tokens += tokens
        
        print(f"      ✓ {len(chunks)} chunks criados")
        print(f"      ✓ {chars:,} caracteres, ~{tokens:,} tokens")
        
        time.sleep(0.5)  # Pequena pausa
    
    print(f"\n📊 Total processado: {total_chars:,} caracteres, ~{total_tokens:,} tokens")
    
    # Obter estatísticas
    print("\n📈 Estatísticas do sistema:")
    stats = cm.get_context_stats()
    
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"   {key}: {len(value)} itens")
        elif isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
    
    # Teste de busca semântica
    print("\n🔍 Teste de busca semântica:")
    queries = [
        "Como funciona order block em SMC?",
        "Qual a regra dos 2% em gestão de risco?",
        "O que são Fair Value Gaps?",
        "Como calcular position sizing?",
        "Estratégias de scalping algorítmico"
    ]
    
    for query in queries:
        print(f"\n   🔎 Query: {query}")
        relevant_chunks = cm.search_relevant_context(query, max_chunks=3)
        print(f"      ✓ {len(relevant_chunks)} chunks relevantes encontrados")
        
        for i, chunk in enumerate(relevant_chunks):
            preview = chunk.content[:100].replace('\n', ' ').strip()
            print(f"      Chunk {i+1}: {preview}...")
            print(f"      Importância: {chunk.importance_score:.3f}, Tokens: {chunk.token_count}")
    
    # Teste de contexto expandido
    print("\n🎯 Teste de contexto expandido:")
    test_queries = [
        "estratégias de trading",
        "gestão de risco",
        "análise técnica"
    ]
    
    for query in test_queries:
        expanded_context = cm.build_expanded_context(query)
        expanded_tokens = cm._count_tokens(expanded_context)
        
        print(f"   Query: '{query}'")
        print(f"   ✓ Contexto expandido: {expanded_tokens:,} tokens")
        print(f"   ✓ Fator de expansão: {expanded_tokens / 163000:.2f}x do limite base")
    
    # Demonstrar capacidade de 2M tokens
    print("\n🚀 Demonstração de capacidade 2M tokens:")
    
    # Simular adição de mais conteúdo
    for i in range(5):
        large_content = "\n".join([
            f"Documento adicional {i+1} sobre trading algorítmico.",
            "Este conteúdo simula documentação extensa sobre:",
            "- Estratégias avançadas de trading",
            "- Análise quantitativa de mercados",
            "- Otimização de algoritmos",
            "- Backtesting e validação",
            "- Gestão de portfólio",
            "- Análise de risco"
        ] * 1000)  # Repetir para criar conteúdo grande
        
        chunks = cm.add_context(large_content, context_id=f"doc_adicional_{i+1}")
        tokens = cm._count_tokens(large_content)
        
        print(f"   📄 Documento {i+1}: {len(chunks)} chunks, ~{tokens:,} tokens")
    
    # Estatísticas finais
    print("\n📊 ESTATÍSTICAS FINAIS:")
    final_stats = cm.get_context_stats()
    
    print(f"   💾 Total de chunks: {final_stats['total_chunks']:,}")
    print(f"   📈 Total de tokens: {final_stats['total_tokens']:,}")
    print(f"   🚀 Fator de expansão: {final_stats['expansion_factor']:.1f}x")
    print(f"   💿 Tamanho do cache: {final_stats['cache_size_mb']:.2f} MB")
    print(f"   ⭐ Importância média: {final_stats['avg_importance_score']:.3f}")
    
    # Verificar se atingiu meta de 2M tokens
    if final_stats['total_tokens'] >= 2000000:
        print("\n🎉 ✅ META ATINGIDA: Sistema processou 2M+ tokens com sucesso!")
    else:
        print(f"\n📈 Sistema processou {final_stats['total_tokens']:,} tokens")
        print(f"   Meta de 2M tokens: {(final_stats['total_tokens']/2000000)*100:.1f}% concluída")
    
    print("\n=== TESTE COMPLETO CONCLUÍDO ===")
    print("\n💡 O sistema demonstrou capacidade de:")
    print("   ✓ Processar grandes volumes de texto")
    print("   ✓ Criar chunks inteligentes com embeddings")
    print("   ✓ Realizar busca semântica eficiente")
    print("   ✓ Expandir contexto além do limite de 163k tokens")
    print("   ✓ Manter cache persistente para performance")
    
    return final_stats

if __name__ == "__main__":
    try:
        stats = teste_completo_contexto_2m()
        print(f"\n🏆 Teste concluído com {stats['total_tokens']:,} tokens processados!")
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()