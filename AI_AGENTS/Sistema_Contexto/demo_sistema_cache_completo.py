#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 DEMO - Sistema de Cache Avançado para R1

Esta demonstração mostra como usar o sistema de cache avançado
com o modelo R1 para obter performance ultra-veloz.

Recursos demonstrados:
- Cache semântico com deduplicação
- Multi-level caching
- Compressão inteligente
- Monitoramento em tempo real
- Cache warming
- Auto-tuning
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

# Importar sistema de cache
from sistema_cache_completo_r1 import CompleteR1CacheSystem

def demo_cache_basico():
    """Demonstração básica do sistema de cache"""
    print("🎯 DEMO 1: CACHE BÁSICO")
    print("=" * 60)

    # Inicializar sistema
    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    # Prompt de teste
    prompt = "Explique o conceito de Order Blocks no trading ICT/SMC"

    print(f"📝 Prompt: {prompt}")
    print("\n🔄 Fazendo primeira requisição (sem cache)...")

    start_time = time.time()
    result1 = system.chat_with_r1(prompt, use_cache=True)
    time1 = time.time() - start_time

    print(".3f"    print(f"📊 Cache hit: {result1['cached']}")
    print(f"🎯 Tokens: {result1['usage']['total_tokens']}")

    print("\n🔄 Fazendo segunda requisição (com cache)...")

    start_time = time.time()
    result2 = system.chat_with_r1(prompt, use_cache=True)
    time2 = time.time() - start_time

    print(".3f"    print(f"📊 Cache hit: {result2['cached']}")
    print(f"🚀 Melhoria: {time1/time2:.1f}x mais rápido!")

    system.stop_system()
    return time1, time2

def demo_deduplicacao_semantica():
    """Demonstra deduplicação semântica"""
    print("\n🎯 DEMO 2: DEDUPLICAÇÃO SEMÂNTICA")
    print("=" * 60)

    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    # Prompts similares semanticamente
    prompts = [
        "O que são Order Blocks?",
        "Explique os Order Blocks no trading",
        "Como funcionam os blocos de ordens?",
        "Order Blocks: conceito e aplicação"
    ]

    print("📝 Testando deduplicação semântica com prompts similares:")
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. {prompt}")
        start_time = time.time()
        result = system.chat_with_r1(prompt, use_cache=True)
        duration = time.time() - start_time

        cache_status = "✅ CACHE HIT" if result['cached'] else "❌ CACHE MISS"
        print(".3f"
        results.append(result['cached'])

    # Estatísticas
    cache_hits = sum(results)
    print("
📊 RESULTADO:"    print(f"   Cache hits: {cache_hits}/{len(prompts)}")
    print(".1f"
    # Mostrar estatísticas do cache
    stats = system.get_cache_stats()
    print(f"   Chunks únicos: {stats['unique_chunks']}")
    print(".1f"
    system.stop_system()

def demo_compressao_inteligente():
    """Demonstra compressão inteligente"""
    print("\n🎯 DEMO 3: COMPRESSÃO INTELIGENTE")
    print("=" * 60)

    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    # Conteúdo grande para teste
    large_content = """
    Estratégias Avançadas de Trading no Mercado Forex

    1. Análise Técnica Fundamental
    A análise técnica é crucial para identificar padrões de preço e tendências.
    Os indicadores técnicos como médias móveis, RSI, MACD, e Bandas de Bollinger
    fornecem insights valiosos sobre a força e direção do mercado.

    2. Smart Money Concepts (SMC)
    Os Smart Money Concepts revelam como instituições financeiras operam nos mercados.
    Conceitos como Order Blocks, Fair Value Gaps, e Liquidity Sweeps são essenciais
    para entender a manipulação institucional do mercado.

    3. Gestão de Risco Profissional
    A gestão de risco é fundamental para a sobrevivência a longo prazo no trading.
    Implementar stop losses adequados, gerenciar tamanho de posições baseado no risco,
    e manter disciplina emocional são práticas essenciais.

    4. Trading Algorítmico e Automação
    O trading algorítmico permite execução precisa e consistente de estratégias.
    Backtesting rigoroso, otimização de parâmetros, e validação out-of-sample
    são necessários para sistemas automatizados robustos.

    5. Psicologia do Trading
    A psicologia representa 80% do sucesso no trading. Desenvolver disciplina,
    controlar emoções como medo e ganância, e manter consistência são desafios
    constantes que traders devem superar.
    """ * 10  # Multiplicar para ter conteúdo grande

    print("📊 Testando compressão com conteúdo extenso:"    print(f"   Tamanho original: {len(large_content)} caracteres")

    # Adicionar contexto
    system.add_context(large_content)

    # Consultar
    query = "Quais são as estratégias de trading mais importantes?"
    result = system.chat_with_r1(query, use_cache=True)

    # Estatísticas de compressão
    stats = system.get_cache_stats()
    print("
📊 ESTATÍSTICAS DE COMPRESSÃO:"    print(".1f"    print(f"   Eficiência: {stats['compression_efficiency']:.1f}%")
    print(f"   Algoritmo usado: {stats['compression_algorithm']}")

    system.stop_system()

def demo_monitoramento_tempo_real():
    """Demonstra monitoramento em tempo real"""
    print("\n🎯 DEMO 4: MONITORAMENTO EM TEMPO REAL")
    print("=" * 60)

    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    print("📊 Carregando dashboard de monitoramento...")
    print("   Acesse: http://localhost:8080 para ver o dashboard")
    print("   (O dashboard será aberto automaticamente)")

    # Simular algumas operações
    operations = [
        "Como usar médias móveis no trading?",
        "Explique o indicador RSI",
        "O que são padrões de candlestick?",
        "Como funciona o volume no trading?",
        "Estratégias de scalping"
    ]

    print("\n🔄 Executando operações de teste...")

    for i, query in enumerate(operations, 1):
        result = system.chat_with_r1(query, use_cache=True)
        print(f"   {i}. {query[:30]}... - {'CACHE' if result['cached'] else 'NOVO'}")
        time.sleep(0.5)  # Pausa para visualização

    # Mostrar estatísticas finais
    stats = system.get_cache_stats()
    print("
📊 ESTATÍSTICAS FINAIS:"    print(f"   Total operações: {stats['total_operations']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(".1f"    print(f"   Tempo médio: {stats['avg_response_time']:.3f}s")

    system.stop_system()

def demo_cache_warming():
    """Demonstra cache warming"""
    print("\n🎯 DEMO 5: CACHE WARMING")
    print("=" * 60)

    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    # Perguntas comuns para pré-carregar
    common_questions = [
        "O que é trading?",
        "Como funciona o mercado Forex?",
        "O que são ordens de compra e venda?",
        "Como calcular risco por trade?",
        "O que é alavancagem?",
        "Como usar stop loss?",
        "O que são indicadores técnicos?",
        "Como fazer análise fundamentalista?"
    ]

    print("🔥 Executando cache warming...")
    print("   Pré-carregando respostas para perguntas comuns...")

    for question in common_questions:
        result = system.chat_with_r1(question, use_cache=True)
        print(f"   ✅ {question}")

    print("
📊 RESULTADO DO CACHE WARMING:"    stats = system.get_cache_stats()
    print(f"   Perguntas pré-carregadas: {len(common_questions)}")
    print(f"   Cache size: {stats['total_size_mb']:.2f} MB")
    print(f"   Chunks em cache: {stats['unique_chunks']}")

    # Testar uma pergunta comum
    print("
🔄 Testando pergunta comum após warming..."    start_time = time.time()
    result = system.chat_with_r1("O que é trading?", use_cache=True)
    duration = time.time() - start_time

    print(".3f"    print(f"   Cache hit: {result['cached']}")

    system.stop_system()

def demo_auto_tuning():
    """Demonstra auto-tuning do sistema"""
    print("\n🎯 DEMO 6: AUTO-TUNING")
    print("=" * 60)

    system = CompleteR1CacheSystem()
    system.initialize_system()
    system.start_system()

    print("🔧 Testando auto-tuning do sistema...")
    print("   O sistema se adapta automaticamente baseado no uso...")

    # Simular diferentes padrões de uso
    patterns = [
        ("trading básico", 3),  # Padrão repetitivo
        ("análise técnica", 5),  # Padrão misto
        ("perguntas aleatórias", 8)  # Padrão diversificado
    ]

    for pattern_name, num_queries in patterns:
        print(f"\n📊 Testando padrão: {pattern_name}")

        for i in range(num_queries):
            if pattern_name == "trading básico":
                query = f"Conceito básico {i+1} do trading"
            elif pattern_name == "análise técnica":
                query = f"Análise técnica {i+1} - indicador {i}"
            else:
                query = f"Pergunta aleatória {i+1} sobre mercados"

            result = system.chat_with_r1(query, use_cache=True)

        # Mostrar estatísticas após cada padrão
        stats = system.get_cache_stats()
        print(f"   Cache hit rate: {stats['hit_rate']:.1f}%")
        print(f"   Estratégia atual: {stats['current_strategy']}")

    print("
🎯 AUTO-TUNING COMPLETO!"    system.stop_system()

def main():
    """Função principal da demonstração"""
    print("🚀 SISTEMA DE CACHE AVANÇADO PARA R1")
    print("=" * 80)
    print("   Demonstração completa das funcionalidades avançadas")
    print("=" * 80)

    try:
        # Executar todas as demonstrações
        demo_cache_basico()
        demo_deduplicacao_semantica()
        demo_compressao_inteligente()
        demo_monitoramento_tempo_real()
        demo_cache_warming()
        demo_auto_tuning()

        print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("✅ Sistema de cache avançado funcionando perfeitamente")
        print("🚀 Performance otimizada para R1")
        print("💾 Cache inteligente com deduplicação semântica")
        print("📊 Monitoramento em tempo real")
        print("🔧 Auto-tuning automático")

    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()