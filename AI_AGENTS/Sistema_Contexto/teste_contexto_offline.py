#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Offline do Sistema de Contexto Expandido - 2M Tokens

Este script testa o sistema sem depender de APIs externas,
focando na funcionalidade de chunking e cache.
"""

import os
import time
import json
from sistema_contexto_expandido_2m import ContextManager

def criar_conteudo_teste():
    """Cria conteúdo de teste para simular 2M tokens."""
    
    # Conteúdo base sobre trading
    base_content = """
    Análise Técnica e Trading Algorítmico
    
    O trading algorítmico representa uma evolução natural dos mercados financeiros,
    combinando análise técnica tradicional com poder computacional moderno.
    
    Conceitos Fundamentais:
    
    1. Análise de Tendência
    - Moving Averages (MA): Suavizam dados de preço
    - Exponential Moving Average (EMA): Reage mais rapidamente
    - MACD: Combina tendência e momentum
    - ADX: Mede força da tendência
    
    2. Osciladores
    - RSI: Identifica sobrecompra/sobrevenda
    - Stochastic: Compara preço de fechamento com range
    - Williams %R: Similar ao Stochastic
    - CCI: Commodity Channel Index
    
    3. Suporte e Resistência
    - Níveis psicológicos importantes
    - Fibonacci retracements
    - Pivot points
    - Volume profile
    
    4. Padrões de Candlestick
    - Doji: Indecisão do mercado
    - Hammer: Possível reversão de baixa
    - Engulfing: Padrão de reversão forte
    - Inside bar: Consolidação
    
    5. Gestão de Risco
    - Stop loss obrigatório
    - Position sizing calculado
    - Risk-reward ratio mínimo 1:2
    - Drawdown máximo controlado
    
    6. Estratégias Algorítmicas
    - Scalping: Operações rápidas
    - Swing trading: Posições de médio prazo
    - Trend following: Seguir tendências
    - Mean reversion: Retorno à média
    
    7. Backtesting
    - Dados históricos confiáveis
    - Simulação realística
    - Métricas de performance
    - Otimização de parâmetros
    
    8. Execução
    - Latência baixa
    - Slippage controlado
    - Spreads considerados
    - Horários de mercado
    
    9. Psicologia do Trading
    - Controle emocional
    - Disciplina na execução
    - Paciência para oportunidades
    - Aceitação de perdas
    
    10. Tecnologia
    - APIs de brokers
    - Feeds de dados em tempo real
    - Infraestrutura robusta
    - Monitoramento contínuo
    """
    
    # Multiplicar conteúdo para atingir volume significativo
    documentos = []
    
    # Criar 50 documentos variados
    for i in range(50):
        doc = f"""
        DOCUMENTO {i+1}: Trading Avançado - Seção {i+1}
        
        {base_content}
        
        Estratégias Específicas para {['Forex', 'Ações', 'Commodities', 'Crypto', 'Índices'][i % 5]}:
        
        - Características únicas do mercado
        - Horários de maior liquidez
        - Spreads típicos
        - Volatilidade esperada
        - Correlações importantes
        - Eventos que afetam preços
        - Estratégias mais eficazes
        - Gestão de risco específica
        
        Timeframes Recomendados:
        - M1: Scalping ultra-rápido
        - M5: Scalping tradicional
        - M15: Swing intraday
        - H1: Swing de curto prazo
        - H4: Swing de médio prazo
        - D1: Posições de longo prazo
        
        Indicadores Técnicos Avançados:
        - Ichimoku Cloud
        - Elliott Wave Theory
        - Harmonic Patterns
        - Market Profile
        - Volume Spread Analysis
        - Smart Money Concepts
        
        Automação e Algoritmos:
        - Machine Learning aplicado
        - Neural Networks
        - Genetic Algorithms
        - Reinforcement Learning
        - Natural Language Processing
        - Sentiment Analysis
        
        " * 20  # Repetir cada seção 20 vezes
        """
        
        documentos.append(doc)
    
    return documentos

def teste_contexto_offline():
    """Executa teste offline do sistema de contexto."""
    print("🔧 TESTE OFFLINE - Sistema de Contexto Expandido")
    print("=" * 60)
    
    # Inicializar sistema (sem API)
    print("\n📊 Inicializando ContextManager (modo offline)...")
    cm = ContextManager(
        base_url="http://localhost:4000",  # Não será usado
        model_name="offline-test",
        cache_dir="./cache_offline_test"
    )
    
    # Criar conteúdo de teste
    print("\n📝 Criando conteúdo de teste...")
    documentos = criar_conteudo_teste()
    
    total_chars = 0
    total_chunks = 0
    
    # Processar documentos
    print("\n🔄 Processando documentos...")
    start_time = time.time()
    
    for i, doc in enumerate(documentos):
        print(f"   📄 Processando documento {i+1}/{len(documentos)}")
        
        try:
            # Adicionar ao contexto (sem resumos que dependem de API)
            chunks = cm.add_context(doc, context_id=f"doc_{i+1}")
            
            chars = len(doc)
            total_chars += chars
            total_chunks += len(chunks)
            
            print(f"      ✓ {len(chunks)} chunks criados, {chars:,} caracteres")
            
        except Exception as e:
            print(f"      ❌ Erro: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ Processamento concluído em {processing_time:.2f} segundos")
    print(f"📊 Total: {total_chars:,} caracteres em {total_chunks} chunks")
    
    # Estimar tokens (aproximadamente 4 chars = 1 token)
    estimated_tokens = total_chars // 4
    print(f"🎯 Tokens estimados: {estimated_tokens:,}")
    
    # Obter estatísticas do sistema
    print("\n📈 Estatísticas do sistema:")
    try:
        stats = cm.get_context_stats()
        
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} itens")
            elif isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
                
    except Exception as e:
        print(f"   ❌ Erro ao obter estatísticas: {e}")
    
    # Teste de busca (sem embeddings)
    print("\n🔍 Teste de busca por palavras-chave:")
    keywords = ['trading', 'algoritmo', 'risco', 'scalping', 'forex']
    
    for keyword in keywords:
        matching_chunks = []
        for chunk_id, chunk in cm.chunk_cache.items():
            if keyword.lower() in chunk.content.lower():
                matching_chunks.append(chunk)
        
        print(f"   🔎 '{keyword}': {len(matching_chunks)} chunks encontrados")
    
    # Verificar cache
    print("\n💾 Informações do cache:")
    cache_dir = cm.cache_dir
    if os.path.exists(cache_dir):
        cache_files = os.listdir(cache_dir)
        print(f"   📁 Diretório: {cache_dir}")
        print(f"   📄 Arquivos: {len(cache_files)}")
        
        total_size = 0
        for file in cache_files:
            file_path = os.path.join(cache_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                print(f"      {file}: {size:,} bytes")
        
        print(f"   💿 Tamanho total: {total_size / 1024 / 1024:.2f} MB")
    
    # Demonstração de capacidade
    print("\n🚀 Demonstração de capacidade:")
    print(f"   ✓ Documentos processados: {len(documentos)}")
    print(f"   ✓ Chunks criados: {total_chunks:,}")
    print(f"   ✓ Caracteres processados: {total_chars:,}")
    print(f"   ✓ Tokens estimados: {estimated_tokens:,}")
    print(f"   ✓ Velocidade: {total_chars/processing_time:,.0f} chars/segundo")
    
    # Verificar meta de 2M tokens
    if estimated_tokens >= 2000000:
        print("\n🎉 ✅ META ATINGIDA: 2M+ tokens processados!")
    else:
        progress = (estimated_tokens / 2000000) * 100
        print(f"\n📈 Progresso: {progress:.1f}% da meta de 2M tokens")
        print(f"   Faltam ~{2000000 - estimated_tokens:,} tokens")
    
    print("\n=== TESTE OFFLINE CONCLUÍDO ===")
    print("\n💡 Funcionalidades testadas:")
    print("   ✓ Chunking hierárquico de texto")
    print("   ✓ Cache persistente")
    print("   ✓ Processamento de grandes volumes")
    print("   ✓ Busca por palavras-chave")
    print("   ✓ Estatísticas do sistema")
    
    return {
        'documentos': len(documentos),
        'chunks': total_chunks,
        'caracteres': total_chars,
        'tokens_estimados': estimated_tokens,
        'tempo_processamento': processing_time
    }

if __name__ == "__main__":
    try:
        resultado = teste_contexto_offline()
        print(f"\n🏆 Teste concluído com sucesso!")
        print(f"   📊 {resultado['tokens_estimados']:,} tokens estimados processados")
        print(f"   ⚡ {resultado['caracteres']/resultado['tempo_processamento']:,.0f} chars/segundo")
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()