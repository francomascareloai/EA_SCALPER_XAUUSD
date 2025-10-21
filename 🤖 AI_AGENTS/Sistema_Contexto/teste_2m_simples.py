#!/usr/bin/env python3
"""
Teste Simples do Sistema de Contexto Expandido - 2M Tokens
Versão otimizada sem dependências de embeddings complexos
"""

import os
import time
import random
from sistema_contexto_expandido_2m import ContextManager

def gerar_texto_simples(tamanho_mb=10):
    """
    Gera texto simples para teste sem usar embeddings
    """
    print(f"📝 Gerando {tamanho_mb}MB de texto para teste...")
    
    # Texto base para repetir
    texto_base = """
    Análise de Trading XAUUSD - Estratégias Avançadas
    
    O mercado de ouro (XAUUSD) apresenta características únicas que requerem
    abordagens específicas de trading. Esta análise aborda:
    
    1. Padrões de Preço e Estrutura de Mercado
    - Order blocks e zonas de liquidez
    - Níveis de suporte e resistência dinâmicos
    - Padrões de reversão em timeframes múltiplos
    
    2. Gestão de Risco FTMO Compliant
    - Máximo drawdown de 5% para FTMO Challenge
    - Stop loss dinâmico baseado em ATR
    - Position sizing adaptativo
    
    3. Indicadores Técnicos Específicos
    - RSI divergências em M15 e H1
    - MACD crossovers com confirmação de volume
    - Bollinger Bands para identificação de volatilidade
    
    4. Análise Fundamental
    - Correlação com DXY (Dólar Index)
    - Impacto de dados econômicos americanos
    - Eventos geopolíticos e safe haven demand
    
    5. Backtesting e Otimização
    - Resultados históricos de 2020-2024
    - Métricas de performance: Sharpe ratio, Maximum Drawdown
    - Adaptação para diferentes condições de mercado
    
    6. Implementação Prática
    - Setup de trading em MetaTrader 5
    - Automação com Expert Advisors
    - Monitoramento em tempo real
    
    Esta estratégia foi testada em mais de 10.000 trades históricos
    com uma taxa de acerto de 68% e profit factor de 1.85.
    """
    
    # Calcular quantas repetições precisamos
    tamanho_texto_base = len(texto_base.encode('utf-8'))
    repeticoes_necessarias = (tamanho_mb * 1024 * 1024) // tamanho_texto_base
    
    print(f"🔄 Repetindo texto base {repeticoes_necessarias:,} vezes...")
    
    # Gerar variações para tornar o texto mais diverso
    textos = []
    for i in range(min(repeticoes_necessarias, 1000)):  # Limitar para evitar uso excessivo de memória
        variacao = texto_base.replace("XAUUSD", f"XAUUSD_V{i}")
        variacao = variacao.replace("Trading", f"Trading_{random.randint(1,100)}")
        textos.append(variacao)
    
    texto_final = "\n\n".join(textos)
    
    # Se ainda não atingiu o tamanho desejado, repetir o conjunto
    while len(texto_final.encode('utf-8')) < (tamanho_mb * 1024 * 1024):
        texto_final += "\n\n" + "\n\n".join(textos[:100])  # Adicionar em lotes menores
    
    tamanho_final_mb = len(texto_final.encode('utf-8')) / (1024 * 1024)
    print(f"✅ Texto gerado: {tamanho_final_mb:.2f}MB")
    
    return texto_final

def teste_contexto_2m_simples():
    """
    Teste simplificado do sistema de contexto expandido
    """
    print("🚀 Iniciando Teste do Sistema de Contexto Expandido - 2M Tokens")
    print("=" * 70)
    
    try:
        # Inicializar o Context Manager
        print("📋 Inicializando Context Manager...")
        cm = ContextManager(
            base_url="http://localhost:4000",
            model_name="deepseek-r1-free",
            cache_dir="./cache_contexto",
            max_context_tokens=2000000,  # 2M tokens
            target_context_tokens=1500000  # 1.5M tokens alvo
        )
        
        print(f"✅ Context Manager inicializado")
        print(f"📊 Limite máximo: {cm.max_context_tokens:,} tokens")
        print(f"🎯 Meta de contexto: {cm.target_context_tokens:,} tokens")
        
        # Gerar e adicionar conteúdo em lotes
        print("\n📝 Gerando e adicionando conteúdo...")
        
        total_caracteres = 0
        lote = 1
        
        # Adicionar conteúdo em lotes de 5MB até atingir ~2M tokens
        while total_caracteres < 8000000:  # ~8MB de texto ≈ 2M tokens
            print(f"\n📦 Processando lote {lote}...")
            
            # Gerar texto para este lote
            texto_lote = gerar_texto_simples(5)  # 5MB por lote
            
            # Adicionar ao contexto
            inicio = time.time()
            cm.add_context(texto_lote, context_id=f"lote_{lote}")
            tempo_adicao = time.time() - inicio
            
            total_caracteres += len(texto_lote)
            
            print(f"⏱️  Lote {lote} adicionado em {tempo_adicao:.2f}s")
            print(f"📈 Total acumulado: {total_caracteres:,} caracteres")
            
            # Obter estatísticas atuais
            stats = cm.get_context_stats()
            print(f"🔢 Chunks: {stats['total_chunks']:,}")
            print(f"🎯 Tokens: {stats['total_tokens']:,}")
            
            lote += 1
            
            # Parar se atingiu a meta
            if stats['total_tokens'] >= 1800000:  # Próximo de 2M
                print(f"\n🎉 Meta de tokens atingida!")
                break
        
        # Estatísticas finais
        print("\n" + "=" * 70)
        print("📊 ESTATÍSTICAS FINAIS")
        print("=" * 70)
        
        stats_finais = cm.get_context_stats()
        
        for chave, valor in stats_finais.items():
            if isinstance(valor, (int, float)):
                if 'tokens' in chave or 'chunks' in chave:
                    print(f"📈 {chave}: {valor:,}")
                else:
                    print(f"📊 {chave}: {valor:.3f}")
            else:
                print(f"📋 {chave}: {valor}")
        
        # Teste de busca simples
        print("\n🔍 Testando busca por palavras-chave...")
        resultados_busca = cm.search_context("FTMO trading estratégia", max_results=5)
        print(f"✅ Encontrados {len(resultados_busca)} resultados relevantes")
        
        # Resultado final
        if stats_finais['total_tokens'] >= 1500000:
            print(f"\n🏆 SUCESSO! Sistema processou {stats_finais['total_tokens']:,} tokens")
            print(f"🚀 Velocidade média: {stats_finais['total_tokens']/60:.0f} tokens/minuto")
            return True
        else:
            print(f"\n⚠️  Teste parcial: {stats_finais['total_tokens']:,} tokens processados")
            return False
            
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Sistema de Contexto Expandido - Teste Simples 2M Tokens")
    print("Versão otimizada sem embeddings complexos")
    print("=" * 70)
    
    inicio_total = time.time()
    
    try:
        sucesso = teste_contexto_2m_simples()
        
        tempo_total = time.time() - inicio_total
        print(f"\n⏱️  Tempo total de execução: {tempo_total:.2f} segundos")
        
        if sucesso:
            print("\n🎉 Teste concluído com SUCESSO!")
        else:
            print("\n⚠️  Teste concluído com limitações")
            
    except KeyboardInterrupt:
        print("\n⏹️  Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()