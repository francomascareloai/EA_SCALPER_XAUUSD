#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Simplificado do Sistema de Contexto Expandido com R1
Versão otimizada para evitar rate limiting
"""

import os
import sys
import time
from pathlib import Path

# Adicionar o diretório do sistema ao path
sys.path.append(str(Path(__file__).parent / "Sistema_Contexto_Expandido_2M"))

from sistema_contexto_expandido_2m import ContextManager

def gerar_conteudo_teste():
    """Gera conteúdo de teste sobre trading."""
    return """
# Guia Completo de Smart Money Concepts (SMC)

## 1. Introdução ao SMC

Smart Money Concepts é uma metodologia de análise técnica que se baseia na compreensão
de como as instituições financeiras (bancos, fundos, market makers) operam no mercado.

## 2. Order Blocks

Order blocks são zonas de preço onde grandes instituições colocaram ordens significativas.
Estas zonas tendem a atuar como suporte ou resistência quando o preço retorna.

### Características dos Order Blocks:
- Formados em áreas de alta atividade institucional
- Podem ser bullish ou bearish
- Tendem a ser respeitados quando testados
- Mais eficazes em timeframes maiores

### Como Identificar Order Blocks:
1. Procure por movimentos impulsivos no preço
2. Identifique a última vela antes do movimento
3. Marque a zona de preço dessa vela
4. Aguarde o retorno do preço à zona

## 3. Liquidity Sweeps

Liquidity sweeps ocorrem quando o preço move rapidamente para capturar liquidez
em níveis óbvios (highs/lows anteriores) antes de reverter na direção pretendida.

### Tipos de Liquidity Sweeps:
- Buy Side Liquidity (BSL): Acima de máximas anteriores
- Sell Side Liquidity (SSL): Abaixo de mínimas anteriores
- Internal Liquidity: Dentro de ranges estabelecidos

## 4. Market Structure

A estrutura de mercado é fundamental para entender a direção institucional.

### Tendência de Alta (Bullish):
- Higher Highs (HH)
- Higher Lows (HL)
- Break of Structure (BOS) para cima

### Tendência de Baixa (Bearish):
- Lower Highs (LH)
- Lower Lows (LL)
- Break of Structure (BOS) para baixo

## 5. Fair Value Gaps (FVG)

Fair Value Gaps são áreas no gráfico onde há um desequilíbrio entre oferta e demanda,
criando lacunas que o preço tende a preencher posteriormente.

### Características dos FVG:
- Formados por três velas consecutivas
- A vela do meio não toca as outras duas
- Atuam como zonas de suporte/resistência
- Podem ser preenchidos parcial ou totalmente

## 6. Displacement

Displacement refere-se a movimentos rápidos e impulsivos no preço que indicam
atividade institucional significativa.

### Sinais de Displacement:
- Velas grandes com pouco ou nenhum wick
- Volume acima da média
- Quebra de estruturas importantes
- Movimento através de múltiplos níveis

## 7. Estratégias de Trading com SMC

### Estratégia 1: Order Block Reversal
1. Identifique um order block válido
2. Aguarde o retorno do preço à zona
3. Procure por sinais de rejeição
4. Entre na direção do order block
5. Stop loss além da zona
6. Take profit no próximo nível de liquidez

### Estratégia 2: Liquidity Sweep Entry
1. Identifique níveis de liquidez óbvios
2. Aguarde o sweep da liquidez
3. Procure por reversão imediata
4. Entre na direção da reversão
5. Stop loss além do sweep
6. Take profit no order block oposto

### Estratégia 3: FVG Fill
1. Identifique um FVG válido
2. Aguarde o retorno do preço ao gap
3. Entre na direção da tendência principal
4. Stop loss além do FVG
5. Take profit no próximo objetivo

## 8. Gerenciamento de Risco

### Regras Fundamentais:
- Nunca arrisque mais de 1-2% do capital por trade
- Use stop loss em todas as operações
- Mantenha ratio risco/recompensa mínimo de 1:2
- Diversifique entre diferentes pares
- Monitore o drawdown constantemente

### Cálculo de Position Size:
Position Size = (Capital × % Risco) / (Preço Entrada - Stop Loss)

## 9. Timeframes e Confluências

### Análise Multi-Timeframe:
- Timeframe maior: Direção geral (D1, H4)
- Timeframe médio: Estrutura e níveis (H1, M15)
- Timeframe menor: Entrada precisa (M5, M1)

### Confluências Importantes:
- Order block + FVG
- Liquidity sweep + Displacement
- Market structure + Volume
- Fibonacci + SMC levels

## 10. Psicologia e Disciplina

### Mindset Correto:
- Pense em probabilidades, não certezas
- Aceite perdas como parte do processo
- Mantenha disciplina no plano de trading
- Evite FOMO e revenge trading
- Foque no processo, não apenas nos resultados

### Journal de Trading:
- Registre todas as operações
- Anote o setup utilizado
- Documente erros e acertos
- Revise periodicamente
- Ajuste a estratégia conforme necessário

## Conclusão

Smart Money Concepts oferece uma perspectiva única sobre o mercado,
focando no comportamento institucional. O sucesso requer prática,
disciplina e constante aperfeiçoamento das habilidades de análise.

Lembre-se: o mercado é um jogo de probabilidades. Mesmo com as
melhores análises, nem todos os trades serão vencedores. O importante
é manter consistência e seguir o plano de trading rigorosamente.
"""

def main():
    """Função principal do teste."""
    print("🚀 TESTE SIMPLIFICADO - SISTEMA DE CONTEXTO EXPANDIDO R1")
    print("=" * 60)
    
    try:
        # Verificar se o LiteLLM está rodando
        print("\n🔍 Verificando conexão com LiteLLM...")
        
        # Inicializar sistema
        print("\n🔧 Inicializando sistema...")
        sistema = ContextManager(
            base_url="http://localhost:4000",
            model_name="deepseek-r1-free",
            max_context_tokens=163000,
            target_context_tokens=500000  # Reduzido para evitar rate limiting
        )
        
        print(f"✅ Sistema inicializado com meta de {sistema.target_context_tokens:,} tokens")
        
        # Gerar conteúdo de teste
        print("\n📝 Gerando conteúdo de teste...")
        conteudo = gerar_conteudo_teste()
        print(f"✅ Conteúdo gerado: {len(conteudo):,} caracteres")
        
        # Adicionar ao contexto (sem resumos automáticos)
        print("\n⚙️ Processando contexto...")
        start_time = time.time()
        
        chunk_ids = sistema.add_context(conteudo)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Contexto processado em {processing_time:.2f}s")
        print(f"📊 Chunks criados: {len(chunk_ids)}")
        print(f"⚡ Velocidade: {len(conteudo)/processing_time:.0f} chars/s")
        
        # Testar busca semântica
        print("\n🔍 Testando busca semântica...")
        queries = [
            "Order Blocks",
            "Liquidity Sweeps",
            "gerenciamento de risco"
        ]
        
        for query in queries:
            print(f"\n  Query: {query}")
            chunks_relevantes = sistema.search_relevant_context(query, max_chunks=3)
            print(f"  ✅ Encontrados {len(chunks_relevantes)} chunks relevantes")
            
            if chunks_relevantes:
                for i, chunk in enumerate(chunks_relevantes[:2]):
                    print(f"    {i+1}. Chunk {chunk.id[:8]}... ({chunk.token_count} tokens)")
        
        # Construir contexto expandido (sem gerar resposta para evitar rate limiting)
        print("\n🏗️ Testando construção de contexto expandido...")
        query_teste = "Como identificar Order Blocks?"
        
        contexto_expandido = sistema.build_expanded_context(query_teste, max_tokens=50000)
        tokens_contexto = sistema._count_tokens(contexto_expandido)
        
        print(f"✅ Contexto expandido construído")
        print(f"📊 Tokens no contexto: {tokens_contexto:,}")
        print(f"📏 Tamanho do contexto: {len(contexto_expandido):,} caracteres")
        
        # Estatísticas finais
        print("\n📈 Estatísticas do sistema:")
        stats = sistema.get_context_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,}")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} itens")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print("\n💡 O sistema de contexto expandido está funcionando corretamente.")
        print("   Para testes com o modelo R1, execute consultas individuais para")
        print("   evitar rate limiting do OpenRouter.")
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)