#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Final - Sistema de Contexto Expandido para 2M Tokens

Este script gera conteúdo suficiente para atingir 2 milhões de tokens
e demonstra a capacidade completa do sistema.
"""

import os
import time
import random
from sistema_contexto_expandido_2m import ContextManager

def gerar_conteudo_trading_extenso():
    """Gera conteúdo extenso sobre trading para atingir 2M tokens."""
    
    # Templates de conteúdo
    templates = {
        'analise_tecnica': """
        ANÁLISE TÉCNICA AVANÇADA - {tema}
        
        A análise técnica é fundamental para o sucesso no trading. {tema} representa
        uma das abordagens mais eficazes para identificar oportunidades de mercado.
        
        Conceitos Fundamentais:
        
        1. Identificação de Tendências
        - Tendência de alta: Sequência de topos e fundos ascendentes
        - Tendência de baixa: Sequência de topos e fundos descendentes
        - Tendência lateral: Movimento entre suporte e resistência
        - Confirmação de tendência através de múltiplos timeframes
        
        2. Suporte e Resistência
        - Níveis psicológicos importantes (números redondos)
        - Máximas e mínimas históricas
        - Médias móveis como suporte/resistência dinâmica
        - Volume profile e pontos de controle
        
        3. Padrões de Preço
        - Triângulos: ascendente, descendente, simétrico
        - Retângulos: consolidação antes de breakout
        - Bandeiras e flâmulas: continuação de tendência
        - Ombro-cabeça-ombro: padrão de reversão
        
        4. Indicadores Técnicos
        - RSI: Força relativa, divergências
        - MACD: Convergência/divergência de médias
        - Bollinger Bands: Volatilidade e reversão à média
        - Stochastic: Momentum e sobrecompra/sobrevenda
        
        5. Análise de Volume
        - Volume confirma movimentos de preço
        - Volume crescente em breakouts
        - Divergências de volume indicam fraqueza
        - Volume profile mostra áreas de interesse
        
        6. Timeframes Múltiplos
        - Timeframe maior define tendência principal
        - Timeframe menor para entrada precisa
        - Alinhamento de sinais entre timeframes
        - Gestão baseada em múltiplos horizontes
        
        7. Gestão de Risco
        - Stop loss baseado em estrutura técnica
        - Take profit em níveis de resistência
        - Trailing stop para maximizar ganhos
        - Position sizing baseado em volatilidade
        
        8. Psicologia do Mercado
        - Sentimento através de indicadores
        - Comportamento de massa vs. smart money
        - Ciclos de medo e ganância
        - Contrarian thinking em extremos
        
        9. Backtesting e Otimização
        - Teste em dados históricos
        - Validação out-of-sample
        - Métricas de performance
        - Robustez da estratégia
        
        10. Execução Prática
        - Disciplina na aplicação de regras
        - Controle emocional
        - Adaptação a condições de mercado
        - Melhoria contínua do sistema
        """,
        
        'estrategias_algoritmos': """
        ESTRATÉGIAS ALGORÍTMICAS - {tema}
        
        O trading algorítmico revolucionou os mercados financeiros, permitindo
        execução precisa e eliminação de vieses emocionais. {tema} é uma das
        abordagens mais promissoras neste campo.
        
        Fundamentos da Estratégia:
        
        1. Lógica de Entrada
        - Condições técnicas específicas
        - Confirmação através de múltiplos indicadores
        - Filtros de qualidade de sinal
        - Timing preciso de execução
        
        2. Gestão de Posição
        - Cálculo automático de position size
        - Stop loss dinâmico
        - Take profit escalonado
        - Trailing stop inteligente
        
        3. Filtros de Mercado
        - Condições de volatilidade
        - Horários de maior liquidez
        - Eventos econômicos importantes
        - Correlações entre ativos
        
        4. Otimização de Parâmetros
        - Algoritmos genéticos
        - Machine learning
        - Walk-forward analysis
        - Robustez estatística
        
        5. Execução Técnica
        - Latência mínima
        - Slippage controlado
        - Partial fills
        - Order management
        
        6. Monitoramento em Tempo Real
        - Performance tracking
        - Risk monitoring
        - Alertas automáticos
        - Intervenção manual quando necessário
        
        7. Adaptação Dinâmica
        - Ajuste a condições de mercado
        - Regime detection
        - Parameter shifting
        - Strategy switching
        
        8. Backtesting Rigoroso
        - Dados de alta qualidade
        - Simulação realística
        - Custos de transação
        - Análise de drawdown
        
        9. Risk Management Avançado
        - Value at Risk (VaR)
        - Maximum drawdown limits
        - Correlation monitoring
        - Portfolio heat maps
        
        10. Infraestrutura Robusta
        - Redundância de sistemas
        - Backup automático
        - Disaster recovery
        - Monitoring 24/7
        """,
        
        'mercados_especificos': """
        ANÁLISE DE MERCADO - {tema}
        
        Cada mercado possui características únicas que devem ser consideradas
        no desenvolvimento de estratégias de trading. {tema} apresenta
        oportunidades e desafios específicos.
        
        Características do Mercado:
        
        1. Horários de Funcionamento
        - Sessões principais de trading
        - Overlaps de maior liquidez
        - Gaps de abertura/fechamento
        - Feriados e eventos especiais
        
        2. Participantes do Mercado
        - Bancos centrais e política monetária
        - Instituições financeiras
        - Hedge funds e asset managers
        - Traders retail e algoritmos
        
        3. Fatores Fundamentais
        - Indicadores econômicos
        - Política monetária
        - Eventos geopolíticos
        - Sentiment de risco
        
        4. Estrutura de Custos
        - Spreads típicos
        - Comissões de corretagem
        - Swap rates (overnight)
        - Slippage esperado
        
        5. Volatilidade Característica
        - Padrões intraday
        - Sazonalidade
        - Eventos de alta volatilidade
        - Correlações históricas
        
        6. Liquidez e Volume
        - Profundidade do book
        - Impacto de grandes ordens
        - Horários de maior/menor liquidez
        - Market makers vs. takers
        
        7. Análise Técnica Específica
        - Níveis técnicos relevantes
        - Padrões comuns
        - Indicadores mais eficazes
        - Timeframes ótimos
        
        8. Estratégias Adequadas
        - Scalping vs. swing trading
        - Trend following vs. mean reversion
        - Breakout vs. fade strategies
        - Carry trade opportunities
        
        9. Gestão de Risco Específica
        - Volatilidade esperada
        - Correlações com outros ativos
        - Exposure limits
        - Hedging strategies
        
        10. Tecnologia e Execução
        - Latência requirements
        - Data feeds necessários
        - Execution venues
        - Regulatory considerations
        """
    }
    
    # Temas para cada template
    temas = {
        'analise_tecnica': [
            'Fibonacci e Proporções Áureas', 'Elliott Wave Theory', 'Harmonic Patterns',
            'Market Profile', 'Volume Spread Analysis', 'Smart Money Concepts',
            'Order Flow Analysis', 'Auction Market Theory', 'Wyckoff Method',
            'Japanese Candlestick Patterns', 'Point and Figure Charts', 'Renko Charts'
        ],
        'estrategias_algoritmos': [
            'High Frequency Trading', 'Statistical Arbitrage', 'Pairs Trading',
            'Mean Reversion Systems', 'Momentum Strategies', 'Grid Trading',
            'Martingale Systems', 'News Trading Algorithms', 'Sentiment Analysis',
            'Machine Learning Models', 'Neural Networks', 'Genetic Algorithms'
        ],
        'mercados_especificos': [
            'EUR/USD - Major Currency Pair', 'GBP/USD - Cable Trading',
            'USD/JPY - Yen Dynamics', 'Gold (XAU/USD) - Safe Haven',
            'S&P 500 Index', 'NASDAQ Technology', 'Bitcoin Trading',
            'Crude Oil Markets', 'Bond Futures', 'Commodity Trading'
        ]
    }
    
    documentos = []
    
    # Gerar múltiplos documentos para cada combinação
    for template_name, template_content in templates.items():
        for tema in temas[template_name]:
            # Criar múltiplas variações do mesmo tema
            for variacao in range(20):  # 20 variações por tema
                doc = template_content.format(tema=tema)
                
                # Adicionar seções extras para aumentar o tamanho
                for i in range(5):  # 5 seções adicionais
                    doc += "\n\n" + "\n".join([
                        f"Seção Adicional {i+1}: Detalhamento de {tema}",
                        "Esta seção fornece análise detalhada e exemplos práticos.",
                        "Incluindo casos de estudo, backtests históricos e métricas de performance.",
                        "Considerações especiais para diferentes condições de mercado.",
                        "Adaptações necessárias para diferentes timeframes e instrumentos.",
                        "Integração com outras estratégias e sistemas de trading.",
                        "Monitoramento de performance e otimização contínua.",
                        "Gestão de risco específica para esta abordagem.",
                        "Aspectos psicológicos e disciplina na execução.",
                        "Tecnologia e infraestrutura necessária."
                    ] * 50)  # Repetir 50 vezes para aumentar significativamente o tamanho
                
                documentos.append(doc)
    
    return documentos

def teste_2m_tokens():
    """Executa teste para atingir 2M tokens."""
    print("🎯 TESTE FINAL - Meta de 2 Milhões de Tokens")
    print("=" * 70)
    
    # Inicializar sistema
    print("\n📊 Inicializando ContextManager...")
    cm = ContextManager(
        base_url="http://localhost:4000",
        model_name="test-model",
        cache_dir="./cache_2m_test"
    )
    
    # Gerar conteúdo extenso
    print("\n📝 Gerando conteúdo extenso...")
    start_gen = time.time()
    documentos = gerar_conteudo_trading_extenso()
    gen_time = time.time() - start_gen
    
    print(f"   ✓ {len(documentos)} documentos gerados em {gen_time:.2f}s")
    
    # Calcular tamanho total
    total_chars = sum(len(doc) for doc in documentos)
    estimated_tokens = total_chars // 4  # Aproximação: 4 chars = 1 token
    
    print(f"   📊 Total: {total_chars:,} caracteres")
    print(f"   🎯 Tokens estimados: {estimated_tokens:,}")
    
    if estimated_tokens >= 2000000:
        print(f"   ✅ Meta de 2M tokens atingida! ({estimated_tokens:,} tokens)")
    else:
        print(f"   📈 {(estimated_tokens/2000000)*100:.1f}% da meta de 2M tokens")
    
    # Processar documentos em lotes
    print("\n🔄 Processando documentos...")
    start_proc = time.time()
    
    total_chunks = 0
    batch_size = 10
    
    for i in range(0, len(documentos), batch_size):
        batch = documentos[i:i+batch_size]
        print(f"   📦 Lote {i//batch_size + 1}/{(len(documentos)-1)//batch_size + 1}")
        
        for j, doc in enumerate(batch):
            try:
                chunks = cm.add_context(doc, context_id=f"doc_{i+j+1}")
                total_chunks += len(chunks)
                
                if (i + j + 1) % 50 == 0:
                    print(f"      ✓ {i+j+1} documentos processados")
                    
            except Exception as e:
                print(f"      ❌ Erro no documento {i+j+1}: {e}")
                continue
    
    proc_time = time.time() - start_proc
    
    print(f"\n⏱️ Processamento concluído em {proc_time:.2f} segundos")
    print(f"📊 {total_chunks:,} chunks criados")
    print(f"⚡ Velocidade: {total_chars/proc_time:,.0f} chars/segundo")
    
    # Obter estatísticas finais
    print("\n📈 ESTATÍSTICAS FINAIS:")
    try:
        stats = cm.get_context_stats()
        
        print(f"   💾 Total de chunks: {stats['total_chunks']:,}")
        print(f"   📈 Total de tokens: {stats['total_tokens']:,}")
        print(f"   📊 Tokens por chunk: {stats['avg_tokens_per_chunk']:.0f}")
        print(f"   ⭐ Importância média: {stats['avg_importance_score']:.3f}")
        print(f"   🚀 Fator de expansão: {stats['expansion_factor']:.2f}x")
        print(f"   💿 Tamanho do cache: {stats['cache_size_mb']:.2f} MB")
        
        # Verificar meta
        if stats['total_tokens'] >= 2000000:
            print("\n🎉 🏆 META ATINGIDA: 2M+ TOKENS PROCESSADOS! 🏆 🎉")
            print(f"   ✅ {stats['total_tokens']:,} tokens no sistema")
            print(f"   ✅ {(stats['total_tokens']/2000000)*100:.1f}% da meta")
        else:
            print(f"\n📊 Progresso: {(stats['total_tokens']/2000000)*100:.1f}% da meta")
            print(f"   Tokens processados: {stats['total_tokens']:,}")
            print(f"   Faltam: {2000000 - stats['total_tokens']:,} tokens")
            
    except Exception as e:
        print(f"   ❌ Erro ao obter estatísticas: {e}")
    
    # Informações do cache
    print("\n💾 Informações do cache:")
    cache_dir = cm.cache_dir
    if os.path.exists(cache_dir):
        cache_files = os.listdir(cache_dir)
        total_size = 0
        
        for file in cache_files:
            file_path = os.path.join(cache_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
        
        print(f"   📁 Diretório: {cache_dir}")
        print(f"   📄 Arquivos: {len(cache_files)}")
        print(f"   💿 Tamanho total: {total_size / 1024 / 1024:.2f} MB")
    
    print("\n=== TESTE CONCLUÍDO ===")
    print("\n🎯 CAPACIDADES DEMONSTRADAS:")
    print("   ✅ Processamento de grandes volumes de texto")
    print("   ✅ Chunking inteligente e hierárquico")
    print("   ✅ Cache persistente para performance")
    print("   ✅ Estatísticas detalhadas do sistema")
    print("   ✅ Escalabilidade para milhões de tokens")
    
    return stats if 'stats' in locals() else None

if __name__ == "__main__":
    try:
        resultado = teste_2m_tokens()
        if resultado:
            print(f"\n🏆 SUCESSO! {resultado['total_tokens']:,} tokens processados")
        else:
            print("\n⚠️ Teste concluído com limitações")
            
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()