#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo Prático: Sistema de Contexto Expandido para 2M Tokens

Este script demonstra como usar o sistema de contexto expandido
para processar documentos grandes que excedem o limite de 163k tokens
do OpenRouter, expandindo efetivamente para 2 milhões de tokens.

Autor: Assistente AI
Data: 2025
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Adicionar o diretório atual ao path para importar o sistema
sys.path.append(str(Path(__file__).parent))

try:
    from sistema_contexto_expandido_2m import ContextManager
except ImportError:
    print("❌ Erro: Não foi possível importar o ContextManager")
    print("Certifique-se de que o arquivo 'sistema_contexto_expandido_2m.py' está no mesmo diretório")
    exit(1)

def criar_documento_exemplo(tamanho_tokens: int = 200000) -> str:
    """
    Cria um documento de exemplo com aproximadamente o número de tokens especificado.
    
    Args:
        tamanho_tokens: Número aproximado de tokens desejado
        
    Returns:
        String com o documento de exemplo
    """
    # Aproximadamente 4 caracteres por token em português
    tamanho_chars = tamanho_tokens * 4
    
    conteudo_base = """
    Este é um documento técnico sobre trading algorítmico e análise de mercado financeiro.
    
    CAPÍTULO 1: FUNDAMENTOS DO TRADING ALGORÍTMICO
    
    O trading algorítmico representa uma revolução na forma como operamos nos mercados financeiros.
    Através de algoritmos sofisticados, podemos analisar grandes volumes de dados em tempo real,
    identificar padrões de mercado e executar operações com precisão milissegunda.
    
    1.1 Conceitos Básicos
    
    Os Expert Advisors (EAs) são programas automatizados que executam estratégias de trading
    baseadas em regras pré-definidas. Estes sistemas podem operar 24/7, eliminando o fator
    emocional das decisões de trading e garantindo consistência na execução das estratégias.
    
    1.2 Tipos de Estratégias
    
    - Scalping: Estratégias de alta frequência que buscam pequenos lucros em timeframes curtos
    - Grid Trading: Sistemas que colocam ordens em intervalos regulares
    - Trend Following: Estratégias que seguem a direção principal do mercado
    - Mean Reversion: Sistemas que apostam no retorno dos preços à média
    - Arbitragem: Exploração de diferenças de preços entre mercados
    
    CAPÍTULO 2: ANÁLISE TÉCNICA AVANÇADA
    
    A análise técnica moderna incorpora conceitos avançados como Smart Money Concepts (SMC)
    e Inner Circle Trader (ICT) methodologies. Estes conceitos focam na identificação de
    zonas de liquidez, order blocks e pontos de interesse institucional.
    
    2.1 Order Blocks
    
    Order blocks são zonas de preço onde grandes instituições colocaram ordens significativas.
    Estas zonas frequentemente atuam como suporte ou resistência e podem ser identificadas
    através da análise de volume e estrutura de mercado.
    
    2.2 Liquidity Zones
    
    As zonas de liquidez representam áreas onde há concentração de stops e ordens pendentes.
    O smart money frequentemente move o preço para estas zonas para coletar liquidez antes
    de iniciar movimentos direcionais significativos.
    
    CAPÍTULO 3: GESTÃO DE RISCO
    
    A gestão de risco é fundamental para o sucesso a longo prazo no trading algorítmico.
    Sistemas robustos incorporam múltiplas camadas de proteção, incluindo:
    
    - Stop Loss dinâmico baseado em volatilidade
    - Position sizing adaptativo
    - Drawdown máximo permitido
    - Correlação entre ativos
    - Exposure máximo por trade
    
    3.1 Compliance FTMO
    
    Para sistemas compatíveis com FTMO e outras prop firms, é essencial implementar:
    - Maximum daily loss protection
    - Maximum total drawdown monitoring
    - News filter para evitar trading durante eventos de alto impacto
    - Trailing stop para proteger lucros
    
    CAPÍTULO 4: OTIMIZAÇÃO E BACKTESTING
    
    O processo de otimização deve ser cuidadoso para evitar overfitting. Técnicas recomendadas:
    
    - Walk-forward analysis
    - Out-of-sample testing
    - Monte Carlo simulation
    - Stress testing em diferentes condições de mercado
    
    4.1 Métricas de Performance
    
    - Profit Factor
    - Sharpe Ratio
    - Maximum Drawdown
    - Win Rate
    - Average Trade Duration
    - Recovery Factor
    
    CAPÍTULO 5: IMPLEMENTAÇÃO PRÁTICA
    
    A implementação de sistemas de trading requer atenção a detalhes técnicos:
    
    - Latência de execução
    - Slippage management
    - Broker compatibility
    - VPS requirements
    - Monitoring e alertas
    
    Este documento continua com análises detalhadas de cada aspecto do trading algorítmico,
    incluindo exemplos de código, estudos de caso e melhores práticas da indústria.
    """
    
    # Repetir o conteúdo até atingir o tamanho desejado
    repeticoes = max(1, tamanho_chars // len(conteudo_base))
    documento = ""
    
    for i in range(repeticoes):
        documento += f"\n\n=== SEÇÃO {i+1} ===\n\n"
        documento += conteudo_base
        
        # Adicionar variações para tornar o conteúdo mais diverso
        if i % 3 == 0:
            documento += "\n\nANÁLISE DE MERCADO ESPECÍFICA:\n"
            documento += f"Nesta seção {i+1}, analisamos padrões específicos do par XAUUSD (Ouro vs Dólar).\n"
            documento += "O ouro apresenta características únicas como safe haven asset.\n"
        elif i % 3 == 1:
            documento += "\n\nESTRATÉGIAS AVANÇADAS:\n"
            documento += f"Implementação de algoritmos de machine learning na seção {i+1}.\n"
            documento += "Uso de redes neurais para predição de movimentos de preço.\n"
        else:
            documento += "\n\nCASOS DE ESTUDO:\n"
            documento += f"Análise de performance histórica - Estudo {i+1}.\n"
            documento += "Resultados de backtesting em diferentes períodos de mercado.\n"
    
    return documento

def demonstrar_processamento_2m_tokens():
    """
    Demonstra o processamento de um documento de 2 milhões de tokens.
    """
    print("🚀 Iniciando demonstração do Sistema de Contexto Expandido para 2M Tokens")
    print("=" * 80)
    
    # Verificar se as dependências estão instaladas
    try:
        import sentence_transformers
        import sklearn
        print("✅ Dependências verificadas com sucesso")
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("Execute: pip install -r requirements.txt")
        return
    
    # Inicializar o sistema
    print("\n📊 Inicializando Sistema de Contexto Expandido...")
    sistema = ContextManager(
        base_url="http://localhost:4000",
        model_name="deepseek-r1-free",
        cache_dir='./cache_contexto_2m'
    )
    
    # Criar documento de exemplo (aproximadamente 500k tokens)
    print("\n📝 Criando documento de exemplo (500k tokens)...")
    documento_grande = criar_documento_exemplo(500000)
    
    print(f"📏 Documento criado com {len(documento_grande):,} caracteres")
    print(f"📊 Estimativa: ~{len(documento_grande) // 4:,} tokens")
    
    # Simular múltiplos documentos para chegar a 2M tokens
    documentos = []
    for i in range(4):  # 4 documentos de 500k = 2M tokens
        doc_variacao = documento_grande.replace(
            "Este é um documento técnico",
            f"Este é o documento técnico #{i+1}"
        )
        documentos.append({
            'id': f'doc_{i+1}',
            'titulo': f'Manual de Trading Algorítmico - Volume {i+1}',
            'conteudo': doc_variacao
        })
    
    print(f"\n📚 Criados {len(documentos)} documentos para totalizar ~2M tokens")
    
    # Processar cada documento
    resultados = []
    tempo_inicio = time.time()
    
    for i, doc in enumerate(documentos, 1):
        print(f"\n🔄 Processando documento {i}/{len(documentos)}: {doc['titulo']}")
        
        # Pergunta de exemplo
        pergunta = f"""
        Com base no documento {i}, responda:
        1. Quais são as principais estratégias de trading mencionadas?
        2. Como implementar gestão de risco adequada?
        3. Quais métricas são importantes para avaliar performance?
        4. Como garantir compliance com FTMO?
        """
        
        try:
            # Processar com o sistema de contexto expandido
            resposta = sistema.processar_contexto_expandido(
                texto=doc['conteudo'],
                pergunta=pergunta,
                max_tokens_resposta=1000
            )
            
            resultado = {
                'documento': doc['titulo'],
                'tokens_processados': len(doc['conteudo']) // 4,
                'resposta': resposta,
                'status': 'sucesso'
            }
            
        except Exception as e:
            resultado = {
                'documento': doc['titulo'],
                'tokens_processados': len(doc['conteudo']) // 4,
                'resposta': f'Erro: {str(e)}',
                'status': 'erro'
            }
        
        resultados.append(resultado)
        
        # Mostrar progresso
        tokens_acumulados = sum(r['tokens_processados'] for r in resultados)
        print(f"📈 Progresso: {tokens_acumulados:,} tokens processados")
        
        # Pausa entre documentos para evitar rate limiting
        if i < len(documentos):
            print("⏳ Aguardando 2 segundos...")
            time.sleep(2)
    
    tempo_total = time.time() - tempo_inicio
    tokens_totais = sum(r['tokens_processados'] for r in resultados)
    
    # Relatório final
    print("\n" + "=" * 80)
    print("📊 RELATÓRIO FINAL - PROCESSAMENTO DE 2M TOKENS")
    print("=" * 80)
    
    print(f"⏱️  Tempo total: {tempo_total:.2f} segundos")
    print(f"📊 Tokens processados: {tokens_totais:,}")
    print(f"🚀 Velocidade: {tokens_totais/tempo_total:.0f} tokens/segundo")
    print(f"✅ Documentos processados: {len([r for r in resultados if r['status'] == 'sucesso'])}/{len(resultados)}")
    
    # Salvar resultados
    arquivo_resultados = 'resultados_contexto_2m.json'
    with open(arquivo_resultados, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.time(),
            'tempo_total_segundos': tempo_total,
            'tokens_totais': tokens_totais,
            'velocidade_tokens_por_segundo': tokens_totais/tempo_total,
            'documentos_processados': len(resultados),
            'sucessos': len([r for r in resultados if r['status'] == 'sucesso']),
            'resultados': resultados
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Resultados salvos em: {arquivo_resultados}")
    
    # Mostrar estatísticas do cache
    stats_cache = sistema.get_context_stats()
    if stats_cache:
        print("\n📈 ESTATÍSTICAS DO CACHE:")
        for chave, valor in stats_cache.items():
            print(f"   {chave}: {valor}")
    
    print("\n🎉 Demonstração concluída com sucesso!")
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. Ajustar parâmetros do sistema conforme necessário")
    print("   2. Implementar processamento de arquivos reais")
    print("   3. Configurar monitoramento de performance")
    print("   4. Otimizar estratégias de chunking para seu caso de uso")

def main():
    """
    Função principal do exemplo.
    """
    print("Sistema de Contexto Expandido - Exemplo de Uso")
    print("Processamento de até 2 milhões de tokens")
    print()
    
    # Verificar variáveis de ambiente
    if not os.getenv('OPENROUTER_API_KEY'):
        print("⚠️  AVISO: OPENROUTER_API_KEY não encontrada no ambiente")
        print("   Configure sua chave de API antes de executar")
        print("   Exemplo: export OPENROUTER_API_KEY='sua-chave-aqui'")
        print()
    
    try:
        demonstrar_processamento_2m_tokens()
    except KeyboardInterrupt:
        print("\n⏹️  Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()