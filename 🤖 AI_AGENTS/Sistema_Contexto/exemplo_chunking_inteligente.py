#!/usr/bin/env python3
"""
Exemplo prático de chunking inteligente para aumentar janela de contexto
Use este script para processar documentos grandes com o LiteLLM
"""

import requests
import json
import time
import re
from typing import List, Dict, Any
from datetime import datetime

# Configurações
BASE_URL = "http://localhost:4000"
MODEL = "deepseek-r1-free"

class ChunkingInteligente:
    def __init__(self, base_url=BASE_URL, model=MODEL):
        self.base_url = base_url
        self.model = model
    
    def chunk_text(self, text: str, chunk_size: int = 30000, overlap: int = 2000) -> List[str]:
        """Divide texto em chunks com sobreposição inteligente"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Último chunk
                chunks.append(text[start:])
                break
            
            # Encontrar quebra natural (final de parágrafo, frase ou palavra)
            chunk_text = text[start:end]
            
            # Procurar por quebra de parágrafo
            last_paragraph = chunk_text.rfind('\n\n')
            if last_paragraph > chunk_size * 0.7:
                end = start + last_paragraph + 2
            else:
                # Procurar por final de frase
                last_sentence = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )
                if last_sentence > chunk_size * 0.8:
                    end = start + last_sentence + 2
                else:
                    # Procurar por espaço (quebra de palavra)
                    last_space = chunk_text.rfind(' ')
                    if last_space > chunk_size * 0.9:
                        end = start + last_space
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def make_request(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> Dict[str, Any]:
        """Faz requisição para o modelo"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'content': data['choices'][0]['message']['content'],
                    'usage': data.get('usage', {})
                }
            elif response.status_code == 429:
                print("⏳ Rate limit detectado, aguardando...")
                time.sleep(10)
                return self.make_request(messages, max_tokens)  # Retry
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_chunk(self, chunk: str, query: str, chunk_num: int, total_chunks: int) -> Dict[str, Any]:
        """Processa um chunk individual"""
        print(f"📝 Processando chunk {chunk_num}/{total_chunks} ({len(chunk):,} caracteres)...")
        
        messages = [
            {
                "role": "system",
                "content": f"Você está analisando a parte {chunk_num} de {total_chunks} de um documento. Responda baseado apenas neste trecho."
            },
            {
                "role": "user",
                "content": f"Pergunta: {query}\n\nTrecho do documento:\n{chunk}"
            }
        ]
        
        result = self.make_request(messages)
        
        if result['success']:
            print(f"  ✅ Chunk {chunk_num} processado com sucesso")
        else:
            print(f"  ❌ Erro no chunk {chunk_num}: {result['error']}")
        
        return result
    
    def combine_results(self, results: List[str], query: str) -> Dict[str, Any]:
        """Combina resultados de múltiplos chunks"""
        print("🔄 Combinando resultados...")
        
        # Filtrar apenas resultados válidos
        valid_results = [r for r in results if r and len(r.strip()) > 0]
        
        if not valid_results:
            return {
                'success': False,
                'error': 'Nenhum resultado válido para combinar'
            }
        
        if len(valid_results) == 1:
            return {
                'success': True,
                'content': valid_results[0]
            }
        
        # Combinar múltiplos resultados
        combined_text = "\n\n".join([f"Análise {i+1}: {result}" for i, result in enumerate(valid_results)])
        
        messages = [
            {
                "role": "system",
                "content": "Você recebeu várias análises parciais de um documento. Combine-as em uma resposta coerente e completa."
            },
            {
                "role": "user",
                "content": f"Pergunta original: {query}\n\nAnálises parciais:\n{combined_text}\n\nPor favor, forneça uma resposta final combinando essas análises:"
            }
        ]
        
        return self.make_request(messages, max_tokens=1200)
    
    def process_large_document(self, document: str, query: str, chunk_size: int = 30000) -> Dict[str, Any]:
        """Processa documento grande usando chunking inteligente"""
        print(f"📄 Processando documento de {len(document):,} caracteres")
        print(f"❓ Pergunta: {query}")
        print("="*60)
        
        # Dividir em chunks
        chunks = self.chunk_text(document, chunk_size)
        print(f"📝 Documento dividido em {len(chunks)} chunks")
        
        # Processar cada chunk
        results = []
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks, 1):
            result = self.process_chunk(chunk, query, i, len(chunks))
            
            if result['success']:
                results.append(result['content'])
                successful_chunks += 1
            else:
                print(f"  ⚠️ Chunk {i} falhou, continuando...")
            
            # Pausa entre chunks para evitar rate limit
            if i < len(chunks):
                time.sleep(2)
        
        print(f"\n📊 Resumo: {successful_chunks}/{len(chunks)} chunks processados com sucesso")
        
        if successful_chunks == 0:
            return {
                'success': False,
                'error': 'Nenhum chunk foi processado com sucesso'
            }
        
        # Combinar resultados
        final_result = self.combine_results(results, query)
        
        if final_result['success']:
            print("\n✅ Processamento concluído com sucesso!")
        else:
            print(f"\n❌ Erro ao combinar resultados: {final_result['error']}")
        
        return final_result

def exemplo_uso():
    """Exemplo de como usar o chunking inteligente"""
    
    # Criar texto de exemplo (simula documento grande)
    texto_exemplo = """
    Este é um exemplo de documento muito longo que precisa ser processado usando chunking inteligente.
    
    Capítulo 1: Introdução
    A inteligência artificial é uma área da ciência da computação que se concentra na criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana. Isso inclui aprendizado, raciocínio, percepção, compreensão de linguagem natural e resolução de problemas.
    
    Capítulo 2: História da IA
    A história da inteligência artificial remonta aos anos 1950, quando Alan Turing propôs o famoso "Teste de Turing" como uma forma de avaliar se uma máquina pode exibir comportamento inteligente equivalente ao de um ser humano.
    
    Capítulo 3: Tipos de IA
    Existem diferentes tipos de inteligência artificial, incluindo IA fraca (ou estreita), que é projetada para realizar tarefas específicas, e IA forte (ou geral), que teria a capacidade de entender, aprender e aplicar conhecimento em uma ampla gama de tarefas.
    
    Capítulo 4: Aplicações Modernas
    Hoje, a IA é usada em muitas aplicações, desde assistentes virtuais como Siri e Alexa até sistemas de recomendação em plataformas de streaming, carros autônomos, diagnósticos médicos e muito mais.
    
    Capítulo 5: Desafios e Considerações Éticas
    Com o avanço da IA, surgem importantes questões éticas e desafios, incluindo preocupações sobre privacidade, viés algorítmico, desemprego tecnológico e a necessidade de regulamentação adequada.
    
    Conclusão
    A inteligência artificial continua a evoluir rapidamente, prometendo transformar muitos aspectos da sociedade. É importante que seu desenvolvimento seja guiado por princípios éticos e considerações sobre seu impacto na humanidade.
    """ * 20  # Repetir para criar documento maior
    
    # Inicializar o processador
    processor = ChunkingInteligente()
    
    # Fazer pergunta sobre o documento
    pergunta = "Quais são os principais tópicos abordados neste documento sobre inteligência artificial?"
    
    # Processar documento
    resultado = processor.process_large_document(texto_exemplo, pergunta)
    
    # Mostrar resultado
    print("\n" + "="*60)
    print("📋 RESULTADO FINAL")
    print("="*60)
    
    if resultado['success']:
        print(f"✅ Resposta: {resultado['content']}")
        
        if 'usage' in resultado:
            usage = resultado['usage']
            print(f"\n📊 Estatísticas:")
            print(f"  - Tokens totais: {usage.get('total_tokens', 'N/A')}")
            print(f"  - Tokens de entrada: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  - Tokens de saída: {usage.get('completion_tokens', 'N/A')}")
    else:
        print(f"❌ Erro: {resultado['error']}")

def processar_arquivo_real():
    """Processa um arquivo real do disco"""
    import os
    
    # Listar arquivos de texto disponíveis
    arquivos_txt = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not arquivos_txt:
        print("❌ Nenhum arquivo .txt encontrado no diretório atual")
        print("💡 Crie um arquivo de texto ou use o exemplo integrado")
        return
    
    print("📁 Arquivos disponíveis:")
    for i, arquivo in enumerate(arquivos_txt, 1):
        tamanho = os.path.getsize(arquivo)
        print(f"  {i}. {arquivo} ({tamanho:,} bytes)")
    
    try:
        escolha = int(input("\nEscolha um arquivo (número): ")) - 1
        arquivo_escolhido = arquivos_txt[escolha]
        
        print(f"📖 Carregando {arquivo_escolhido}...")
        
        with open(arquivo_escolhido, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        pergunta = input("❓ Qual sua pergunta sobre este documento? ")
        
        processor = ChunkingInteligente()
        resultado = processor.process_large_document(conteudo, pergunta)
        
        print("\n" + "="*60)
        print("📋 RESULTADO FINAL")
        print("="*60)
        
        if resultado['success']:
            print(f"✅ Resposta: {resultado['content']}")
        else:
            print(f"❌ Erro: {resultado['error']}")
            
    except (ValueError, IndexError):
        print("❌ Escolha inválida")
    except FileNotFoundError:
        print("❌ Arquivo não encontrado")
    except Exception as e:
        print(f"❌ Erro: {e}")

def main():
    """Função principal"""
    print("🚀 CHUNKING INTELIGENTE PARA CONTEXTO EXPANDIDO")
    print("="*60)
    print("Este script permite processar documentos grandes usando chunking inteligente")
    print("")
    
    while True:
        print("\n📋 Opções:")
        print("1. Usar exemplo integrado")
        print("2. Processar arquivo do disco")
        print("3. Sair")
        
        escolha = input("\nEscolha uma opção: ").strip()
        
        if escolha == '1':
            exemplo_uso()
        elif escolha == '2':
            processar_arquivo_real()
        elif escolha == '3':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()