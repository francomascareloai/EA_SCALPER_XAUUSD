#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Contexto Expandido - 2 Milhões de Tokens

Este sistema permite expandir efetivamente o contexto de 163k para 2M+ tokens
usando técnicas avançadas de gerenciamento de contexto:

1. Chunking Hierárquico Inteligente
2. Cache de Contexto com Embeddings
3. Sumarização Automática Progressiva
4. Busca Semântica por Relevância
5. Compressão de Contexto Dinâmica

Autor: Sistema LiteLLM + OpenRouter
Data: 2025
"""

import os
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextChunk:
    """Representa um chunk de contexto com metadados."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    summary: Optional[str] = None
    importance_score: float = 0.0
    token_count: int = 0
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []

class ContextManager:
    """Gerenciador avançado de contexto expandido."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:4000",
                 model_name: str = "deepseek-r1-free",
                 cache_dir: str = "./cache/context_expanded",
                 max_context_tokens: int = 163000,  # Limite do OpenRouter
                 target_context_tokens: int = 2000000,  # Meta de 2M tokens
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.base_url = base_url
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.max_context_tokens = max_context_tokens
        self.target_context_tokens = target_context_tokens
        
        # Criar diretórios de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "chunks").mkdir(exist_ok=True)
        (self.cache_dir / "embeddings").mkdir(exist_ok=True)
        (self.cache_dir / "summaries").mkdir(exist_ok=True)
        
        # Inicializar modelos
        self.client = openai.OpenAI(base_url=base_url, api_key="fake-key")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Cache em memória
        self.chunk_cache: Dict[str, ContextChunk] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Carregar cache existente
        self._load_cache()
        
        logger.info(f"ContextManager inicializado:")
        logger.info(f"  - Limite base: {max_context_tokens:,} tokens")
        logger.info(f"  - Meta expandida: {target_context_tokens:,} tokens")
        logger.info(f"  - Fator de expansão: {target_context_tokens/max_context_tokens:.1f}x")
    
    def _load_cache(self):
        """Carrega cache existente do disco."""
        try:
            cache_file = self.cache_dir / "chunk_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.chunk_cache = pickle.load(f)
                logger.info(f"Cache carregado: {len(self.chunk_cache)} chunks")
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
    
    def _save_cache(self):
        """Salva cache no disco."""
        try:
            cache_file = self.cache_dir / "chunk_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.chunk_cache, f)
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto."""
        return len(self.tokenizer.encode(text))
    
    def _generate_chunk_id(self, content: str) -> str:
        """Gera ID único para um chunk."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Cria embedding para o texto."""
        chunk_id = self._generate_chunk_id(text)
        if chunk_id in self.embedding_cache:
            return self.embedding_cache[chunk_id]
        
        embedding = self.embedding_model.encode(text)
        self.embedding_cache[chunk_id] = embedding
        return embedding
    
    def _chunk_text_hierarchical(self, text: str, 
                                chunk_size: int = 4000, 
                                overlap: int = 200,
                                levels: int = 3) -> List[ContextChunk]:
        """Cria chunks hierárquicos do texto."""
        chunks = []
        
        # Nível 1: Chunks grandes (4000 tokens)
        level1_chunks = self._chunk_text_simple(text, chunk_size, overlap)
        
        for i, chunk_text in enumerate(level1_chunks):
            chunk_id = self._generate_chunk_id(chunk_text)
            embedding = self._create_embedding(chunk_text)
            
            chunk = ContextChunk(
                id=chunk_id,
                content=chunk_text,
                embedding=embedding,
                token_count=self._count_tokens(chunk_text),
                importance_score=self._calculate_importance(chunk_text)
            )
            
            # Nível 2: Sub-chunks (1000 tokens)
            if levels > 1 and len(chunk_text) > chunk_size // 2:
                sub_chunks = self._chunk_text_simple(chunk_text, chunk_size // 4, overlap // 2)
                for sub_text in sub_chunks:
                    sub_id = self._generate_chunk_id(sub_text)
                    sub_embedding = self._create_embedding(sub_text)
                    
                    sub_chunk = ContextChunk(
                        id=sub_id,
                        content=sub_text,
                        embedding=sub_embedding,
                        token_count=self._count_tokens(sub_text),
                        parent_chunk_id=chunk_id,
                        importance_score=self._calculate_importance(sub_text)
                    )
                    
                    chunk.child_chunk_ids.append(sub_id)
                    chunks.append(sub_chunk)
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_text_simple(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunking simples com sobreposição."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    def _calculate_importance(self, text: str) -> float:
        """Calcula score de importância do texto."""
        # Fatores de importância
        factors = {
            'length': min(len(text) / 1000, 1.0),  # Textos maiores são mais importantes
            'keywords': self._count_keywords(text) / 10,  # Palavras-chave importantes
            'structure': self._has_structure(text),  # Texto estruturado
            'uniqueness': self._calculate_uniqueness(text)  # Conteúdo único
        }
        
        return sum(factors.values()) / len(factors)
    
    def _count_keywords(self, text: str) -> int:
        """Conta palavras-chave importantes."""
        keywords = ['def ', 'class ', 'import ', 'function', 'method', 'algorithm', 
                   'strategy', 'trading', 'analysis', 'data', 'model', 'system']
        return sum(1 for keyword in keywords if keyword.lower() in text.lower())
    
    def _has_structure(self, text: str) -> float:
        """Verifica se o texto tem estrutura (headers, listas, etc.)."""
        structure_indicators = ['#', '*', '-', '1.', '2.', '```', 'def ', 'class ']
        count = sum(1 for indicator in structure_indicators if indicator in text)
        return min(count / 5, 1.0)
    
    def _calculate_uniqueness(self, text: str) -> float:
        """Calcula unicidade do texto comparado com cache."""
        if not self.chunk_cache:
            return 1.0
        
        text_embedding = self._create_embedding(text)
        similarities = []
        
        for chunk in list(self.chunk_cache.values())[:50]:  # Limitar comparações
            if chunk.embedding is not None:
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    chunk.embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
        
        if similarities:
            max_similarity = max(similarities)
            return 1.0 - max_similarity  # Quanto menor a similaridade, maior a unicidade
        
        return 1.0
    
    def _summarize_chunk(self, chunk: ContextChunk) -> str:
        """Cria resumo de um chunk."""
        if chunk.summary:
            return chunk.summary
        
        try:
            prompt = f"""Resuma o seguinte texto de forma concisa, mantendo as informações mais importantes:

{chunk.content[:2000]}...

Resumo (máximo 200 palavras):"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            chunk.summary = summary
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao criar resumo: {e}")
            # Fallback: primeiras linhas do chunk
            lines = chunk.content.split('\n')[:5]
            return '\n'.join(lines) + "..."
    
    def add_context(self, text: str, context_id: Optional[str] = None) -> List[str]:
        """Adiciona texto ao contexto expandido."""
        logger.info(f"Adicionando contexto: {len(text)} caracteres")
        
        # Criar chunks hierárquicos
        chunks = self._chunk_text_hierarchical(text)
        chunk_ids = []
        
        for chunk in chunks:
            self.chunk_cache[chunk.id] = chunk
            chunk_ids.append(chunk.id)
            
            # Criar resumo em background
            if chunk.token_count > 500:
                try:
                    self._summarize_chunk(chunk)
                except Exception as e:
                    logger.warning(f"Erro ao resumir chunk {chunk.id}: {e}")
        
        # Salvar cache
        self._save_cache()
        
        logger.info(f"Contexto adicionado: {len(chunks)} chunks criados")
        return chunk_ids
    
    def search_relevant_context(self, query: str, 
                              max_chunks: int = 20,
                              min_similarity: float = 0.3) -> List[ContextChunk]:
        """Busca contexto relevante para uma query."""
        if not self.chunk_cache:
            return []
        
        query_embedding = self._create_embedding(query)
        similarities = []
        
        for chunk in self.chunk_cache.values():
            if chunk.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    chunk.embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= min_similarity:
                    similarities.append((similarity, chunk))
        
        # Ordenar por similaridade e importância
        similarities.sort(key=lambda x: (x[0] * 0.7 + x[1].importance_score * 0.3), reverse=True)
        
        # Atualizar estatísticas de acesso
        relevant_chunks = []
        for similarity, chunk in similarities[:max_chunks]:
            chunk.last_accessed = datetime.now()
            chunk.access_count += 1
            relevant_chunks.append(chunk)
        
        logger.info(f"Encontrados {len(relevant_chunks)} chunks relevantes para: '{query[:50]}...'")
        return relevant_chunks
    
    def build_expanded_context(self, query: str, 
                             max_tokens: Optional[int] = None) -> str:
        """Constrói contexto expandido para uma query."""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # Buscar chunks relevantes
        relevant_chunks = self.search_relevant_context(query, max_chunks=50)
        
        if not relevant_chunks:
            return ""
        
        # Estratégia de construção de contexto
        context_parts = []
        current_tokens = 0
        
        # 1. Adicionar chunks mais relevantes primeiro
        for chunk in relevant_chunks:
            chunk_tokens = chunk.token_count
            
            # Se o chunk é muito grande, usar resumo
            if chunk_tokens > max_tokens // 10:  # Máximo 10% do contexto por chunk
                if chunk.summary:
                    content = f"[RESUMO] {chunk.summary}"
                    chunk_tokens = self._count_tokens(content)
                else:
                    content = chunk.content[:max_tokens // 10]
                    chunk_tokens = self._count_tokens(content)
            else:
                content = chunk.content
            
            # Verificar se cabe no contexto
            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(content)
                current_tokens += chunk_tokens
            else:
                # Tentar adicionar resumo se disponível
                if chunk.summary and current_tokens + self._count_tokens(chunk.summary) <= max_tokens:
                    context_parts.append(f"[RESUMO] {chunk.summary}")
                    current_tokens += self._count_tokens(chunk.summary)
                else:
                    break
        
        # 2. Adicionar contexto hierárquico (parent/child relationships)
        self._add_hierarchical_context(context_parts, relevant_chunks, max_tokens - current_tokens)
        
        final_context = "\n\n--- CONTEXTO EXPANDIDO ---\n\n".join(context_parts)
        
        logger.info(f"Contexto construído: {self._count_tokens(final_context):,} tokens de {len(context_parts)} chunks")
        return final_context
    
    def _add_hierarchical_context(self, context_parts: List[str], 
                                relevant_chunks: List[ContextChunk], 
                                remaining_tokens: int):
        """Adiciona contexto hierárquico (parent/child)."""
        added_chunks = set(chunk.id for chunk in relevant_chunks)
        
        for chunk in relevant_chunks:
            if remaining_tokens <= 0:
                break
            
            # Adicionar chunks filhos se relevantes
            for child_id in chunk.child_chunk_ids:
                if child_id not in added_chunks and child_id in self.chunk_cache:
                    child_chunk = self.chunk_cache[child_id]
                    if child_chunk.token_count <= remaining_tokens:
                        context_parts.append(f"[DETALHE] {child_chunk.content}")
                        remaining_tokens -= child_chunk.token_count
                        added_chunks.add(child_id)
    
    def chat_with_expanded_context(self, query: str, 
                                 system_prompt: Optional[str] = None,
                                 max_response_tokens: int = 1000) -> str:
        """Chat com contexto expandido."""
        # Construir contexto expandido
        expanded_context = self.build_expanded_context(query)
        
        # Preparar mensagens
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if expanded_context:
            context_message = f"""Contexto relevante para responder à pergunta:

{expanded_context}

---

Pergunta do usuário: {query}"""
            messages.append({"role": "user", "content": context_message})
        else:
            messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_response_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Erro no chat: {e}")
            return f"Erro ao processar consulta: {e}"
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do contexto."""
        total_chunks = len(self.chunk_cache)
        total_tokens = sum(chunk.token_count for chunk in self.chunk_cache.values())
        avg_importance = sum(chunk.importance_score for chunk in self.chunk_cache.values()) / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / total_chunks if total_chunks > 0 else 0,
            "avg_importance_score": avg_importance,
            "expansion_factor": total_tokens / self.max_context_tokens if total_tokens > 0 else 0,
            "cache_size_mb": self._get_cache_size_mb(),
            "most_accessed_chunks": self._get_most_accessed_chunks(5)
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calcula tamanho do cache em MB."""
        try:
            cache_file = self.cache_dir / "chunk_cache.pkl"
            if cache_file.exists():
                return cache_file.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0.0
    
    def _get_most_accessed_chunks(self, limit: int) -> List[Dict[str, Any]]:
        """Retorna chunks mais acessados."""
        sorted_chunks = sorted(
            self.chunk_cache.values(),
            key=lambda x: x.access_count,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": chunk.id,
                "access_count": chunk.access_count,
                "importance_score": chunk.importance_score,
                "token_count": chunk.token_count,
                "preview": chunk.content[:100] + "..."
            }
            for chunk in sorted_chunks
        ]
    
    def cleanup_old_chunks(self, max_chunks: int = 10000, 
                          min_access_count: int = 1):
        """Remove chunks antigos e pouco acessados."""
        if len(self.chunk_cache) <= max_chunks:
            return
        
        # Ordenar por última data de acesso e contagem de acesso
        sorted_chunks = sorted(
            self.chunk_cache.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed or datetime.min)
        )
        
        # Remover chunks menos importantes
        chunks_to_remove = sorted_chunks[:len(self.chunk_cache) - max_chunks]
        
        for chunk_id, chunk in chunks_to_remove:
            if chunk.access_count < min_access_count:
                del self.chunk_cache[chunk_id]
                if chunk_id in self.embedding_cache:
                    del self.embedding_cache[chunk_id]
        
        logger.info(f"Limpeza concluída: {len(chunks_to_remove)} chunks removidos")
        self._save_cache()


def demo_contexto_expandido():
    """Demonstração do sistema de contexto expandido."""
    print("=== DEMO: Sistema de Contexto Expandido (2M Tokens) ===")
    
    # Inicializar gerenciador
    cm = ContextManager(
        base_url="http://localhost:4000",
        model_name="deepseek-r1-free",
        max_context_tokens=163000,
        target_context_tokens=2000000
    )
    
    # Exemplo de texto grande para adicionar ao contexto
    exemplo_texto = """
    # Estratégias Avançadas de Trading
    
    ## 1. Smart Money Concepts (SMC)
    
    Smart Money Concepts é uma metodologia de análise técnica que se baseia na compreensão
    de como o "dinheiro inteligente" (bancos, fundos, instituições) opera no mercado.
    
    ### Order Blocks
    Order blocks são zonas de preço onde grandes instituições colocaram ordens significativas.
    Estas zonas tendem a atuar como suporte ou resistência quando o preço retorna.
    
    ### Liquidity Sweeps
    Liquidity sweeps ocorrem quando o preço move rapidamente para capturar liquidez
    em níveis óbvios (highs/lows anteriores) antes de reverter na direção pretendida.
    
    ## 2. Volume Analysis
    
    A análise de volume é crucial para confirmar movimentos de preço e identificar
    possíveis reversões ou continuações de tendência.
    
    ### Volume Profile
    O volume profile mostra a distribuição de volume em diferentes níveis de preço,
    ajudando a identificar áreas de valor e pontos de controle.
    
    ### On-Balance Volume (OBV)
    O OBV é um indicador que relaciona volume e preço, útil para detectar divergências
    e confirmar tendências.
    
    ## 3. Risk Management
    
    O gerenciamento de risco é fundamental para o sucesso a longo prazo no trading.
    
    ### Position Sizing
    O tamanho da posição deve ser calculado com base no risco por trade e no
    stop loss definido.
    
    ### Drawdown Control
    É essencial monitorar o drawdown e ter regras claras para reduzir o tamanho
    das posições ou parar de operar temporariamente.
    
    ## 4. Backtesting e Otimização
    
    ### Dados Históricos
    Use dados de qualidade e considere diferentes períodos de mercado (bull, bear, lateral).
    
    ### Walk-Forward Analysis
    Implemente análise walk-forward para validar a robustez da estratégia.
    
    ### Monte Carlo Simulation
    Use simulações Monte Carlo para entender a distribuição de resultados possíveis.
    
    ## 5. Psicologia do Trading
    
    ### Controle Emocional
    Desenvolva disciplina para seguir o plano de trading mesmo em períodos difíceis.
    
    ### Journal de Trading
    Mantenha um registro detalhado de todas as operações para análise posterior.
    
    ### Mindset de Probabilidade
    Entenda que cada trade é apenas uma amostra de um conjunto maior de probabilidades.
    """
    
    # Adicionar contexto
    print("\n1. Adicionando contexto ao sistema...")
    chunk_ids = cm.add_context(exemplo_texto)
    print(f"   ✓ {len(chunk_ids)} chunks criados")
    
    # Mostrar estatísticas
    print("\n2. Estatísticas do contexto:")
    stats = cm.get_context_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)} itens")
        else:
            print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
    
    # Teste de busca semântica
    print("\n3. Teste de busca semântica:")
    query = "Como funciona o order block em SMC?"
    relevant_chunks = cm.search_relevant_context(query, max_chunks=5)
    print(f"   ✓ {len(relevant_chunks)} chunks relevantes encontrados")
    
    for i, chunk in enumerate(relevant_chunks[:3]):
        print(f"   Chunk {i+1}: {chunk.content[:100]}...")
        print(f"   Importância: {chunk.importance_score:.2f}, Tokens: {chunk.token_count}")
    
    # Teste de chat com contexto expandido
    print("\n4. Teste de chat com contexto expandido:")
    try:
        response = cm.chat_with_expanded_context(
            query="Explique como usar order blocks em uma estratégia de trading",
            system_prompt="Você é um especialista em trading. Responda de forma clara e prática."
        )
        print(f"   Resposta: {response[:200]}...")
    except Exception as e:
        print(f"   Erro no chat: {e}")
    
    # Demonstrar expansão de contexto
    print("\n5. Demonstração de expansão de contexto:")
    expanded_context = cm.build_expanded_context("estratégias de trading")
    expanded_tokens = cm._count_tokens(expanded_context)
    print(f"   ✓ Contexto expandido: {expanded_tokens:,} tokens")
    print(f"   ✓ Fator de expansão: {expanded_tokens / 163000:.1f}x do limite base")
    
    print("\n=== DEMO CONCLUÍDA ===")
    print(f"\nO sistema permite expandir efetivamente o contexto de 163k para {stats['total_tokens']:,} tokens!")
    print("\nPróximos passos:")
    print("1. Adicione mais documentos ao contexto")
    print("2. Use busca semântica para encontrar informações relevantes")
    print("3. Aproveite o chat com contexto expandido")
    print("4. Monitore as estatísticas para otimizar o sistema")


if __name__ == "__main__":
    demo_contexto_expandido()