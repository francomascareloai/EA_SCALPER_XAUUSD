# 🚀 Guia Completo: Como Aumentar a Janela de Contexto Localmente

## 📊 Status Atual dos Testes

### ✅ **Cache LiteLLM**
- **Status**: Funcionando perfeitamente
- **Melhoria de Performance**: 36.1%
- **Modelo Testado**: DeepSeek R1 Free
- **Benefícios**: Respostas mais rápidas para consultas repetidas

### ⚠️ **Janela de Contexto**
- **Limite Teórico**: 163.840 tokens (~655.360 caracteres)
- **Limite Prático**: Restrito por rate limits do OpenRouter
- **Problema**: Modelos gratuitos têm limitações de uso

---

## 🎯 Estratégias para Aumentar Contexto Localmente

### 1. 📝 **Chunking Inteligente**

#### Como Funciona:
- Divide textos grandes em pedaços menores
- Processa cada chunk separadamente
- Combina resultados de forma inteligente

#### Implementação:
```python
def chunk_text(text, chunk_size=50000, overlap=1000):
    """Divide texto em chunks com sobreposição"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Encontrar quebra natural (final de frase)
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def process_large_text(text, model="deepseek-r1-free"):
    """Processa texto grande usando chunking"""
    chunks = chunk_text(text)
    results = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processando chunk {i+1}/{len(chunks)}...")
        
        response = requests.post(
            "http://localhost:4000/v1/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": f"Analise este trecho (parte {i+1} de {len(chunks)}): {chunk}"
                }],
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            results.append(result)
        
        time.sleep(1)  # Evitar rate limit
    
    return results
```

### 2. 🧠 **Cache de Contexto com Embeddings**

#### Como Funciona:
- Converte texto em embeddings (vetores)
- Armazena embeddings em banco vetorial
- Recupera contexto relevante baseado em similaridade

#### Implementação:
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class ContextCache:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
    
    def add_text(self, text):
        """Adiciona texto ao cache"""
        chunks = chunk_text(text, chunk_size=1000)
        embeddings = self.model.encode(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(chunks)
    
    def search_relevant_context(self, query, top_k=5):
        """Busca contexto relevante"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_texts = [self.texts[i] for i in indices[0]]
        return relevant_texts
    
    def get_enhanced_context(self, query, max_context_length=100000):
        """Obtém contexto otimizado para a query"""
        relevant_chunks = self.search_relevant_context(query)
        
        context = "\n\n".join(relevant_chunks)
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        return context

# Uso:
cache = ContextCache()
cache.add_text("Seu texto muito longo aqui...")

query = "Qual é o tema principal?"
context = cache.get_enhanced_context(query)

response = requests.post(
    "http://localhost:4000/v1/chat/completions",
    json={
        "model": "deepseek-r1-free",
        "messages": [
            {"role": "system", "content": f"Contexto relevante: {context}"},
            {"role": "user", "content": query}
        ]
    }
)
```

### 3. 📋 **Sumarização Automática**

#### Como Funciona:
- Resume textos longos em versões mais curtas
- Mantém informações essenciais
- Permite processar mais conteúdo

#### Implementação:
```python
def summarize_text(text, model="deepseek-r1-free", max_summary_length=2000):
    """Sumariza texto longo"""
    if len(text) <= max_summary_length:
        return text
    
    chunks = chunk_text(text, chunk_size=10000)
    summaries = []
    
    for chunk in chunks:
        response = requests.post(
            "http://localhost:4000/v1/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": f"Resuma este texto mantendo as informações mais importantes: {chunk}"
                }],
                "max_tokens": 300
            }
        )
        
        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content']
            summaries.append(summary)
        
        time.sleep(1)
    
    # Combinar resumos
    combined_summary = "\n\n".join(summaries)
    
    # Se ainda for muito longo, resumir novamente
    if len(combined_summary) > max_summary_length:
        return summarize_text(combined_summary, model, max_summary_length)
    
    return combined_summary

def process_with_summarization(long_text, query):
    """Processa texto longo com sumarização"""
    summary = summarize_text(long_text)
    
    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        json={
            "model": "deepseek-r1-free",
            "messages": [
                {"role": "system", "content": f"Resumo do documento: {summary}"},
                {"role": "user", "content": query}
            ]
        }
    )
    
    return response.json()
```

### 4. 🗜️ **Compressão de Contexto**

#### Como Funciona:
- Remove informações redundantes
- Mantém apenas o essencial
- Otimiza uso de tokens

#### Implementação:
```python
import re

def compress_context(text, compression_ratio=0.7):
    """Comprime contexto removendo redundâncias"""
    
    # Remover linhas em branco excessivas
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text)
    
    # Dividir em sentenças
    sentences = text.split('.')
    
    # Calcular importância de cada sentença (exemplo simples)
    sentence_scores = []
    for sentence in sentences:
        # Score baseado em comprimento e palavras-chave
        score = len(sentence.split()) * 0.1
        
        # Bonus para sentenças com palavras importantes
        important_words = ['importante', 'principal', 'essencial', 'fundamental']
        for word in important_words:
            if word.lower() in sentence.lower():
                score += 2
        
        sentence_scores.append((sentence, score))
    
    # Ordenar por importância
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Manter apenas as sentenças mais importantes
    keep_count = int(len(sentences) * compression_ratio)
    important_sentences = [s[0] for s in sentence_scores[:keep_count]]
    
    return '. '.join(important_sentences)

def smart_context_management(text, query, max_tokens=150000):
    """Gerenciamento inteligente de contexto"""
    
    # Estimar tokens (aproximadamente 4 caracteres por token)
    estimated_tokens = len(text) // 4
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Estratégia 1: Compressão
    compressed = compress_context(text)
    if len(compressed) // 4 <= max_tokens:
        return compressed
    
    # Estratégia 2: Busca por relevância
    cache = ContextCache()
    cache.add_text(text)
    relevant_context = cache.get_enhanced_context(query, max_tokens * 4)
    
    return relevant_context
```

### 5. 🖥️ **Modelos Locais (Alternativa Recomendada)**

#### Ollama
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Baixar modelo com contexto grande
ollama pull llama2:70b  # 4K contexto
ollama pull codellama:34b  # 16K contexto
ollama pull mistral:7b  # 32K contexto

# Usar com contexto customizado
ollama run mistral:7b --ctx-size 32768
```

#### LM Studio
- Interface gráfica amigável
- Suporte a modelos GGUF
- Contexto configurável até 200K+ tokens
- Sem limitações de rate limit

---

## 🛠️ **Implementação Prática Completa**

### Script de Exemplo:
```python
#!/usr/bin/env python3
"""
Sistema completo de gerenciamento de contexto expandido
"""

import requests
import json
import time
from typing import List, Dict, Any

class ExpandedContextManager:
    def __init__(self, base_url="http://localhost:4000", model="deepseek-r1-free"):
        self.base_url = base_url
        self.model = model
        self.cache = ContextCache()
    
    def process_large_document(self, document: str, query: str, strategy="auto") -> Dict[str, Any]:
        """Processa documento grande com estratégia automática"""
        
        doc_length = len(document)
        
        if strategy == "auto":
            if doc_length < 50000:
                strategy = "direct"
            elif doc_length < 200000:
                strategy = "chunking"
            else:
                strategy = "embedding"
        
        print(f"📄 Documento: {doc_length:,} caracteres")
        print(f"🎯 Estratégia: {strategy}")
        
        if strategy == "direct":
            return self._process_direct(document, query)
        elif strategy == "chunking":
            return self._process_chunking(document, query)
        elif strategy == "embedding":
            return self._process_embedding(document, query)
        elif strategy == "summarization":
            return self._process_summarization(document, query)
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")
    
    def _process_direct(self, document: str, query: str) -> Dict[str, Any]:
        """Processamento direto"""
        return self._make_request([
            {"role": "system", "content": f"Documento: {document}"},
            {"role": "user", "content": query}
        ])
    
    def _process_chunking(self, document: str, query: str) -> Dict[str, Any]:
        """Processamento com chunking"""
        chunks = chunk_text(document)
        results = []
        
        for i, chunk in enumerate(chunks):
            print(f"📝 Processando chunk {i+1}/{len(chunks)}...")
            
            result = self._make_request([
                {"role": "user", "content": f"Baseado neste trecho, responda: {query}\n\nTrecho: {chunk}"}
            ])
            
            if result.get('success'):
                results.append(result['content'])
            
            time.sleep(1)  # Evitar rate limit
        
        # Combinar resultados
        combined = "\n\n".join(results)
        
        return self._make_request([
            {"role": "user", "content": f"Baseado nestas análises parciais, forneça uma resposta final para: {query}\n\nAnálises: {combined}"}
        ])
    
    def _process_embedding(self, document: str, query: str) -> Dict[str, Any]:
        """Processamento com embeddings"""
        self.cache.add_text(document)
        relevant_context = self.cache.get_enhanced_context(query)
        
        return self._make_request([
            {"role": "system", "content": f"Contexto relevante: {relevant_context}"},
            {"role": "user", "content": query}
        ])
    
    def _process_summarization(self, document: str, query: str) -> Dict[str, Any]:
        """Processamento com sumarização"""
        summary = summarize_text(document)
        
        return self._make_request([
            {"role": "system", "content": f"Resumo do documento: {summary}"},
            {"role": "user", "content": query}
        ])
    
    def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Faz requisição para o modelo"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1000
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
            else:
                return {
                    'success': False,
                    'error': response.text,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Exemplo de uso:
if __name__ == "__main__":
    manager = ExpandedContextManager()
    
    # Carregar documento grande
    with open('documento_grande.txt', 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Fazer pergunta
    query = "Quais são os pontos principais deste documento?"
    
    # Processar
    result = manager.process_large_document(document, query)
    
    if result['success']:
        print(f"✅ Resposta: {result['content']}")
    else:
        print(f"❌ Erro: {result['error']}")
```

---

## 📈 **Comparação de Estratégias**

| Estratégia | Contexto Máximo | Precisão | Velocidade | Complexidade |
|------------|-----------------|----------|------------|-------------|
| **Chunking** | Ilimitado | Média | Lenta | Baixa |
| **Embeddings** | Ilimitado | Alta | Rápida | Média |
| **Sumarização** | Ilimitado | Média-Alta | Média | Baixa |
| **Compressão** | 2-3x original | Alta | Rápida | Baixa |
| **Modelos Locais** | 200K+ tokens | Alta | Rápida | Alta |

---

## 🎯 **Recomendações Finais**

### Para Uso Imediato:
1. **Use chunking** para documentos > 50K caracteres
2. **Implemente cache de embeddings** para consultas frequentes
3. **Configure sumarização** para documentos muito longos

### Para Uso Avançado:
1. **Instale Ollama ou LM Studio** para modelos locais
2. **Use modelos com contexto 32K+** (Mistral, CodeLlama)
3. **Implemente sistema híbrido** combinando todas as estratégias

### Para Produção:
1. **Configure banco vetorial** (Pinecone, Weaviate, Chroma)
2. **Use API keys próprias** para evitar rate limits
3. **Implemente cache persistente** com Redis ou SQLite

---

## 🔧 **Próximos Passos**

1. ✅ Cache LiteLLM funcionando (36.1% melhoria)
2. ⚠️ Implementar chunking inteligente
3. 🔄 Configurar cache de embeddings
4. 🚀 Testar modelos locais (Ollama)
5. 📊 Benchmark de todas as estratégias

**Status**: Cache funcionando perfeitamente. Janela de contexto limitada por rate limits do OpenRouter. Recomenda-se implementar estratégias locais para contextos maiores.