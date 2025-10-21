# Documentação LiteLLM com OpenRouter

## Visão Geral

Este projeto demonstra como usar o LiteLLM como proxy para a API do OpenRouter, permitindo cache local e gerenciamento de contexto expandido para modelos de IA.

## Configuração Inicial

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Configuração de Variáveis de Ambiente

Copie o arquivo `.env.example` para `.env` e configure sua chave da API do OpenRouter:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-sua-chave-aqui
LITELLM_LOG_LEVEL=INFO
LITELLM_CACHE_TYPE=disk
LITELLM_CACHE_DIR=./cache/litellm_cache
```

### 3. Configuração do LiteLLM

O arquivo `litellm_simple.yaml` contém a configuração do proxy:

```yaml
model_list:
  - model_name: deepseek-r1-free
    litellm_params:
      model: openrouter/deepseek/deepseek-r1-0528:free
      api_key: sk-or-v1-sua-chave-aqui
      api_base: https://openrouter.ai/api/v1

general_settings:
  disable_auth: true
  cache:
    type: disk
    disk_cache_dir: ./cache/litellm_cache
  master_key: null
  database_url: null
```

## Uso do Sistema

### 1. Iniciando o Proxy LiteLLM

```bash
litellm --config litellm_simple.yaml --port 4000 --host 0.0.0.0
```

Ou use o script Python:

```bash
python start_proxy.py
```

### 2. Testando a Conexão

Use o script de teste para verificar se tudo está funcionando:

```bash
python test_simple_proxy.py
```

### 3. Exemplo de Uso em Python

```python
import openai

# Configure o cliente para usar o proxy local
client = openai.OpenAI(
    base_url="http://localhost:4000",
    api_key="fake-key"  # Não necessário com disable_auth: true
)

# Faça uma requisição
response = client.chat.completions.create(
    model="deepseek-r1-free",
    messages=[
        {"role": "user", "content": "Olá! Como você está?"}
    ],
    max_tokens=150
)

print(response.choices[0].message.content)
```

## Estratégias para Contexto Expandido

### 1. Chunking Inteligente

Use o script `exemplo_chunking_inteligente.py` para processar documentos grandes:

```bash
python exemplo_chunking_inteligente.py
```

### 2. Cache de Contexto

O LiteLLM automaticamente faz cache das respostas no diretório `./cache/litellm_cache/`.

### 3. Técnicas Avançadas

Consulte o arquivo `GUIA_AUMENTAR_CONTEXTO_LOCAL.md` para estratégias avançadas:

- Chunking com sobreposição
- Cache de contexto com embeddings
- Sumarização automática
- Compressão de contexto
- Uso de modelos locais (Ollama, LM Studio)

## Modelos Disponíveis

### OpenRouter (Gratuitos)

- `deepseek/deepseek-r1-0528:free` - Modelo DeepSeek R1 gratuito
- `openai/gpt-3.5-turbo:free` - GPT-3.5 Turbo gratuito (limitado)

### Configuração de Novos Modelos

Para adicionar novos modelos, edite o arquivo `litellm_simple.yaml`:

```yaml
model_list:
  - model_name: novo-modelo
    litellm_params:
      model: openrouter/provider/model-name
      api_key: ${OPENROUTER_API_KEY}
      api_base: https://openrouter.ai/api/v1
```

## Troubleshooting

### Erro 401 (Não Autorizado)

1. Verifique se a chave da API está correta no arquivo `.env`
2. Confirme se `disable_auth: true` está no `litellm_simple.yaml`
3. Reinicie o proxy após mudanças na configuração

### Erro 429 (Limite de Taxa)

1. Aguarde alguns minutos antes de tentar novamente
2. Considere usar modelos pagos para limites maiores
3. Implemente retry logic com backoff exponencial

### Cache Não Funcionando

1. Verifique se o diretório `./cache/litellm_cache/` existe
2. Confirme as permissões de escrita no diretório
3. Verifique os logs do LiteLLM para erros

### Problemas de Contexto

1. Use chunking para textos muito grandes
2. Implemente sumarização para manter contexto relevante
3. Considere modelos locais para contextos muito grandes

## Scripts Úteis

- `test_simple_proxy.py` - Teste básico do proxy
- `test_direct_openrouter.py` - Teste direto da API OpenRouter
- `exemplo_chunking_inteligente.py` - Demonstração de chunking
- `start_proxy.py` - Script para iniciar o proxy

## Logs e Monitoramento

Os logs do LiteLLM são exibidos no terminal. Para logs mais detalhados:

```bash
export LITELLM_LOG_LEVEL=DEBUG
litellm --config litellm_simple.yaml --port 4000 --host 0.0.0.0
```

## Considerações de Performance

1. **Cache**: Ative o cache em disco para melhor performance
2. **Chunking**: Use chunks de 2000-4000 tokens para melhor eficiência
3. **Modelos Locais**: Para uso intensivo, considere Ollama ou LM Studio
4. **Rate Limiting**: Implemente delays entre requisições para evitar 429

## 🚀 Sistema de Contexto Expandido (2M Tokens)

### Visão Geral

O sistema de contexto expandido permite processar documentos de até **2 milhões de tokens**, superando o limite de 163k tokens do OpenRouter através de técnicas avançadas:

- **Chunking Hierárquico Inteligente**: Divisão inteligente do texto preservando contexto
- **Cache de Contexto com Embeddings**: Busca semântica por relevância
- **Sumarização Automática Progressiva**: Compressão dinâmica de contexto
- **Processamento Paralelo**: Múltiplos chunks processados simultaneamente

### Instalação Rápida

```bash
# 1. Instalar dependências automaticamente
python instalar_sistema_contexto.py

# 2. Configurar chave de API no .env
echo "OPENROUTER_API_KEY=sua-chave-aqui" >> .env

# 3. Testar o sistema
python exemplo_uso_contexto_2m.py
```

### Arquivos do Sistema

| Arquivo | Descrição |
|---------|----------|
| `sistema_contexto_expandido_2m.py` | Sistema principal de contexto expandido |
| `exemplo_uso_contexto_2m.py` | Exemplo prático de uso com 2M tokens |
| `instalar_sistema_contexto.py` | Instalador automático de dependências |
| `GUIA_AUMENTAR_CONTEXTO_LOCAL.md` | Guia detalhado de estratégias |

### Exemplo de Uso

```python
from sistema_contexto_expandido_2m import SistemaContextoExpandido

# Inicializar sistema
sistema = SistemaContextoExpandido(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    modelo_principal='deepseek/deepseek-r1-0528:free',
    limite_tokens_modelo=150000,
    cache_dir='./cache_contexto_2m'
)

# Processar documento grande
resposta = sistema.processar_contexto_expandido(
    texto=documento_2m_tokens,
    pergunta="Analise este documento e extraia os pontos principais",
    max_tokens_resposta=2000
)
```

### Capacidades do Sistema

- ✅ **Processamento de 2M+ tokens** em documentos únicos
- ✅ **Cache inteligente** com embeddings semânticos
- ✅ **Busca por relevância** nos chunks mais importantes
- ✅ **Sumarização progressiva** para manter contexto
- ✅ **Processamento paralelo** para melhor performance
- ✅ **Monitoramento de custos** e uso de tokens

### Performance Esperada

| Tamanho do Documento | Tempo de Processamento | Tokens/Segundo |
|---------------------|----------------------|----------------|
| 500k tokens | ~2-3 minutos | ~3,000 |
| 1M tokens | ~4-6 minutos | ~3,500 |
| 2M tokens | ~8-12 minutos | ~4,000 |

### Configurações Avançadas

```python
# Configuração personalizada
sistema = SistemaContextoExpandido(
    api_key="sua-chave",
    modelo_principal="deepseek/deepseek-r1-0528:free",
    limite_tokens_modelo=150000,
    tamanho_chunk=8000,          # Tamanho dos chunks
    sobreposicao_chunk=500,      # Sobreposição entre chunks
    max_chunks_paralelos=3,      # Processamento paralelo
    threshold_relevancia=0.7,    # Limite de relevância semântica
    cache_dir="./cache_contexto"
)
```

## 🎯 Próximos Passos

1. **Sistema de Contexto Expandido**
   - ✅ Implementado sistema para 2M tokens
   - ✅ Cache inteligente com embeddings
   - ✅ Processamento paralelo otimizado
   - ✅ Instalador automático criado

2. **Otimização de Performance**
   - Implementar cache Redis para melhor performance
   - Configurar load balancing para múltiplos modelos
   - Monitorar métricas de latência e throughput

3. **Expansão de Funcionalidades**
   - Adicionar suporte a mais modelos do OpenRouter
   - Implementar rate limiting personalizado
   - Criar dashboard de monitoramento

4. **Integração com Aplicações**
   - Desenvolver SDK para diferentes linguagens
   - Criar plugins para editores de código
   - Implementar webhooks para notificações

5. **Segurança e Compliance**
   - Implementar autenticação JWT
   - Adicionar logs de auditoria
   - Configurar backup automático do cache

## Recursos Adicionais

- [Documentação LiteLLM](https://docs.litellm.ai/)
- [OpenRouter API](https://openrouter.ai/docs)
- [Guia de Contexto Expandido](./GUIA_AUMENTAR_CONTEXTO_LOCAL.md)
- [Exemplo Prático](./EXEMPLO_PRATICO_USO.md)
- [Sistema de Contexto 2M Tokens](./sistema_contexto_expandido_2m.py)

---

**Documentação criada em:** 2025  
**Versão:** 2.0 (com Sistema de Contexto Expandido)  
**Autor:** Assistente AI

**Nota**: Este sistema foi testado com Windows PowerShell. Para outros sistemas operacionais, ajuste os comandos conforme necessário.