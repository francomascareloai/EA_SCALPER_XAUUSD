# Sistema de Contexto Expandido R1

## Descrição
Esta pasta contém todos os arquivos do Sistema de Contexto Expandido configurado para trabalhar com o modelo DeepSeek R1 através do OpenRouter.

## Arquivos Principais

### Scripts de Teste
- `teste_contexto_2m_r1.py` - Script principal de teste com o modelo R1
- `teste_contexto_simples_r1.py` - Versão simplificada para testes básicos
- `teste_r1_simples.py` - Teste direto do modelo R1

### Sistema Principal
- `sistema_contexto_expandido_2m.py` - Classe ContextManager principal
- `instalar_sistema_contexto.py` - Script de instalação automática

### Configuração
- `.env` - Variáveis de ambiente (configurar API_KEY)
- `.env.example` - Exemplo de configuração
- `litellm_simple.yaml` - Configuração do LiteLLM
- `requirements.txt` - Dependências Python

### Documentação
- `DOCUMENTACAO_LITELLM_OPENROUTER.md` - Guia de uso do LiteLLM
- `GUIA_AUMENTAR_CONTEXTO_LOCAL.md` - Como expandir contexto
- `RESUMO_COMPLETO_PROJETO.md` - Visão geral completa
- `INDICE_ARQUIVOS.md` - Índice de todos os arquivos

## Como Usar

1. **Configurar ambiente:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Editar .env com sua API_KEY do OpenRouter
   ```

2. **Executar teste básico:**
   ```bash
   python teste_contexto_simples_r1.py
   ```

## Recursos

-  Suporte a até 2 milhões de tokens
-  Cache inteligente de chunks
-  Busca semântica com embeddings
-  Integração com DeepSeek R1
-  Proteção contra rate limiting

---
*Pasta criada automaticamente em: 24/08/2025*
