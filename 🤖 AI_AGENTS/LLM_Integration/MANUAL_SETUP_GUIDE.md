# 📘 Guia de Configuração Manual

## 🔍 Diagnóstico do Problema:
O sistema não conseguiu encontrar o Python instalado. Isso pode ser devido a:
- Python não instalado
- Python não adicionado ao PATH do sistema
- Versão muito antiga do Python

## 🛠️ Solução Passo a Passo:

### 1. Verificar instalação do Python
Abra o prompt de comando e execute:
```cmd
python --version
```

Se retornar uma versão (ex: Python 3.11.4), prossiga para o passo 3.

### 2. Instalar Python (se necessário)
Baixe e instale o Python:
- Acesse [python.org/downloads](https://python.org/downloads)
- Instale a versão mais recente
- **IMPORTANTE:** Marque a opção "Add Python to PATH" durante a instalação

### 3. Instalar dependências
No prompt de comando:
```cmd
pip install litellm==1.0.0 diskcache==5.6.1
```

### 4. Configurar variável de ambiente
```cmd
setx OPENAI_API_KEY "sua_api_key_aqui"
```

### 5. Testar o sistema
Navegue até a pasta `LLM_Integration` e execute:
```cmd
python litellm_prompt_cache.py
```

## 💡 Dicas Importantes:
- Se encontrar erros de permissão, execute o prompt como administrador
- Para problemas persistentes, reinicie o computador após instalar o Python
- Atualize o pip se necessário: `python -m pip install --upgrade pip`

## 📞 Suporte:
Caso ainda tenha problemas, colete estas informações:
1. Saída de `python --version`
2. Saída de `pip --version`
3. Captura de tela do erro