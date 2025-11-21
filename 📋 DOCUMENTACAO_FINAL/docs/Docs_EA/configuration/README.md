# Documentação de Configuração - EA_SCALPER_XAUUSD

## 📋 Visão Geral

Esta seção contém a documentação completa de configurações e variáveis do projeto EA_SCALPER_XAUUSD, abrangendo todos os aspectos necessários para configurar, deploy e manter o sistema de trading automatizado.

## 📚 Estrutura da Documentação

### 🗂️ Documentos Principais

| Documento | Descrição | Status |
|------------|-----------|---------|
| [01-environment-variables.md](./01-environment-variables.md) | Variáveis de ambiente completas | ✅ Completo |
| [02-api-configuration.md](./02-api-configuration.md) | Guia de configuração de APIs | ✅ Completo |
| [03-ea-parameters.md](./03-ea-parameters.md) | Parâmetros dos Expert Advisors | ✅ Completo |
| [04-file-configuration.md](./04-file-configuration.md) | Configurações YAML/JSON/TOML | ✅ Completo |
| [05-global-constants.md](./05-global-constants.md) | Variáveis globais e constantes | ✅ Completo |
| [06-practical-examples.md](./06-practical-examples.md) | Exemplos práticos e troubleshooting | ✅ Completo |

## 🚀 Início Rápido

### 1. Setup Básico

```bash
# Clonar projeto
git clone <repository-url>
cd EA_SCALPER_XAUUSD

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar ambiente
cp .env.example .env
# Editar .env com suas chaves

# Validar configuração
python scripts/validate_config.py
```

### 2. Configuração Mínima

Edite o arquivo `.env` com as seguintes variáveis obrigatórias:

```bash
# Configuração essencial
OPENROUTER_API_KEY=sua_chave_aqui
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
```

### 3. Executar Sistema

```bash
# Modo desenvolvimento
python main.py --mode development

# Modo produção
python main.py --mode production

# Com configuração customizada
python main.py --config configs/custom_config.json
```

## 🔧 Principais Componentes

### 📊 Configuração de Trading

- **Risk Management**: Configuração de risco por trade e limites diários
- **Strategy Selection**: Habilitação/desabilitação de estratégias
- **Asset Configuration**: Parâmetros específicos por ativo (XAUUSD, Forex, Crypto)
- **Session Management**: Configuração de sessões de trading

### 🤖 Configuração de APIs

- **OpenRouter**: Configuração de modelos de linguagem
- **LiteLLM**: Proxy e cache de requisições
- **MCP**: Integração com Model Context Protocol
- **Notifications**: Telegram, Discord, Slack

### 🧠 Configuração de Machine Learning

- **Model Parameters**: Configuração de modelos ML
- **Feature Engineering**: Definição de features para treinamento
- **Prediction Thresholds**: Limiares de confiança para decisões
- **Update Frequency**: Frequência de atualização de modelos

## 📁 Estrutura de Arquivos

```
docs/configuration/
├── README.md                     # Este arquivo
├── 01-environment-variables.md   # Variáveis de ambiente
├── 02-api-configuration.md       # Configuração de APIs
├── 03-ea-parameters.md           # Parâmetros dos EAs
├── 04-file-configuration.md      # Arquivos YAML/JSON/TOML
├── 05-global-constants.md        # Constantes globais
└── 06-practical-examples.md      # Exemplos e troubleshooting
```

## 🔍 Por Onde Começar

### Para Novos Usuários

1. **Leia [01-environment-variables.md](./01-environment-variables.md)** para entender as variáveis de ambiente essenciais
2. **Siga o Quick Start** para configuração básica
3. **Consulte [06-practical-examples.md](./06-practical-examples.md)** para exemplos práticos

### Para Deploy em Produção

1. **Estude [02-api-configuration.md](./02-api-configuration.md)** para configuração segura de APIs
2. **Revise [03-ea-parameters.md](./03-ea-parameters.md)** para configuração otimizada dos EAs
3. **Use os checklists** em [06-practical-examples.md](./06-practical-examples.md)

### Para Desenvolvedores

1. **Consulte [05-global-constants.md](./05-global-constants.md)** para convenções de código
2. **Estude [04-file-configuration.md](./04-file-configuration.md)** para estrutura de configurações
3. **Use os scripts de automação** disponíveis

## ⚙️ Configurações Recomendadas

### Ambiente de Desenvolvimento

```bash
# .env.development
DEBUG_MODE=true
LOG_LEVEL=DEBUG
TESTING_MODE=true
CACHE_TYPE=local
```

### Ambiente de Produção

```bash
# .env.production
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_AUDIT_LOG=true
CACHE_TYPE=redis
REDIS_URL=redis://prod-redis:6379/0
```

## 🔧 Ferramentas e Scripts

### Scripts de Validação

- `validate_config.py` - Validação completa de configuração
- `test_apis.py` - Teste de conectividade das APIs
- `memory_profiler.py` - Análise de consumo de memória

### Scripts de Automação

- `backup_system.py` - Backup automatizado do sistema
- `config_manager.py` - Gerenciamento multi-ambiente
- `setup_production.sh` - Setup automatizado de produção

## 🚨 Troubleshooting Comum

### Problemas Frequentes

1. **API Key Inválida**
   - Verifique formato da chave em [01-environment-variables.md](./01-environment-variables.md)
   - Use script de debug em [06-practical-examples.md](./06-practical-examples.md)

2. **Alta Latência**
   - Configure cache Redis em [02-api-configuration.md](./02-api-configuration.md)
   - Otimize parâmetros em [03-ea-parameters.md](./03-ea-parameters.md)

3. **Erro de Configuração**
   - Execute script de validação
   - Consulte guia de troubleshooting em [06-practical-examples.md](./06-practical-examples.md)

## 📞 Suporte

### Recursos

- **Documentação Completa**: Todos os guias detalhados
- **Scripts de Debug**: Ferramentas de diagnóstico
- **Checklists**: Validação passo a passo
- **Exemplos Práticos**: Cenários reais implementados

### Contato

- **Issues**: GitHub repository issues
- **Documentação**: Revisar guias específicos
- **Examples**: Consultar exemplos práticos

## 🔄 Atualizações

### Versão Atual: 2.0.0

- ✅ Documentação completa de configurações
- ✅ Exemplos práticos implementados
- ✅ Scripts de automação disponíveis
- ✅ Checklists de validação

### Próximas Versões

- 📋 Template generator automático
- 📋 Configuration wizard CLI
- 📋 Integration tests automáticos
- 📋 Performance dashboard

---

## 📝 Notas Importantes

1. **Segurança**: Nunca commit arquivos `.env` com chaves reais
2. **Backup**: Sempre mantenha backup das configurações
3. **Testes**: Valide configurações em ambiente de desenvolvimento antes de produção
4. **Monitoramento**: Mantenha monitoring ativo em produção

## 🎯 Próximos Passos

1. **Configure seu ambiente** seguindo os guias
2. **Execute os scripts de validação**
3. **Teste em ambiente de desenvolvimento**
4. **Implante em produção seguindo os checklists**

---

**Última Atualização**: 18/10/2025
**Versão**: 2.0.0
**Status**: ✅ Completo e Testado