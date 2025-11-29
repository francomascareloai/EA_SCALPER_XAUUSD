# Guia de Uso Diário - EA_SCALPER_XAUUSD

## 📋 Índice
1. [Rotina Matinal de Preparação](#rotina-matinal)
2. [Monitoramento dos EAs](#monitoramento-eas)
3. [Uso do Sistema Multi-Agente](#sistema-multiagente)
4. [Análise de Resultados](#analise-resultados)
5. [Manutenção do Sistema](#manutencao-sistema)
6. [Relatórios e Documentação](#relatorios)
7. [Finalização do Dia](#finalizacao-dia)
8. [Checklists Diários](#checklists)

---

## 🌅 Rotina Matinal de Preparação (15 minutos)

### 8:00 - Verificação de Conexões

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows

# 2. Verificar API Keys
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENROUTER_API_KEY')
print(f'API Key configurada: {bool(api_key)}')
print(f'Primeiros caracteres: {api_key[:10]}...' if api_key else 'Sem API Key')
"

# 3. Testar conexão com OpenRouter
curl -s http://localhost:4000/health | python -m json.tool
```

### 8:05 - Iniciar Serviços Essenciais

#### Iniciar Proxy Server
```bash
# Terminal 1: Iniciar proxy
python scripts/python/simple_trading_proxy.py &
echo "Proxy iniciado PID: $!"
```

#### Iniciar MCP Servers (se utilizado)
```bash
# Terminal 2: MCP Code Checker
cd "🤖 AI_AGENTS/MCP_Code_Checker"
python -m mcp_code_checker --port 8001 &

# Terminal 3: MCP MetaTrader (se configurado)
cd "📚 LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/MCP_Debug"
python main.py &
```

### 8:10 - Preparar MetaTrader

1. **Abrir MetaTrader 5**
2. **Verificar conexão**:
   - Status deve mostrar "Conectado"
   - Ping < 500ms
3. **Abrir gráficos**:
   - XAUUSD M5 (principal)
   - XAUUSD M15 (análise)
   - EURUSD M5 (diversificação)

### 8:12 - Verificar Condições de Mercado

```bash
# Script de verificação de mercado
python -c "
import requests
import json

# Verificar notícias econômicas
response = requests.get('https://api.example.com/forex-calendar')
print('📅 Calendário Econômico Hoje:')
# Adicionar lógica de verificação de notícias

# Verificar horários de negociação
import datetime
now = datetime.datetime.now()
print(f'⏰ Hora atual: {now.strftime("%H:%M")}')
print(f'📊 Abertura NY: {now.replace(hour=13, minute=0) > now}')
"
```

### 8:15 - Ativar EAs

1. **Verificar arquivos de log**:
   ```bash
   tail -20 logs/system.log
   ```

2. **Ativar EAs nos gráficos**:
   - Arrastar EA_FTMO_SCALPER_ELITE para XAUUSD M5
   - Configurar parâmetros do dia
   - Acompanhar primeira inicialização

---

## 📊 Monitoramento dos EAs (Ao longo do dia)

### Monitoramento em Tempo Real

#### A cada 30 minutos
```bash
# Script de verificação rápida
python -c "
import MetaTrader5 as mt5
import time
from datetime import datetime

if mt5.initialize():
    positions = mt5.positions_get(symbol='XAUUSD')
    print(f'📈 Posições abertas: {len(positions) if positions else 0}')

    if positions:
        for pos in positions:
            profit = pos.profit
            print(f'   #{pos.ticket}: {pos.type} {pos.volume} P&L: {profit:.2f}')

    account = mt5.account_info()
    if account:
        print(f'💰 Saldo: {account.balance} | Equity: {account.equity}')
        print(f'📉 Drawdown: {((account.balance - account.equity) / account.balance * 100):.2f}%')

    mt5.shutdown()
"
```

#### A cada 2 horas
```bash
# Verificação completa do sistema
python 🔧\ WORKSPACE/Development/Core/monitor_tempo_real.py --mode=quick

# Gerar relatório de status
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py --type=daily --quick
```

### Análise de Performance

#### Métricas a observar:
- **Win Rate**: % > 55%
- **Profit Factor**: > 1.5
- **Maximum Drawdown**: < 10%
- **Sharpe Ratio**: > 1.0
- **Average Trade**: Positivo

#### Comandos de análise:
```bash
# Análise de performance dos EAs
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_avaliacao_ftmo_rigoroso.py

# Otimização rápida
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/performance_optimizer.py --quick
```

### Gerenciamento de Risco

#### Monitorar Drawdown
```bash
# Alerta de drawdown
python -c "
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '10.0'))

if mt5.initialize():
    account = mt5.account_info()
    if account:
        current_dd = abs(account.balance - account.equity) / account.balance * 100
        if current_dd > MAX_DRAWDOWN:
            print(f'⚠️ ALERTA: Drawdown atual {current_dd:.2f}% > limite {MAX_DRAWDOWN}%')
            # Enviar notificação (implementar)
        else:
            print(f'✅ Drawdown sob controle: {current_dd:.2f}%')
    mt5.shutdown()
"
```

---

## 🤖 Uso do Sistema Multi-Agente

### Iniciar Agentes de Análise

#### Agente Classificador
```bash
# Classificar novos arquivos/estratégias
python 🔧\ WORKSPACE/Development/Core/classificador_qualidade_maxima.py \
  --input="novas_estrategias/" \
  --output="data/reports/classificacao_diaria.json"

# Modo batch para múltiplos arquivos
python 🔧\ WORKSPACE/Development/Core/classificador_lote_avancado.py \
  --batch-size=10 \
  --auto-classify
```

#### Agente de Otimização
```bash
# Otimização de parâmetros
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_otimizacao_continua.py \
  --symbol=XAUUSD \
  --timeframe=M5 \
  --optimization-type=quick

# Otimização completa (fim de semana)
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_otimizacao_continua.py \
  --full-optimization \
  --duration=3600
```

#### Agente de Análise de Mercado
```bash
# Análise via MCP Claude Code
# 1. Iniciar Claude Code com MCP servers
# 2. Usar prompts predefinidos

# Prompt exemplo:
"""
Analise as condições atuais do mercado XAUUSD usando os dados disponíveis.
Considere:
1. Tendências de curto prazo (M5, M15)
2. Níveis de suporte e resistência
3. Notícias econômicas do dia
4. Indicadores técnicos relevantes

Forneça uma recomendação de trading com:
- Direção (buy/sell/hold)
- Nível de confiança (0-100%)
- Justificativa técnica
- Níveis de entrada, stop loss e take profit
"""
```

### Integração com Claude Code

#### Uso do MCP Code Checker
```bash
# Verificar qualidade do código
# No Claude Code:
/check-code 📚\ LIBRARY\02_Strategies_Legacy\EA_FTMO_SCALPER_ELITE\MQL5_Source\

# Executar testes automatizados
/run-tests 🤖\ AI_AGENTS\MCP_Code_Checker\tests\
```

#### Uso do MCP GitHub
```bash
# Sincronizar repositório
# No Claude Code:
/sync-repo
/create-branch daily-update-$(date +%Y%m%d)
/commit "Daily updates and performance reports"
```

---

## 📈 Análise de Resultados

### Relatório Diário de Performance

#### Gerar relatório automático
```bash
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py \
  --type=daily \
  --output="data/reports/daily_report_$(date +%Y%m%d).html" \
  --include-charts
```

#### Métricas do relatório:
- **Resultado financeiro do dia**
- **Número de trades**
- **Win rate diária**
- **Maior perda/ganho**
- **Horários de maior atividade**
- **Análise de erros**

### Análise Semanal

#### Todo domingo (preparação para semana):
```bash
# Relatório semanal completo
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py \
  --type=weekly \
  --period=7 \
  --deep-analysis

# Otimização semanal de parâmetros
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_otimizacao_continua.py \
  --weekly-optimization

# Backup semanal
python 🔧\ WORKSPACE/Development/Scripts/git_auto_backup.py --weekly
```

### Análise de Estratégias

#### Comparar performance de estratégias:
```bash
# Análise comparativa
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_analise_critica_avancado.py \
  --compare-strategies \
  --period=30d \
  --output="data/reports/strategy_comparison_$(date +%Y%m%d).json"
```

---

## 🔧 Manutenção do Sistema

### Manutenção Diária (5 minutos)

#### Limpeza de logs
```bash
# Manter apenas últimos 7 dias
find logs/ -name "*.log" -mtime +7 -delete

# Comprimir logs antigos
find logs/ -name "*.log" -mtime +1 -exec gzip {} \;
```

#### Verificação de espaço em disco
```bash
# Verificar uso de espaço
df -h

# Limpar cache antigo
find temp/ -type f -mtime +1 -delete
```

#### Atualização de dependências
```bash
# Verificar atualizações semanais
pip list --outdated

# Atualizar se necessário
pip install --upgrade package_name
```

### Manutenção Semanal

#### Backup automático
```bash
# Backup completo do sistema
python 🔧\ WORKSPACE/Development/Scripts/git_auto_backup.py \
  --full-backup \
  --compress \
  --upload-to-cloud  # se configurado
```

#### Otimização do banco de dados
```bash
# Otimizar arquivos de dados
python 🔧\ WORKSPACE/Development/Utils/otimizador_dados.py \
  --compress-old \
  --archive-readonly
```

### Manutenção Mensal

#### Atualização do sistema
```bash
# Verificar atualizações do projeto
git fetch origin
git log HEAD..origin/main --oneline

# Atualizar se necessário
git pull origin main
pip install -r requirements.txt --upgrade
```

#### Revisão de segurança
```bash
# Verificar vulnerabilidades
pip-audit

# Verificar configurações de segurança
python 🔧\ WORKSPACE/Development/Utils/security_audit.py
```

---

## 📋 Relatórios e Documentação

### Gerar Relatórios Automáticos

#### Relatório de Trading Diário
```bash
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py \
  --type=trading \
  --period=today \
  --format=html,pdf \
  --email-to=user@example.com
```

#### Relatório de Sistema
```bash
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py \
  --type=system \
  --include-performance,errors,usage \
  --format=json
```

### Documentação de Mudanças

#### Registrar alterações diárias
```bash
# Adicionar entrada no changelog
echo "$(date +%Y-%m-%d): $(git log -1 --pretty=%s)" >> 📋\ DOCUMENTACAO_FINAL\LOGS\CHANGELOG.md

# Documentar novos parâmetros testados
echo "$(date +%Y-%m-%d) - Novos parâmetros: [listar]" >> 📋\ DOCUMENTACAO_FINAL\RELATORIOS\PARAMETERS_TESTED.md
```

### Backup de Configurações

#### Salvar configurações do dia
```bash
mkdir -p data/backups/configs/$(date +%Y%m%d)
cp .env data/backups/configs/$(date +%Y%m%d)/
cp 📚\ LIBRARY\02_Strategies_Legacy\EA_FTMO_SCALPER_ELITE\*.set data/backups/configs/$(date +%Y%m%d)/ 2>/dev/null || true
```

---

## 🌙 Finalização do Dia (15 minutos)

### 22:00 - Parada dos EAs

1. **Fechar posições abertas** (se configurado):
```bash
python -c "
import MetaTrader5 as mt5
if mt5.initialize():
    positions = mt5.positions_get(symbol='XAUUSD')
    if positions:
        for pos in positions:
            mt5.close_position(pos)
            print(f'Fechada posição #{pos.ticket}')
    mt5.shutdown()
"
```

2. **Desativar EAs**:
   - Clique no botão "AutoTrading" no MetaTrader
   - Aguarde todos os EAs finalizarem

### 22:05 - Gerar Relatórios Finais

```bash
# Relatório do dia
python 🔧\ WORKSPACE/Development/Core/gerador_relatorios_avancados.py \
  --type=daily-final \
  --auto-save \
  --archive

# Relatório de performance
python 📚\ LIBRARY/02_Strategies_Legacy/EA_FTMO_SCALPER_ELITE/Tools/sistema_avaliacao_ftmo_rigoroso.py \
  --daily-summary
```

### 22:10 - Backup Final

```bash
# Backup dos dados do dia
python 🔧\ WORKSPACE/Development/Scripts/git_auto_backup.py \
  --daily \
  --include-logs,reports,config

# Commit para o repositório
git add .
git commit -m "Daily updates - $(date +%Y-%m-%d)"
git push origin main
```

### 22:12 - Limpeza do Sistema

```bash
# Parar serviços
pkill -f "simple_trading_proxy.py"
pkill -f "mcp_code_checker"

# Limpar arquivos temporários
rm -rf temp/*
find logs/ -name "*.tmp" -delete

# Compactar logs do dia
tar -czf logs/daily_logs_$(date +%Y%m%d).tar.gz logs/*.log
```

### 22:15 - Verificação Final

```bash
# Verificar status do sistema
python scripts/python/quick_test.py --final-check

# Verificar integridade dos dados
python 🔧\ WORKSPACE/Development/Utils/integrity_check.py

# Preparar para o próximo dia
python 🔧\ WORKSPACE/Development/Scripts/prepare_next_day.py
```

---

## ✅ Checklists Diários

### Checklist Matinal (8:00-8:15)

- [ ] Ambiente virtual ativado
- [ ] Conexões com APIs testadas
- [ ] Proxy server iniciado
- [ ] MCP servers ativos (se utilizado)
- [ ] MetaTrader conectado
- [ ] Gráficos preparados
- [ ] Condições de mercado verificadas
- [ ] EAs ativados e funcionando
- [ ] Logs sem erros críticos

### Checklist de Monitoramento (Ao longo do dia)

- [ ] Verificar posições a cada 30 min
- [ ] Monitorar drawdown
- [ ] Analisar performance a cada 2 horas
- [ ] Revisar alertas e notificações
- [ ] Ajustar parâmetros se necessário
- [ ] Documentar eventos importantes

### Checklist de Finalização (22:00-22:15)

- [ ] Fechar posições (se configurado)
- [ ] Desativar EAs
- [ ] Gerar relatórios do dia
- [ ] Executar backup diário
- [ ] Limpar arquivos temporários
- [ ] Verificar integridade do sistema
- [ ] Preparar ambiente para o próximo dia
- [ ] Documentar eventos do dia

---

## 📱 Notificações e Alertas

### Configurar Alertas Automáticos

```python
# Adicionar em scripts de monitoramento
def send_alert(message, priority="normal"):
    """Enviar alerta via email/push notification"""
    if priority == "critical":
        # Implementar notificação imediata
        send_email(message, priority="high")
        send_push_notification(message)
    elif priority == "warning":
        send_email(message, priority="normal")
    else:
        # Apenas log
        log_message(message)
```

### Tipos de Alertas

1. **Críticos**:
   - Drawdown > 8%
   - Sem conexão com broker > 5 min
   - Erro grave nos EAs

2. **Aviso**:
   - Drawdown > 5%
   - Win rate < 40% (dia)
   - API com latência alta

3. **Informativos**:
   - Metas diárias alcançadas
   - Novos parâmetros otimizados
   - Atualizações do sistema

---

## 🔄 Fluxo de Trabalho Integrado

### Integração com Claude Code

1. **Manhã**: Usar Claude para análise de mercado
2. **Durante o dia**: Consultar Claude para decisões
3. **Fim do dia**: Gerar resumos com Claude

### Automação com Scripts

1. **Executar scripts em cron** (Linux/macOS):
```bash
# crontab -e
0 8 * * * /path/to/EA_SCALPER_XAUUSD/scripts/daily_start.sh
*/30 8-22 * * * /path/to/EA_SCALPER_XAUUSD/scripts/check_positions.sh
0 22 * * * /path/to/EA_SCALPER_XAUUSD/scripts/daily_end.sh
```

2. **Agendador de tarefas** (Windows):
   - Configurar Task Scheduler
   - Executar scripts em horários definidos

Este guia garante o uso eficiente e seguro do sistema EA_SCALPER_XAUUSD no dia a dia, maximizando a performance e minimizando riscos operacionais.