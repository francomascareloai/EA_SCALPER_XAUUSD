# Relatório de Análise de Rede - EA_SCALPER_XAUUSD
**Data**: 2025-10-19 | **Ambiente**: WSL2/Linux | **Propósito**: Trading Algorítmico XAUUSD

---

## 1. ANÁLISE DE CONECTIVIDADE E LATÊNCIA

### Diagnóstico Atual
```
Latência Google DNS (8.8.8.8): 45-134ms (variação alta)
Conexão MT5 (trade.mql5.com): ✓ ATIVA
- DNS Resolution: 75.7ms
- TCP Connect: 112.9ms
- TLS Handshake: 257.5ms ⚠️ ALTO
- Total RTT: 296.6ms ⚠️ CRÍTICO para HFT

Servidor Broker: br.sa.web.mql5.com (177.154.156.123) - Brasil
Rota: WSL2 → Windows Host → Internet
```

### Problemas Identificados
- **Latência TLS excessiva** (257ms) - impacta execução de ordens
- **Variação de ping** (45-134ms) - indica jitter na rede
- **Sem TCP timestamps otimizados** - buffer limitado (212KB)
- **WSL2 overhead** - camada NAT adiciona 5-15ms

### Otimizações Recomendadas

#### A) VPS Dedicado de Trading (ESSENCIAL para HFT)
```bash
# Localizações por Broker:
# - XM, FXTM: Londres (Equinix LD4/LD5) - 1-5ms
# - IC Markets: NY4 Equinix ou SG1 Singapore - 0.5-2ms
# - FXPro: Londres LD4 - 1-3ms

# Provider recomendado: Contabo, Vultr High Frequency, AWS Lightsail
# Região: Europe (Frankfurt/London) ou US-East (NY)
# Specs: 2vCPU, 4GB RAM, SSD NVMe, 1Gbps dedicado
```

#### B) Otimização TCP/IP Kernel (para VPS Linux)
```bash
# /etc/sysctl.conf - Tuning para trading de baixa latência
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_fastopen = 3
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_mtu_probing = 1

# Aplicar: sudo sysctl -p
```

#### C) DNS Caching Local
```bash
# Instalar dnsmasq para resolver DNS instantaneamente
sudo apt install dnsmasq
echo "server=1.1.1.1" >> /etc/dnsmasq.conf
echo "cache-size=1000" >> /etc/dnsmasq.conf

# Reduz DNS lookup de 75ms → <1ms
```

#### D) Monitoramento Contínuo de Latência
```bash
#!/bin/bash
# /home/franco/projetos/EA_SCALPER_XAUUSD/scripts/monitor_latency.sh

BROKER="trade.mql5.com"
LOG="/var/log/trading_latency.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    PING=$(ping -c 1 -W 1 $BROKER | grep time= | cut -d'=' -f4 | cut -d' ' -f1)

    if (( $(echo "$PING > 50" | bc -l) )); then
        echo "[$TIMESTAMP] ALERTA: Latência ${PING}ms - Acima do limite!" | tee -a $LOG
        # Enviar alerta (email/telegram/webhook)
    fi

    echo "[$TIMESTAMP] Latência: ${PING}ms" >> $LOG
    sleep 5
done
```

---

## 2. SEGURANÇA DE REDE - PROTEÇÃO DE APIs E TOKENS

### Vulnerabilidades Detectadas
```
⚠️ CRÍTICO: Arquivo .env versionado com tokens expostos
⚠️ MÉDIO: Portas MCP abertas em localhost (15711, 45229, etc)
⚠️ BAIXO: Sem firewall UFW configurado
```

### Implementação de Segurança em Camadas

#### A) Vault de Credenciais (.env seguro)
```bash
# /home/franco/projetos/EA_SCALPER_XAUUSD/.env.vault
# Usar git-crypt ou SOPS para criptografar

# Instalar git-crypt
sudo apt install git-crypt
cd /home/franco/projetos/EA_SCALPER_XAUUSD
git-crypt init
echo ".env" >> .gitattributes
echo ".env filter=git-crypt diff=git-crypt" >> .gitattributes
git-crypt lock
```

#### B) Firewall UFW - Regras Mínimas
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Permitir apenas conexões essenciais
sudo ufw allow from 127.0.0.1 to any port 4000 proto tcp  # LiteLLM local
sudo ufw allow out 443/tcp  # HTTPS brokers
sudo ufw allow out 80/tcp   # HTTP APIs

# Bloquear tráfego suspeito de brokers falsos
sudo ufw deny from 185.0.0.0/8  # Bloquear IPs russos conhecidos por scam
sudo ufw enable
```

#### C) TLS Mutual Authentication (mTLS)
```python
# Para APIs críticas - autenticação de certificado cliente
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
import ssl

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_cert_chain('/path/to/client.crt', '/path/to/client.key')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', TLSAdapter())
```

#### D) Rotação Automática de API Keys
```bash
# Cronjob para rotacionar chaves a cada 30 dias
# /home/franco/projetos/EA_SCALPER_XAUUSD/scripts/rotate_keys.sh

#!/bin/bash
API_KEY_FILE="/home/franco/projetos/EA_SCALPER_XAUUSD/.env"
BACKUP_DIR="/home/franco/projetos/EA_SCALPER_XAUUSD/.env.backups"

# Backup da chave antiga
mkdir -p $BACKUP_DIR
cp $API_KEY_FILE "$BACKUP_DIR/.env.$(date +%Y%m%d)"

# Gerar nova chave (exemplo para API genérica)
NEW_KEY=$(openssl rand -hex 32)
sed -i "s/API_KEY=.*/API_KEY=$NEW_KEY/" $API_KEY_FILE

# Notificar sistema de trading para reconectar
systemctl restart trading_bot.service
```

#### E) Network Segmentation (Docker)
```yaml
# /home/franco/projetos/EA_SCALPER_XAUUSD/docker-compose.yml
version: '3.8'
services:
  trading_engine:
    image: ea_scalper:latest
    networks:
      - trading_net
    environment:
      - BROKER_API_KEY=${BROKER_API_KEY}
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true

networks:
  trading_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16
    driver_opts:
      com.docker.network.bridge.name: br_trading
```

---

## 3. TROUBLESHOOTING - DIAGNÓSTICO DE FALHAS DE CONEXÃO

### Checklist de Diagnóstico Sistemático

#### Camada 1-3: Conectividade Física/IP
```bash
# 1. Verificar interface de rede
ip link show
ip addr show

# 2. Testar gateway
ping -c 3 $(ip route | grep default | awk '{print $3}')

# 3. Verificar rotas
ip route show
mtr --report-cycles 10 trade.mql5.com
```

#### Camada 4: Transporte TCP
```bash
# 1. Verificar portas abertas do broker
nmap -p 443,8443,17000-17100 trade.mql5.com

# 2. Testar conexão TCP raw
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/trade.mql5.com/443' && echo "OK" || echo "FALHA"

# 3. Capturar handshake TCP (requer root)
sudo tcpdump -i any -nn 'host trade.mql5.com and tcp[tcpflags] & (tcp-syn|tcp-ack) != 0' -c 10
```

#### Camada 7: Aplicação MT5/API
```bash
# 1. Verificar certificado SSL do broker
echo | openssl s_client -connect trade.mql5.com:443 -servername trade.mql5.com 2>/dev/null | \
  openssl x509 -noout -dates -subject -issuer

# 2. Testar API REST do broker
curl -v -X GET "https://trade.mql5.com/api/v1/status" \
  -H "Authorization: Bearer $API_TOKEN" \
  --max-time 10

# 3. Verificar quota/rate limit
curl -I "https://api.broker.com/v1/account" \
  -H "X-API-Key: $KEY" | grep -i "x-ratelimit"
```

### Script de Diagnóstico Automatizado
```bash
#!/bin/bash
# /home/franco/projetos/EA_SCALPER_XAUUSD/scripts/network_diagnostics.sh

echo "=== DIAGNÓSTICO DE REDE PARA TRADING ==="
echo "Data: $(date)"
echo ""

# 1. Latência básica
echo "[1/6] Testando latência..."
BROKERS=("trade.mql5.com" "mt5.fxpro.com" "icmarkets-mt5.com")
for broker in "${BROKERS[@]}"; do
    PING=$(ping -c 3 -W 2 $broker 2>/dev/null | grep avg | cut -d'/' -f5)
    if [ -n "$PING" ]; then
        echo "  $broker: ${PING}ms"
    else
        echo "  $broker: FALHA ❌"
    fi
done

# 2. DNS resolução
echo ""
echo "[2/6] Verificando DNS..."
dig +short trade.mql5.com @1.1.1.1 | head -1

# 3. Portas TCP
echo ""
echo "[3/6] Testando portas do broker..."
nc -zv -w 3 trade.mql5.com 443 2>&1 | grep succeeded

# 4. Path MTU
echo ""
echo "[4/6] Verificando MTU..."
ping -c 1 -M do -s 1472 trade.mql5.com > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  MTU: OK (1500 bytes)"
else
    echo "  MTU: Fragmentação detectada ⚠️"
fi

# 5. Packet loss
echo ""
echo "[5/6] Teste de perda de pacotes (30s)..."
LOSS=$(ping -c 30 -i 0.2 trade.mql5.com 2>/dev/null | grep loss | awk '{print $6}')
echo "  Perda: $LOSS"

# 6. Conexões ativas
echo ""
echo "[6/6] Conexões MT5 ativas..."
ss -tn | grep -E "(443|17[0-9]{3})" | wc -l
```

### Códigos de Erro MT5 Comuns
```
ERR_NO_CONNECTION (6):        Sem internet - verificar firewall
ERR_TRADE_TIMEOUT (10):       Latência > 5s - trocar VPS
ERR_INVALID_PRICE (135):      Slippage alto - melhorar conectividade
ERR_OFF_QUOTES (136):         Market closed ou gap de rede
ERR_REQUOTE (138):            Preço mudou - latência excessiva
ERR_CONNECTION_FAILED (4756): Broker bloqueou IP - verificar whitelist
```

---

## 4. ARQUITETURA DE REDE PARA VPS DE ALTA FREQUÊNCIA

### Topologia Recomendada

```
┌─────────────────────────────────────────────────────────────┐
│                      BROKER SERVERS                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Equinix LD4  │  │ Equinix NY4  │  │ Equinix SG1  │      │
│  │ Londres      │  │ Nova York    │  │ Singapura    │      │
│  │ 1-3ms RTT    │  │ 1-5ms RTT    │  │ 2-8ms RTT    │      │
│  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘      │
└──────────┼─────────────────┼─────────────────┼─────────────┘
           │                 │                 │
        ┌──▼─────────────────▼─────────────────▼──┐
        │     DEDICATED 10Gbps BACKBONE           │
        │  (Low-latency fiber: Frankfurt-London)  │
        └──┬──────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │       VPS TRADING (Frankfurt/London)        │
    │  ┌────────────────────────────────────────┐ │
    │  │   HOST HYPERVISOR (KVM/Proxmox)        │ │
    │  │   - CPU Pinning (Cores dedicados)      │ │
    │  │   - NUMA awareness                     │ │
    │  │   - Kernel bypass (DPDK optional)      │ │
    │  └────────────────────────────────────────┘ │
    │                                              │
    │  ┌────────────────┐  ┌──────────────────┐   │
    │  │   EA SCALPER   │  │   MONITORING     │   │
    │  │   Container    │  │   (Prometheus)   │   │
    │  │                │  │                  │   │
    │  │ ┌────────────┐ │  │ ┌──────────────┐ │   │
    │  │ │ MT5 Client │ │  │ │ Grafana      │ │   │
    │  │ │ + Expert   │ │  │ │ + Alerts     │ │   │
    │  │ │ Advisor    │ │  │ └──────────────┘ │   │
    │  │ └────────────┘ │  └──────────────────┘   │
    │  │                │                          │
    │  │ ┌────────────┐ │  ┌──────────────────┐   │
    │  │ │ Risk Mgmt  │ │  │  LOG COLLECTOR   │   │
    │  │ │ Service    │ │  │  (Loki/ELK)      │   │
    │  │ └────────────┘ │  └──────────────────┘   │
    │  └────────────────┘                          │
    │                                              │
    │  NETWORK: Bridge dedicada - 172.25.0.0/16   │
    │  FIREWALL: iptables/nftables rules          │
    └──────────────────────────────────────────────┘
           │
    ┌──────▼───────────────────────────┐
    │   BACKUP CHANNEL (4G/5G LTE)     │
    │   Failover automático < 500ms    │
    └──────────────────────────────────┘
```

### Stack Tecnológico Ideal

#### A) Sistema Operacional Otimizado
```bash
# Ubuntu Server 22.04 LTS - Kernel Real-Time (PREEMPT_RT)
uname -r  # Verificar se é kernel RT
# Exemplo: 5.15.0-rt48-generic

# Se não for RT, compilar kernel custom:
sudo apt install build-essential libncurses-dev bison flex libssl-dev
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.15.tar.xz
wget https://cdn.kernel.org/pub/linux/kernel/projects/rt/5.15/patch-5.15-rt48.patch.xz

# Configurações críticas no boot:
# /etc/default/grub
GRUB_CMDLINE_LINUX="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3 intel_pstate=disable"
```

#### B) Container Runtime Otimizado
```dockerfile
# /home/franco/projetos/EA_SCALPER_XAUUSD/Dockerfile.production
FROM ubuntu:22.04

# Install Wine para rodar MT5 no Linux
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y wine64 wine32 xvfb

# Otimizações de rede
RUN echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf && \
    echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf

# Copiar EA e configurações
COPY ./MQL5/Experts/EA_SCALPER_XAUUSD.ex5 /root/.wine/drive_c/MT5/MQL5/Experts/
COPY ./config/mt5_config.ini /root/.wine/drive_c/MT5/config/

# Healthcheck para garantir conectividade
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
  CMD ping -c 1 -W 1 trade.mql5.com || exit 1

ENTRYPOINT ["xvfb-run", "wine", "/root/.wine/drive_c/MT5/terminal64.exe"]
```

#### C) Network Policies (Kubernetes/Docker Swarm)
```yaml
# /home/franco/projetos/EA_SCALPER_XAUUSD/k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ea-scalper-netpol
spec:
  podSelector:
    matchLabels:
      app: ea-scalper
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # HTTPS para brokers
  - to:
    - ipBlock:
        cidr: 177.154.156.0/24  # IP do broker MQL5
    ports:
    - protocol: TCP
      port: 443
```

#### D) Monitoramento de Latência em Tempo Real
```yaml
# /home/franco/projetos/EA_SCALPER_XAUUSD/monitoring/prometheus.yml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'blackbox_broker_latency'
    metrics_path: /probe
    params:
      module: [icmp]
    static_configs:
      - targets:
        - trade.mql5.com
        - mt5.fxpro.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - target_label: __address__
        replacement: blackbox-exporter:9115

  - job_name: 'trading_metrics'
    static_configs:
      - targets: ['localhost:9091']
        labels:
          service: 'ea_scalper'
```

#### E) Alertas Críticos (AlertManager)
```yaml
# /home/franco/projetos/EA_SCALPER_XAUUSD/monitoring/alerts.yml
groups:
- name: trading_alerts
  interval: 10s
  rules:
  - alert: HighLatencyToBroker
    expr: probe_duration_seconds{job="blackbox_broker_latency"} > 0.050
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Latência acima de 50ms para {{ $labels.instance }}"
      description: "RTT atual: {{ $value }}s - PERDA DE VANTAGEM COMPETITIVA"

  - alert: PacketLossDetected
    expr: rate(probe_icmp_packets_lost[1m]) > 0.01
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Perda de pacotes detectada: {{ $value }}%"

  - alert: BrokerConnectionFailed
    expr: probe_success{job="blackbox_broker_latency"} == 0
    for: 15s
    labels:
      severity: critical
    annotations:
      summary: "CONEXÃO COM BROKER PERDIDA - {{ $labels.instance }}"
      description: "Trocar para broker backup imediatamente!"
```

### Benchmarks de Performance Esperados

```
MÉTRICA                    │ OBJETIVO HFT  │ ACEITÁVEL  │ CRÍTICO
────────────────────────────┼───────────────┼────────────┼─────────
Latência one-way (broker)  │ < 1ms         │ < 10ms     │ > 50ms
Jitter (variação)          │ < 0.5ms       │ < 5ms      │ > 20ms
Packet loss                │ 0%            │ < 0.1%     │ > 1%
Order execution time       │ < 5ms         │ < 50ms     │ > 200ms
Throughput                 │ > 1000 msg/s  │ > 100/s    │ < 10/s
Connection uptime          │ 99.99%        │ 99.9%      │ < 99%
TLS handshake              │ < 10ms        │ < 100ms    │ > 250ms
DNS resolution (cached)    │ < 1ms         │ < 10ms     │ > 50ms
```

---

## 5. CHECKLIST DE IMPLEMENTAÇÃO IMEDIATA

### Prioridade CRÍTICA (Implementar hoje)
- [ ] Migrar .env para git-crypt ou remover do repositório
- [ ] Configurar firewall UFW com regras mínimas
- [ ] Criar script de monitoramento de latência (cron a cada 5min)
- [ ] Testar conexão com múltiplos brokers (failover)

### Prioridade ALTA (Próximos 7 dias)
- [ ] Contratar VPS em Frankfurt/Londres (Contabo/Vultr)
- [ ] Aplicar otimizações TCP sysctl.conf
- [ ] Configurar Prometheus + Grafana para métricas
- [ ] Implementar health check automatizado

### Prioridade MÉDIA (Próximos 30 dias)
- [ ] Dockerizar EA com Wine + MT5
- [ ] Configurar backup de conexão (4G/LTE failover)
- [ ] Implementar rotação de API keys
- [ ] Criar playbook Ansible para deploy

### Prioridade BAIXA (Otimizações futuras)
- [ ] Compilar kernel RT para latência submilissegundo
- [ ] Implementar DPDK para kernel bypass
- [ ] Configurar NUMA pinning para CPU cores
- [ ] Migrar para Kubernetes com network policies

---

## 6. CUSTOS ESTIMADOS (Mensal)

```
ITEM                              │ CUSTO/MÊS  │ PROVIDER
──────────────────────────────────┼────────────┼────────────────────
VPS Frankfurt (2vCPU, 4GB, SSD)  │ €7-15      │ Contabo, Hetzner
VPS London HFT (dedicated core)  │ $50-100    │ Vultr High Freq
Backup 4G LTE (dados ilimitados) │ €20-30     │ Vodafone, O2
Monitoring (Grafana Cloud)       │ $0-49      │ Free tier ou Pro
SSL Certificates (Wildcard)      │ $0         │ Let's Encrypt
DDoS Protection (Cloudflare)     │ $0-20      │ Free ou Pro
──────────────────────────────────┼────────────┼────────────────────
TOTAL BÁSICO                     │ €27-115/mês
TOTAL PREMIUM (HFT)              │ €150-250/mês
```

---

## CONCLUSÃO E PRÓXIMOS PASSOS

**Status Atual**: ⚠️ INADEQUADO PARA SCALPING DE ALTA FREQUÊNCIA
- Latência de 296ms é CRÍTICA (objetivo: <10ms)
- Ambiente WSL2 adiciona overhead desnecessário
- Sem monitoramento de rede em tempo real
- Credenciais expostas em .env não criptografado

**Ação Imediata**: Migrar para VPS dedicado em Europa (Frankfurt/London) próximo aos servidores do broker.

**ROI Esperado**: Redução de 250ms na latência pode melhorar taxa de preenchimento de ordens em 15-30%, equivalente a ~$500-2000/mês em slippage evitado para contas de $10k+.

---

**Gerado por**: Claude Network Engineer
**Contato**: Para suporte, consulte /home/franco/projetos/EA_SCALPER_XAUUSD/docs/SUPPORT.md
