# Exemplos Práticos e Troubleshooting - EA_SCALPER_XAUUSD

## Overview

Este documento fornece exemplos práticos completos de configuração do projeto EA_SCALPER_XAUUSD, incluindo cenários reais, troubleshooting comum e soluções para problemas frequentes.

## Sumário

1. [Exemplos de Configuração Completa](#exemplos-de-configuração-completa)
2. [Cenários de Implementação](#cenários-de-implementação)
3. [Troubleshooting Comum](#troubleshooting-comum)
4. [Debug e Diagnóstico](#debug-e-diagnóstico)
5. [Performance Optimization](#performance-optimization)
6. [Migração e Atualização](#migração-e-atualização)
7. [Scripts de Automação](#scripts-de-automação)
8. [Checklists de Validação](#checklists-de-validação)

---

## Exemplos de Configuração Completa

### 1. Ambiente de Desenvolvimento Local

#### .env (Desenvolvimento)
```bash
# === API CONFIGURATION ===
OPENROUTER_API_KEY=sk-or-v1-1234567890abcdef1234567890abcdef1234567890abcdef1234567890
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD_Dev"
OPENROUTER_SITE_URL="http://localhost:3000"

# === MODEL CONFIGURATION ===
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
BACKUP_MODEL=openrouter/openai/gpt-4o

# === DEVELOPMENT SETTINGS ===
DEBUG_MODE=true
LOG_LEVEL=DEBUG
TESTING_MODE=true
ENABLE_AUDIT_LOG=false

# === LOCAL CACHE ===
OPENAI_API_BASE=http://localhost:4000
CACHE_TYPE=local
PROMPT_CACHE_TTL=300
RESPONSE_CACHE_TTL=150

# === NOTIFICATIONS (Development) ===
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-1001234567890
```

#### config_desenvolvimento.json
```json
{
  "version": "6.0",
  "environment": "development",
  "batch_size": 10,
  "max_threads": 2,
  "timeout_per_file": 120,
  "enable_real_processing": false,
  "enable_ftmo_validation": true,
  "enable_auto_backup": false,
  "debug": {
    "verbose_logging": true,
    "mock_apis": true,
    "simulate_failures": false,
    "performance_profiling": true
  },
  "testing": {
    "use_test_data": true,
    "deterministic_results": true,
    "validate_all_outputs": true
  }
}
```

#### trading_dev.yaml
```yaml
# Development Trading Configuration
project:
  name: "EA_SCALPER_XAUUSD"
  version: "2.0.0-dev"
  environment: "development"

assets:
  xauusd:
    symbol: "XAUUSD"
    risk_management:
      max_risk_percent: 2.0        # Mais agressivo para testes
      max_daily_loss: 5.0
      max_positions: 3              # Permitir múltiplas posições

    strategies:
      ml_scalping:
        enabled: true
        confidence_threshold: 0.60  # Mais permissivo
      smart_money:
        enabled: true
      volatility_breakout:
        enabled: true

    sessions:
      all_sessions:
        enabled: true               # Testar todas as sessões
        max_spread: 100            # Mais permissivo

notifications:
  telegram:
    enabled: true
    alerts:
      - trade_entry
      - trade_exit
      - system_errors
      - debug_info

monitoring:
  log_level: "DEBUG"
  performance_profiling: true
  memory_tracking: true
```

### 2. Ambiente de Produção

#### .env (Produção)
```bash
# === API CONFIGURATION ===
OPENROUTER_API_KEY=sk-or-v1-prod-key-with-higher-limits-abcdef1234567890
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD_Production"
OPENROUTER_SITE_URL="https://github.com/your-org/EA_SCALPER_XAUUSD"

# === MODEL CONFIGURATION ===
DEFAULT_MODEL=openrouter/anthropic/claude-3-5-sonnet
BACKUP_MODEL=openrouter/openai/gpt-4o

# === PRODUCTION SETTINGS ===
DEBUG_MODE=false
LOG_LEVEL=INFO
TESTING_MODE=false
ENABLE_AUDIT_LOG=true

# === REDIS CACHE ===
REDIS_URL=redis://prod-redis-cluster:6379/0
CACHE_TYPE=redis
PROMPT_CACHE_TTL=7200
RESPONSE_CACHE_TTL=3600

# === SECURITY ===
MAX_REQUEST_RATE=120
ENCRYPTION_KEY=your-32-character-encryption-key-here

# === NOTIFICATIONS ===
TELEGRAM_BOT_TOKEN=prod-bot-token
TELEGRAM_CHAT_ID=prod-chat-id
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/prod-url
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/prod-url

# === MONITORING ===
SENTRY_DSN=https://your-sentry-dsn-here
DATADOG_API_KEY=your-datadog-key-here
```

#### config_producao.json
```json
{
  "version": "6.0",
  "environment": "production",
  "batch_size": 100,
  "max_threads": 8,
  "timeout_per_file": 30,
  "enable_real_processing": true,
  "enable_ftmo_validation": true,
  "enable_auto_backup": true,
  "security": {
    "enable_audit_log": true,
    "rate_limiting": {
      "requests_per_minute": 120,
      "burst_size": 20
    },
    "encryption": {
      "enabled": true,
      "algorithm": "AES-256-GCM"
    }
  },
  "monitoring": {
    "enable_metrics": true,
    "log_level": "INFO",
    "alert_thresholds": {
      "error_rate": 0.02,
      "response_time": 500,
      "memory_usage": 0.8,
      "cpu_usage": 0.7
    },
    "health_checks": {
      "interval_seconds": 30,
      "timeout_seconds": 10,
      "retries": 3
    }
  },
  "backup": {
    "enabled": true,
    "frequency": "daily",
    "retention_days": 30,
    "compression": true,
    "encryption": true
  }
}
```

#### trading_production.yaml
```yaml
# Production Trading Configuration
project:
  name: "EA_SCALPER_XAUUSD"
  version: "2.1.0"
  environment: "production"

assets:
  xauusd:
    symbol: "XAUUSD"
    risk_management:
      max_risk_percent: 0.5        # Conservador
      max_daily_loss: 1.0
      max_positions: 1              # Apenas uma posição
      max_drawdown: 5.0

    strategies:
      ml_scalping:
        enabled: true
        confidence_threshold: 0.85  # Alta confiança
        update_frequency_hours: 24
      smart_money:
        enabled: true
      volatility_breakout:
        enabled: false              # Desativado em produção

    sessions:
      asian:
        enabled: false              # Evitar volatilidade
      european:
        enabled: true
        start: "08:00"
        end: "16:00"
        max_spread: 25
      american:
        enabled: true
        start: "13:00"
        end: "20:00"
        max_spread: 20

    news_filtering:
      enabled: true
      minutes_before: 30
      minutes_after: 30
      high_impact_only: true

risk_management:
  position_sizing:
    method: "fixed_percentage"
    risk_per_trade: 0.005
    max_lot_size: 0.1
    min_lot_size: 0.01

  stop_management:
    trailing_stop:
      enabled: true
      activation_profit: 50
      trail_distance: 30
      trail_step: 10

notifications:
  telegram:
    enabled: true
    alerts:
      - trade_entry
      - trade_exit
      - risk_alerts
      - system_errors
      - daily_summary

  discord:
    enabled: true
    webhook_url: "${DISCORD_WEBHOOK_URL}"
    alerts:
      - performance_metrics
      - system_status

monitoring:
  performance:
    latency_threshold_ms: 100
    execution_timeout_ms: 50
    enable_profiling: false

  health:
    check_interval: 60
    auto_restart: false
    alert_on_errors: true
```

### 3. Ambiente de Staging/Testes

#### .env (Staging)
```bash
# === API CONFIGURATION ===
OPENROUTER_API_KEY=sk-or-v1-staging-key-for-testing-abcdef1234567890
OPENROUTER_APP_NAME="EA_SCALPER_XAUUSD_Staging"

# === STAGING SETTINGS ===
DEBUG_MODE=true
LOG_LEVEL=INFO
TESTING_MODE=true
ENABLE_AUDIT_LOG=true

# === TEST CACHE ===
REDIS_URL=redis://staging-redis:6379/1
CACHE_TYPE=redis
PROMPT_CACHE_TTL=1800
RESPONSE_CACHE_TTL=900

# === MONITORING ===
MAX_REQUEST_RATE=60
```

---

## Cenários de Implementação

### 1. Configuração Rápida (Quick Start)

#### Script de Setup Automático
```bash
#!/bin/bash
# quick_setup.sh - Setup rápido do projeto

echo "🚀 EA_SCALPER_XAUUSD - Quick Setup"

# 1. Verificar dependências
echo "📋 Verificando dependências..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required"; exit 1; }
command -v redis-cli >/dev/null 2>&1 || { echo "⚠️ Redis CLI not found (optional)" }

# 2. Criar ambiente virtual
echo "🐍 Criando ambiente virtual..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
echo "📦 Instalando dependências Python..."
pip install -r requirements.txt

# 4. Criar arquivo .env
echo "⚙️ Criando arquivo de configuração..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Arquivo .env criado. Edite com suas chaves de API."
fi

# 5. Iniciar Redis
echo "🔴 Iniciando Redis..."
docker run -d --name ea-redis -p 6379:6379 redis:alpine

# 6. Validar configuração
echo "✅ Validando configuração..."
python scripts/validate_config.py

echo "🎉 Setup completo! Execute 'python main.py' para iniciar."
```

#### Configuração Mínima Funcional
```python
# minimal_config.py - Configuração mínima para teste
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class MinimalConfig:
    """Configuração mínima para funcionamento básico"""

    # API Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    DEFAULT_MODEL = "openrouter/anthropic/claude-3-5-sonnet"

    # Basic Trading
    SYMBOL = "XAUUSD"
    TIMEFRAME = "M5"
    MAX_RISK_PERCENT = 1.0

    # Simple Cache
    CACHE_TYPE = "local"
    ENABLE_LOGGING = True

    @classmethod
    def validate(cls):
        """Valida configuração mínima"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY é obrigatório")

        print("✅ Configuração mínima validada")
        return True

# Uso
if __name__ == "__main__":
    MinimalConfig.validate()
    print("🚀 Sistema pronto para uso mínimo")
```

### 2. Setup de Produção com Docker

#### Dockerfile
```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 eauser && chown -R eauser:eauser /app
USER eauser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py

# Start command
CMD ["python", "main.py"]
```

#### docker-compose.yml (Produção)
```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: ea-scaller-xauusd
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env.production
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - ea-network

  redis:
    image: redis:7-alpine
    container_name: ea-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ea-network

  postgres:
    image: postgres:15-alpine
    container_name: ea-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ea_trading
      POSTGRES_USER: ea_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ea-network

  nginx:
    image: nginx:alpine
    container_name: ea-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - ea-network

volumes:
  redis_data:
  postgres_data:

networks:
  ea-network:
    driver: bridge
```

### 3. Configuração Multi-Ambiente

#### config_manager.py
```python
# config_manager.py - Gerenciador de configurações multi-ambiente
import os
import json
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """Gerenciador central de configurações"""

    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config = {}
        self.load_all_configs()

    def load_all_configs(self):
        """Carrega todas as configurações"""
        # Carregar variáveis de ambiente
        self.load_env_vars()

        # Carregar configurações específicas do ambiente
        self.load_env_config()

        # Carregar configurações de trading
        self.load_trading_config()

        # Validar configurações
        self.validate_configs()

    def load_env_vars(self):
        """Carrega variáveis de ambiente"""
        from dotenv import load_dotenv
        load_dotenv()

        self.config.update({
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'openrouter/anthropic/claude-3-5-sonnet'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'DEBUG_MODE': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        })

    def load_env_config(self):
        """Carrega configuração do ambiente específico"""
        config_file = f"config_{self.environment}.json"
        config_path = Path("configs") / config_file

        if config_path.exists():
            with open(config_path, 'r') as f:
                env_config = json.load(f)
                self.config.update(env_config)

    def load_trading_config(self):
        """Carrega configuração de trading"""
        trading_file = f"trading_{self.environment}.yaml"
        trading_path = Path("configs") / trading_file

        if trading_path.exists():
            with open(trading_path, 'r') as f:
                trading_config = yaml.safe_load(f)
                self.config['trading'] = trading_config

    def validate_configs(self):
        """Valida todas as configurações"""
        required_keys = ['OPENROUTER_API_KEY', 'DEFAULT_MODEL']

        for key in required_keys:
            if not self.config.get(key):
                raise ValueError(f"Configuração obrigatória ausente: {key}")

        print(f"✅ Configurações validadas para ambiente: {self.environment}")

    def get(self, key: str, default=None):
        """Obtém valor da configuração"""
        return self.config.get(key, default)

    def get_trading_config(self, symbol: str = "XAUUSD"):
        """Obtém configuração de trading para símbolo específico"""
        trading_config = self.config.get('trading', {})
        assets = trading_config.get('assets', {})
        return assets.get(symbol.lower(), {})

    def reload(self):
        """Recarrega configurações"""
        self.config.clear()
        self.load_all_configs()

# Uso
config = ConfigManager('production')
api_key = config.get('OPENROUTER_API_KEY')
trading_config = config.get_trading_config('XAUUSD')
```

---

## Troubleshooting Comum

### 1. Problemas de Conexão e API

#### Problema: OpenRouter API Não Responde
```python
# debug_openrouter.py - Debug de conexão OpenRouter
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

async def debug_openrouter():
    """Debug completo da conexão OpenRouter"""
    load_dotenv()

    api_key = os.getenv('OPENROUTER_API_KEY')
    base_url = "https://openrouter.ai/api/v1"

    print("🔍 Debug OpenRouter API")
    print(f"API Key: {'*' * 10}{api_key[-10:] if api_key else 'None'}")
    print(f"Base URL: {base_url}")

    # Teste 1: Conexão básica
    print("\n1️⃣ Testando conexão básica...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    print("✅ Conexão bem-sucedida")
                    models = await response.json()
                    print(f"📊 Modelos disponíveis: {len(models.get('data', []))}")
                else:
                    print(f"❌ Erro na conexão: {response.status}")
                    print(f"Response: {await response.text()}")
    except Exception as e:
        print(f"❌ Erro de conexão: {e}")

    # Teste 2: Autenticação
    print("\n2️⃣ Testando autenticação...")
    if not api_key:
        print("❌ API Key não encontrada")
        return

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "openrouter/anthropic/claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                }
            ) as response:
                if response.status == 200:
                    print("✅ Autenticação bem-sucedida")
                    result = await response.json()
                    print(f"💬 Resposta: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
                else:
                    print(f"❌ Erro na autenticação: {response.status}")
                    print(f"Response: {await response.text()}")
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")

    # Teste 3: Latência
    print("\n3️⃣ Testando latência...")
    try:
        import time
        start_time = time.time()

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "openrouter/anthropic/claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 3
                }
            ) as response:
                end_time = time.time()
                latency = (end_time - start_time) * 1000

                print(f"🚀 Latência: {latency:.0f}ms")

                if latency > 5000:
                    print("⚠️ Latência alta detectada")
                elif latency > 1000:
                    print("⚠️ Latência moderada")
                else:
                    print("✅ Latência boa")

    except Exception as e:
        print(f"❌ Erro no teste de latência: {e}")

if __name__ == "__main__":
    asyncio.run(debug_openrouter())
```

#### Problema: Redis Cache Não Funciona
```python
# debug_redis.py - Debug de conexão Redis
import redis
import os
from dotenv import load_dotenv

def debug_redis():
    """Debug completo da conexão Redis"""
    load_dotenv()

    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    print("🔍 Debug Redis Cache")
    print(f"Redis URL: {redis_url}")

    try:
        # Teste 1: Conexão básica
        print("\n1️⃣ Testando conexão básica...")
        r = redis.from_url(redis_url)

        pong = r.ping()
        print(f"🏓 Ping: {pong}")

        if pong:
            print("✅ Conexão Redis estabelecida")
        else:
            print("❌ Falha no ping Redis")
            return

        # Teste 2: Operações básicas
        print("\n2️⃣ Testando operações básicas...")

        # SET
        r.set("test_key", "test_value", ex=10)
        print("✅ SET operation successful")

        # GET
        value = r.get("test_key")
        print(f"📖 GET result: {value}")

        # DELETE
        r.delete("test_key")
        print("✅ DELETE operation successful")

        # Teste 3: Informações do servidor
        print("\n3️⃣ Informações do servidor Redis...")
        info = r.info()

        print(f"📊 Redis Version: {info.get('redis_version')}")
        print(f"💾 Used Memory: {info.get('used_memory_human')}")
        print(f"🔗 Connected Clients: {info.get('connected_clients')}")
        print(f"📈 Total Commands Processed: {info.get('total_commands_processed')}")

        # Teste 4: Performance
        print("\n4️⃣ Testando performance...")
        import time

        start_time = time.time()

        for i in range(1000):
            r.set(f"perf_test_{i}", f"value_{i}", ex=60)

        set_time = time.time() - start_time

        start_time = time.time()

        for i in range(1000):
            r.get(f"perf_test_{i}")

        get_time = time.time() - start_time

        print(f"⚡ SET 1000 operations: {set_time:.3f}s ({1000/set_time:.0f} ops/s)")
        print(f"⚡ GET 1000 operations: {get_time:.3f}s ({1000/get_time:.0f} ops/s)")

        # Limpeza
        for i in range(1000):
            r.delete(f"perf_test_{i}")

        print("✅ Redis está funcionando perfeitamente")

    except redis.ConnectionError as e:
        print(f"❌ Erro de conexão Redis: {e}")
        print("\nSoluções possíveis:")
        print("1. Verifique se Redis está rodando: redis-server")
        print("2. Verifique URL no .env")
        print("3. Teste com: redis-cli ping")

    except Exception as e:
        print(f"❌ Erro Redis: {e}")

if __name__ == "__main__":
    debug_redis()
```

### 2. Problemas de Performance

#### Problema: Alto Consumo de Memória
```python
# memory_profiler.py - Análise de consumo de memória
import psutil
import time
import os
from memory_profiler import profile

class MemoryProfiler:
    """Analisa consumo de memória do sistema"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()

    def get_memory_usage(self):
        """Obtém uso atual de memória"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        return {
            'rss_mb': memory_mb,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

    def monitor_memory(self, duration_seconds=60, interval=5):
        """Monitora memória por período específico"""
        print(f"🔍 Monitorando memória por {duration_seconds} segundos...")

        start_time = time.time()
        peak_memory = 0

        while time.time() - start_time < duration_seconds:
            memory = self.get_memory_usage()
            peak_memory = max(peak_memory, memory['rss_mb'])

            print(f"📊 Memória: {memory['rss_mb']:.1f}MB ({memory['percent']:.1f}%) - "
                  f"Pico: {peak_memory:.1f}MB - Disponível: {memory['available_mb']:.1f}MB")

            # Alertas
            if memory['rss_mb'] > 1024:  # > 1GB
                print("⚠️ Alto consumo de memória detectado!")

            if memory['percent'] > 80:
                print("🚨 Uso de memória crítico!")

            time.sleep(interval)

        print(f"\n📈 Resumo do monitoramento:")
        print(f"   Pico de memória: {peak_memory:.1f}MB")
        print(f"   Duração: {time.time() - start_time:.1f}s")

    @profile
    def profile_function(self, data_size=10000):
        """Profile de função específica"""
        # Simula processamento intensivo
        data = [i ** 2 for i in range(data_size)]
        result = sum(data)
        return result

# Uso
if __name__ == "__main__":
    profiler = MemoryProfiler()

    # Monitorar memória atual
    current_memory = profiler.get_memory_usage()
    print(f"💾 Memória atual: {current_memory['rss_mb']:.1f}MB")

    # Monitorar por período
    profiler.monitor_memory(duration_seconds=30, interval=2)

    # Profile função
    print("\n🔬 Profile de função:")
    profiler.profile_function()
```

#### Problema: Latência Alta em Operações
```python
# latency_analyzer.py - Análise de latência do sistema
import time
import statistics
import asyncio
import aiohttp
from typing import List, Dict

class LatencyAnalyzer:
    """Analisa latência de várias operações"""

    def __init__(self):
        self.measurements = {}

    async def measure_api_latency(self, url: str, headers: Dict = None, samples: int = 10):
        """Mede latência de API"""
        latencies = []

        print(f"🚀 Medindo latência para: {url}")

        async with aiohttp.ClientSession(headers=headers) as session:
            for i in range(samples):
                start_time = time.time()

                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            end_time = time.time()
                            latency_ms = (end_time - start_time) * 1000
                            latencies.append(latency_ms)

                            print(f"   Amostra {i+1}: {latency_ms:.0f}ms")
                        else:
                            print(f"   Amostra {i+1}: Erro HTTP {response.status}")

                except Exception as e:
                    print(f"   Amostra {i+1}: Erro - {e}")

        if latencies:
            self.measurements[url] = {
                'avg_ms': statistics.mean(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'median_ms': statistics.median(latencies),
                'std_dev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'samples': len(latencies)
            }

            self.print_latency_stats(url)

    def measure_function_latency(self, func, *args, samples: int = 100):
        """Mede latência de função"""
        latencies = []

        print(f"⚡ Medindo latência da função: {func.__name__}")

        for i in range(samples):
            start_time = time.time()

            try:
                result = func(*args)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            except Exception as e:
                print(f"   Amostra {i+1}: Erro - {e}")

        if latencies:
            self.measurements[func.__name__] = {
                'avg_ms': statistics.mean(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'median_ms': statistics.median(latencies),
                'std_dev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'samples': len(latencies)
            }

            self.print_latency_stats(func.__name__)

    def print_latency_stats(self, operation_name: str):
        """Imprime estatísticas de latência"""
        stats = self.measurements[operation_name]

        print(f"\n📊 Estatísticas de Latência - {operation_name}:")
        print(f"   Média: {stats['avg_ms']:.1f}ms")
        print(f"   Mediana: {stats['median_ms']:.1f}ms")
        print(f"   Mínimo: {stats['min_ms']:.1f}ms")
        print(f"   Máximo: {stats['max_ms']:.1f}ms")
        print(f"   Desvio Padrão: {stats['std_dev_ms']:.1f}ms")
        print(f"   Amostras: {stats['samples']}")

        # Classificação de performance
        avg_latency = stats['avg_ms']
        if avg_latency < 100:
            print("   🟢 Performance: Excelente")
        elif avg_latency < 500:
            print("   🟡 Performance: Boa")
        elif avg_latency < 1000:
            print("   🟠 Performance: Aceitável")
        else:
            print("   🔴 Performance: Ruim")

    def get_recommendations(self, operation_name: str):
        """Fornece recomendações baseadas na latência"""
        if operation_name not in self.measurements:
            return

        stats = self.measurements[operation_name]
        avg_latency = stats['avg_ms']

        recommendations = []

        if avg_latency > 1000:
            recommendations.append("Implementar cache para reduzir latência")
            recommendations.append("Otimizar algoritmos ou usar assíncrono")

        if stats['std_dev_ms'] > stats['avg_ms'] * 0.5:
            recommendations.append("Investigar variabilidade alta na latência")

        if stats['max_ms'] > stats['avg_ms'] * 3:
            recommendations.append("Identificar e corrigir picos de latência")

        if recommendations:
            print(f"\n💡 Recomendações para {operation_name}:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print(f"\n✅ Latência de {operation_name} está dentro dos limites aceitáveis")

# Exemplo de uso
async def main():
    analyzer = LatencyAnalyzer()

    # Medir latência de APIs
    await analyzer.measure_api_latency("https://openrouter.ai/api/v1/models", samples=5)

    # Medir latência de funções locais
    def sample_function():
        time.sleep(0.01)  # Simula processamento
        return "result"

    analyzer.measure_function_latency(sample_function, samples=50)

    # Obter recomendações
    for operation in analyzer.measurements:
        analyzer.get_recommendations(operation)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Problemas de Configuração

#### Problema: Validação de Configuração Falha
```python
# config_validator.py - Validador abrangente de configuração
import json
import yaml
import os
from typing import Dict, List, Any
from pathlib import Path

class ConfigValidator:
    """Validador completo de configuração"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def validate_all(self, config_path: str = "."):
        """Valida todas as configurações do projeto"""
        print("🔍 Validando configuração completa do projeto...")

        config_path = Path(config_path)

        # Validar arquivo .env
        self.validate_env_file(config_path)

        # Validar arquivos JSON
        self.validate_json_files(config_path)

        # Validar arquivos YAML
        self.validate_yaml_files(config_path)

        # Validar arquivos TOML
        self.validate_toml_files(config_path)

        # Validar estrutura de diretórios
        self.validate_directory_structure(config_path)

        # Validar dependências
        self.validate_dependencies()

        # Imprimir resumo
        self.print_summary()

        return len(self.errors) == 0

    def validate_env_file(self, config_path: Path):
        """Valida arquivo .env"""
        env_file = config_path / ".env"

        if not env_file.exists():
            self.errors.append("Arquivo .env não encontrado")
            return

        print("\n📄 Validando .env...")

        # Carregar variáveis
        from dotenv import load_dotenv
        load_dotenv(env_file)

        # Validar variáveis obrigatórias
        required_vars = [
            'OPENROUTER_API_KEY',
            'DEFAULT_MODEL'
        ]

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Variável obrigatória ausente: {var}")
            else:
                self.info.append(f"✅ {var}: configurada")

        # Validar formatos
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key and not api_key.startswith('sk-or-v1-'):
            self.errors.append("OPENROUTER_API_KEY formato inválido")

        # Validar valores opcionais
        optional_vars = {
            'REDIS_URL': self._validate_url,
            'MAX_REQUEST_RATE': self._validate_positive_int,
            'DEBUG_MODE': self._validate_boolean
        }

        for var, validator in optional_vars.items():
            value = os.getenv(var)
            if value and not validator(value):
                self.warnings.append(f"Valor inválido para {var}: {value}")

    def validate_json_files(self, config_path: Path):
        """Valida arquivos JSON"""
        print("\n📄 Validando arquivos JSON...")

        json_files = list(config_path.rglob("*.json"))

        for json_file in json_files:
            if "node_modules" in str(json_file):
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Validações específicas
                if json_file.name == "config_sistema.json":
                    self._validate_config_sistema(data, json_file)
                elif json_file.name == "config_multi_agente.json":
                    self._validate_config_multi_agente(data, json_file)

                self.info.append(f"✅ JSON válido: {json_file.name}")

            except json.JSONDecodeError as e:
                self.errors.append(f"JSON inválido {json_file.name}: {e}")
            except Exception as e:
                self.errors.append(f"Erro ao validar {json_file.name}: {e}")

    def validate_yaml_files(self, config_path: Path):
        """Valida arquivos YAML"""
        print("\n📄 Validando arquivos YAML...")

        yaml_files = list(config_path.rglob("*.yaml")) + list(config_path.rglob("*.yml"))

        for yaml_file in yaml_files:
            if "node_modules" in str(yaml_file):
                continue

            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                self.info.append(f"✅ YAML válido: {yaml_file.name}")

            except yaml.YAMLError as e:
                self.errors.append(f"YAML inválido {yaml_file.name}: {e}")
            except Exception as e:
                self.errors.append(f"Erro ao validar {yaml_file.name}: {e}")

    def validate_toml_files(self, config_path: Path):
        """Valida arquivos TOML"""
        print("\n📄 Validando arquivos TOML...")

        try:
            import toml
        except ImportError:
            self.warnings.append("Biblioteca toml não instalada - pulando validação TOML")
            return

        toml_files = list(config_path.rglob("*.toml"))

        for toml_file in toml_files:
            if "node_modules" in str(toml_file):
                continue

            try:
                with open(toml_file, 'r', encoding='utf-8') as f:
                    data = toml.load(f)

                self.info.append(f"✅ TOML válido: {toml_file.name}")

            except toml.TomlDecodeError as e:
                self.errors.append(f"TOML inválido {toml_file.name}: {e}")
            except Exception as e:
                self.errors.append(f"Erro ao validar {toml_file.name}: {e}")

    def validate_directory_structure(self, config_path: Path):
        """Valida estrutura de diretórios"""
        print("\n📁 Validando estrutura de diretórios...")

        required_dirs = [
            "docs/configuration",
            "configs",
            "📋 DOCUMENTACAO_FINAL",
            "🤖 AI_AGENTS"
        ]

        for dir_name in required_dirs:
            dir_path = config_path / dir_name
            if dir_path.exists():
                self.info.append(f"✅ Diretório encontrado: {dir_name}")
            else:
                self.warnings.append(f"Diretório ausente: {dir_name}")

    def validate_dependencies(self):
        """Valida dependências Python"""
        print("\n🐍 Validando dependências...")

        required_packages = [
            "python-dotenv",
            "requests",
            "aiohttp",
            "pyyaml",
            "redis"
        ]

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.info.append(f"✅ Pacote encontrado: {package}")
            except ImportError:
                self.warnings.append(f"Pacote ausente: {package}")

    def _validate_config_sistema(self, data: Dict, file_path: Path):
        """Valida config_sistema.json"""
        required_fields = ['version', 'batch_size', 'max_threads']

        for field in required_fields:
            if field not in data:
                self.errors.append(f"Campo obrigatório ausente em {file_path.name}: {field}")

        # Validar tipos
        if 'batch_size' in data and not isinstance(data['batch_size'], int):
            self.errors.append(f"batch_size deve ser inteiro em {file_path.name}")

        if 'max_threads' in data and not isinstance(data['max_threads'], int):
            self.errors.append(f"max_threads deve ser inteiro em {file_path.name}")

    def _validate_config_multi_agente(self, data: Dict, file_path: Path):
        """Valida config_multi_agente.json"""
        required_sections = ['versao', 'sistema', 'agentes']

        for section in required_sections:
            if section not in data:
                self.errors.append(f"Seção obrigatória ausente em {file_path.name}: {section}")

        # Validar agentes
        if 'agentes' in data and isinstance(data['agentes'], list):
            for i, agente in enumerate(data['agentes']):
                required_agent_fields = ['Terminal', 'Nome', 'Modelo']
                for field in required_agent_fields:
                    if field not in agente:
                        self.errors.append(f"Agente {i}: campo obrigatório ausente: {field}")

    def _validate_url(self, url: str) -> bool:
        """Valida formato de URL"""
        return url.startswith(('http://', 'https://'))

    def _validate_positive_int(self, value: str) -> bool:
        """Valida inteiro positivo"""
        try:
            return int(value) > 0
        except ValueError:
            return False

    def _validate_boolean(self, value: str) -> bool:
        """Valida valor booleano"""
        return value.lower() in ('true', 'false', '1', '0', '')

    def print_summary(self):
        """Imprime resumo da validação"""
        print(f"\n{'='*50}")
        print("📊 RESUMO DA VALIDAÇÃO")
        print(f"{'='*50}")
        print(f"✅ Informações: {len(self.info)}")
        print(f"⚠️ Avisos: {len(self.warnings)}")
        print(f"❌ Erros: {len(self.errors)}")

        if self.errors:
            print(f"\n❌ ERROS ENCONTRADOS:")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print(f"\n⚠️ AVISOS:")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.info:
            print(f"\n✅ INFORMAÇÕES:")
            for info in self.info[:10]:  # Limitar a 10 informações
                print(f"   • {info}")
            if len(self.info) > 10:
                print(f"   ... e mais {len(self.info) - 10} itens")

        # Verdict final
        if not self.errors:
            print(f"\n🎉 CONFIGURAÇÃO VÁLIDA!")
            print("O sistema está pronto para uso.")
        else:
            print(f"\n❌ CONFIGURAÇÃO INVÁLIDA!")
            print("Corrija os erros antes de prosseguir.")

# Uso
if __name__ == "__main__":
    validator = ConfigValidator()
    is_valid = validator.validate_all(".")

    exit(0 if is_valid else 1)
```

---

## Scripts de Automação

### 1. Script de Backup Automático

```python
# backup_system.py - Backup completo do sistema
import os
import shutil
import json
import datetime
import tarfile
from pathlib import Path

class SystemBackup:
    """Sistema de backup completo"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def create_full_backup(self, include_large_files: bool = False):
        """Cria backup completo do projeto"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"ea_scalper_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        print(f"🚀 Criando backup completo: {backup_name}")

        # Criar diretório de backup
        backup_path.mkdir(exist_ok=True)

        # Arquivos e diretórios para backup
        backup_items = {
            "configs": "Configurações",
            "docs": "Documentação",
            "📋 DOCUMENTACAO_FINAL": "Documentação Final",
            "🤖 AI_AGENTS": "Agentes AI",
            "🚀 MAIN_EAS": "EAs Principais",
            "📚 LIBRARY": "Biblioteca",
            "scripts": "Scripts",
            ".env.example": "Template .env",
            "requirements.txt": "Dependências Python",
            "pyproject.toml": "Configuração Python",
        }

        # Backup de arquivos de configuração
        for item, description in backup_items.items():
            source = self.project_root / item
            if source.exists():
                dest = backup_path / item
                if source.is_file():
                    shutil.copy2(source, dest)
                else:
                    shutil.copytree(source, dest, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
                print(f"✅ {description}: copiado")

        # Gerar manifesto do backup
        self.create_backup_manifest(backup_path, timestamp)

        # Compactar backup
        if include_large_files:
            self.compress_backup(backup_path)

        # Limpar backups antigos
        self.cleanup_old_backups(keep_count=10)

        print(f"🎉 Backup concluído: {backup_name}")
        return backup_path

    def create_config_backup(self):
        """Cria backup apenas das configurações"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        print(f"⚙️ Criando backup de configurações: {backup_name}")

        backup_path.mkdir(exist_ok=True)

        # Backup apenas de configurações
        config_items = [
            ".env",
            "configs",
            "📋 DOCUMENTACAO_FINAL/CONFIGURACOES",
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml"
        ]

        for pattern in config_items:
            if pattern.startswith("📋"):
                # Diretório especial
                source = self.project_root / pattern
                if source.exists():
                    dest = backup_path / pattern.replace("/", "_")
                    shutil.copytree(source, dest)
                    print(f"✅ {pattern}: copiado")
            else:
                # Usar glob para patterns
                for source in self.project_root.glob(pattern):
                    if source.is_file():
                        dest = backup_path / source.name
                        shutil.copy2(source, dest)
                        print(f"✅ {source.name}: copiado")

        print(f"🎉 Backup de configurações concluído: {backup_name}")
        return backup_path

    def create_backup_manifest(self, backup_path: Path, timestamp: str):
        """Cria manifesto do backup"""
        manifest = {
            "backup_name": backup_path.name,
            "timestamp": timestamp,
            "created_at": datetime.datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "files": []
        }

        # Listar todos os arquivos no backup
        for file_path in backup_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(backup_path)
                file_info = {
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                manifest["files"].append(file_info)

        # Salvar manifesto
        manifest_path = backup_path / "backup_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"📋 Manifesto criado: {len(manifest['files'])} arquivos")

    def compress_backup(self, backup_path: Path):
        """Compacta backup em tar.gz"""
        compressed_path = backup_path.with_suffix('.tar.gz')

        print(f"🗜️ Compactando backup...")

        with tarfile.open(compressed_path, 'w:gz') as tar:
            tar.add(backup_path, arcname=backup_path.name)

        # Remover diretório original
        shutil.rmtree(backup_path)

        # Informar tamanho
        size_mb = compressed_path.stat().st_size / 1024 / 1024
        print(f"✅ Backup compactado: {size_mb:.1f}MB")

    def cleanup_old_backups(self, keep_count: int = 10):
        """Remove backups antigos, mantendo os mais recentes"""
        backups = sorted(self.backup_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)

        if len(backups) > keep_count:
            for old_backup in backups[keep_count:]:
                if old_backup.is_dir():
                    shutil.rmtree(old_backup)
                else:
                    old_backup.unlink()
                print(f"🗑️ Backup antigo removido: {old_backup.name}")

    def restore_backup(self, backup_name: str, target_dir: str = None):
        """Restaura backup específico"""
        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            # Tentar encontrar versão compactada
            compressed_path = backup_path.with_suffix('.tar.gz')
            if compressed_path.exists():
                self.extract_backup(compressed_path)
                backup_path = self.backup_dir / backup_name
            else:
                raise FileNotFoundError(f"Backup não encontrado: {backup_name}")

        target = Path(target_dir) if target_dir else self.project_root

        print(f"🔄 Restaurando backup: {backup_name}")

        # Verificar manifesto
        manifest_path = backup_path / "backup_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Backup de: {manifest['created_at']}")
            print(f"📁 Arquivos: {len(manifest['files'])}")

        # Copiar arquivos
        for item in backup_path.iterdir():
            if item.name == "backup_manifest.json":
                continue

            dest = target / item.name

            if dest.exists():
                if dest.is_file():
                    backup_dest = dest.with_suffix(f'.backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
                    shutil.move(dest, backup_dest)
                    print(f"📦 Arquivo existente backupado: {backup_dest}")
                else:
                    backup_dest = dest.with_name(f"{dest.name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    shutil.move(dest, backup_dest)
                    print(f"📦 Diretório existente backupado: {backup_dest}")

            if item.is_file():
                shutil.copy2(item, dest)
            else:
                shutil.copytree(item, dest)

            print(f"✅ Restaurado: {item.name}")

        print(f"🎉 Backup restaurado com sucesso!")

# Uso
if __name__ == "__main__":
    import sys

    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    backup_system = SystemBackup(project_root)

    # Criar backup completo
    backup_path = backup_system.create_full_backup(include_large_files=True)

    # Criar backup de configurações
    config_backup = backup_system.create_config_backup()

    print(f"\n🎉 Backup concluído!")
    print(f"📁 Backup completo: {backup_path}")
    print(f"⚙️ Backup configurações: {config_backup}")
```

---

## Checklists de Validação

### 1. Checklist de Setup Inicial

```markdown
# 🚀 Checklist de Setup Inicial - EA_SCALPER_XAUUSD

## 📋 Pré-requisitos
- [ ] Python 3.11+ instalado
- [ ] Docker instalado e rodando
- [ ] Redis instalado ou Docker Redis disponível
- [ ] Conta OpenRouter com API key
- [ ] Git configurado

## ⚙️ Configuração do Ambiente
- [ ] Clonar repositório
- [ ] Criar ambiente virtual: `python -m venv .venv`
- [ ] Ativar ambiente virtual
- [ ] Instalar dependências: `pip install -r requirements.txt`
- [ ] Copiar `.env.example` para `.env`
- [ ] Configurar variáveis no `.env`

## 🔑 Configuração de APIs
- [ ] OPENROUTER_API_KEY configurada
- [ ] DEFAULT_MODEL definido
- [ ] Validar formato da API key
- [ ] Testar conexão com OpenRouter

## 🗄️ Configuração de Cache
- [ ] Redis rodando (local ou Docker)
- [ ] REDIS_URL configurada
- [ ] Testar conexão Redis
- [ ] Configurar CACHE_TYPE

## 📁 Estrutura de Diretórios
- [ ] `docs/configuration/` criado
- [ ] `configs/` criado
- [ ] Logs directory criado
- [ ] Data directory criado
- [ ] Models directory criado

## 🧪 Validação
- [ ] Executar script de validação: `python scripts/validate_config.py`
- [ ] Todos os testes passando
- [ ] Sem erros de configuração
- [ ] APIs respondendo corretamente

## 📱 Notificações (Opcional)
- [ ] Telegram bot configurado
- [ ] Chat ID obtido
- [ ] Testar envio de mensagem
- [ ] Discord webhook configurado (se necessário)

## 🚀 Teste Final
- [ ] Executar aplicação: `python main.py`
- [ ] Verificar logs de inicialização
- [ ] Testar funcionalidades básicas
- [ ] Monitorar consumo de recursos

## ✅ Critérios de Sucesso
- Todos os itens acima marcados
- Aplicação inicia sem erros
- APIs funcionando corretamente
- Logs sendo gerados
- Sistema pronto para uso
```

### 2. Checklist de Deploy Produção

```markdown
# 🚀 Checklist de Deploy Produção - EA_SCALPER_XAUUSD

## 🔒 Segurança
- [ ] Chaves de API de produção configuradas
- [ ] Variáveis sensíveis no .env.production
- [ ] HTTPS configurado
- [ ] Firewall regras aplicadas
- [ ] SSL/TLS certificados válidos

## 🏗️ Infraestrutura
- [ ] Servidor preparado (requisitos mínimos)
- [ ] Docker instalado e configurado
- [ ] Redis cluster configurado
- [ ] PostgreSQL configurado
- [ ] Nginx configurado
- [ ] Balanceador de carga (se necessário)

## 📦 Build e Deploy
- [ ] Código atualizado para latest stable
- [ ] Tests executados e passando
- [ ] Build de imagem Docker
- [ ] Push para registry
- [ ] docker-compose.yml atualizado
- [ ] Serviços reiniciados

## 🔧 Configuração
- [ ] .env.production configurado
- [ ] Configurações de produção validadas
- [ ] Rate limiting configurado
- [ ] Monitoring configurado
- [ ] Backup automatizado
- [ ] Logs centralizados

## 📊 Monitoring
- [ ] Health checks configurados
- [ ] Métricas coletando
- [ ] Alertas configurados
- [ ] Dashboard funcionando
- [ ] Logs agregados
- [ ] Performance monitorada

## 🧪 Testes Pós-Deploy
- [ ] Health check passando
- [ ] APIs respondendo
- [ ] Cache funcionando
- [ ] Database conectando
- [ ] Notificações enviadas
- [ ] Performance aceitável

## 📋 Documentação
- [ ] Runbook atualizado
- [ ] Diagrama de arquitetura
- [ ] Contatos de emergência
- [ ] Procedimentos de rollback
- [ ] Documentação de APIs

## 🔁 Rollback Plan
- [ ] Backup atual do sistema
- [ ] Versão anterior disponível
- [ ] Script de rollback testado
- [ ] Janela de manutenção definida
- [ ] Usuários notificados

## ✅ Critérios de Sucesso
- Sistema estável em produção
- Todos os serviços rodando
- Métricas dentro dos limites
- Sem erros críticos
- Backup funcionando
- Monitoramento ativo
```

Este guia completo de exemplos práticos e troubleshooting cobre todos os cenários reais de implementação, debug e resolução de problemas para o projeto EA_SCALPER_XAUUSD, fornecendo ferramentas e scripts prontos para uso.