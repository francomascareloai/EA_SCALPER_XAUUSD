#!/bin/bash
# Real-time Latency Monitor for Trading
# Monitors broker connectivity and alerts on high latency

BROKER="trade.mql5.com"
LOG_DIR="/home/franco/projetos/EA_SCALPER_XAUUSD/logs"
LOG_FILE="$LOG_DIR/trading_latency.log"
ALERT_THRESHOLD=50  # ms

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Iniciando monitoramento de latência para $BROKER"
echo "Limite de alerta: ${ALERT_THRESHOLD}ms"
echo "Log: $LOG_FILE"
echo "Pressione Ctrl+C para parar"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Measure ping latency
    PING_OUTPUT=$(ping -c 1 -W 2 $BROKER 2>&1)

    if echo "$PING_OUTPUT" | grep -q "time="; then
        LATENCY=$(echo "$PING_OUTPUT" | grep time= | sed 's/.*time=\([0-9.]*\).*/\1/')

        # Check if latency exceeds threshold
        if (( $(echo "$LATENCY > $ALERT_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
            MSG="[$TIMESTAMP] ⚠️  ALERTA: Latência ${LATENCY}ms (limite: ${ALERT_THRESHOLD}ms)"
            echo -e "\033[1;31m$MSG\033[0m"  # Red color
            echo "$MSG" >> "$LOG_FILE"
        else
            MSG="[$TIMESTAMP] ✓ Latência: ${LATENCY}ms"
            echo -e "\033[1;32m$MSG\033[0m"  # Green color
            echo "$MSG" >> "$LOG_FILE"
        fi
    else
        MSG="[$TIMESTAMP] ✗ ERRO: Falha ao conectar com $BROKER"
        echo -e "\033[1;31m$MSG\033[0m"
        echo "$MSG" >> "$LOG_FILE"
    fi

    sleep 5
done
