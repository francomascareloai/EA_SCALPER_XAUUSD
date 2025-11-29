# Ficha Técnica do EA (Resumo)

- Nome: 
- Versão: 
- Status: Produção | Dev | Teste
- Símbolo/TF: XAUUSD / M5 (exemplo)
- Objetivo: Scalping | Tendência | SMC | Híbrido
- Dependências: nenhuma | libs internas

## Parâmetros-chave
- Risco por trade: 1.0% (recomendado: 0.5–2.0%)
- SL/TP: SL dinâmico (ATR) | TP parcial
- Filtros: tempo de sessão | notícias

## Lógica (alto nível)
- Sinais: [ex.: confluência RSI+VWAP]
- Entrada/Saída: [ex.: break-even + trailing]
- Gestão de posição: [ex.: parcial 50% em RR 1:1]

## Operação
- Pré-requisitos: MT5 build, VPS, latência
- Passos: anexar ao gráfico, habilitar auto-trading
- Observações: horários preferenciais (Londres/NY)

## Avaliação
- Métricas alvo: Win Rate, PF, DD máx.
- Backtest: período, qualidade, modelagem
- Forward: conta demo, duração

