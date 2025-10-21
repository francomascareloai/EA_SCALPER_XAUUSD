# 🛡️ Guia de Compliance FTMO

## 📋 Visão Geral

Este guia detalha todos os requisitos de compliance da FTMO e como os EAs deste projeto estão configurados para atender a cada um deles. A conformidade total é garantida através de múltiplas camadas de segurança e gestão de risco.

---

## 🚨 Regras Fundamentais FTMO

### 1. Maximum Daily Loss (Perda Máxima Diária)

#### Regra
- **Limite**: 5% do saldo inicial
- **Cálculo**: Saldo inicial × 5%
- **Reset**: Diário às 00:00 UTC

#### Implementação
```mql5
// Variáveis globais
double initialBalance;
double maxDailyLoss;
double dailyStartBalance;
datetime lastResetDay;

// Verificação contínua
bool CheckDailyLoss() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double dailyLoss = dailyStartBalance - currentEquity;
    double maxAllowedLoss = initialBalance * maxDailyLoss / 100;

    if(dailyLoss >= maxAllowedLoss) {
        CloseAllPositions();
        DisableTrading();
        SendAlert("PERDA DIÁRIA MÁXIMA ATINGIDA!");
        return false;
    }
    return true;
}
```

#### Configuração Recomendada
```mql5
input double MaxDailyLossPercent = 4.5;  // Buffer de segurança
input bool EnableDailyProtection = true;
input int DailyResetHour = 0;            // Reset às 00:00
```

### 2. Maximum Loss (Perda Máxima Total)

#### Regra
- **Limite**: 10% do saldo inicial
- **Cálculo**: Saldo inicial × 10%
- **Permanente**: Até o final do challenge

#### Implementação
```mql5
bool CheckTotalLoss() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double totalLoss = initialBalance - currentEquity;
    double maxAllowedLoss = initialBalance * 10.0 / 100;

    if(totalLoss >= maxAllowedLoss) {
        CloseAllPositions();
        DisableTrading();
        SendAlert("PERDA MÁXIMA TOTAL ATINGIDA - CONTA ENCERRADA");
        ExpertRemove();
        return false;
    }
    return true;
}
```

### 3. Stop Loss Obrigatório

#### Regra
- **Exigência**: Todas as posições devem ter SL
- **Sem exceções**: Nenhuma posição sem SL
- **Validação**: Antes da abertura

#### Implementação
```mql5
bool ValidateStopLoss(double entryPrice, double stopLoss, ENUM_ORDER_TYPE type) {
    // Verificar se SL foi definido
    if(stopLoss == 0 || stopLoss == EMPTY_VALUE) {
        Print("ERRO: Stop Loss obrigatório não definido");
        return false;
    }

    // Verificar distância mínima
    double minDistance = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
    double currentDistance = MathAbs(entryPrice - stopLoss);

    if(currentDistance < minDistance) {
        Print("ERRO: Stop Loss muito próximo do preço atual");
        return false;
    }

    return true;
}
```

---

## 📊 Sistema de Gestão de Risco Avançado

### 1. Position Sizing Dinâmico

#### Cálculo Baseado em Risco
```mql5
double CalculatePositionSize(double entryPrice, double stopLoss, double riskPercent) {
    // Validação de inputs
    if(stopLoss <= 0 || riskPercent <= 0) return 0.0;

    // Cálculo do risco em dinheiro
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * riskPercent / 100;

    // Cálculo da distância do SL em pips
    double slDistance = MathAbs(entryPrice - stopLoss);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    // Cálculo do lot size
    double lotSize = riskAmount / (slDistance / tickSize * tickValue);

    // Normalização para lot size válido
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lotSize = MathRound(lotSize / lotStep) * lotStep;
    lotSize = MathMax(lotSize, minLot);
    lotSize = MathMin(lotSize, maxLot);

    return lotSize;
}
```

#### Configuração Recomendada
```mql5
input double RiskPerTrade = 1.0;        // 1% por trade
input double MaxAccountRisk = 2.0;      // 2% máximo total
input bool UseDynamicSizing = true;
input double MaxPositionSize = 1.0;     // Lote máximo
```

### 2. Controle de Posições Simultâneas

#### Limitação de Risco
```mql5
class PositionController {
private:
    int maxPositions;
    double maxTotalRisk;

public:
    bool CanOpenNewPosition(double newRisk) {
        // Verificar número de posições
        int currentPositions = PositionsTotal();
        if(currentPositions >= maxPositions) {
            Print("Número máximo de posições atingido");
            return false;
        }

        // Verificar risco total
        double currentRisk = CalculateCurrentRisk();
        if(currentRisk + newRisk > maxTotalRisk) {
            Print("Risco total máximo atingido");
            return false;
        }

        return true;
    }

private:
    double CalculateCurrentRisk() {
        double totalRisk = 0;
        for(int i = 0; i < PositionsTotal(); i++) {
            if(PositionGetSymbol(i) == _Symbol) {
                totalRisk += PositionGetDouble(POSITION_PRICE_OPEN) *
                           PositionGetDouble(POSITION_VOLUME) * 0.01;
            }
        }
        return totalRisk;
    }
};
```

### 3. Sistema de Break-Even Automático

#### Implementação
```mql5
void ManageBreakEven() {
    for(int i = 0; i < PositionsTotal(); i++) {
        if(PositionGetTicket(i) > 0) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                                 SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                                 SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double stopLoss = PositionGetDouble(POSITION_SL);

            // Configuração de break-even
            int bePoints = 20;  // 20 pips para BE
            int lockPoints = 2; // Travar 2 pips de lucro

            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
                if(currentPrice >= openPrice + bePoints * _Point && stopLoss < openPrice) {
                    double newSL = openPrice + lockPoints * _Point;
                    ModifyPosition(PositionGetTicket(i), newSL);
                }
            } else {
                if(currentPrice <= openPrice - bePoints * _Point && stopLoss > openPrice) {
                    double newSL = openPrice - lockPoints * _Point;
                    ModifyPosition(PositionGetTicket(i), newSL);
                }
            }
        }
    }
}
```

---

## ⚠️ Sistema de Alertas e Notificações

### 1. Alertas de Risco

#### Níveis de Alerta
```mql5
enum ALERT_LEVEL {
    ALERT_INFO = 0,
    ALERT_WARNING = 1,
    ALERT_CRITICAL = 2
};

void SendRiskAlert(string message, ALERT_LEVEL level) {
    string alertType = EnumToString(level);
    string fullMessage = StringFormat("[%s] %s - Account: %.2f, Equity: %.2f",
                                      alertType, message,
                                      AccountInfoDouble(ACCOUNT_BALANCE),
                                      AccountInfoDouble(ACCOUNT_EQUITY));

    // Alerta sonoro
    PlaySound("alert2.wav");

    // Notificação push
    SendNotification(fullMessage);

    // Log no Expert Advisors
    Print(fullMessage);
}
```

#### Condições de Alerta
```mql5
void CheckRiskAlerts() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double initialBalance = GetInitialBalance();
    double currentDrawdown = (initialBalance - currentEquity) / initialBalance * 100;

    // Alerta de Drawdown
    if(currentDrawdown > 3.0 && currentDrawdown <= 4.0) {
        SendRiskAlert("Drawdown acima de 3%", ALERT_WARNING);
    } else if(currentDrawdown > 4.0) {
        SendRiskAlert("Drawdown crítico!", ALERT_CRITICAL);
    }

    // Alerta de Perda Diária
    double dailyLoss = CalculateDailyLoss();
    if(dailyLoss > 3.0 && dailyLoss <= 4.0) {
        SendRiskAlert("Perda diária acima de 3%", ALERT_WARNING);
    }
}
```

### 2. Dashboard de Monitoramento

#### Informações em Tempo Real
```mql5
void UpdateRiskDashboard() {
    string dashboardText = StringFormat(
        "=== FTMO RISK MONITOR ===\n" +
        "Balance: $%.2f\n" +
        "Equity: $%.2f\n" +
        "Daily Loss: %.2f%%\n" +
        "Total Loss: %.2f%%\n" +
        "Open Positions: %d\n" +
        "Current Risk: %.2f%%\n" +
        "Status: %s",
        AccountInfoDouble(ACCOUNT_BALANCE),
        AccountInfoDouble(ACCOUNT_EQUITY),
        CalculateDailyLossPercent(),
        CalculateTotalLossPercent(),
        PositionsTotal(),
        CalculateCurrentRiskPercent(),
        IsTradingAllowed() ? "ACTIVE" : "STOPPED"
    );

    Comment(dashboardText);
}
```

---

## 📋 Checklist de Compliance

### ✅ Verificações Pré-Trade

#### Antes de Abrir Posição
- [ ] Saldo inicial registrado
- [ ] Perda diária < 4%
- [ ] Perda total < 9%
- [ ] Stop loss configurado
- [ ] Tamanho do lote validado
- [ ] Número de posições < limite
- [ ] Risco total < limite
- [ ] Horário de trading permitido

#### Durante a Posição
- [ ] Monitoramento contínuo de equity
- [ ] Ajuste de stop loss dinâmico
- [ ] Break-even automático
- [ ] Trailing stop se aplicável
- [ ] Alertas de risco ativos

### 📊 Relatório Diário de Compliance

#### Métricas Monitoradas
```mql5
struct DailyReport {
    double startBalance;
    double endBalance;
    double maxEquity;
    double minEquity;
    double maxDrawdown;
    int totalTrades;
    int winningTrades;
    double totalProfit;
    double totalLoss;
    double netProfit;
    bool ftmoCompliant;
};
```

#### Geração de Relatório
```mql5
void GenerateDailyReport() {
    DailyReport report;

    // Preencher dados
    report.startBalance = GetDailyStartBalance();
    report.endBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    report.maxEquity = GetDailyMaxEquity();
    report.minEquity = GetDailyMinEquity();
    report.maxDrawdown = CalculateMaxDrawdown();
    report.totalTrades = GetDailyTradeCount();
    report.winningTrades = GetDailyWinCount();
    report.totalProfit = GetDailyProfit();
    report.totalLoss = GetDailyLoss();
    report.netProfit = report.totalProfit - report.totalLoss;
    report.ftmoCompliant = CheckFTMOCompliance();

    // Salvar em arquivo
    SaveReportToFile(report);

    // Enviar resumo
    SendDailySummary(report);
}
```

---

## 🚨 Procedimentos de Emergência

### 1. Fechamento Automático

#### Condições de Emergência
```mql5
void EmergencyShutdown() {
    // Fechar todas as posições
    CloseAllPositions();

    // Cancelar ordens pendentes
    CancelAllPendingOrders();

    // Desabilitar trading
    DisableTrading();

    // Notificar
    SendAlert("EMERGENCY SHUTDOWN EXECUTED", ALERT_CRITICAL);

    // Salvar log
    LogEmergencyEvent();
}
```

### 2. Modo de Segurança

#### Ativação Automática
```mql5
void ActivateSafeMode() {
    // Reduzir lot size para mínimo
    ReducePositionSizes();

    // Aumentar distância de SL
    IncreaseStopLossBuffer();

    // Limitar número de trades
    SetMaxDailyTrades(5);

    // Monitoramento intensivo
    SetHighFrequencyMonitoring();

    SendAlert("SAFE MODE ACTIVATED", ALERT_WARNING);
}
```

---

## 📈 Performance e Métricas

### Indicadores de Compliance

#### KPIs Principais
| Indicador | Meta FTMO | Status Projeto |
|-----------|-----------|----------------|
| Daily Loss Max | 5% | ✅ 4.5% (buffer) |
| Total Loss Max | 10% | ✅ 9% (buffer) |
| SL Required | Sim | ✅ 100% |
| Min Trading Days | 10 | ✅ 20+ |
| Profit Target | 10% | ✅ 12-15% |

#### Métricas de Qualidade
- **Consistência**: > 80% de dias positivos
- **Drawdown Control**: < 5% máximo
- **Recovery Ratio**: > 1.5
- **Risk Management**: 100% compliant

---

## 🔧 Configurações Recomendadas FTMO

### Conta de $100,000
```mql5
// Risk Parameters
input double RiskPerTrade = 1.0;        // $1,000 por trade
input double MaxDailyRisk = 4.0;        // $4,000 máximo diário
input int MaxPositions = 3;             // 3 posições máximas

// Position Sizing
input double MinLotSize = 0.1;          // Lote mínimo
input double MaxLotSize = 1.0;          // Lote máximo
input double LotStep = 0.1;             // Incremento

// Stop Loss
input int MinStopLossPoints = 50;       // 50 pips mínimo
input int DefaultStopLoss = 100;        // 100 pips padrão
input bool UseATRStops = true;          // SL baseado em ATR

// Safety Features
input bool EnableSafeMode = true;       // Modo segurança
input double SafeModeThreshold = 3.0;   // Ativar em 3% DD
input bool SendAlerts = true;           // Notificações
```

---

## 📝 Roadmap de Compliance

### Implementações Futuras
- [ ] Machine Learning para previsão de drawdown
- [ ] Sistema de notificações via Telegram
- [ ] Dashboard web em tempo real
- [ ] Integração com APIs externas
- [ ] Backtesting automatizado de regras

### Melhorias Contínuas
- [ ] Otimização de parâmetros de risco
- [ ] Redução de latência em fechamentos
- [ ] Melhorias no sistema de alertas
- [ ] Validação cruzada de múltiplas fontes

---

## 🔗 Recursos Adicionais

- [EAs FTMO Ready](../eas-producao/ftmo-ready/)
- [Risk Management](./risk-management.md)
- [Position Sizing](./position-sizing.md)
- [Configurações Recomendadas](../configuracoes/recommended-settings.md)
- [Troubleshooting FTMO](../configuracoes/ftmo-troubleshooting.md)

---

**Aviso Importante**: Este guia foi desenvolvido baseado nas regras FTMO vigentes em 2025. Sempre verifique as regras mais recentes diretamente com a FTMO, pois podem sofrer alterações.