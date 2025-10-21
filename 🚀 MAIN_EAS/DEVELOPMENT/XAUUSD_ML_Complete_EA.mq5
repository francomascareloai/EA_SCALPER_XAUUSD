//+------------------------------------------------------------------+
//|                                    XAUUSD_ML_Complete_EA.mq5     |
//|                                     Copyright 2024, Elite Trading |
//|                           🥇 XAUUSD ML Trading Bot - Versão Única |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Elite Trading"
#property version   "3.00"
#property description "🥇 XAUUSD ML Trading Bot - Sistema Completo em um Arquivo"

#include <Trade\Trade.mqh>

//--- Parâmetros de Entrada
input group "=== CONFIGURAÇÃO PRINCIPAL ==="
input double   RiskPercent = 0.01;                          // Risco por Trade (%)
input double   MaxDailyRisk = 0.02;                         // Risco Diário Máximo (%)
input double   MaxDrawdown = 0.03;                          // Drawdown Máximo (%)
input bool     EnableFTMO = true;                           // Conformidade FTMO
input int      MagicNumber = 123456;                        // Número Mágico

input group "=== MACHINE LEARNING ==="
input bool     EnableML = true;                             // Ativar ML
input double   MLConfidence = 0.75;                         // Confiança ML Mínima
input bool     EnableSmartMoney = true;                     // Estratégia Smart Money
input bool     EnableScalping = true;                       // Scalping ML

input group "=== OTIMIZAÇÃO ==="
input int      MaxLatency = 120;                            // Latência Máxima (ms)
input double   SlippagePips = 2.0;                          // Tolerância Slippage
input bool     EnableVisual = true;                         // Interface Visual

//--- Enumerações
enum ENUM_SIGNAL { SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD };
enum ENUM_REGIME { REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_VOLATILE };
enum ENUM_STRATEGY { STRATEGY_SMART_MONEY, STRATEGY_SCALPING, STRATEGY_BREAKOUT };

//--- Estruturas
struct SMarketData {
    double atr, rsi, macd_main, macd_signal;
    double ema21, ema50, ema200;
    double current_price, spread;
    datetime time;
};

struct SMLFeatures {
    double momentum[3];
    double volatility_ratio;
    double rsi_div;
    double macd_hist;
    double order_block_strength;
    double session_factor;
};

struct SSignal {
    ENUM_SIGNAL type;
    double entry, sl, tp;
    double confidence;
    string strategy;
    datetime time;
};

//--- Variáveis Globais
CTrade g_trade;
int g_atr_handle, g_rsi_handle, g_macd_handle;
int g_ema21_handle, g_ema50_handle, g_ema200_handle;

bool g_trading_enabled = true;
double g_daily_pnl = 0.0;
double g_current_dd = 0.0;
int g_total_trades = 0;
int g_winning_trades = 0;

datetime g_last_ml_update = 0;
datetime g_daily_reset = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 Inicializando XAUUSD ML Trading Bot...");
    
    // Configurar Trade
    g_trade.SetExpertMagicNumber(MagicNumber);
    g_trade.SetDeviationInPoints((int)SlippagePips);
    
    // Inicializar Indicadores
    g_atr_handle = iATR(_Symbol, PERIOD_M15, 14);
    g_rsi_handle = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
    g_macd_handle = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
    g_ema21_handle = iMA(_Symbol, PERIOD_M15, 21, 0, MODE_EMA, PRICE_CLOSE);
    g_ema50_handle = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
    g_ema200_handle = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_EMA, PRICE_CLOSE);
    
    if(g_atr_handle == INVALID_HANDLE || g_rsi_handle == INVALID_HANDLE || 
       g_macd_handle == INVALID_HANDLE || g_ema21_handle == INVALID_HANDLE ||
       g_ema50_handle == INVALID_HANDLE || g_ema200_handle == INVALID_HANDLE)
    {
        Print("❌ Erro ao inicializar indicadores");
        return INIT_FAILED;
    }
    
    // Interface Visual
    if(EnableVisual)
    {
        CreateVisualInterface();
    }
    
    g_last_ml_update = TimeCurrent();
    g_daily_reset = TimeCurrent();
    
    Print("✅ XAUUSD ML Trading Bot inicializado com sucesso!");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Liberar handles
    IndicatorRelease(g_atr_handle);
    IndicatorRelease(g_rsi_handle);
    IndicatorRelease(g_macd_handle);
    IndicatorRelease(g_ema21_handle);
    IndicatorRelease(g_ema50_handle);
    IndicatorRelease(g_ema200_handle);
    
    // Limpar interface
    if(EnableVisual)
    {
        ObjectsDeleteAll(0, "XAUUSD_");
    }
    
    Print("✅ EA finalizado");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Reset diário
    CheckDailyReset();
    
    // Atualizar interface
    if(EnableVisual)
    {
        UpdateVisualInterface();
    }
    
    // Verificar condições de trading
    if(!g_trading_enabled || !IsTradeAllowed())
        return;
    
    // Verificar limites de risco
    if(!CheckRiskLimits())
    {
        g_trading_enabled = false;
        Print("🚫 Trading desabilitado - Limite de risco atingido");
        return;
    }
    
    // Análise de mercado
    SMarketData market_data;
    if(!GetMarketData(market_data))
        return;
    
    // ML Features
    SMLFeatures features;
    ExtractMLFeatures(market_data, features);
    
    // Predição ML
    double ml_confidence;
    ENUM_SIGNAL ml_signal = GetMLPrediction(features, ml_confidence);
    
    // Estratégia de trading
    SSignal signal;
    if(GenerateTradingSignal(market_data, features, ml_signal, ml_confidence, signal))
    {
        ExecuteSignal(signal);
    }
    
    // Gerenciar posições
    ManagePositions();
}

//+------------------------------------------------------------------+
//| Obter Dados de Mercado                                          |
//+------------------------------------------------------------------+
bool GetMarketData(SMarketData &data)
{
    double atr[1], rsi[1], macd_main[1], macd_signal[1];
    double ema21[1], ema50[1], ema200[1];
    
    if(CopyBuffer(g_atr_handle, 0, 0, 1, atr) <= 0) return false;
    if(CopyBuffer(g_rsi_handle, 0, 0, 1, rsi) <= 0) return false;
    if(CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) <= 0) return false;
    if(CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) <= 0) return false;
    if(CopyBuffer(g_ema21_handle, 0, 0, 1, ema21) <= 0) return false;
    if(CopyBuffer(g_ema50_handle, 0, 0, 1, ema50) <= 0) return false;
    if(CopyBuffer(g_ema200_handle, 0, 0, 1, ema200) <= 0) return false;
    
    data.atr = atr[0];
    data.rsi = rsi[0];
    data.macd_main = macd_main[0];
    data.macd_signal = macd_signal[0];
    data.ema21 = ema21[0];
    data.ema50 = ema50[0];
    data.ema200 = ema200[0];
    data.current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    data.spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * Point;
    data.time = TimeCurrent();
    
    return true;
}

//+------------------------------------------------------------------+
//| Extrair Características ML                                       |
//+------------------------------------------------------------------+
void ExtractMLFeatures(const SMarketData &data, SMLFeatures &features)
{
    // Momentum de preço
    for(int i = 0; i < 3; i++)
    {
        double current = iClose(_Symbol, PERIOD_M15, i);
        double previous = iClose(_Symbol, PERIOD_M15, i + 1);
        features.momentum[i] = (current - previous) / Point;
    }
    
    // Características técnicas
    features.volatility_ratio = data.atr / data.current_price;
    features.rsi_div = data.rsi - 50.0;
    features.macd_hist = data.macd_main - data.macd_signal;
    
    // Order blocks (simplificado)
    double volume_current = iVolume(_Symbol, PERIOD_M15, 0);
    double volume_avg = 0;
    for(int i = 1; i <= 10; i++)
        volume_avg += iVolume(_Symbol, PERIOD_M15, i);
    volume_avg /= 10.0;
    
    features.order_block_strength = (volume_current > volume_avg * 1.5) ? 1.0 : 0.0;
    
    // Fator de sessão
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    bool is_london = (dt.hour >= 8 && dt.hour <= 17);
    bool is_ny = (dt.hour >= 13 && dt.hour <= 22);
    features.session_factor = (is_london || is_ny) ? 1.0 : 0.5;
}

//+------------------------------------------------------------------+
//| Predição ML                                                      |
//+------------------------------------------------------------------+
ENUM_SIGNAL GetMLPrediction(const SMLFeatures &features, double &confidence)
{
    if(!EnableML)
    {
        confidence = 0.5;
        return SIGNAL_HOLD;
    }
    
    // Ensemble ML simplificado
    double score = 0.0;
    
    // Random Forest simulado
    if(features.momentum[0] > 5.0) score += 0.3;
    if(features.rsi_div > 0) score += 0.2;
    if(features.macd_hist > 0) score += 0.2;
    if(features.order_block_strength > 0.5) score += 0.3;
    
    // Ajuste por sessão
    score *= features.session_factor;
    
    // Confiança baseada na consistência
    confidence = MathAbs(score - 0.5) * 2.0;
    
    if(score > 0.6 && confidence > MLConfidence)
        return SIGNAL_BUY;
    else if(score < 0.4 && confidence > MLConfidence)
        return SIGNAL_SELL;
    
    return SIGNAL_HOLD;
}

//+------------------------------------------------------------------+
//| Gerar Sinal de Trading                                          |
//+------------------------------------------------------------------+
bool GenerateTradingSignal(const SMarketData &data, const SMLFeatures &features, 
                          ENUM_SIGNAL ml_signal, double ml_conf, SSignal &signal)
{
    // Estratégia Smart Money
    if(EnableSmartMoney && ml_signal != SIGNAL_HOLD)
    {
        // Confirmação de tendência
        bool uptrend = (data.ema21 > data.ema50) && (data.ema50 > data.ema200);
        bool downtrend = (data.ema21 < data.ema50) && (data.ema50 < data.ema200);
        
        if((ml_signal == SIGNAL_BUY && uptrend) || (ml_signal == SIGNAL_SELL && downtrend))
        {
            signal.type = ml_signal;
            signal.entry = data.current_price;
            signal.confidence = ml_conf;
            signal.strategy = "Smart Money ML";
            signal.time = TimeCurrent();
            
            // Stops e targets baseados em ATR
            if(signal.type == SIGNAL_BUY)
            {
                signal.sl = signal.entry - (data.atr * 2.0);
                signal.tp = signal.entry + (data.atr * 3.0);
            }
            else
            {
                signal.sl = signal.entry + (data.atr * 2.0);
                signal.tp = signal.entry - (data.atr * 3.0);
            }
            
            return true;
        }
    }
    
    // Estratégia Scalping
    if(EnableScalping && IsScalpingTime() && ml_conf > 0.8)
    {
        double spread_limit = data.atr * 0.3;
        if(data.spread <= spread_limit && ml_signal != SIGNAL_HOLD)
        {
            signal.type = ml_signal;
            signal.entry = data.current_price;
            signal.confidence = ml_conf;
            signal.strategy = "ML Scalping";
            signal.time = TimeCurrent();
            
            // Scalping com targets menores
            if(signal.type == SIGNAL_BUY)
            {
                signal.sl = signal.entry - (data.atr * 1.0);
                signal.tp = signal.entry + (data.atr * 1.5);
            }
            else
            {
                signal.sl = signal.entry + (data.atr * 1.0);
                signal.tp = signal.entry - (data.atr * 1.5);
            }
            
            return true;
        }
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Executar Sinal                                                   |
//+------------------------------------------------------------------+
void ExecuteSignal(const SSignal &signal)
{
    // Calcular tamanho da posição
    double lot_size = CalculateLotSize(signal);
    if(lot_size <= 0) return;
    
    // Aplicar otimização de latência se necessário
    if(MaxLatency > 100)
    {
        if(!ValidateLatencyConditions(signal))
        {
            Print("🚫 Condições de latência não atendidas");
            return;
        }
    }
    
    bool result = false;
    
    if(signal.type == SIGNAL_BUY)
    {
        result = g_trade.Buy(lot_size, _Symbol, 0, signal.sl, signal.tp, signal.strategy);
    }
    else if(signal.type == SIGNAL_SELL)
    {
        result = g_trade.Sell(lot_size, _Symbol, 0, signal.sl, signal.tp, signal.strategy);
    }
    
    if(result)
    {
        Print("✅ Trade executado: ", signal.strategy, " - Confiança: ", signal.confidence);
        g_total_trades++;
    }
    else
    {
        Print("❌ Falha na execução: ", g_trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Calcular Tamanho da Posição                                     |
//+------------------------------------------------------------------+
double CalculateLotSize(const SSignal &signal)
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = equity * RiskPercent;
    
    double sl_pips = MathAbs(signal.entry - signal.sl) / Point;
    if(sl_pips <= 0) return 0.0;
    
    double pip_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lot_size = risk_amount / (sl_pips * pip_value);
    
    // Ajustar por confiança ML
    lot_size *= signal.confidence;
    
    // Limites do símbolo
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lot_size = MathMax(min_lot, lot_size);
    lot_size = MathMin(max_lot, lot_size);
    lot_size = MathRound(lot_size / step_lot) * step_lot;
    
    return lot_size;
}

//+------------------------------------------------------------------+
//| Verificar Limites de Risco                                      |
//+------------------------------------------------------------------+
bool CheckRiskLimits()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    // Calcular drawdown atual
    if(balance > equity)
    {
        g_current_dd = (balance - equity) / balance;
        if(g_current_dd > MaxDrawdown)
        {
            Print("🚨 Drawdown máximo excedido: ", g_current_dd * 100, "%");
            return false;
        }
    }
    
    // Verificar perda diária (FTMO)
    if(EnableFTMO)
    {
        double daily_loss_pct = -g_daily_pnl / equity;
        if(daily_loss_pct > MaxDailyRisk)
        {
            Print("🚨 Limite de perda diária excedido: ", daily_loss_pct * 100, "%");
            return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Gerenciar Posições                                              |
//+------------------------------------------------------------------+
void ManagePositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByIndex(i) && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
        {
            // Trailing stop simplificado
            double atr = iATR(_Symbol, PERIOD_M15, 14);
            double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double position_profit = PositionGetDouble(POSITION_PROFIT);
            
            if(position_profit > 0) // Posição em lucro
            {
                double new_sl = 0;
                if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    new_sl = current_price - atr;
                    if(new_sl > PositionGetDouble(POSITION_SL))
                    {
                        g_trade.PositionModify(PositionGetInteger(POSITION_TICKET), new_sl, PositionGetDouble(POSITION_TP));
                    }
                }
                else
                {
                    new_sl = current_price + atr;
                    if(new_sl < PositionGetDouble(POSITION_SL))
                    {
                        g_trade.PositionModify(PositionGetInteger(POSITION_TICKET), new_sl, PositionGetDouble(POSITION_TP));
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Validar Condições de Latência                                   |
//+------------------------------------------------------------------+
bool ValidateLatencyConditions(const SSignal &signal)
{
    // Verificar se preço não se moveu muito
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double price_change = MathAbs(current_price - signal.entry) / Point;
    
    return (price_change <= SlippagePips);
}

//+------------------------------------------------------------------+
//| Reset Diário                                                     |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
    MqlDateTime current_dt, reset_dt;
    TimeToStruct(TimeCurrent(), current_dt);
    TimeToStruct(g_daily_reset, reset_dt);
    
    if(current_dt.day != reset_dt.day)
    {
        // Reset métricas diárias
        double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        static double daily_start_equity = current_equity;
        
        g_daily_pnl = current_equity - daily_start_equity;
        daily_start_equity = current_equity;
        g_daily_reset = TimeCurrent();
        
        Print("📅 Reset diário - P&L: $", g_daily_pnl);
    }
}

//+------------------------------------------------------------------+
//| Interface Visual                                                 |
//+------------------------------------------------------------------+
void CreateVisualInterface()
{
    ObjectCreate(0, "XAUUSD_Title", OBJ_LABEL, 0, 0, 0);
    ObjectSetString(0, "XAUUSD_Title", OBJPROP_TEXT, "🥇 XAUUSD ML Trading Bot");
    ObjectSetInteger(0, "XAUUSD_Title", OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(0, "XAUUSD_Title", OBJPROP_YDISTANCE, 30);
    ObjectSetInteger(0, "XAUUSD_Title", OBJPROP_COLOR, clrGold);
    ObjectSetInteger(0, "XAUUSD_Title", OBJPROP_FONTSIZE, 12);
    
    ObjectCreate(0, "XAUUSD_Status", OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, "XAUUSD_Status", OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(0, "XAUUSD_Status", OBJPROP_YDISTANCE, 60);
    ObjectSetInteger(0, "XAUUSD_Status", OBJPROP_COLOR, clrLime);
    
    ObjectCreate(0, "XAUUSD_Performance", OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, "XAUUSD_Performance", OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(0, "XAUUSD_Performance", OBJPROP_YDISTANCE, 90);
    ObjectSetInteger(0, "XAUUSD_Performance", OBJPROP_COLOR, clrYellow);
}

void UpdateVisualInterface()
{
    // Atualizar status
    string status = StringFormat("Status: %s | Trades: %d | DD: %.1f%%", 
                                g_trading_enabled ? "Ativo" : "Inativo",
                                g_total_trades, g_current_dd * 100);
    ObjectSetString(0, "XAUUSD_Status", OBJPROP_TEXT, status);
    
    // Atualizar performance
    double win_rate = (g_total_trades > 0) ? (double)g_winning_trades / g_total_trades * 100 : 0;
    string performance = StringFormat("P&L Diário: $%.2f | Win Rate: %.1f%%", 
                                    g_daily_pnl, win_rate);
    ObjectSetString(0, "XAUUSD_Performance", OBJPROP_TEXT, performance);
}

//+------------------------------------------------------------------+
//| Funções Auxiliares                                               |
//+------------------------------------------------------------------+
bool IsScalpingTime()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    // Londres e NY
    return (dt.hour >= 8 && dt.hour <= 17) || (dt.hour >= 13 && dt.hour <= 22);
}

bool IsNewsTime()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    // Horários típicos de notícias importantes
    return (dt.min >= 28 && dt.min <= 32) && 
           (dt.hour == 8 || dt.hour == 10 || dt.hour == 14);
}

bool IsActiveSession()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    return (dt.hour >= 8 && dt.hour <= 22); // Londres + NY
}