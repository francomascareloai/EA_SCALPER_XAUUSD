//+------------------------------------------------------------------+
//|                                         EA_Advanced_Template.mq5 |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"
#property strict
#property robot_indicators

#include <RiskManager.mqh>
#include <OrderManager.mqh>
#include <PerformanceTracker.mqh>

// Input parameters
input double InitialLotSize = 0.01;
input int StopLoss = 100;
input int TakeProfit = 200;
input int MaxSpread = 30;
input bool UseTrailingStop = true;
input int TrailingStopPoints = 50;
input bool UseMartingale = false;
input double MartingaleMultiplier = 2.0;

// Global variables
datetime lastTradeTime = 0;
int magicNumber = 123457;
RiskManager riskManager;
OrderManager orderManager;
PerformanceTracker perfTracker;
double lotSize = InitialLotSize;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize expert advisor
    Print("EA Advanced Template initialized");
    
    // Initialize components
    riskManager.initialize();
    orderManager.initialize();
    perfTracker.initialize();
    
    // Check if symbol is available
    if (!SymbolInfoInteger(Symbol(), SYMBOL_TRADE_ALLOWED)) {
        Print("Trading not allowed for ", Symbol());
        return(INIT_FAILED);
    }
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Deinitialize expert advisor
    Print("EA Advanced Template deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Update components
    orderManager.update();
    perfTracker.update();
    
    // Check if it's time to trade
    if (!IsTradeAllowed()) {
        return;
    }
    
    // Manage existing orders
    ManageExistingOrders();
    
    // Main trading logic
    ExecuteTradingLogic();
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed() {
    // Check for new bar
    static datetime lastBar = 0;
    datetime currentBar = iTime(Symbol(), PERIOD_CURRENT, 0);
    if (currentBar == lastBar) {
        return false;
    }
    lastBar = currentBar;
    
    // Check spread
    if (SymbolInfoInteger(Symbol(), SYMBOL_SPREAD) > MaxSpread) {
        return false;
    }
    
    // Check if market is open
    if (!IsTradeAllowed()) {
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Manage existing orders                                           |
//+------------------------------------------------------------------+
void ManageExistingOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == Symbol() && OrderMagicNumber() == magicNumber) {
            // Update trailing stop
            if (UseTrailingStop) {
                UpdateTrailingStop(OrderTicket());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Update trailing stop                                             |
//+------------------------------------------------------------------+
void UpdateTrailingStop(int ticket) {
    if (OrderSelect(ticket, SELECT_BY_TICKET)) {
        double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
        double trailingStopPrice = 0;
        
        if (OrderType() == ORDER_TYPE_BUY) {
            trailingStopPrice = Bid - TrailingStopPoints * point;
            if (trailingStopPrice > OrderStopLoss() && trailingStopPrice > OrderOpenPrice()) {
                OrderModify(OrderTicket(), OrderOpenPrice(), trailingStopPrice, OrderTakeProfit(), 0, clrNONE);
            }
        } else if (OrderType() == ORDER_TYPE_SELL) {
            trailingStopPrice = Ask + TrailingStopPoints * point;
            if (trailingStopPrice < OrderStopLoss() && trailingStopPrice < OrderOpenPrice()) {
                OrderModify(OrderTicket(), OrderOpenPrice(), trailingStopPrice, OrderTakeProfit(), 0, clrNONE);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic() {
    // Advanced indicator-based strategy
    double rsi = iRSI(Symbol(), 0, 14, PRICE_CLOSE, 0);
    double macdMain, macdSignal;
    int macdHandle = iMACD(Symbol(), 0, 12, 26, 9, PRICE_CLOSE);
    if (macdHandle != INVALID_HANDLE) {
        double macdData[];
        if (CopyBuffer(macdHandle, 0, 0, 1, macdData) > 0) {
            macdMain = macdData[0];
        }
        if (CopyBuffer(macdHandle, 1, 0, 1, macdData) > 0) {
            macdSignal = macdData[0];
        }
        IndicatorRelease(macdHandle);
    }
    
    // Check for buy signal
    if (rsi < 30 && macdMain > macdSignal) {
        CloseSellOrders();
        if (NoOrders()) {
            ExecuteBuyOrder();
        }
    }
    // Check for sell signal
    else if (rsi > 70 && macdMain < macdSignal) {
        CloseBuyOrders();
        if (NoOrders()) {
            ExecuteSellOrder();
        }
    }
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder() {
    double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    double sl = price - StopLoss * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double tp = price + TakeProfit * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    
    if (riskManager.checkRiskRules(Symbol(), lotSize)) {
        int ticket = orderManager.placeOrder(Symbol(), ORDER_TYPE_BUY, lotSize, sl, tp);
        if (ticket > 0) {
            perfTracker.logTrade(ticket, Symbol(), ORDER_TYPE_BUY, lotSize);
        }
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder() {
    double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double sl = price + StopLoss * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double tp = price - TakeProfit * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    
    if (riskManager.checkRiskRules(Symbol(), lotSize)) {
        int ticket = orderManager.placeOrder(Symbol(), ORDER_TYPE_SELL, lotSize, sl, tp);
        if (ticket > 0) {
            perfTracker.logTrade(ticket, Symbol(), ORDER_TYPE_SELL, lotSize);
        }
    }
}

//+------------------------------------------------------------------+
//| Close all buy orders                                             |
//+------------------------------------------------------------------+
void CloseBuyOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == Symbol() && OrderType() == ORDER_TYPE_BUY && OrderMagicNumber() == magicNumber) {
            if (!orderManager.closeOrder(OrderTicket())) {
                Print("OrderClose failed with error #", GetLastError());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Close all sell orders                                            |
//+------------------------------------------------------------------+
void CloseSellOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == Symbol() && OrderType() == ORDER_TYPE_SELL && OrderMagicNumber() == magicNumber) {
            if (!orderManager.closeOrder(OrderTicket())) {
                Print("OrderClose failed with error #", GetLastError());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check if there are no orders                                     |
//+------------------------------------------------------------------+
bool NoOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == Symbol() && OrderMagicNumber() == magicNumber) {
            return false;
        }
    }
    return true;
}

//+------------------------------------------------------------------+
//| Handle order close event                                         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if (trans.type == TRADE_TRANSACTION_ORDER_CLOSE) {
        // Adjust lot size based on result for martingale
        if (UseMartingale) {
            if (result.retcode == TRADE_RETCODE_DONE) {
                if (result.profit < 0) {
                    // Loss - increase lot size
                    lotSize *= MartingaleMultiplier;
                } else {
                    // Profit - reset to initial lot size
                    lotSize = InitialLotSize;
                }
            }
        }
    }
}