//+------------------------------------------------------------------+
//|                                            EA_Basic_Template.mq5 |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"
#property strict
#property robot_indicators

// Input parameters
input double LotSize = 0.01;
input int StopLoss = 100;
input int TakeProfit = 200;
input int MaxSpread = 30;

// Global variables
datetime lastTradeTime = 0;
int magicNumber = 123456;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize expert advisor
    Print("EA Basic Template initialized");
    
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
    Print("EA Basic Template deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Check if it's time to trade
    if (!IsTradeAllowed()) {
        return;
    }
    
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
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic() {
    // Simple moving average crossover strategy
    double fastMA = iMA(Symbol(), 0, 10, 0, MODE_SMA, PRICE_CLOSE, 0);
    double slowMA = iMA(Symbol(), 0, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
    double prevFastMA = iMA(Symbol(), 0, 10, 0, MODE_SMA, PRICE_CLOSE, 1);
    double prevSlowMA = iMA(Symbol(), 0, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
    
    // Check for crossover
    if (fastMA > slowMA && prevFastMA <= prevSlowMA) {
        // Buy signal
        CloseSellOrders();
        if (NoOrders()) {
            ExecuteBuyOrder();
        }
    } else if (fastMA < slowMA && prevFastMA >= prevSlowMA) {
        // Sell signal
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
    
    int ticket = OrderSend(Symbol(), ORDER_TYPE_BUY, LotSize, price, 3, sl, tp, "Basic EA Buy", magicNumber, 0, clrGreen);
    if (ticket < 0) {
        Print("OrderSend failed with error #", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder() {
    double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double sl = price + StopLoss * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double tp = price - TakeProfit * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    
    int ticket = OrderSend(Symbol(), ORDER_TYPE_SELL, LotSize, price, 3, sl, tp, "Basic EA Sell", magicNumber, 0, clrRed);
    if (ticket < 0) {
        Print("OrderSend failed with error #", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Close all buy orders                                             |
//+------------------------------------------------------------------+
void CloseBuyOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == Symbol() && OrderType() == ORDER_TYPE_BUY && OrderMagicNumber() == magicNumber) {
            if (!OrderClose(OrderTicket())) {
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
            if (!OrderClose(OrderTicket())) {
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