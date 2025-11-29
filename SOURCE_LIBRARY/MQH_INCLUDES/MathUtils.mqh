//+------------------------------------------------------------------+
//|                                                    MathUtils.mqh |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"

// Round to nearest tick size
double RoundToTick(double price, string symbol = NULL) {
    if (symbol == NULL) {
        symbol = Symbol();
    }
    
    double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    if (tickSize <= 0) {
        tickSize = 0.00001;
    }
    
    return MathRound(price / tickSize) * tickSize;
}

// Calculate pip value
double CalculatePipValue(string symbol = NULL, double lotSize = 0.01) {
    if (symbol == NULL) {
        symbol = Symbol();
    }
    
    double pipSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE) * 10;
    double contractSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    
    return lotSize * pipSize * contractSize;
}

// Normalize price to point format
string DoubleToStringP(double value, int digits = 5) {
    return DoubleToString(value, digits);
}

// Calculate percentage
double Percentage(double part, double whole) {
    if (whole == 0) return 0;
    return (part / whole) * 100;
}

// Linear interpolation
double LinearInterpolate(double x, double x1, double x2, double y1, double y2) {
    if (x2 == x1) return y1;
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}