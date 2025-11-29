//+------------------------------------------------------------------+
//|                                                    TimeUtils.mqh |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"

// Check if it's a new bar
bool IsNewBar() {
    static datetime lastTime = 0;
    datetime currentTime = Time[0];
    
    if (currentTime != lastTime) {
        lastTime = currentTime;
        return true;
    }
    
    return false;
}

// Get session start time
datetime GetSessionStart(int sessionType) {
    // 0 = Asian, 1 = European, 2 = American
    int hour = 0;
    int minute = 0;
    
    switch(sessionType) {
        case 0: // Asian session
            hour = 0;
            minute = 0;
            break;
        case 1: // European session
            hour = 7;
            minute = 0;
            break;
        case 2: // American session
            hour = 13;
            minute = 30;
            break;
        default:
            hour = 0;
            minute = 0;
    }
    
    return TimeHourToDateTime(hour, minute);
}

// Convert hour and minute to datetime
datetime TimeHourToDateTime(int hour, int minute) {
    datetime now = TimeCurrent();
    return StrToTime(StringFormat("%04d.%02d.%02d %02d:%02d", 
                                  TimeYear(now), TimeMonth(now), TimeDay(now), hour, minute));
}

// Check if market is open
bool IsMarketOpen(string symbol = NULL) {
    if (symbol == NULL) {
        symbol = Symbol();
    }
    
    return SymbolInfoInteger(symbol, SYMBOL_SESSION_TRADE);
}

// Get time until next bar
int TimeUntilNextBar() {
    datetime currentTime = TimeCurrent();
    datetime nextBarTime = Time[0] + Period() * 60;
    return (int)(nextBarTime - currentTime);
}