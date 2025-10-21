//+------------------------------------------------------------------+
//|                                                  TradingEngine.mqh |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"

#include <RiskManager.mqh>
#include <OrderManager.mqh>
#include <PerformanceTracker.mqh>

class TradingEngine {
private:
    RiskManager riskManager;
    OrderManager orderManager;
    PerformanceTracker perfTracker;

public:
    TradingEngine() {
        // Constructor
    }

    void initialize() {
        // Initialize trading engine components
        riskManager.initialize();
        orderManager.initialize();
        perfTracker.initialize();
    }

    bool executeTrade(string symbol, int orderType, double lotSize, double stopLoss, double takeProfit) {
        // Check risk management rules
        if (!riskManager.checkRiskRules(symbol, lotSize)) {
            return false;
        }

        // Execute the trade
        int ticket = orderManager.placeOrder(symbol, orderType, lotSize, stopLoss, takeProfit);
        if (ticket > 0) {
            perfTracker.logTrade(ticket, symbol, orderType, lotSize);
            return true;
        }
        
        return false;
    }

    void update() {
        // Update all engine components
        orderManager.update();
        perfTracker.update();
    }
};