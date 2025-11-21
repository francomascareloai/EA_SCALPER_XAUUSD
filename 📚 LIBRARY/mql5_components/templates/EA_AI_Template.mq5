//+------------------------------------------------------------------+
//|                                              EA_AI_Template.mq5 |
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
input bool UseMCP = true;
input string MCPHost = "localhost";
input int MCPPort = 3000;

// Global variables
datetime lastTradeTime = 0;
int magicNumber = 123458;
RiskManager riskManager;
OrderManager orderManager;
PerformanceTracker perfTracker;
double lotSize = InitialLotSize;

// MCP connection variables
bool mcpConnected = false;
int mcpSocket = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize expert advisor
    Print("EA AI Template initialized");
    
    // Initialize components
    riskManager.initialize();
    orderManager.initialize();
    perfTracker.initialize();
    
    // Connect to MCP server if enabled
    if (UseMCP) {
        ConnectToMCP();
    }
    
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
    // Disconnect from MCP
    if (mcpSocket != -1) {
        SocketClose(mcpSocket);
    }
    
    // Deinitialize expert advisor
    Print("EA AI Template deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Update components
    orderManager.update();
    perfTracker.update();
    
    // Check MCP connection
    if (UseMCP && !mcpConnected) {
        ConnectToMCP();
    }
    
    // Check if it's time to trade
    if (!IsTradeAllowed()) {
        return;
    }
    
    // Manage existing orders
    ManageExistingOrders();
    
    // Get AI trading decision
    ExecuteAITradingLogic();
}

//+------------------------------------------------------------------+
//| Connect to MCP server                                            |
//+------------------------------------------------------------------+
void ConnectToMCP() {
    mcpSocket = SocketCreate();
    if (mcpSocket != -1) {
        if (SocketConnect(mcpSocket, MCPHost, MCPPort)) {
            mcpConnected = true;
            Print("Connected to MCP server at ", MCPHost, ":", MCPPort);
        } else {
            Print("Failed to connect to MCP server");
            mcpConnected = false;
        }
    }
}

//+------------------------------------------------------------------+
//| Send request to MCP server                                       |
//+------------------------------------------------------------------+
string SendMCPRequest(string request) {
    if (!mcpConnected) {
        return "";
    }
    
    if (SocketSend(mcpSocket, request) > 0) {
        char buffer[1024];
        int received = SocketRead(mcpSocket, buffer, 1024, 5000); // 5 second timeout
        if (received > 0) {
            string response = CharArrayToString(buffer, 0, received);
            return response;
        }
    }
    
    return "";
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
            // Simple order management
        }
    }
}

//+------------------------------------------------------------------+
//| Execute AI trading logic                                         |
//+------------------------------------------------------------------+
void ExecuteAITradingLogic() {
    string decision = "HOLD";
    
    // Get decision from MCP if connected
    if (UseMCP && mcpConnected) {
        string request = StringFormat("{\"action\":\"get_trading_decision\",\"symbol\":\"%s\",\"timeframe\":\"M1\"}", Symbol());
        string response = SendMCPRequest(request);
        
        if (response != "") {
            // Parse JSON response (simplified)
            if (StringFind(response, "\"decision\":\"BUY\"") != -1) {
                decision = "BUY";
            } else if (StringFind(response, "\"decision\":\"SELL\"") != -1) {
                decision = "SELL";
            }
        }
    } else {
        // Fallback to simple technical analysis
        double rsi = iRSI(Symbol(), 0, 14, PRICE_CLOSE, 0);
        double ma = iMA(Symbol(), 0, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
        double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
        
        if (rsi < 30 && price > ma) {
            decision = "BUY";
        } else if (rsi > 70 && price < ma) {
            decision = "SELL";
        }
    }
    
    // Execute based on decision
    if (decision == "BUY") {
        CloseSellOrders();
        if (NoOrders()) {
            ExecuteBuyOrder();
        }
    } else if (decision == "SELL") {
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
            
            // Notify MCP of trade
            if (UseMCP && mcpConnected) {
                string notification = StringFormat("{\"action\":\"trade_executed\",\"symbol\":\"%s\",\"type\":\"BUY\",\"ticket\":%d}", Symbol(), ticket);
                SendMCPRequest(notification);
            }
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
            
            // Notify MCP of trade
            if (UseMCP && mcpConnected) {
                string notification = StringFormat("{\"action\":\"trade_executed\",\"symbol\":\"%s\",\"type\":\"SELL\",\"ticket\":%d}", Symbol(), ticket);
                SendMCPRequest(notification);
            }
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