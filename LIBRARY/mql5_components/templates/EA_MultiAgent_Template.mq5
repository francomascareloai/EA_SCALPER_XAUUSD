//+------------------------------------------------------------------+
//|                                        EA_MultiAgent_Template.mq5 |
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
input string AgentID = "agent_01";
input string CoordinatorHost = "localhost";
input int CoordinatorPort = 3001;

// Global variables
datetime lastTradeTime = 0;
int magicNumber = 123459;
RiskManager riskManager;
OrderManager orderManager;
PerformanceTracker perfTracker;
double lotSize = InitialLotSize;

// Coordinator connection variables
bool coordinatorConnected = false;
int coordinatorSocket = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize expert advisor
    Print("EA Multi-Agent Template initialized for agent: ", AgentID);
    
    // Initialize components
    riskManager.initialize();
    orderManager.initialize();
    perfTracker.initialize();
    
    // Connect to coordinator
    ConnectToCoordinator();
    
    // Check if symbol is available
    if (!SymbolInfoInteger(Symbol(), SYMBOL_TRADE_ALLOWED)) {
        Print("Trading not allowed for ", Symbol());
        return(INIT_FAILED);
    }
    
    // Register with coordinator
    RegisterWithCoordinator();
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Disconnect from coordinator
    if (coordinatorSocket != -1) {
        SocketClose(coordinatorSocket);
    }
    
    // Deinitialize expert advisor
    Print("EA Multi-Agent Template deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Update components
    orderManager.update();
    perfTracker.update();
    
    // Check coordinator connection
    if (!coordinatorConnected) {
        ConnectToCoordinator();
    }
    
    // Check if it's time to trade
    if (!IsTradeAllowed()) {
        return;
    }
    
    // Manage existing orders
    ManageExistingOrders();
    
    // Get multi-agent trading decision
    ExecuteMultiAgentTradingLogic();
}

//+------------------------------------------------------------------+
//| Connect to coordinator                                           |
//+------------------------------------------------------------------+
void ConnectToCoordinator() {
    coordinatorSocket = SocketCreate();
    if (coordinatorSocket != -1) {
        if (SocketConnect(coordinatorSocket, CoordinatorHost, CoordinatorPort)) {
            coordinatorConnected = true;
            Print("Connected to coordinator at ", CoordinatorHost, ":", CoordinatorPort);
        } else {
            Print("Failed to connect to coordinator");
            coordinatorConnected = false;
        }
    }
}

//+------------------------------------------------------------------+
//| Register with coordinator                                        |
//+------------------------------------------------------------------+
void RegisterWithCoordinator() {
    if (coordinatorConnected) {
        string registration = StringFormat("{\"action\":\"register_agent\",\"agent_id\":\"%s\",\"symbol\":\"%s\"}", AgentID, Symbol());
        SendCoordinatorRequest(registration);
    }
}

//+------------------------------------------------------------------+
//| Send request to coordinator                                      |
//+------------------------------------------------------------------+
string SendCoordinatorRequest(string request) {
    if (!coordinatorConnected) {
        return "";
    }
    
    if (SocketSend(coordinatorSocket, request) > 0) {
        char buffer[2048];
        int received = SocketRead(coordinatorSocket, buffer, 2048, 10000); // 10 second timeout
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
//| Execute multi-agent trading logic                                |
//+------------------------------------------------------------------+
void ExecuteMultiAgentTradingLogic() {
    // Request decision from coordinator
    string request = StringFormat("{\"action\":\"get_trading_decision\",\"agent_id\":\"%s\",\"symbol\":\"%s\",\"timeframe\":\"M1\"}", AgentID, Symbol());
    string response = SendCoordinatorRequest(request);
    
    string decision = "HOLD";
    string reason = "";
    
    if (response != "") {
        // Parse JSON response (simplified)
        if (StringFind(response, "\"decision\":\"BUY\"") != -1) {
            decision = "BUY";
            // Extract reason if available
            int reasonStart = StringFind(response, "\"reason\":\"");
            if (reasonStart != -1) {
                reasonStart += 10; // Length of "\"reason\":\""
                int reasonEnd = StringFind(response, "\"", reasonStart);
                if (reasonEnd != -1) {
                    reason = StringSubstr(response, reasonStart, reasonEnd - reasonStart);
                }
            }
        } else if (StringFind(response, "\"decision\":\"SELL\"") != -1) {
            decision = "SELL";
            // Extract reason if available
            int reasonStart = StringFind(response, "\"reason\":\"");
            if (reasonStart != -1) {
                reasonStart += 10; // Length of "\"reason\":\""
                int reasonEnd = StringFind(response, "\"", reasonStart);
                if (reasonEnd != -1) {
                    reason = StringSubstr(response, reasonStart, reasonEnd - reasonStart);
                }
            }
        }
    }
    
    // Execute based on decision
    if (decision == "BUY") {
        Print("Executing BUY order for agent ", AgentID, ". Reason: ", reason);
        CloseSellOrders();
        if (NoOrders()) {
            ExecuteBuyOrder();
        }
    } else if (decision == "SELL") {
        Print("Executing SELL order for agent ", AgentID, ". Reason: ", reason);
        CloseBuyOrders();
        if (NoOrders()) {
            ExecuteSellOrder();
        }
    }
    
    // Report performance to coordinator
    ReportPerformance();
}

//+------------------------------------------------------------------+
//| Report performance to coordinator                                |
//+------------------------------------------------------------------+
void ReportPerformance() {
    string report = StringFormat("{\"action\":\"report_performance\",\"agent_id\":\"%s\",\"symbol\":\"%s\",\"equity\":%.2f,\"balance\":%.2f}",
                                AgentID, Symbol(), AccountInfoDouble(ACCOUNT_EQUITY), AccountInfoDouble(ACCOUNT_BALANCE));
    SendCoordinatorRequest(report);
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
            
            // Notify coordinator of trade
            string notification = StringFormat("{\"action\":\"trade_executed\",\"agent_id\":\"%s\",\"symbol\":\"%s\",\"type\":\"BUY\",\"ticket\":%d}", AgentID, Symbol(), ticket);
            SendCoordinatorRequest(notification);
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
            
            // Notify coordinator of trade
            string notification = StringFormat("{\"action\":\"trade_executed\",\"agent_id\":\"%s\",\"symbol\":\"%s\",\"type\":\"SELL\",\"ticket\":%d}", AgentID, Symbol(), ticket);
            SendCoordinatorRequest(notification);
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