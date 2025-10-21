//+------------------------------------------------------------------+
//|                                          XAUUSD_ML_Trading_Bot.mq5 |
//|                                     Copyright 2024, Elite Trading |
//|                                  https://github.com/elite-trading |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Elite Trading"
#property link      "https://github.com/elite-trading"
#property version   "1.00"
#property description "Advanced XAUUSD ML Trading Bot - Elite Intelligent Trading System"

// Include necessary files
#include "XAUUSD_ML_Core.mqh"
#include "XAUUSD_ML_Strategies.mqh"
#include "XAUUSD_ML_Risk.mqh"
#include "XAUUSD_ML_Visual.mqh"

//--- Input parameters
input group "=== ML CONFIGURATION ==="
input bool     EnableMLPrediction = true;                    // Enable ML Prediction
input double   MLConfidenceThreshold = 0.75;                 // ML Confidence Threshold (0.0-1.0)
input int      MLModelUpdateHours = 24;                      // ML Model Update Frequency (hours)

input group "=== RISK MANAGEMENT ==="
input double   BaseRiskPercent = 0.01;                       // Base Risk per Trade (%)
input double   MaxDailyRisk = 0.02;                          // Maximum Daily Risk (%)
input double   MaxDrawdownPercent = 0.03;                    // Maximum Drawdown (%)
input bool     EnableFTMOCompliance = true;                  // Enable FTMO Compliance

input group "=== STRATEGY SELECTION ==="
input bool     EnableSmartMoney = true;                      // Enable ICT Smart Money Strategy
input bool     EnableMLScalping = true;                      // Enable ML Adaptive Scalping
input bool     EnableVolatilityBreakout = true;              // Enable Volatility Breakout
input bool     EnableMultiTimeframe = true;                  // Enable Multi-Timeframe Analysis

input group "=== LATENCY OPTIMIZATION ==="
input int      MaxLatencyMS = 120;                           // Maximum Acceptable Latency (ms)
input bool     EnablePreStops = true;                        // Enable Pre-Stop Validation
input bool     EnableOrderBuffering = true;                  // Enable Order Buffering
input double   SlippageTolerancePips = 2.0;                  // Slippage Tolerance (pips)

input group "=== VISUAL INTERFACE ==="
input bool     EnableVisualInterface = true;                 // Enable Visual Dashboard
input bool     ShowRealTimeAnalysis = true;                  // Show Real-Time Analysis
input bool     ShowDecisionProcess = true;                   // Show Decision Process
input color    InterfaceColor = clrCyan;                     // Interface Color

//--- Global variables
CXAUUSD_MLCore*        g_mlCore;
CXAUUSD_MLStrategies*  g_strategies;
CXAUUSD_MLRisk*        g_riskManager;
CXAUUSD_MLVisual*      g_visualInterface;

datetime               g_lastMLUpdate;
double                 g_dailyPnL;
double                 g_maxDailyDrawdown;
bool                   g_tradingEnabled;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 XAUUSD ML Trading Bot - Elite System Initializing...");
    
    // Initialize core components
    g_mlCore = new CXAUUSD_MLCore();
    g_strategies = new CXAUUSD_MLStrategies();
    g_riskManager = new CXAUUSD_MLRisk();
    g_visualInterface = new CXAUUSD_MLVisual();
    
    // Configure parameters
    if(!ConfigureSystem())
    {
        Print("❌ System configuration failed!");
        return INIT_FAILED;
    }
    
    // Initialize ML models
    if(EnableMLPrediction && !g_mlCore.LoadMLModels())
    {
        Print("⚠️ ML Models initialization failed - continuing with traditional strategies");
    }
    
    // Initialize visual interface
    if(EnableVisualInterface)
    {
        g_visualInterface.Initialize(InterfaceColor);
        g_visualInterface.UpdateStatus("System Initialized", "Ready for Trading");
    }
    
    g_tradingEnabled = true;
    g_lastMLUpdate = TimeCurrent();
    
    Print("✅ XAUUSD ML Trading Bot - Elite System Initialized Successfully!");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🔄 XAUUSD ML Trading Bot - Shutting down...");
    
    // Clean up resources
    if(g_mlCore != NULL)
    {
        delete g_mlCore;
        g_mlCore = NULL;
    }
    
    if(g_strategies != NULL)
    {
        delete g_strategies;
        g_strategies = NULL;
    }
    
    if(g_riskManager != NULL)
    {
        delete g_riskManager;
        g_riskManager = NULL;
    }
    
    if(g_visualInterface != NULL)
    {
        g_visualInterface.Cleanup();
        delete g_visualInterface;
        g_visualInterface = NULL;
    }
    
    Print("✅ XAUUSD ML Trading Bot - Shutdown completed");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Update visual interface
    if(EnableVisualInterface && g_visualInterface != NULL)
    {
        g_visualInterface.UpdateRealTime();
    }
    
    // Check if trading is enabled
    if(!g_tradingEnabled || !IsTradeAllowed())
    {
        return;
    }
    
    // Update ML models if needed
    if(EnableMLPrediction && ShouldUpdateMLModel())
    {
        g_mlCore.UpdateModels();
        g_lastMLUpdate = TimeCurrent();
    }
    
    // Perform market analysis
    SMarketAnalysis marketAnalysis;
    if(!g_mlCore.AnalyzeMarket(marketAnalysis))
    {
        return;
    }
    
    // Update visual interface with analysis
    if(EnableVisualInterface)
    {
        g_visualInterface.UpdateAnalysis(marketAnalysis);
    }
    
    // Check risk limits
    if(!g_riskManager.ValidateRiskLimits())
    {
        g_tradingEnabled = false;
        if(EnableVisualInterface)
        {
            g_visualInterface.UpdateStatus("Trading Disabled", "Risk Limits Exceeded");
        }
        return;
    }
    
    // Strategy execution
    ExecuteTradingStrategies(marketAnalysis);
    
    // Monitor existing positions
    MonitorPositions();
}

//+------------------------------------------------------------------+
//| Configure system parameters                                      |
//+------------------------------------------------------------------+
bool ConfigureSystem()
{
    // Configure ML Core
    g_mlCore.SetConfidenceThreshold(MLConfidenceThreshold);
    g_mlCore.SetUpdateFrequency(MLModelUpdateHours);
    
    // Configure Risk Manager
    g_riskManager.SetBaseRisk(BaseRiskPercent);
    g_riskManager.SetMaxDailyRisk(MaxDailyRisk);
    g_riskManager.SetMaxDrawdown(MaxDrawdownPercent);
    g_riskManager.SetFTMOCompliance(EnableFTMOCompliance);
    
    // Configure Strategies
    g_strategies.EnableSmartMoney(EnableSmartMoney);
    g_strategies.EnableMLScalping(EnableMLScalping);
    g_strategies.EnableVolatilityBreakout(EnableVolatilityBreakout);
    g_strategies.EnableMultiTimeframe(EnableMultiTimeframe);
    
    // Configure Latency Optimization
    g_mlCore.SetMaxLatency(MaxLatencyMS);
    g_mlCore.SetPreStopsEnabled(EnablePreStops);
    g_mlCore.SetOrderBuffering(EnableOrderBuffering);
    g_mlCore.SetSlippageTolerance(SlippageTolerancePips);
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trading strategies                                        |
//+------------------------------------------------------------------+
void ExecuteTradingStrategies(const SMarketAnalysis &analysis)
{
    if(g_strategies == NULL) return;
    
    // Get optimal strategy based on market conditions
    ENUM_STRATEGY_TYPE optimalStrategy = g_strategies.SelectOptimalStrategy(analysis);
    
    if(EnableVisualInterface)
    {
        g_visualInterface.UpdateStrategy(EnumToString(optimalStrategy));
    }
    
    // Execute strategy
    STradeSignal signal;
    bool hasSignal = false;
    
    switch(optimalStrategy)
    {
        case STRATEGY_SMART_MONEY:
            hasSignal = g_strategies.GenerateSmartMoneySignal(analysis, signal);
            break;
            
        case STRATEGY_ML_SCALPING:
            hasSignal = g_strategies.GenerateMLScalpingSignal(analysis, signal);
            break;
            
        case STRATEGY_VOLATILITY_BREAKOUT:
            hasSignal = g_strategies.GenerateVolatilityBreakoutSignal(analysis, signal);
            break;
            
        default:
            return;
    }
    
    // Execute signal if valid
    if(hasSignal && ValidateSignal(signal))
    {
        ExecuteTradeSignal(signal);
    }
}

//+------------------------------------------------------------------+
//| Validate trade signal                                            |
//+------------------------------------------------------------------+
bool ValidateSignal(const STradeSignal &signal)
{
    // Risk validation
    if(!g_riskManager.ValidateSignal(signal))
    {
        if(EnableVisualInterface)
        {
            g_visualInterface.UpdateStatus("Signal Rejected", "Risk Validation Failed");
        }
        return false;
    }
    
    // Latency validation for high-latency environments
    if(MaxLatencyMS > 100 && !ValidateLatencyConditions(signal))
    {
        if(EnableVisualInterface)
        {
            g_visualInterface.UpdateStatus("Signal Rejected", "Latency Conditions Not Met");
        }
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trade signal                                             |
//+------------------------------------------------------------------+
void ExecuteTradeSignal(const STradeSignal &signal)
{
    double lotSize = g_riskManager.CalculateOptimalLotSize(signal);
    
    if(EnableVisualInterface)
    {
        string msg = StringFormat("Executing %s - Lot: %.2f", 
                                EnumToString(signal.type), lotSize);
        g_visualInterface.UpdateStatus("Trade Execution", msg);
    }
    
    // Execute with latency optimization
    bool success = g_mlCore.ExecuteTradeWithLatencyOptimization(signal, lotSize);
    
    if(EnableVisualInterface)
    {
        g_visualInterface.UpdateTradeResult(success);
    }
}

//+------------------------------------------------------------------+
//| Check if ML model should be updated                             |
//+------------------------------------------------------------------+
bool ShouldUpdateMLModel()
{
    return (TimeCurrent() - g_lastMLUpdate) >= (MLModelUpdateHours * 3600);
}

//+------------------------------------------------------------------+
//| Validate latency conditions for high-latency environments       |
//+------------------------------------------------------------------+
bool ValidateLatencyConditions(const STradeSignal &signal)
{
    // For 120ms+ latency, ensure signal is still valid after delay
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double priceChange = MathAbs(currentPrice - signal.entryPrice) / Point;
    
    // Reject if price moved more than tolerance
    if(priceChange > SlippageTolerancePips)
    {
        return false;
    }
    
    // Additional validations for high-latency environment
    if(!g_mlCore.ValidatePreStopConditions(signal))
    {
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Monitor existing positions                                        |
//+------------------------------------------------------------------+
void MonitorPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionSelectByTicket(ticket))
        {
            // Update position management
            g_strategies.ManagePosition(ticket);
            
            // Update visual interface
            if(EnableVisualInterface)
            {
                g_visualInterface.UpdatePositionInfo(ticket);
            }
        }
    }
}