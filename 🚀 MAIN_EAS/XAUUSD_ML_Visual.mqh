//+------------------------------------------------------------------+
//|                                       XAUUSD_ML_Visual.mqh      |
//|                                     Copyright 2024, Elite Trading |
//+------------------------------------------------------------------+

#include <ChartObjects\ChartObjectsTxtControls.mqh>
#include "XAUUSD_ML_Core.mqh"

//+------------------------------------------------------------------+
//| Advanced Visual Interface for XAUUSD ML Trading Bot            |
//+------------------------------------------------------------------+
class CXAUUSD_MLVisual
{
private:
    // Interface Configuration
    color             m_interfaceColor;
    int               m_fontSize;
    string            m_fontName;
    bool              m_showRealTimeAnalysis;
    bool              m_showDecisionProcess;
    
    // Dashboard Components
    CChartObjectLabel m_lblTitle;
    CChartObjectLabel m_lblStatus;
    CChartObjectLabel m_lblStrategy;
    CChartObjectLabel m_lblRiskMetrics;
    CChartObjectLabel m_lblMarketAnalysis;
    CChartObjectLabel m_lblMLPrediction;
    CChartObjectLabel m_lblPerformance;
    CChartObjectLabel m_lblDecisionLog;
    
    // Real-time Data Display
    struct SVisualData
    {
        string        currentStrategy;
        string        marketRegime;
        double        mlConfidence;
        string        riskStatus;
        double        dailyPnL;
        double        currentDrawdown;
        int           totalTrades;
        double        winRate;
        string        lastDecision;
        datetime      lastUpdate;
    };
    
    SVisualData       m_visualData;
    
    // Decision Log
    struct SDecisionEntry
    {
        datetime      timestamp;
        string        decision;
        string        reasoning;
        double        confidence;
        string        outcome;
    };
    
    SDecisionEntry    m_decisionLog[20]; // Keep last 20 decisions
    int               m_logIndex;
    
    // Chart Objects
    string            m_objectPrefix;
    int               m_chartID;
    
    // Animation and Updates
    datetime          m_lastUpdateTime;
    int               m_updateInterval;
    
public:
                     CXAUUSD_MLVisual();
                    ~CXAUUSD_MLVisual();
    
    // Initialization
    bool             Initialize(color interfaceColor);
    void             Cleanup();
    
    // Real-time Updates
    void             UpdateRealTime();
    void             UpdateStatus(string status, string details);
    void             UpdateStrategy(string strategy);
    void             UpdateAnalysis(const SMarketAnalysis &analysis);
    void             UpdateRiskMetrics();
    void             UpdatePerformance();
    void             UpdatePositionInfo(ulong ticket);
    void             UpdateTradeResult(bool success);
    
    // Decision Tracking
    void             LogDecision(string decision, string reasoning, double confidence);
    void             UpdateDecisionOutcome(string outcome);
    void             DisplayDecisionProcess();
    
    // Market Scenario Display
    void             DisplayMarketScenario(const SMarketAnalysis &analysis);
    void             ShowMLAnalysisSteps(const SMLFeatures &features);
    void             DisplayRiskCalculations();
    
private:
    // Interface Setup
    bool             CreateDashboard();
    void             SetupLabels();
    void             PositionElements();
    
    // Display Helpers
    string           FormatMLConfidence(double confidence);
    string           FormatRiskStatus();
    string           FormatPerformanceMetrics();
    string           FormatMarketRegime(ENUM_MARKET_REGIME regime);
    color            GetStatusColor(string status);
    
    // Object Management
    void             DeleteAllObjects();
    string           GenerateObjectName(string suffix);
    bool             CreateLabel(CChartObjectLabel &label, string name, string text, int x, int y, color clr);
    
    // Animation and Effects
    void             AnimateUpdate(string objectName);
    void             FlashAlert(string message);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CXAUUSD_MLVisual::CXAUUSD_MLVisual()
{
    m_interfaceColor = clrCyan;
    m_fontSize = 10;
    m_fontName = "Consolas";
    m_showRealTimeAnalysis = true;
    m_showDecisionProcess = true;
    
    m_objectPrefix = "XAUUSD_ML_";
    m_chartID = ChartID();
    m_logIndex = 0;
    m_lastUpdateTime = 0;
    m_updateInterval = 1; // Update every second
    
    ZeroMemory(m_visualData);
    ZeroMemory(m_decisionLog);
}

//+------------------------------------------------------------------+
//| Initialize Visual Interface                                      |
//+------------------------------------------------------------------+
bool CXAUUSD_MLVisual::Initialize(color interfaceColor)
{
    m_interfaceColor = interfaceColor;
    
    Print("🎨 Initializing XAUUSD ML Visual Interface...");
    
    // Clean up any existing objects
    DeleteAllObjects();
    
    // Create dashboard
    if(!CreateDashboard())
    {
        Print("❌ Failed to create visual dashboard");
        return false;
    }
    
    // Initial display
    UpdateStatus("System Initialized", "Ready for Trading");
    m_visualData.lastUpdate = TimeCurrent();
    
    Print("✅ Visual Interface initialized successfully");
    return true;
}

//+------------------------------------------------------------------+
//| Create Dashboard                                                 |
//+------------------------------------------------------------------+
bool CXAUUSD_MLVisual::CreateDashboard()
{
    int baseX = 20;
    int baseY = 50;
    int lineHeight = 25;
    
    // Title
    if(!CreateLabel(m_lblTitle, "Title", "🥇 XAUUSD ML Trading Bot - Elite System", 
                   baseX, baseY, clrGold))
        return false;
    
    // System Status
    if(!CreateLabel(m_lblStatus, "Status", "Status: Initializing...", 
                   baseX, baseY + lineHeight, m_interfaceColor))
        return false;
    
    // Current Strategy
    if(!CreateLabel(m_lblStrategy, "Strategy", "Strategy: None", 
                   baseX, baseY + lineHeight * 2, clrLime))
        return false;
    
    // Risk Metrics
    if(!CreateLabel(m_lblRiskMetrics, "Risk", "Risk: Calculating...", 
                   baseX, baseY + lineHeight * 3, clrOrange))
        return false;
    
    // Market Analysis
    if(!CreateLabel(m_lblMarketAnalysis, "Market", "Market: Analyzing...", 
                   baseX, baseY + lineHeight * 4, clrWhite))
        return false;
    
    // ML Prediction
    if(!CreateLabel(m_lblMLPrediction, "ML", "ML Confidence: 0%", 
                   baseX, baseY + lineHeight * 5, clrDeepSkyBlue))
        return false;
    
    // Performance
    if(!CreateLabel(m_lblPerformance, "Performance", "Performance: Tracking...", 
                   baseX, baseY + lineHeight * 6, clrYellow))
        return false;
    
    // Decision Log
    if(!CreateLabel(m_lblDecisionLog, "Decisions", "🧠 Decision Log:", 
                   baseX, baseY + lineHeight * 8, clrSilver))
        return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Update Real-time Display                                        |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdateRealTime()
{
    if(TimeCurrent() - m_lastUpdateTime < m_updateInterval)
        return;
    
    // Update all components
    UpdateRiskMetrics();
    UpdatePerformance();
    
    // Update timestamp
    m_visualData.lastUpdate = TimeCurrent();
    m_lastUpdateTime = TimeCurrent();
    
    // Refresh chart
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Update Status                                                    |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdateStatus(string status, string details)
{
    string statusText = StringFormat("📊 Status: %s | %s", status, details);
    m_lblStatus.Description(statusText);
    
    // Set color based on status
    color statusColor = GetStatusColor(status);
    m_lblStatus.Color(statusColor);
    
    m_visualData.riskStatus = status;
    
    // Log important status changes
    if(StringFind(status, "Alert") >= 0 || StringFind(status, "Error") >= 0)
    {
        LogDecision("Status Change", details, 1.0);
    }
}

//+------------------------------------------------------------------+
//| Update Strategy Display                                          |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdateStrategy(string strategy)
{
    string strategyText = StringFormat("🎯 Strategy: %s | Confidence: %.1f%%", 
                                     strategy, m_visualData.mlConfidence * 100);
    m_lblStrategy.Description(strategyText);
    m_visualData.currentStrategy = strategy;
    
    // Animate strategy change
    AnimateUpdate("Strategy");
}

//+------------------------------------------------------------------+
//| Update Market Analysis Display                                   |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdateAnalysis(const SMarketAnalysis &analysis)
{
    // Update market regime
    string regimeText = FormatMarketRegime(analysis.regime);
    string volatilityText = StringFormat("%.1f pips", analysis.volatility / Point);
    
    string analysisText = StringFormat("📈 Market: %s | Volatility: %s | Session: %s", 
                                     regimeText, volatilityText, 
                                     analysis.isSessionActive ? "Active" : "Inactive");
    
    m_lblMarketAnalysis.Description(analysisText);
    m_visualData.marketRegime = regimeText;
    
    // Update ML confidence
    m_visualData.mlConfidence = analysis.confidence;
    string mlText = StringFormat("🧠 ML Confidence: %s | News: %s", 
                                FormatMLConfidence(analysis.confidence),
                                analysis.isNewsTime ? "High Impact" : "Clear");
    
    m_lblMLPrediction.Description(mlText);
    
    // Show detailed ML analysis steps if enabled
    if(m_showRealTimeAnalysis)
    {
        ShowMLAnalysisSteps(analysis.features);
        DisplayMarketScenario(analysis);
    }
}

//+------------------------------------------------------------------+
//| Update Risk Metrics                                             |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdateRiskMetrics()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (balance > equity) ? (balance - equity) / balance * 100 : 0.0;
    
    m_visualData.currentDrawdown = drawdown;
    
    string riskText = StringFormat("🛡️ Risk: DD %.1f%% | Equity: $%.2f | Positions: %d", 
                                 drawdown, equity, PositionsTotal());
    
    m_lblRiskMetrics.Description(riskText);
    
    // Color coding based on risk level
    color riskColor = clrLime;
    if(drawdown > 2.0) riskColor = clrOrange;
    if(drawdown > 3.0) riskColor = clrRed;
    
    m_lblRiskMetrics.Color(riskColor);
}

//+------------------------------------------------------------------+
//| Update Performance Metrics                                       |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::UpdatePerformance()
{
    // Calculate daily P&L
    static double dailyStartBalance = 0;
    if(dailyStartBalance == 0)
        dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    m_visualData.dailyPnL = currentBalance - dailyStartBalance;
    
    // Calculate win rate (simplified)
    int totalTrades = 0, winningTrades = 0;
    
    // Count today's trades from history
    datetime today = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    HistorySelect(today, TimeCurrent());
    
    for(int i = 0; i < HistoryDealsTotal(); i++)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealSelect(ticket))
        {
            if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
            {
                totalTrades++;
                double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
                if(profit > 0) winningTrades++;
            }
        }
    }
    
    m_visualData.totalTrades = totalTrades;
    m_visualData.winRate = (totalTrades > 0) ? (double)winningTrades / totalTrades : 0.0;
    
    string perfText = StringFormat("📊 Performance: P&L $%.2f | Trades: %d | Win Rate: %.1f%%", 
                                 m_visualData.dailyPnL, totalTrades, m_visualData.winRate * 100);
    
    m_lblPerformance.Description(perfText);
    
    // Color based on performance
    color perfColor = (m_visualData.dailyPnL >= 0) ? clrLime : clrOrange;
    m_lblPerformance.Color(perfColor);
}

//+------------------------------------------------------------------+
//| Log Trading Decision                                             |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::LogDecision(string decision, string reasoning, double confidence)
{
    m_decisionLog[m_logIndex].timestamp = TimeCurrent();
    m_decisionLog[m_logIndex].decision = decision;
    m_decisionLog[m_logIndex].reasoning = reasoning;
    m_decisionLog[m_logIndex].confidence = confidence;
    m_decisionLog[m_logIndex].outcome = "Pending";
    
    m_logIndex = (m_logIndex + 1) % 20; // Circular buffer
    
    DisplayDecisionProcess();
}

//+------------------------------------------------------------------+
//| Display Decision Process                                         |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::DisplayDecisionProcess()
{
    if(!m_showDecisionProcess) return;
    
    string decisionText = "🧠 Recent Decisions:\n";
    
    // Show last 5 decisions
    for(int i = 0; i < 5; i++)
    {
        int idx = (m_logIndex - 1 - i + 20) % 20;
        if(m_decisionLog[idx].timestamp > 0)
        {
            string timeStr = TimeToString(m_decisionLog[idx].timestamp, TIME_MINUTES);
            decisionText += StringFormat("%s: %s (%.0f%%)\n", 
                                       timeStr, 
                                       m_decisionLog[idx].decision,
                                       m_decisionLog[idx].confidence * 100);
        }
    }
    
    // Create or update decision log object
    string objName = GenerateObjectName("DecisionDetail");
    ObjectDelete(m_chartID, objName);
    
    ObjectCreate(m_chartID, objName, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(m_chartID, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(m_chartID, objName, OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(m_chartID, objName, OBJPROP_YDISTANCE, 250);
    ObjectSetString(m_chartID, objName, OBJPROP_TEXT, decisionText);
    ObjectSetInteger(m_chartID, objName, OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(m_chartID, objName, OBJPROP_COLOR, clrSilver);
}

//+------------------------------------------------------------------+
//| Show ML Analysis Steps                                           |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::ShowMLAnalysisSteps(const SMLFeatures &features)
{
    string analysisSteps = "🔍 ML Analysis Steps:\n";
    
    analysisSteps += StringFormat("• Price Momentum: %.1f pips\n", features.price_momentum[0]);
    analysisSteps += StringFormat("• Volatility Ratio: %.3f\n", features.volatility_ratio);
    analysisSteps += StringFormat("• RSI Divergence: %.1f\n", features.rsi_divergence);
    analysisSteps += StringFormat("• Order Block Strength: %.1f%%\n", features.order_block_strength * 100);
    analysisSteps += StringFormat("• Institutional Flow: %.1f\n", features.institutional_flow);
    analysisSteps += StringFormat("• Session Activity: %.1f\n", features.session_volatility);
    
    // Create analysis steps object
    string objName = GenerateObjectName("AnalysisSteps");
    ObjectDelete(m_chartID, objName);
    
    ObjectCreate(m_chartID, objName, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(m_chartID, objName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
    ObjectSetInteger(m_chartID, objName, OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(m_chartID, objName, OBJPROP_YDISTANCE, 50);
    ObjectSetString(m_chartID, objName, OBJPROP_TEXT, analysisSteps);
    ObjectSetInteger(m_chartID, objName, OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(m_chartID, objName, OBJPROP_COLOR, clrDeepSkyBlue);
}

//+------------------------------------------------------------------+
//| Display Market Scenario                                         |
//+------------------------------------------------------------------+
void CXAUUSD_MLVisual::DisplayMarketScenario(const SMarketAnalysis &analysis)
{
    string scenario = "📊 Current Market Scenario:\n";
    
    // Determine scenario
    if(analysis.regime == REGIME_VOLATILE && analysis.isNewsTime)
    {
        scenario += "⚡ HIGH VOLATILITY + NEWS EVENT\n";
        scenario += "Action: Reduced position sizing\n";
        scenario += "Strategy: Wait for volatility to settle\n";
    }
    else if(analysis.regime == REGIME_LOW_VOLATILITY)
    {
        scenario += "😴 LOW VOLATILITY ENVIRONMENT\n";
        scenario += "Action: Mean reversion strategy active\n";
        scenario += "Strategy: Range trading with tight stops\n";
    }
    else if(analysis.regime == REGIME_TRENDING_UP || analysis.regime == REGIME_TRENDING_DOWN)
    {
        scenario += "📈 TRENDING MARKET DETECTED\n";
        scenario += "Action: Trend following strategy active\n";
        scenario += "Strategy: Momentum-based entries\n";
    }
    else if(analysis.regime == REGIME_RANGING)
    {
        scenario += "↔️ RANGING MARKET CONDITIONS\n";
        scenario += "Action: Support/resistance trading\n";
        scenario += "Strategy: Mean reversion with oscillators\n";
    }
    
    // Add session information
    if(analysis.isSessionActive)
    {
        scenario += "🌍 Active trading session\n";
    }
    else
    {
        scenario += "🌙 Low activity session\n";
    }
    
    // Create scenario object
    string objName = GenerateObjectName("MarketScenario");
    ObjectDelete(m_chartID, objName);
    
    ObjectCreate(m_chartID, objName, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(m_chartID, objName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
    ObjectSetInteger(m_chartID, objName, OBJPROP_XDISTANCE, 20);
    ObjectSetInteger(m_chartID, objName, OBJPROP_YDISTANCE, 200);
    ObjectSetString(m_chartID, objName, OBJPROP_TEXT, scenario);
    ObjectSetInteger(m_chartID, objName, OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(m_chartID, objName, OBJPROP_COLOR, clrYellow);
}

//+------------------------------------------------------------------+
//| Helper Methods                                                   |
//+------------------------------------------------------------------+
string CXAUUSD_MLVisual::FormatMLConfidence(double confidence)
{
    if(confidence > 0.85) return StringFormat("🟢 %.1f%% (High)", confidence * 100);
    if(confidence > 0.70) return StringFormat("🟡 %.1f%% (Medium)", confidence * 100);
    return StringFormat("🔴 %.1f%% (Low)", confidence * 100);
}

string CXAUUSD_MLVisual::FormatMarketRegime(ENUM_MARKET_REGIME regime)
{
    switch(regime)
    {
        case REGIME_TRENDING_UP: return "📈 Trending Up";
        case REGIME_TRENDING_DOWN: return "📉 Trending Down";
        case REGIME_RANGING: return "↔️ Ranging";
        case REGIME_VOLATILE: return "⚡ Volatile";
        case REGIME_LOW_VOLATILITY: return "😴 Low Volatility";
        default: return "❓ Unknown";
    }
}

color CXAUUSD_MLVisual::GetStatusColor(string status)
{
    if(StringFind(status, "Error") >= 0 || StringFind(status, "Emergency") >= 0) return clrRed;
    if(StringFind(status, "Warning") >= 0 || StringFind(status, "Alert") >= 0) return clrOrange;
    if(StringFind(status, "Success") >= 0 || StringFind(status, "Ready") >= 0) return clrLime;
    return m_interfaceColor;
}

bool CXAUUSD_MLVisual::CreateLabel(CChartObjectLabel &label, string name, string text, int x, int y, color clr)
{
    string objName = GenerateObjectName(name);
    
    if(!label.Create(m_chartID, objName, 0, x, y))
        return false;
    
    label.Description(text);
    label.FontSize(m_fontSize);
    label.Font(m_fontName);
    label.Color(clr);
    label.Corner(CORNER_LEFT_UPPER);
    
    return true;
}

string CXAUUSD_MLVisual::GenerateObjectName(string suffix)
{
    return m_objectPrefix + suffix;
}

void CXAUUSD_MLVisual::DeleteAllObjects()
{
    for(int i = ObjectsTotal(m_chartID) - 1; i >= 0; i--)
    {
        string objName = ObjectName(m_chartID, i);
        if(StringFind(objName, m_objectPrefix) == 0)
        {
            ObjectDelete(m_chartID, objName);
        }
    }
}

void CXAUUSD_MLVisual::AnimateUpdate(string objectName)
{
    // Simple flash animation
    string objName = GenerateObjectName(objectName);
    ObjectSetInteger(m_chartID, objName, OBJPROP_COLOR, clrWhite);
    Sleep(100);
    ObjectSetInteger(m_chartID, objName, OBJPROP_COLOR, m_interfaceColor);
}

void CXAUUSD_MLVisual::Cleanup()
{
    DeleteAllObjects();
    Print("🎨 Visual Interface cleaned up");
}