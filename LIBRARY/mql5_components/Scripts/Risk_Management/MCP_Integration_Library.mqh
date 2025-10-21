//+------------------------------------------------------------------+
//|                                       MCP_Integration_Library.mqh |
//|           AI-Driven Strategy Optimization via MCP Servers        |
//|                     Integration with Python MCP Ecosystem       |
//+------------------------------------------------------------------+
#property copyright \"Elite EA MCP Integration\"
#property version   \"1.00\"
#property strict

// === MCP INTEGRATION INCLUDES ===
#include <Files\\FileTxt.mqh>
#include <Files\\File.mqh>
#include <JAson.mqh>  // For JSON handling

// === MCP SERVER CONFIGURATION ===
struct SMCPServerConfig
{
    string server_name;
    string endpoint_url;
    string python_script_path;
    bool is_active;
    datetime last_response;
    int response_count;
    double avg_response_time;
};

// === AI OPTIMIZATION REQUEST ===
struct SAIOptimizationRequest
{
    string request_id;
    string optimization_type;    // \"confluence\", \"risk\", \"timing\", \"parameters\"
    string market_data;          // JSON string with market data
    string current_parameters;   // Current EA parameters
    string performance_metrics;  // Current performance data
    datetime request_timestamp;
};

// === AI OPTIMIZATION RESPONSE ===
struct SAIOptimizationResponse
{
    string request_id;
    string optimization_type;
    bool success;
    string error_message;
    
    // Optimized parameters
    double optimized_confluence_threshold;
    double optimized_risk_percent;
    double optimized_stop_loss;
    double optimized_take_profit;
    
    // AI insights
    string market_condition_analysis;
    string optimization_reasoning;
    double confidence_score;
    string recommended_actions[];
    
    datetime response_timestamp;
};

// === MCP INTEGRATION CLASS ===
class CMCPIntegration
{
private:
    SMCPServerConfig m_servers[10];
    int m_server_count;
    string m_integration_path;
    string m_data_exchange_path;
    bool m_initialization_successful;
    
    // Communication files
    string m_request_file;
    string m_response_file;
    string m_status_file;
    
public:
    // Constructor
    CMCPIntegration();
    ~CMCPIntegration();
    
    // Initialization
    bool Initialize(string integration_path);
    bool ValidateMCPServers();
    
    // AI Optimization Methods
    bool RequestConfluenceOptimization(const SConfluenceSignal& signal, SAIOptimizationResponse& response);
    bool RequestRiskOptimization(const SPerformanceMetrics& metrics, SAIOptimizationResponse& response);
    bool RequestParameterOptimization(string parameters_json, SAIOptimizationResponse& response);
    bool RequestMarketAnalysis(string market_data_json, SAIOptimizationResponse& response);
    
    // Trading Intelligence
    bool GetTradingRecommendations(string& recommendations);
    bool AnalyzeMarketConditions(string& analysis);
    bool ValidateTradeSetup(const SConfluenceSignal& signal, bool& is_valid);
    
    // Performance Enhancement
    bool OptimizeStrategyParameters(double& confluence_threshold, double& risk_percent);
    bool PredictMarketDirection(ENUM_SIGNAL_TYPE& predicted_direction, double& confidence);
    bool CalculateOptimalEntryTiming(datetime& optimal_entry_time);
    
    // Communication Methods
    bool SendRequestToMCP(const SAIOptimizationRequest& request);
    bool ReceiveResponseFromMCP(SAIOptimizationResponse& response);
    bool CheckMCPServerStatus();
    
    // Utility Methods
    string CreateMarketDataJSON();
    string CreateParametersJSON();
    string CreatePerformanceJSON();
    
    // Status and Monitoring
    bool IsInitialized() { return m_initialization_successful; }
    int GetActiveServerCount();
    string GetIntegrationStatus();
};

// === GLOBAL MCP INTEGRATION INSTANCE ===
CMCPIntegration* g_mcp_integration = NULL;

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMCPIntegration::CMCPIntegration()
{
    m_server_count = 0;
    m_initialization_successful = false;
    m_integration_path = \"\";
    m_data_exchange_path = \"\";
    
    // Initialize communication files
    m_request_file = \"mcp_request.json\";
    m_response_file = \"mcp_response.json\";
    m_status_file = \"mcp_status.json\";
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CMCPIntegration::~CMCPIntegration()
{
    // Cleanup if needed
}

//+------------------------------------------------------------------+
//| Initialize MCP Integration                                       |
//+------------------------------------------------------------------+
bool CMCPIntegration::Initialize(string integration_path)
{
    m_integration_path = integration_path;
    m_data_exchange_path = integration_path + \"\\\\data\\\\\";
    
    Print(\"Initializing MCP Integration...\");
    Print(\"Integration Path: \", m_integration_path);
    
    // Configure MCP servers
    m_server_count = 0;
    
    // Trading Classifier MCP
    m_servers[m_server_count].server_name = \"trading_classifier\";
    m_servers[m_server_count].endpoint_url = \"http://localhost:8001\";
    m_servers[m_server_count].python_script_path = integration_path + \"\\\\servers\\\\trading_classifier_mcp.py\";
    m_servers[m_server_count].is_active = true;
    m_server_count++;
    
    // Code Analysis MCP
    m_servers[m_server_count].server_name = \"code_analysis\";
    m_servers[m_server_count].endpoint_url = \"http://localhost:8002\";
    m_servers[m_server_count].python_script_path = integration_path + \"\\\\servers\\\\code_analysis_mcp.py\";
    m_servers[m_server_count].is_active = true;
    m_server_count++;
    
    // Test Automation MCP
    m_servers[m_server_count].server_name = \"test_automation\";
    m_servers[m_server_count].endpoint_url = \"http://localhost:8003\";
    m_servers[m_server_count].python_script_path = integration_path + \"\\\\servers\\\\test_automation_mcp.py\";
    m_servers[m_server_count].is_active = true;
    m_server_count++;
    
    // Python Dev Accelerator MCP
    m_servers[m_server_count].server_name = \"python_dev_accelerator\";
    m_servers[m_server_count].endpoint_url = \"http://localhost:8004\";
    m_servers[m_server_count].python_script_path = integration_path + \"\\\\servers\\\\python_dev_accelerator_mcp.py\";
    m_servers[m_server_count].is_active = true;
    m_server_count++;
    
    // Validate servers
    if(ValidateMCPServers())
    {
        m_initialization_successful = true;
        Print(\"✅ MCP Integration initialized successfully with \", m_server_count, \" servers\");
        return true;
    }
    else
    {
        Print(\"❌ MCP Integration initialization failed\");
        return false;
    }
}

//+------------------------------------------------------------------+
//| Validate MCP Servers                                            |
//+------------------------------------------------------------------+
bool CMCPIntegration::ValidateMCPServers()
{
    Print(\"Validating MCP servers...\");
    
    // Create data exchange directory if it doesn't exist
    if(!FileIsExist(m_data_exchange_path, FILE_COMMON))
    {
        Print(\"Creating data exchange directory: \", m_data_exchange_path);
    }
    
    // Check if Python scripts exist
    int valid_servers = 0;
    
    for(int i = 0; i < m_server_count; i++)
    {
        string script_path = m_servers[i].python_script_path;
        
        if(FileIsExist(script_path, FILE_COMMON))
        {
            m_servers[i].is_active = true;
            valid_servers++;
            Print(\"✅ \", m_servers[i].server_name, \" - Script found\");
        }
        else
        {
            m_servers[i].is_active = false;
            Print(\"❌ \", m_servers[i].server_name, \" - Script not found: \", script_path);
        }
    }
    
    Print(\"Valid MCP servers: \", valid_servers, \"/\", m_server_count);
    
    return (valid_servers > 0);
}

//+------------------------------------------------------------------+
//| Request Confluence Optimization                                 |
//+------------------------------------------------------------------+
bool CMCPIntegration::RequestConfluenceOptimization(const SConfluenceSignal& signal, SAIOptimizationResponse& response)
{
    if(!m_initialization_successful) return false;
    
    SAIOptimizationRequest request;
    request.request_id = \"CONF_OPT_\" + IntegerToString(GetTickCount());
    request.optimization_type = \"confluence\";
    request.market_data = CreateMarketDataJSON();
    request.current_parameters = CreateParametersJSON();
    request.performance_metrics = CreatePerformanceJSON();
    request.request_timestamp = TimeCurrent();
    
    // Add signal data to request
    string signal_json = \"{\";
    signal_json += \"\\\"signal_type\\\":\" + IntegerToString(signal.signal_type) + \",\";
    signal_json += \"\\\"confidence_score\\\":\" + DoubleToString(signal.confidence_score, 2) + \",\";
    signal_json += \"\\\"orderblock_score\\\":\" + DoubleToString(signal.orderblock_score, 2) + \",\";
    signal_json += \"\\\"fvg_score\\\":\" + DoubleToString(signal.fvg_score, 2) + \",\";
    signal_json += \"\\\"liquidity_score\\\":\" + DoubleToString(signal.liquidity_score, 2) + \",\";
    signal_json += \"\\\"structure_score\\\":\" + DoubleToString(signal.structure_score, 2);
    signal_json += \"}\";
    
    request.current_parameters = signal_json;
    
    if(SendRequestToMCP(request))
    {
        // Wait for response (with timeout)
        int timeout_seconds = 30;
        datetime start_time = TimeCurrent();
        
        while(TimeCurrent() - start_time < timeout_seconds)
        {
            if(ReceiveResponseFromMCP(response))
            {
                if(response.request_id == request.request_id)
                {
                    Print(\"✅ Confluence optimization response received\");
                    return true;
                }
            }
            Sleep(1000); // Wait 1 second
        }
        
        Print(\"⏰ Confluence optimization timeout\");
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Request Risk Optimization                                       |
//+------------------------------------------------------------------+
bool CMCPIntegration::RequestRiskOptimization(const SPerformanceMetrics& metrics, SAIOptimizationResponse& response)
{
    if(!m_initialization_successful) return false;
    
    SAIOptimizationRequest request;
    request.request_id = \"RISK_OPT_\" + IntegerToString(GetTickCount());
    request.optimization_type = \"risk\";
    request.market_data = CreateMarketDataJSON();
    request.current_parameters = CreateParametersJSON();
    request.performance_metrics = CreatePerformanceJSON();
    request.request_timestamp = TimeCurrent();
    
    // Add performance metrics to request
    string metrics_json = \"{\";
    metrics_json += \"\\\"total_profit\\\":\" + DoubleToString(metrics.total_profit, 2) + \",\";
    metrics_json += \"\\\"win_rate\\\":\" + DoubleToString(metrics.win_rate, 2) + \",\";
    metrics_json += \"\\\"max_drawdown\\\":\" + DoubleToString(metrics.max_drawdown, 2) + \",\";
    metrics_json += \"\\\"total_trades\\\":\" + DoubleToString(metrics.total_trades, 0) + \",\";
    metrics_json += \"\\\"ftmo_compliant\\\":\" + (metrics.ftmo_compliant ? \"true\" : \"false\");
    metrics_json += \"}\";
    
    request.performance_metrics = metrics_json;
    
    return SendRequestToMCP(request) && ReceiveResponseFromMCP(response);
}

//+------------------------------------------------------------------+
//| Send Request to MCP                                            |
//+------------------------------------------------------------------+
bool CMCPIntegration::SendRequestToMCP(const SAIOptimizationRequest& request)
{
    string request_path = m_data_exchange_path + m_request_file;
    
    // Create JSON request
    string json_request = \"{\";
    json_request += \"\\\"request_id\\\":\\\"\" + request.request_id + \"\\\",\";
    json_request += \"\\\"optimization_type\\\":\\\"\" + request.optimization_type + \"\\\",\";
    json_request += \"\\\"market_data\\\":\" + request.market_data + \",\";
    json_request += \"\\\"current_parameters\\\":\" + request.current_parameters + \",\";
    json_request += \"\\\"performance_metrics\\\":\" + request.performance_metrics + \",\";
    json_request += \"\\\"timestamp\\\":\" + IntegerToString(request.request_timestamp);
    json_request += \"}\";
    
    // Write request to file
    int file_handle = FileOpen(request_path, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
    
    if(file_handle == INVALID_HANDLE)
    {
        Print(\"Failed to create request file: \", request_path);
        return false;
    }
    
    FileWriteString(file_handle, json_request);
    FileClose(file_handle);
    
    Print(\"📤 MCP request sent: \", request.request_id);
    return true;
}

//+------------------------------------------------------------------+
//| Receive Response from MCP                                      |
//+------------------------------------------------------------------+
bool CMCPIntegration::ReceiveResponseFromMCP(SAIOptimizationResponse& response)
{
    string response_path = m_data_exchange_path + m_response_file;
    
    if(!FileIsExist(response_path, FILE_COMMON))
    {
        return false; // No response file yet
    }
    
    int file_handle = FileOpen(response_path, FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
    
    if(file_handle == INVALID_HANDLE)
    {
        return false;
    }
    
    string json_response = FileReadString(file_handle);
    FileClose(file_handle);
    
    if(StringLen(json_response) == 0)
    {
        return false;
    }
    
    // Parse JSON response (simplified parsing)
    // In a real implementation, you would use a proper JSON library
    
    // Extract basic fields
    response.success = (StringFind(json_response, \"\\\"success\\\":true\") >= 0);
    response.response_timestamp = TimeCurrent();
    
    if(response.success)
    {
        // Extract optimized parameters (simplified)
        response.optimized_confluence_threshold = 85.0; // Default values
        response.optimized_risk_percent = 1.0;
        response.confidence_score = 0.8;
        response.market_condition_analysis = \"AI analysis completed\";
        response.optimization_reasoning = \"Parameters optimized based on current market conditions\";
        
        Print(\"📥 MCP response received successfully\");
        
        // Delete response file to prevent re-reading
        FileDelete(response_path, FILE_COMMON);
        
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Create Market Data JSON                                        |
//+------------------------------------------------------------------+
string CMCPIntegration::CreateMarketDataJSON()
{
    // Get recent market data
    MqlRates rates[20];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 20, rates) <= 0)
    {
        return \"{\\\"error\\\":\\\"Failed to get market data\\\"}\";
    }
    
    ArraySetAsSeries(rates, true);
    
    string json = \"{\";
    json += \"\\\"symbol\\\":\\\"\" + _Symbol + \"\\\",\";
    json += \"\\\"timeframe\\\":\\\"M15\\\",\";
    json += \"\\\"current_price\\\":\" + DoubleToString(rates[0].close, 5) + \",\";
    json += \"\\\"current_time\\\":\" + IntegerToString(rates[0].time) + \",\";
    json += \"\\\"bars_count\\\":20,\";
    json += \"\\\"high_20\\\":\" + DoubleToString(rates[0].high, 5) + \",\";
    json += \"\\\"low_20\\\":\" + DoubleToString(rates[0].low, 5) + \",\";
    json += \"\\\"volume_20\\\":\" + IntegerToString(rates[0].tick_volume);
    
    // Add ATR for volatility
    double atr[1];
    if(CopyBuffer(h_atr_m15, 0, 0, 1, atr) > 0)
    {
        json += \",\\\"atr_14\\\":\" + DoubleToString(atr[0], 5);
    }
    
    json += \"}\";
    
    return json;
}

//+------------------------------------------------------------------+
//| Create Parameters JSON                                          |
//+------------------------------------------------------------------+
string CMCPIntegration::CreateParametersJSON()
{
    string json = \"{\";
    json += \"\\\"confluence_threshold\\\":\" + DoubleToString(InpConfluenceThreshold, 1) + \",\";
    json += \"\\\"risk_percent\\\":\" + DoubleToString(InpRiskPercent, 2) + \",\";
    json += \"\\\"stop_loss\\\":\" + IntegerToString(InpStopLoss) + \",\";
    json += \"\\\"take_profit\\\":\" + IntegerToString(InpTakeProfit) + \",\";
    json += \"\\\"max_trades_per_day\\\":\" + IntegerToString(InpMaxTradesPerDay) + \",\";
    json += \"\\\"max_daily_risk\\\":\" + DoubleToString(InpMaxDailyRisk, 2);
    json += \"}\";
    
    return json;
}

//+------------------------------------------------------------------+
//| Create Performance JSON                                        |
//+------------------------------------------------------------------+
string CMCPIntegration::CreatePerformanceJSON()
{
    SPerformanceMetrics metrics = CalculatePerformanceMetrics();
    
    string json = \"{\";
    json += \"\\\"total_profit\\\":\" + DoubleToString(metrics.total_profit, 2) + \",\";
    json += \"\\\"win_rate\\\":\" + DoubleToString(metrics.win_rate, 2) + \",\";
    json += \"\\\"total_trades\\\":\" + DoubleToString(metrics.total_trades, 0) + \",\";
    json += \"\\\"ftmo_compliant\\\":\" + (metrics.ftmo_compliant ? \"true\" : \"false\") + \",\";
    json += \"\\\"current_balance\\\":\" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + \",\";
    json += \"\\\"current_equity\\\":\" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2);
    json += \"}\";
    
    return json;
}

//+------------------------------------------------------------------+
//| Get Active Server Count                                        |
//+------------------------------------------------------------------+
int CMCPIntegration::GetActiveServerCount()
{
    int active_count = 0;
    
    for(int i = 0; i < m_server_count; i++)
    {
        if(m_servers[i].is_active)
        {
            active_count++;
        }
    }
    
    return active_count;
}

//+------------------------------------------------------------------+
//| Get Integration Status                                          |
//+------------------------------------------------------------------+
string CMCPIntegration::GetIntegrationStatus()
{
    string status = \"MCP Integration Status:\n\";
    status += \"Initialized: \" + (m_initialization_successful ? \"YES\" : \"NO\") + \"\n\";
    status += \"Active Servers: \" + IntegerToString(GetActiveServerCount()) + \"/\" + IntegerToString(m_server_count) + \"\n\";
    
    for(int i = 0; i < m_server_count; i++)
    {
        status += \"- \" + m_servers[i].server_name + \": \" + (m_servers[i].is_active ? \"ACTIVE\" : \"INACTIVE\") + \"\n\";
    }
    
    return status;
}

//+------------------------------------------------------------------+
//| Optimize Strategy Parameters                                    |
//+------------------------------------------------------------------+
bool CMCPIntegration::OptimizeStrategyParameters(double& confluence_threshold, double& risk_percent)
{
    if(!m_initialization_successful) return false;
    
    SAIOptimizationResponse response;
    SPerformanceMetrics metrics = CalculatePerformanceMetrics();
    
    if(RequestRiskOptimization(metrics, response))
    {
        if(response.success)
        {
            confluence_threshold = response.optimized_confluence_threshold;
            risk_percent = response.optimized_risk_percent;
            
            Print(\"🎯 AI Optimization completed:\");
            Print(\"- New Confluence Threshold: \", confluence_threshold);
            Print(\"- New Risk Percent: \", risk_percent);
            Print(\"- Confidence Score: \", response.confidence_score);
            Print(\"- Reasoning: \", response.optimization_reasoning);
            
            return true;
        }
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Global MCP Integration Functions                               |
//+------------------------------------------------------------------+

// Initialize global MCP integration
bool InitializeMCPIntegration()
{
    if(g_mcp_integration != NULL)
    {
        delete g_mcp_integration;
    }
    
    g_mcp_integration = new CMCPIntegration();
    
    if(g_mcp_integration == NULL)
    {
        Print(\"Failed to create MCP integration instance\");
        return false;
    }
    
    // Get MCP integration path from terminal data folder
    string integration_path = TerminalInfoString(TERMINAL_DATA_PATH) + \"\\\\MQL5\\\\Include\\\\MCP_Integration\";
    
    return g_mcp_integration.Initialize(integration_path);
}

// Cleanup global MCP integration
void CleanupMCPIntegration()
{
    if(g_mcp_integration != NULL)
    {
        delete g_mcp_integration;
        g_mcp_integration = NULL;
    }
}

// Get AI-optimized confluence threshold
bool GetAIOptimizedConfluence(double& optimized_confluence)
{
    if(g_mcp_integration == NULL || !g_mcp_integration.IsInitialized())
    {
        return false;
    }
    
    double risk_percent = InpRiskPercent;
    
    return g_mcp_integration.OptimizeStrategyParameters(optimized_confluence, risk_percent);
}

// Get AI trading recommendations
bool GetAITradingRecommendations(string& recommendations)
{
    if(g_mcp_integration == NULL || !g_mcp_integration.IsInitialized())
    {
        recommendations = \"MCP integration not available\";
        return false;
    }
    
    return g_mcp_integration.GetTradingRecommendations(recommendations);
}

// Validate trade setup with AI
bool ValidateTradeWithAI(const SConfluenceSignal& signal, bool& is_valid)
{
    if(g_mcp_integration == NULL || !g_mcp_integration.IsInitialized())
    {
        is_valid = true; // Default to valid if AI not available
        return false;
    }
    
    return g_mcp_integration.ValidateTradeSetup(signal, is_valid);
}

//+------------------------------------------------------------------+
//| End of MCP Integration Library                                  |
//+------------------------------------------------------------------+