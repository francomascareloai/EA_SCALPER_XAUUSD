//+------------------------------------------------------------------+
//|                                           FTMO_RiskManager.mqh |
//|                           Autonomous Expert Advisor for XAUUSD Trading |
//|                                     Risk Management Component |
//+------------------------------------------------------------------+
#property copyright "Developed by Autonomous AI Agent - FTMO Elite Trading System"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/AccountInfo.mqh>
#include <Trade/PositionInfo.mqh>
#include <Trade/SymbolInfo.mqh>
#include "Definitions.mqh"

// === FTMO RISK MANAGER CLASS ===
class CFTMORiskManager
{
private:
    SFTMOCompliance     m_compliance;
    
    // External references (pointers to global objects)
    CTrade*             m_trade;
    CAccountInfo*       m_account;
    CPositionInfo*      m_position;
    CSymbolInfo*        m_symbol;
    
public:
    CFTMORiskManager();
    ~CFTMORiskManager();
    
    // Initialization
    void Init(CTrade* trade_obj, CAccountInfo* account_obj, CPositionInfo* position_obj, CSymbolInfo* symbol_obj);
    void ResetDailyFTMOTracking();
    
    // Main Compliance Checks
    void CheckFTMOCompliance();
    bool ValidateFTMOTradeCompliance(const SConfluenceSignal& signal, double lot_size);
    bool IsTradingAllowed();
    
    // Specific Checks
    bool CheckDailyLossLimit();
    bool CheckMaxDrawdownLimit();
    bool CheckTotalOpenRisk();
    bool CheckCorrelationRisk();
    bool IsWeekendGapRisk();
    
    // Risk Calculations
    double CalculateTotalOpenRisk();
    double CalculatePositionRisk(ulong ticket);
    double CalculatePotentialLoss(const SConfluenceSignal& signal, double lot_size);
    
    // Emergency Handling
    void HaltTradingEmergency(string reason);
    void CloseAllPositionsEmergency(string reason);
    
    // Reporting
    void LogFTMOViolation(string reason);
    void TrackFTMOTradeExecution(ulong ticket, double lot_size, double risk);
    string GetFTMOComplianceReport();
    
    // Getters/Setters
    void SetDailyLossLimit(double limit) { m_compliance.daily_loss_limit = limit; }
    void SetMaxDrawdownLimit(double limit) { m_compliance.max_drawdown_limit = limit; }
    void SetMaxTradesPerDay(int max_trades) { m_compliance.max_trades_per_day = max_trades; }
    void SetMaxRiskPerTrade(double risk_percent) { m_compliance.max_risk_per_trade = risk_percent; }
    bool IsCompliant() { return m_compliance.is_compliant; }
    bool IsHalted() { return m_compliance.trading_halted; }
};

//+------------------------------------------------------------------+
//| FTMO Risk Manager Implementation                                 |
//+------------------------------------------------------------------+

// Constructor
CFTMORiskManager::CFTMORiskManager()
{
    m_trade = NULL;
    m_account = NULL;
    m_position = NULL;
    m_symbol = NULL;
    
    // Initialize default values
    m_compliance.daily_loss_limit = 5.0;
    m_compliance.max_drawdown_limit = 10.0;
    m_compliance.max_trades_per_day = 5;
    m_compliance.max_risk_per_trade = 1.0;
    m_compliance.safety_buffer = 0.2; // 20% buffer
    
    m_compliance.is_compliant = true;
    m_compliance.trading_halted = false;
    m_compliance.violation_count = 0;
}

// Destructor
CFTMORiskManager::~CFTMORiskManager()
{
}

// Initialization
void CFTMORiskManager::Init(CTrade* trade_obj, CAccountInfo* account_obj, CPositionInfo* position_obj, CSymbolInfo* symbol_obj)
{
    m_trade = trade_obj;
    m_account = account_obj;
    m_position = position_obj;
    m_symbol = symbol_obj;
    
    ResetDailyFTMOTracking();
}

// Reset Daily Tracking
void CFTMORiskManager::ResetDailyFTMOTracking()
{
    if(m_account == NULL) return;
    
    m_compliance.daily_starting_balance = m_account.Balance();
    m_compliance.daily_loss_current = 0.0;
    m_compliance.trades_today_count = 0;
    m_compliance.violation_count = 0;
    m_compliance.daily_reset_time = TimeCurrent();
    
    // Reset daily flags but keep overall compliance status
    m_compliance.daily_limit_breached = false;
    
    Print("📅 FTMO Daily Tracking Reset | Starting Balance: ", m_compliance.daily_starting_balance);
}

// Main Compliance Check
void CFTMORiskManager::CheckFTMOCompliance()
{
    if(m_account == NULL) return;
    
    datetime current_time = TimeCurrent();
    
    // Check if new day (reset daily stats)
    MqlDateTime dt;
    TimeToStruct(current_time, dt);
    static int last_day = -1;
    
    if(last_day != -1 && last_day != dt.day)
    {
        ResetDailyFTMOTracking();
    }
    last_day = dt.day;
    
    double current_equity = m_account.Equity();
    double current_balance = m_account.Balance();
    
    // Update daily loss calculation
    m_compliance.daily_loss_current = 
        ((m_compliance.daily_starting_balance - current_equity) / m_compliance.daily_starting_balance) * 100.0;
    
    // Update high water mark
    if(current_balance > m_compliance.account_high_water_mark)
    {
        m_compliance.account_high_water_mark = current_balance;
    }
    
    // Update current drawdown
    m_compliance.max_drawdown_current = 
        ((m_compliance.account_high_water_mark - current_equity) / m_compliance.account_high_water_mark) * 100.0;
    
    // Update total open risk
    m_compliance.total_open_risk = CalculateTotalOpenRisk();
    
    m_compliance.last_check_time = current_time;
    
    // Perform Checks
    if(!CheckDailyLossLimit()) HaltTradingEmergency("Daily Loss Limit Breached");
    if(!CheckMaxDrawdownLimit()) HaltTradingEmergency("Max Drawdown Limit Breached");
    if(!CheckTotalOpenRisk()) LogFTMOViolation("Total Open Risk Warning");
}

// Check Daily Loss Limit
bool CFTMORiskManager::CheckDailyLossLimit()
{
    double effective_limit = m_compliance.daily_loss_limit * (1.0 - m_compliance.safety_buffer);
    
    if(m_compliance.daily_loss_current >= effective_limit)
    {
        LogFTMOViolation("Daily loss limit approached: " + 
                        DoubleToString(m_compliance.daily_loss_current, 2) + "% >= " + 
                        DoubleToString(effective_limit, 2) + "%");
        return false;
    }
    
    return true;
}

// Check Maximum Drawdown Limit
bool CFTMORiskManager::CheckMaxDrawdownLimit()
{
    double effective_limit = m_compliance.max_drawdown_limit * (1.0 - m_compliance.safety_buffer);
    
    if(m_compliance.max_drawdown_current >= effective_limit)
    {
        LogFTMOViolation("Maximum drawdown limit approached: " + 
                        DoubleToString(m_compliance.max_drawdown_current, 2) + "% >= " + 
                        DoubleToString(effective_limit, 2) + "%");
        return false;
    }
    
    return true;
}

// Check Total Open Risk
bool CFTMORiskManager::CheckTotalOpenRisk()
{
    if(m_account == NULL) return true;
    
    double max_total_risk = m_account.Balance() * (m_compliance.max_risk_per_trade / 100.0) * 2.0; // Max 2 positions worth of risk
    
    if(m_compliance.total_open_risk >= max_total_risk)
    {
        LogFTMOViolation("Total open risk too high: " + 
                        DoubleToString(m_compliance.total_open_risk, 2) + " >= " + 
                        DoubleToString(max_total_risk, 2));
        return false;
    }
    
    return true;
}

// Check Correlation Risk
bool CFTMORiskManager::CheckCorrelationRisk()
{
    if(m_position == NULL || m_symbol == NULL) return true;
    
    // For XAUUSD EA, this would check if multiple XAUUSD positions exist
    int xauusd_positions = 0;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(m_position.SelectByIndex(i))
        {
            if(m_position.Symbol() == m_symbol.Name())
            {
                xauusd_positions++;
            }
        }
    }
    
    // Allow maximum 2 XAUUSD positions
    if(xauusd_positions >= 2)
    {
        LogFTMOViolation("Too many correlated positions: " + IntegerToString(xauusd_positions));
        return false;
    }
    
    return true;
}

// Calculate Total Open Risk
double CFTMORiskManager::CalculateTotalOpenRisk()
{
    if(m_position == NULL) return 0.0;
    
    double total_risk = 0.0;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(m_position.SelectByIndex(i))
        {
            // Assuming magic number check is handled by caller or we check all positions
            double position_risk = CalculatePositionRisk(m_position.Ticket());
            total_risk += position_risk;
        }
    }
    
    return total_risk;
}

// Calculate Position Risk
double CFTMORiskManager::CalculatePositionRisk(ulong ticket)
{
    if(m_position == NULL || m_symbol == NULL) return 0.0;
    
    if(!m_position.SelectByTicket(ticket)) return 0.0;
    
    double entry_price = m_position.PriceOpen();
    double sl_price = m_position.StopLoss();
    double volume = m_position.Volume();
    
    if(sl_price == 0.0) return 0.0; // No SL set
    
    double sl_distance = MathAbs(entry_price - sl_price);
    double tick_value = m_symbol.TickValue();
    
    return (sl_distance / _Point) * tick_value * volume;
}

// Validate Trade Before Execution
bool CFTMORiskManager::ValidateFTMOTradeCompliance(const SConfluenceSignal& signal, double lot_size)
{
    if(m_account == NULL || m_symbol == NULL) return false;
    
    // 1. Check if trading is halted
    if(m_compliance.trading_halted)
    {
        return false;
    }
    
    // 2. Check daily trade limit
    if(m_compliance.trades_today_count >= m_compliance.max_trades_per_day)
    {
        return false;
    }
    
    // 3. Calculate potential trade risk
    double potential_risk = CalculatePotentialLoss(signal, lot_size);
    double risk_percentage = (potential_risk / m_account.Balance()) * 100.0;
    
    // 4. Check individual trade risk
    if(risk_percentage > m_compliance.max_risk_per_trade)
    {
        LogFTMOViolation("Trade risk too high: " + DoubleToString(risk_percentage, 2) + "% > " + 
                        DoubleToString(m_compliance.max_risk_per_trade, 2) + "%");
        return false;
    }
    
    // 5. Check combined risk after this trade
    double total_risk_after = m_compliance.total_open_risk + potential_risk;
    double max_combined_risk = m_account.Balance() * (m_compliance.max_risk_per_trade / 100.0) * 2.0;
    
    if(total_risk_after > max_combined_risk)
    {
        LogFTMOViolation("Combined risk would be too high: " + DoubleToString(total_risk_after, 2));
        return false;
    }
    
    // 6. Check if trade would cause daily loss limit approach
    double current_equity = m_account.Equity();
    double potential_daily_loss = ((m_compliance.daily_starting_balance - (current_equity - potential_risk)) / 
                                   m_compliance.daily_starting_balance) * 100.0;
    
    double safe_daily_limit = m_compliance.daily_loss_limit * (1.0 - m_compliance.safety_buffer);
    
    if(potential_daily_loss > safe_daily_limit)
    {
        LogFTMOViolation("Trade could cause daily limit breach: " + DoubleToString(potential_daily_loss, 2) + "%");
        return false;
    }
    
    // 7. Check lot size limits
    double max_lot = m_symbol.LotsMax();
    double conservative_max_lot = MathMin(max_lot, 10.0); 
    
    if(lot_size > conservative_max_lot)
    {
        LogFTMOViolation("Lot size too large: " + DoubleToString(lot_size, 2) + " > " + 
                        DoubleToString(conservative_max_lot, 2));
        return false;
    }
    
    return true;
}

// Calculate Potential Loss
double CFTMORiskManager::CalculatePotentialLoss(const SConfluenceSignal& signal, double lot_size)
{
    if(m_symbol == NULL) return 0.0;
    
    double entry_price = signal.entry_price;
    double sl_price = signal.stop_loss;
    
    double sl_distance = MathAbs(entry_price - sl_price);
    double tick_value = m_symbol.TickValue();
    
    return (sl_distance / _Point) * tick_value * lot_size;
}

// Log FTMO Violation
void CFTMORiskManager::LogFTMOViolation(string reason)
{
    if(m_account == NULL) return;
    
    m_compliance.violation_count++;
    m_compliance.last_violation_reason = reason;
    m_compliance.last_violation_time = TimeCurrent();
    
    Print("🚨 FTMO VIOLATION #", m_compliance.violation_count, ": ", reason);
    
    // Log to file for audit trail
    int file_handle = FileOpen("FTMO_Violations_" + IntegerToString(m_account.Login()) + ".log", 
                               FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_APPEND, "\t");
    if(file_handle != INVALID_HANDLE)
    {
        FileWrite(file_handle, TimeToString(TimeCurrent()), reason, 
                  DoubleToString(m_account.Balance(), 2), 
                  DoubleToString(m_account.Equity(), 2));
        FileClose(file_handle);
    }
}

// Emergency Trading Halt
void CFTMORiskManager::HaltTradingEmergency(string reason)
{
    if(m_account == NULL) return;
    
    m_compliance.trading_halted = true;
    m_compliance.is_compliant = false;
    
    Print("🛑 EMERGENCY TRADING HALT: ", reason);
    Print("🛑 All trading suspended to protect FTMO account");
    
    // Close all open positions immediately
    CloseAllPositionsEmergency(reason);
    
    // Log emergency halt
    int file_handle = FileOpen("FTMO_Emergency_Halts_" + IntegerToString(m_account.Login()) + ".log", 
                               FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_APPEND, "\t");
    if(file_handle != INVALID_HANDLE)
    {
        FileWrite(file_handle, TimeToString(TimeCurrent()), "EMERGENCY HALT", reason, 
                  DoubleToString(m_account.Balance(), 2), 
                  DoubleToString(m_account.Equity(), 2));
        FileClose(file_handle);
    }
    
    // Send alert if possible
    Alert("FTMO EMERGENCY HALT: ", reason);
}

// Close All Positions Emergency
void CFTMORiskManager::CloseAllPositionsEmergency(string reason)
{
    if(m_position == NULL || m_trade == NULL) return;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(m_position.SelectByIndex(i))
        {
            // Assuming we close all positions for this account/magic
            ulong ticket = m_position.Ticket();
            if(m_trade.PositionClose(ticket))
            {
                Print("🛑 Emergency close: Ticket ", ticket, " - Reason: ", reason);
            }
        }
    }
}

// Track Trade Execution
void CFTMORiskManager::TrackFTMOTradeExecution(ulong ticket, double lot_size, double risk)
{
    m_compliance.trades_today_count++;
    
    Print("📊 FTMO Trade Executed: Ticket ", ticket, 
          " | Trades Today: ", m_compliance.trades_today_count, "/", m_compliance.max_trades_per_day,
          " | Risk: ", DoubleToString(risk, 2));
}

// Get FTMO Compliance Report
string CFTMORiskManager::GetFTMOComplianceReport()
{
    string report = "\n=== FTMO COMPLIANCE STATUS ===";
    report += "\nOverall Status: " + (m_compliance.is_compliant ? "✅ COMPLIANT" : "❌ NON-COMPLIANT");
    report += "\nTrading Status: " + (m_compliance.trading_halted ? "🛑 HALTED" : "✅ ACTIVE");
    
    report += "\n\n--- Daily Limits ---";
    report += "\nDaily Loss: " + DoubleToString(m_compliance.daily_loss_current, 2) + "% / " + 
              DoubleToString(m_compliance.daily_loss_limit, 2) + "%";
    report += "\nTrades Today: " + IntegerToString(m_compliance.trades_today_count) + " / " + 
              IntegerToString(m_compliance.max_trades_per_day);
    
    report += "\n\n--- Overall Limits ---";
    report += "\nMax Drawdown: " + DoubleToString(m_compliance.max_drawdown_current, 2) + "% / " + 
              DoubleToString(m_compliance.max_drawdown_limit, 2) + "%";
    report += "\nTotal Open Risk: $" + DoubleToString(m_compliance.total_open_risk, 2);
    
    if(m_compliance.violation_count > 0)
    {
        report += "\n\n--- Violations ---";
        report += "\nViolations Today: " + IntegerToString(m_compliance.violation_count);
        report += "\nLast Violation: " + m_compliance.last_violation_reason;
    }
    
    report += "\n========================\n";
    
    return report;
}

// Check Weekend Gap Risk
bool CFTMORiskManager::IsWeekendGapRisk()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // Stop trading on Friday after 20:00 GMT
    if(dt.day_of_week == FRIDAY && dt.hour >= 20)
    {
        return true;
    }
    
    // Don't trade on Monday before 08:00 GMT (wait for gap to settle)
    if(dt.day_of_week == MONDAY && dt.hour < 8)
    {
        return true;
    }
    
    return false;
}

// Is Trading Allowed
bool CFTMORiskManager::IsTradingAllowed()
{
    if(m_compliance.trading_halted) return false;
    if(m_compliance.daily_limit_breached) return false;
    
    return true;
}
