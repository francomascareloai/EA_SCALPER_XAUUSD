//+------------------------------------------------------------------+
//|                                             FTMO_RiskManager.mqh |
//|                                                           Franco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Franco"
#property link      "https://www.mql5.com"
#property strict

#include "../Core/Definitions.mqh"

//+------------------------------------------------------------------+
//| Class: CFTMO_RiskManager                                         |
//| Purpose: Manages risk according to Prop Firm (FTMO) rules.       |
//+------------------------------------------------------------------+
class CFTMO_RiskManager
{
private:
   //--- Limits
   double            m_risk_per_trade_percent;
   double            m_max_daily_loss_percent;
   double            m_max_total_loss_percent;
   double            m_soft_stop_percent;
   int               m_max_trades_per_day;

   //--- State
   double            m_initial_equity;
   double            m_daily_start_equity;
   double            m_current_daily_loss;
   double            m_current_total_loss;
   int               m_trades_today;
   bool              m_trading_halted;
   datetime          m_last_day_check;

public:
                     CFTMO_RiskManager();
                    ~CFTMO_RiskManager();

   //--- Initialization
   bool              Init(double risk_per_trade, double max_daily_loss, double max_total_loss, int max_trades, double soft_stop);

   //--- Core Logic
   void              OnTick();
   bool              CanOpenNewTrade();
   double            CalculateLotSize(double sl_points);
   void              OnTradeExecuted();
   
   //--- Getters
   bool              IsTradingHalted() const { return m_trading_halted; }
   double            GetCurrentDailyLoss() const { return m_current_daily_loss; }

private:
   void              CheckNewDay();
   void              CheckDrawdownLimits();
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CFTMO_RiskManager::CFTMO_RiskManager() :
   m_risk_per_trade_percent(0.5),
   m_max_daily_loss_percent(5.0),
   m_max_total_loss_percent(10.0),
   m_soft_stop_percent(4.0),
   m_max_trades_per_day(20),
   m_initial_equity(0.0),
   m_daily_start_equity(0.0),
   m_current_daily_loss(0.0),
   m_current_total_loss(0.0),
   m_trades_today(0),
   m_trading_halted(false),
   m_last_day_check(0)
{
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CFTMO_RiskManager::~CFTMO_RiskManager()
{
}

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
bool CFTMO_RiskManager::Init(double risk_per_trade, double max_daily_loss, double max_total_loss, int max_trades, double soft_stop)
{
   m_risk_per_trade_percent = risk_per_trade;
   m_max_daily_loss_percent = max_daily_loss;
   m_max_total_loss_percent = max_total_loss;
   m_max_trades_per_day = max_trades;
   m_soft_stop_percent = soft_stop;

   m_initial_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   m_daily_start_equity = m_initial_equity;
   m_last_day_check = TimeCurrent();

   return true;
}

//+------------------------------------------------------------------+
//| OnTick: Monitor Drawdown                                         |
//+------------------------------------------------------------------+
void CFTMO_RiskManager::OnTick()
{
   CheckNewDay();
   CheckDrawdownLimits();
}

//+------------------------------------------------------------------+
//| Check for New Day (Reset Daily Stats)                            |
//+------------------------------------------------------------------+
void CFTMO_RiskManager::CheckNewDay()
{
   datetime current_time = TimeCurrent();
   MqlDateTime dt_struct;
   TimeToStruct(current_time, dt_struct);
   
   MqlDateTime last_dt_struct;
   TimeToStruct(m_last_day_check, last_dt_struct);

   if(dt_struct.day != last_dt_struct.day)
   {
      // New Day Detected
      m_daily_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      m_trades_today = 0;
      m_current_daily_loss = 0.0;
      
      // Reset Halt ONLY if not in Total Loss Breach
      if(m_current_total_loss < m_max_total_loss_percent)
      {
         m_trading_halted = false;
      }
      
      m_last_day_check = current_time;
      Print("RiskManager: New Day Reset. Daily Start Equity: ", m_daily_start_equity);
   }
}

//+------------------------------------------------------------------+
//| Check Drawdown Limits                                            |
//+------------------------------------------------------------------+
void CFTMO_RiskManager::CheckDrawdownLimits()
{
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // 1. Calculate Daily Drawdown
   // FTMO: Daily Loss is calculated based on the starting equity of the day
   double daily_dd = 0.0;
   if(current_equity < m_daily_start_equity)
   {
      daily_dd = ((m_daily_start_equity - current_equity) / m_daily_start_equity) * 100.0;
   }
   m_current_daily_loss = daily_dd;

   // 2. Calculate Total Drawdown
   // FTMO: Total Loss is usually based on Initial Balance (or High Water Mark depending on specific rules)
   // Assuming Initial Balance for simplicity here, but standard is often relative to initial deposit.
   double total_dd = 0.0;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE); // Or Initial Balance if static
   // Let's use m_initial_equity as the baseline for Total Loss
   if(current_equity < m_initial_equity)
   {
      total_dd = ((m_initial_equity - current_equity) / m_initial_equity) * 100.0;
   }
   m_current_total_loss = total_dd;

   // 3. Check Limits
   if(m_current_daily_loss >= m_max_daily_loss_percent)
   {
      if(!m_trading_halted) Print("CRITICAL: Max Daily Loss Breached! Halting Trading.");
      m_trading_halted = true;
   }
   
   if(m_current_total_loss >= m_max_total_loss_percent)
   {
      if(!m_trading_halted) Print("CRITICAL: Max Total Loss Breached! Halting Trading.");
      m_trading_halted = true;
   }

   // 4. Soft Stop Check
   if(m_current_daily_loss >= m_soft_stop_percent)
   {
      // We don't halt, but we might want to reduce risk or stop opening NEW trades
      // For now, let's treat Soft Stop as a "Stop New Trades" trigger
      if(!m_trading_halted) Print("WARNING: Soft Stop Level Reached. Pausing New Trades.");
      m_trading_halted = true; 
   }
}

//+------------------------------------------------------------------+
//| Can Open New Trade?                                              |
//+------------------------------------------------------------------+
bool CFTMO_RiskManager::CanOpenNewTrade()
{
   if(m_trading_halted) return false;
   if(m_trades_today >= m_max_trades_per_day) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate Lot Size based on Risk %                               |
//+------------------------------------------------------------------+
double CFTMO_RiskManager::CalculateLotSize(double sl_points)
{
   if(sl_points <= 0) return 0.0;

   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (m_risk_per_trade_percent / 100.0);
   
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(tick_size == 0 || tick_value == 0) return 0.0;

   // Lot = Risk / (SL_Points * TickValue_Per_Point)
   // SL is in points. TickValue is per Tick.
   // Value per point = TickValue * (Point / TickSize)
   
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double value_per_point = tick_value * (point / tick_size);
   
   double lot_size = risk_amount / (sl_points * value_per_point);
   
   // Normalize Lot Size
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot_size = MathFloor(lot_size / step_lot) * step_lot;
   
   if(lot_size < min_lot) lot_size = min_lot; // Or 0 if strict
   if(lot_size > max_lot) lot_size = max_lot;
   
   return lot_size;
}

//+------------------------------------------------------------------+
//| Update Stats after Trade Execution                               |
//+------------------------------------------------------------------+
void CFTMO_RiskManager::OnTradeExecuted()
{
   m_trades_today++;
}
