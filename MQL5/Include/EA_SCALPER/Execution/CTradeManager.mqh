//+------------------------------------------------------------------+
//|                                               CTradeManager.mqh |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|                        Trade Management State Machine for FTMO    |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property strict

#include <Trade\Trade.mqh>
#include "../Core/Definitions.mqh"

// === TRADE STATE ENUMERATION ===
enum ENUM_TRADE_STATE
{
   TRADE_STATE_IDLE = 0,           // No active trade
   TRADE_STATE_ENTRY_PENDING,      // Waiting for entry confirmation
   TRADE_STATE_POSITION_OPEN,      // Position is open
   TRADE_STATE_BREAKEVEN,          // At breakeven (SL moved to entry)
   TRADE_STATE_PARTIAL_TP,         // Partial TP taken
   TRADE_STATE_TRAILING,           // Trailing stop active
   TRADE_STATE_CLOSING,            // In process of closing
   TRADE_STATE_CLOSED              // Trade completed
};

// === TRADE MANAGEMENT MODE ===
enum ENUM_MANAGEMENT_MODE
{
   MGMT_FIXED_RR,                  // Fixed R:R ratio
   MGMT_TRAILING_ATR,              // ATR-based trailing
   MGMT_PARTIAL_TP,                // Multiple TPs with partial close
   MGMT_DYNAMIC                    // Dynamic based on market conditions
};

// === TRADE INFO STRUCTURE ===
struct STradeInfo
{
   ulong                ticket;              // Position ticket
   ENUM_TRADE_STATE     state;               // Current state
   ENUM_POSITION_TYPE   type;                // BUY or SELL
   double               entry_price;         // Entry price
   double               initial_sl;          // Initial stop loss
   double               current_sl;          // Current stop loss
   double               tp1;                 // Take profit 1 (partial)
   double               tp2;                 // Take profit 2 (partial)
   double               tp3;                 // Take profit 3 (final)
   double               initial_lots;        // Initial position size
   double               current_lots;        // Current position size
   double               highest_price;       // Highest price since entry (for trailing)
   double               lowest_price;        // Lowest price since entry (for trailing)
   datetime             entry_time;          // Entry timestamp
   double               entry_score;         // Confluence score at entry
   double               realized_pnl;        // Realized P/L from partials
   int                  partials_taken;      // Number of partial TPs taken
   string               entry_reason;        // Entry reason for logging
};

// === TRADE MANAGER CLASS ===
class CTradeManager
{
private:
   // Trade object
   CTrade               m_trade;
   
   // Active trade info
   STradeInfo           m_active_trade;
   bool                 m_has_active_trade;
   
   // Settings
   int                  m_magic_number;
   int                  m_slippage;
   string               m_symbol;
   ENUM_MANAGEMENT_MODE m_mgmt_mode;
   
   // Management parameters
   double               m_breakeven_trigger;     // R multiple to move to BE (e.g., 1.0 = 1R)
   double               m_partial1_trigger;      // R multiple for first partial (e.g., 1.5R)
   double               m_partial1_percent;      // Percent to close at TP1 (e.g., 0.33)
   double               m_partial2_trigger;      // R multiple for second partial
   double               m_partial2_percent;      // Percent to close at TP2
   double               m_trailing_start;        // R multiple to start trailing
   double               m_trailing_step;         // ATR multiple for trailing step
   
   // ATR for dynamic calculations
   int                  m_atr_handle;
   double               m_current_atr;
   
   // Internal methods
   void                 UpdateATR();
   double               GetCurrentRMultiple();
   double               CalculateTrailingStop();
   bool                 MoveToBreakeven();
   bool                 TakePartialProfit(double percent, int partial_num);
   bool                 UpdateTrailingStop();
   void                 LogStateChange(ENUM_TRADE_STATE old_state, ENUM_TRADE_STATE new_state);
   
public:
   CTradeManager();
   ~CTradeManager();
   
   // Initialization
   bool                 Init(int magic, int slippage, string symbol = NULL);
   void                 SetManagementMode(ENUM_MANAGEMENT_MODE mode);
   void                 SetBreakevenTrigger(double r_multiple) { m_breakeven_trigger = r_multiple; }
   void                 SetPartialTP1(double r_mult, double pct) { m_partial1_trigger = r_mult; m_partial1_percent = pct; }
   void                 SetPartialTP2(double r_mult, double pct) { m_partial2_trigger = r_mult; m_partial2_percent = pct; }
   void                 SetTrailingParams(double start_r, double step_atr) { m_trailing_start = start_r; m_trailing_step = step_atr; }
   
   // Quick config: 40/30/30 split at specified R levels
   void                 ConfigurePartials(double tp1_r, double tp2_r, double tp1_pct = 0.40, double tp2_pct = 0.50)
   {
      m_partial1_trigger = tp1_r;
      m_partial1_percent = tp1_pct;
      m_partial2_trigger = tp2_r;
      m_partial2_percent = tp2_pct;
      m_trailing_start = tp2_r;
   }
   
   // Trade operations
   bool                 OpenTrade(ENUM_SIGNAL_TYPE signal, double lots, double sl, double tp, 
                                  double score, string reason);
   bool                 OpenTradeWithTPs(ENUM_SIGNAL_TYPE signal, double lots, double sl,
                                         double tp1, double tp2, double tp3,
                                         double score, string reason);
   bool                 CloseTrade(string reason = "Manual close");
   bool                 ClosePartial(double percent, string reason = "Partial TP");
   
   // State machine update (call on every tick)
   void                 OnTick();
   
   // Accessors
   bool                 HasActiveTrade() { return m_has_active_trade; }
   ENUM_TRADE_STATE     GetState() { return m_active_trade.state; }
   STradeInfo           GetTradeInfo() { return m_active_trade; }
   double               GetUnrealizedPnL();
   double               GetTotalPnL();
   
   // Position sync (after EA restart)
   bool                 SyncWithExistingPosition();
};

// === IMPLEMENTATION ===

CTradeManager::CTradeManager()
{
   m_has_active_trade = false;
   m_magic_number = 0;
   m_slippage = 50;
   m_symbol = _Symbol;
   m_mgmt_mode = MGMT_PARTIAL_TP;
   m_atr_handle = INVALID_HANDLE;
   m_current_atr = 0;
   
   // Default management parameters (FTMO optimized)
   // Strategy: 40% at TP1, 30% at TP2, trail remaining 30%
   m_breakeven_trigger = 1.0;      // Move to BE at 1R profit
   m_partial1_trigger = 1.5;       // First partial at 1.5R
   m_partial1_percent = 0.40;      // Close 40% at TP1
   m_partial2_trigger = 2.5;       // Second partial at 2.5R
   m_partial2_percent = 0.50;      // Close 50% of remaining (= 30% of original)
   m_trailing_start = 2.5;         // Start trailing after TP2
   m_trailing_step = 0.5;          // Trail by 0.5 ATR
   
   ZeroMemory(m_active_trade);
}

CTradeManager::~CTradeManager()
{
   if(m_atr_handle != INVALID_HANDLE)
      IndicatorRelease(m_atr_handle);
}

bool CTradeManager::Init(int magic, int slippage, string symbol)
{
   m_magic_number = magic;
   m_slippage = slippage;
   m_symbol = (symbol == NULL) ? _Symbol : symbol;
   
   m_trade.SetExpertMagicNumber(magic);
   m_trade.SetDeviationInPoints(slippage);
   m_trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Initialize ATR
   m_atr_handle = iATR(m_symbol, PERIOD_H1, 14);
   if(m_atr_handle == INVALID_HANDLE)
   {
      Print("CTradeManager: Failed to create ATR indicator");
      return false;
   }
   
   // Sync with existing position if any
   SyncWithExistingPosition();
   
   Print("CTradeManager: Initialized for ", m_symbol, " Magic: ", m_magic_number);
   return true;
}

void CTradeManager::SetManagementMode(ENUM_MANAGEMENT_MODE mode)
{
   m_mgmt_mode = mode;
   Print("CTradeManager: Management mode set to ", EnumToString(mode));
}

void CTradeManager::UpdateATR()
{
   double atr[];
   ArrayResize(atr, 1);
   if(CopyBuffer(m_atr_handle, 0, 0, 1, atr) > 0)
      m_current_atr = atr[0];
}

double CTradeManager::GetCurrentRMultiple()
{
   if(!m_has_active_trade) return 0;
   
   double current_price = (m_active_trade.type == POSITION_TYPE_BUY) 
      ? SymbolInfoDouble(m_symbol, SYMBOL_BID) 
      : SymbolInfoDouble(m_symbol, SYMBOL_ASK);
   
   double risk = MathAbs(m_active_trade.entry_price - m_active_trade.initial_sl);
   if(risk <= 0) return 0;
   
   double profit = (m_active_trade.type == POSITION_TYPE_BUY)
      ? current_price - m_active_trade.entry_price
      : m_active_trade.entry_price - current_price;
   
   return profit / risk;
}

bool CTradeManager::OpenTrade(ENUM_SIGNAL_TYPE signal, double lots, double sl, double tp, 
                               double score, string reason)
{
   if(m_has_active_trade)
   {
      Print("CTradeManager: Cannot open trade - already have active position");
      return false;
   }
   
   if(signal == SIGNAL_NONE) return false;
   
   double price = (signal == SIGNAL_BUY) 
      ? SymbolInfoDouble(m_symbol, SYMBOL_ASK) 
      : SymbolInfoDouble(m_symbol, SYMBOL_BID);
   
   // Validate SL/TP
   double min_stop = SymbolInfoInteger(m_symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(MathAbs(price - sl) < min_stop || MathAbs(tp - price) < min_stop)
   {
      Print("CTradeManager: SL/TP too close to price");
      return false;
   }
   
   bool result = false;
   if(signal == SIGNAL_BUY)
      result = m_trade.Buy(lots, m_symbol, price, sl, tp, reason);
   else
      result = m_trade.Sell(lots, m_symbol, price, sl, tp, reason);
   
   if(!result)
   {
      Print("CTradeManager: Trade failed - ", m_trade.ResultRetcodeDescription());
      return false;
   }
   
   // Initialize trade info
   m_active_trade.ticket = m_trade.ResultDeal();
   m_active_trade.state = TRADE_STATE_POSITION_OPEN;
   m_active_trade.type = (signal == SIGNAL_BUY) ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
   m_active_trade.entry_price = m_trade.ResultPrice();
   m_active_trade.initial_sl = sl;
   m_active_trade.current_sl = sl;
   m_active_trade.tp1 = tp;
   m_active_trade.tp2 = 0;
   m_active_trade.tp3 = 0;
   m_active_trade.initial_lots = lots;
   m_active_trade.current_lots = lots;
   m_active_trade.highest_price = m_active_trade.entry_price;
   m_active_trade.lowest_price = m_active_trade.entry_price;
   m_active_trade.entry_time = TimeCurrent();
   m_active_trade.entry_score = score;
   m_active_trade.realized_pnl = 0;
   m_active_trade.partials_taken = 0;
   m_active_trade.entry_reason = reason;
   
   m_has_active_trade = true;
   
   Print("CTradeManager: Trade opened - ", (signal == SIGNAL_BUY ? "BUY" : "SELL"),
         " @ ", m_active_trade.entry_price, " SL: ", sl, " TP: ", tp,
         " Score: ", score, " Reason: ", reason);
   
   return true;
}

bool CTradeManager::OpenTradeWithTPs(ENUM_SIGNAL_TYPE signal, double lots, double sl,
                                      double tp1, double tp2, double tp3,
                                      double score, string reason)
{
   // Open trade with TP1 as initial take profit (broker order)
   // TP2 and TP3 stored for partial management
   if(!OpenTrade(signal, lots, sl, tp1, score, reason))
      return false;
   
   // Store additional TP levels for partial management
   m_active_trade.tp1 = tp1;
   m_active_trade.tp2 = tp2;
   m_active_trade.tp3 = tp3;
   
   // Calculate R multiples for each TP
   double risk = MathAbs(m_active_trade.entry_price - sl);
   if(risk > 0)
   {
      double r1 = MathAbs(tp1 - m_active_trade.entry_price) / risk;
      double r2 = MathAbs(tp2 - m_active_trade.entry_price) / risk;
      
      // Auto-configure partials based on TP levels
      m_partial1_trigger = r1;
      m_partial2_trigger = r2;
      m_trailing_start = r2;
   }
   
   Print("CTradeManager: Multi-TP trade - TP1: ", tp1, " (40%), TP2: ", tp2, " (30%), TP3: ", tp3, " (trail)");
   return true;
}

bool CTradeManager::CloseTrade(string reason)
{
   if(!m_has_active_trade) return false;
   
   // Find position by magic number
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            if(m_trade.PositionClose(PositionGetTicket(i)))
            {
               ENUM_TRADE_STATE old_state = m_active_trade.state;
               m_active_trade.state = TRADE_STATE_CLOSED;
               m_has_active_trade = false;
               LogStateChange(old_state, TRADE_STATE_CLOSED);
               Print("CTradeManager: Trade closed - ", reason);
               return true;
            }
         }
      }
   }
   return false;
}

bool CTradeManager::ClosePartial(double percent, string reason)
{
   if(!m_has_active_trade) return false;
   if(percent <= 0 || percent > 1) return false;
   
   double lots_to_close = NormalizeDouble(m_active_trade.current_lots * percent, 2);
   double min_lot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
   
   if(lots_to_close < min_lot)
   {
      Print("CTradeManager: Partial close amount too small");
      return false;
   }
   
   // Find and partially close
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            ulong ticket = PositionGetTicket(i);
            if(m_trade.PositionClosePartial(ticket, lots_to_close))
            {
               // Capture profit from this partial only; prefer trade result, fallback to proportional unrealized
               double partial_profit = m_trade.ResultProfit();
               if(partial_profit == 0.0)
                  partial_profit = PositionGetDouble(POSITION_PROFIT) * percent;

               m_active_trade.current_lots -= lots_to_close;
               m_active_trade.partials_taken++;
               m_active_trade.realized_pnl += partial_profit;
               
               Print("CTradeManager: Partial close - ", lots_to_close, " lots. Reason: ", reason);
               return true;
            }
         }
      }
   }
   return false;
}

void CTradeManager::OnTick()
{
   if(!m_has_active_trade) return;
   
   UpdateATR();
   
   // Check if position still exists
   bool position_exists = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            position_exists = true;
            break;
         }
      }
   }
   
   if(!position_exists)
   {
      // Position was closed (by SL/TP or manually)
      ENUM_TRADE_STATE old_state = m_active_trade.state;
      m_active_trade.state = TRADE_STATE_CLOSED;
      m_has_active_trade = false;
      LogStateChange(old_state, TRADE_STATE_CLOSED);
      Print("CTradeManager: Position no longer exists - closed externally");
      return;
   }
   
   // Update high/low tracking
   double current_price = (m_active_trade.type == POSITION_TYPE_BUY)
      ? SymbolInfoDouble(m_symbol, SYMBOL_BID)
      : SymbolInfoDouble(m_symbol, SYMBOL_ASK);
   
   if(current_price > m_active_trade.highest_price)
      m_active_trade.highest_price = current_price;
   if(current_price < m_active_trade.lowest_price)
      m_active_trade.lowest_price = current_price;
   
   // State machine logic
   double r_multiple = GetCurrentRMultiple();
   ENUM_TRADE_STATE old_state = m_active_trade.state;
   
   switch(m_active_trade.state)
   {
      case TRADE_STATE_POSITION_OPEN:
         // Check for breakeven
         if(r_multiple >= m_breakeven_trigger && m_mgmt_mode != MGMT_FIXED_RR)
         {
            if(MoveToBreakeven())
            {
               m_active_trade.state = TRADE_STATE_BREAKEVEN;
               LogStateChange(old_state, TRADE_STATE_BREAKEVEN);
            }
         }
         break;
         
      case TRADE_STATE_BREAKEVEN:
         // Check for partial TP1
         if(r_multiple >= m_partial1_trigger && m_active_trade.partials_taken == 0 &&
            (m_mgmt_mode == MGMT_PARTIAL_TP || m_mgmt_mode == MGMT_DYNAMIC))
         {
            if(TakePartialProfit(m_partial1_percent, 1))
            {
               m_active_trade.state = TRADE_STATE_PARTIAL_TP;
               LogStateChange(old_state, TRADE_STATE_PARTIAL_TP);
            }
         }
         // Or start trailing
         else if(r_multiple >= m_trailing_start && m_mgmt_mode == MGMT_TRAILING_ATR)
         {
            m_active_trade.state = TRADE_STATE_TRAILING;
            LogStateChange(old_state, TRADE_STATE_TRAILING);
         }
         break;
         
      case TRADE_STATE_PARTIAL_TP:
         // Check for partial TP2
         if(r_multiple >= m_partial2_trigger && m_active_trade.partials_taken == 1)
         {
            TakePartialProfit(m_partial2_percent, 2);
         }
         // Start trailing after partials
         if(r_multiple >= m_trailing_start)
         {
            m_active_trade.state = TRADE_STATE_TRAILING;
            LogStateChange(old_state, TRADE_STATE_TRAILING);
         }
         break;
         
      case TRADE_STATE_TRAILING:
         // Update trailing stop
         UpdateTrailingStop();
         break;
   }
}

bool CTradeManager::MoveToBreakeven()
{
   if(!m_has_active_trade) return false;
   
   // Add small buffer (2 points) to ensure profit
   double buffer = 2 * _Point;
   double new_sl = (m_active_trade.type == POSITION_TYPE_BUY)
      ? m_active_trade.entry_price + buffer
      : m_active_trade.entry_price - buffer;
   
   // Already at or better than BE
   if(m_active_trade.type == POSITION_TYPE_BUY && m_active_trade.current_sl >= new_sl)
      return true;
   if(m_active_trade.type == POSITION_TYPE_SELL && m_active_trade.current_sl <= new_sl)
      return true;
   
   // Modify position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            double current_tp = PositionGetDouble(POSITION_TP);
            if(m_trade.PositionModify(PositionGetTicket(i), new_sl, current_tp))
            {
               m_active_trade.current_sl = new_sl;
               Print("CTradeManager: Moved to breakeven @ ", new_sl);
               return true;
            }
         }
      }
   }
   return false;
}

bool CTradeManager::TakePartialProfit(double percent, int partial_num)
{
   if(!m_has_active_trade) return false;
   
   string reason = StringFormat("Partial TP %d @ %.2fR", partial_num, GetCurrentRMultiple());
   return ClosePartial(percent, reason);
}

bool CTradeManager::UpdateTrailingStop()
{
   if(!m_has_active_trade || m_current_atr <= 0) return false;
   
   double new_sl = CalculateTrailingStop();
   
   // Only move SL in profitable direction
   bool should_update = false;
   if(m_active_trade.type == POSITION_TYPE_BUY && new_sl > m_active_trade.current_sl)
      should_update = true;
   if(m_active_trade.type == POSITION_TYPE_SELL && new_sl < m_active_trade.current_sl)
      should_update = true;
   
   if(!should_update) return false;
   
   // Modify position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            double current_tp = PositionGetDouble(POSITION_TP);
            if(m_trade.PositionModify(PositionGetTicket(i), new_sl, current_tp))
            {
               m_active_trade.current_sl = new_sl;
               Print("CTradeManager: Trailing stop updated to ", new_sl);
               return true;
            }
         }
      }
   }
   return false;
}

double CTradeManager::CalculateTrailingStop()
{
   double trail_distance = m_current_atr * m_trailing_step;
   
   if(m_active_trade.type == POSITION_TYPE_BUY)
   {
      // Trail below highest price
      return m_active_trade.highest_price - trail_distance;
   }
   else
   {
      // Trail above lowest price
      return m_active_trade.lowest_price + trail_distance;
   }
}

double CTradeManager::GetUnrealizedPnL()
{
   if(!m_has_active_trade) return 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            return PositionGetDouble(POSITION_PROFIT);
         }
      }
   }
   return 0;
}

double CTradeManager::GetTotalPnL()
{
   return m_active_trade.realized_pnl + GetUnrealizedPnL();
}

bool CTradeManager::SyncWithExistingPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magic_number &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            // Found existing position - sync state
            m_active_trade.ticket = PositionGetTicket(i);
            m_active_trade.type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            m_active_trade.entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            m_active_trade.current_sl = PositionGetDouble(POSITION_SL);
            m_active_trade.initial_sl = m_active_trade.current_sl;
            m_active_trade.tp1 = PositionGetDouble(POSITION_TP);
            m_active_trade.current_lots = PositionGetDouble(POSITION_VOLUME);
            m_active_trade.initial_lots = m_active_trade.current_lots;
            m_active_trade.entry_time = (datetime)PositionGetInteger(POSITION_TIME);
            m_active_trade.highest_price = m_active_trade.entry_price;
            m_active_trade.lowest_price = m_active_trade.entry_price;
            
            // Determine current state
            double r = GetCurrentRMultiple();
            if(r >= m_trailing_start)
               m_active_trade.state = TRADE_STATE_TRAILING;
            else if(m_active_trade.current_sl >= m_active_trade.entry_price - _Point && 
                    m_active_trade.type == POSITION_TYPE_BUY)
               m_active_trade.state = TRADE_STATE_BREAKEVEN;
            else if(m_active_trade.current_sl <= m_active_trade.entry_price + _Point && 
                    m_active_trade.type == POSITION_TYPE_SELL)
               m_active_trade.state = TRADE_STATE_BREAKEVEN;
            else
               m_active_trade.state = TRADE_STATE_POSITION_OPEN;
            
            m_has_active_trade = true;
            Print("CTradeManager: Synced with existing position #", m_active_trade.ticket,
                  " State: ", EnumToString(m_active_trade.state));
            return true;
         }
      }
   }
   return false;
}

void CTradeManager::LogStateChange(ENUM_TRADE_STATE old_state, ENUM_TRADE_STATE new_state)
{
   Print("CTradeManager: State change ",
         EnumToString(old_state), " -> ", EnumToString(new_state),
         " @ ", GetCurrentRMultiple(), "R");
}
