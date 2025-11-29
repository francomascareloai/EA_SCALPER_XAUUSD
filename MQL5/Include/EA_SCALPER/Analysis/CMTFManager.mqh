//+------------------------------------------------------------------+
//|                                                 CMTFManager.mqh |
//|                    Multi-Timeframe Analysis Manager              |
//|                    HTF=H1, MTF=M15, LTF=M5                       |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property strict

#include "..\Core\Definitions.mqh"

//--- MTF Configuration
#define MTF_HTF     PERIOD_H1    // High Timeframe - Direction
#define MTF_MTF     PERIOD_M15   // Medium Timeframe - Structure  
#define MTF_LTF     PERIOD_M5    // Low Timeframe - Execution

//--- Trend Direction Enum
enum ENUM_MTF_TREND
{
   MTF_TREND_BULLISH = 1,      // Clear bullish trend
   MTF_TREND_BEARISH = -1,     // Clear bearish trend
   MTF_TREND_NEUTRAL = 0,      // No clear direction
   MTF_TREND_RANGING = 2       // Range-bound market
};

//--- MTF Alignment Quality
enum ENUM_MTF_ALIGNMENT
{
   MTF_ALIGN_PERFECT = 3,      // All TFs aligned
   MTF_ALIGN_GOOD = 2,         // 2 TFs aligned
   MTF_ALIGN_WEAK = 1,         // Only 1 TF confirms
   MTF_ALIGN_NONE = 0          // No alignment
};

//--- Structure for HTF Analysis
struct SHTFAnalysis
{
   ENUM_MTF_TREND    trend;               // H1 trend direction
   double            trend_strength;       // Trend strength (0-100)
   double            ma_20;               // H1 MA(20)
   double            ma_50;               // H1 MA(50)
   double            atr;                 // H1 ATR(14)
   double            rsi;                 // H1 RSI(14)
   bool              is_trending;         // True if H > 0.55 or < 0.45
   double            hurst;               // Hurst exponent estimate
   datetime          last_update;         // Last update time
   
   // Key levels from H1
   double            swing_high;          // Last H1 swing high
   double            swing_low;           // Last H1 swing low
   double            daily_high;          // Today's high
   double            daily_low;           // Today's low
   
   void Reset()
   {
      trend = MTF_TREND_NEUTRAL;
      trend_strength = 0;
      ma_20 = 0;
      ma_50 = 0;
      atr = 0;
      rsi = 50;
      is_trending = false;
      hurst = 0.5;
      last_update = 0;
      swing_high = 0;
      swing_low = 0;
      daily_high = 0;
      daily_low = 0;
   }
};

//--- Structure for MTF (M15) Analysis
struct SMTFAnalysis
{
   ENUM_MTF_TREND    trend;               // M15 trend direction
   double            trend_strength;       // Trend strength (0-100)
   
   // Structure levels
   bool              has_active_ob;        // Active Order Block present
   bool              has_active_fvg;       // Active FVG present
   bool              has_liquidity_pool;   // Liquidity pool nearby
   
   double            nearest_ob_high;      // Nearest OB high
   double            nearest_ob_low;       // Nearest OB low
   double            nearest_fvg_high;     // Nearest FVG high
   double            nearest_fvg_low;      // Nearest FVG low
   
   // Market structure
   double            last_swing_high;      // M15 swing high
   double            last_swing_low;       // M15 swing low
   bool              bos_bullish;          // Break of structure bullish
   bool              bos_bearish;          // Break of structure bearish
   bool              choch_detected;       // Change of character
   
   datetime          last_update;
   
   void Reset()
   {
      trend = MTF_TREND_NEUTRAL;
      trend_strength = 0;
      has_active_ob = false;
      has_active_fvg = false;
      has_liquidity_pool = false;
      nearest_ob_high = 0;
      nearest_ob_low = 0;
      nearest_fvg_high = 0;
      nearest_fvg_low = 0;
      last_swing_high = 0;
      last_swing_low = 0;
      bos_bullish = false;
      bos_bearish = false;
      choch_detected = false;
      last_update = 0;
   }
};

//--- Structure for LTF (M5) Analysis
struct SLTFAnalysis
{
   ENUM_MTF_TREND    trend;               // M5 trend direction
   double            trend_strength;       // Trend strength (0-100)
   
   // Entry signals
   bool              has_confirmation;     // Confirmation candle present
   bool              bullish_engulf;       // Bullish engulfing
   bool              bearish_engulf;       // Bearish engulfing
   bool              pin_bar_bullish;      // Bullish pin bar
   bool              pin_bar_bearish;      // Bearish pin bar
   
   // Momentum
   double            rsi;                  // M5 RSI
   double            momentum;             // Price momentum
   bool              momentum_aligned;     // Momentum with trend
   
   // Entry precision
   double            optimal_entry_long;   // Best entry for longs
   double            optimal_entry_short;  // Best entry for shorts
   double            tight_sl_long;        // Tight SL for longs
   double            tight_sl_short;       // Tight SL for shorts
   
   datetime          last_update;
   
   void Reset()
   {
      trend = MTF_TREND_NEUTRAL;
      trend_strength = 0;
      has_confirmation = false;
      bullish_engulf = false;
      bearish_engulf = false;
      pin_bar_bullish = false;
      pin_bar_bearish = false;
      rsi = 50;
      momentum = 0;
      momentum_aligned = false;
      optimal_entry_long = 0;
      optimal_entry_short = 0;
      tight_sl_long = 0;
      tight_sl_short = 0;
      last_update = 0;
   }
};

//--- MTF Confluence Result
struct SMTFConfluence
{
   ENUM_MTF_ALIGNMENT alignment;          // Overall alignment
   ENUM_SIGNAL_TYPE   signal;             // Trade signal
   double             confidence;          // Confidence score (0-100)
   double             position_size_mult;  // Position size multiplier (0-1)
   
   // Individual TF signals
   ENUM_MTF_TREND     htf_trend;
   ENUM_MTF_TREND     mtf_trend;
   ENUM_MTF_TREND     ltf_trend;
   
   // Entry details
   double             entry_price;
   double             stop_loss;
   double             take_profit;
   double             risk_reward;
   
   // Confluence factors
   bool               htf_aligned;         // H1 supports direction
   bool               mtf_structure;       // M15 has OB/FVG zone
   bool               ltf_confirmed;       // M5 confirmation
   bool               session_ok;          // Active session
   int                confluence_count;    // Number of factors aligned
   
   string             reason;              // Trade reason
   
   void Reset()
   {
      alignment = MTF_ALIGN_NONE;
      signal = SIGNAL_NONE;
      confidence = 0;
      position_size_mult = 0;
      htf_trend = MTF_TREND_NEUTRAL;
      mtf_trend = MTF_TREND_NEUTRAL;
      ltf_trend = MTF_TREND_NEUTRAL;
      entry_price = 0;
      stop_loss = 0;
      take_profit = 0;
      risk_reward = 0;
      htf_aligned = false;
      mtf_structure = false;
      ltf_confirmed = false;
      session_ok = false;
      confluence_count = 0;
      reason = "";
   }
};

//+------------------------------------------------------------------+
//| CMTFManager - Multi-Timeframe Analysis Manager                   |
//+------------------------------------------------------------------+
class CMTFManager
{
private:
   string            m_symbol;
   
   // Analysis results
   SHTFAnalysis      m_htf;                // H1 analysis
   SMTFAnalysis      m_mtf;                // M15 analysis  
   SLTFAnalysis      m_ltf;                // M5 analysis
   SMTFConfluence    m_confluence;         // Combined result
   
   // Indicator handles
   int               m_htf_ma20_handle;
   int               m_htf_ma50_handle;
   int               m_htf_atr_handle;
   int               m_htf_rsi_handle;
   int               m_mtf_atr_handle;
   int               m_ltf_rsi_handle;
   int               m_ltf_atr_handle;
   
   // Update intervals
   datetime          m_htf_last_bar;
   datetime          m_mtf_last_bar;
   datetime          m_ltf_last_bar;
   
   // Configuration
   double            m_min_trend_strength;  // Min trend strength to confirm
   double            m_min_confluence;      // Min confluence score
   int               m_lookback_bars;       // Bars to analyze
   
   // Private methods
   void              AnalyzeHTF();
   void              AnalyzeMTF();
   void              AnalyzeLTF();
   ENUM_MTF_TREND    DetermineTrend(double ma_fast, double ma_slow, double price, double atr);
   double            CalculateTrendStrength(ENUM_TIMEFRAMES tf);
   void              FindSwingPoints(ENUM_TIMEFRAMES tf, double &swing_high, double &swing_low);
   bool              DetectConfirmationCandle(ENUM_SIGNAL_TYPE expected_direction);
   
public:
                     CMTFManager();
                    ~CMTFManager();
   
   // Initialization
   bool              Init(string symbol);
   void              Deinit();
   
   // Update methods
   void              Update();              // Full update
   void              UpdateHTF();           // Update H1 only (on new H1 bar)
   void              UpdateMTF();           // Update M15 only (on new M15 bar)
   void              UpdateLTF();           // Update M5 on every tick
   
   // Analysis getters
   SHTFAnalysis      GetHTFAnalysis() const { return m_htf; }
   SMTFAnalysis      GetMTFAnalysis() const { return m_mtf; }
   SLTFAnalysis      GetLTFAnalysis() const { return m_ltf; }
   SMTFConfluence    GetConfluence();
   
   // Quick checks
   bool              IsHTFBullish() const { return m_htf.trend == MTF_TREND_BULLISH; }
   bool              IsHTFBearish() const { return m_htf.trend == MTF_TREND_BEARISH; }
   bool              IsTrending() const { return m_htf.is_trending; }
   bool              HasMTFStructure() const { return m_mtf.has_active_ob || m_mtf.has_active_fvg; }
   bool              HasLTFConfirmation(ENUM_SIGNAL_TYPE direction);
   
   // Trade permission
   bool              CanTradeLong();
   bool              CanTradeShort();
   ENUM_MTF_ALIGNMENT GetAlignment() const { return m_confluence.alignment; }
   double            GetPositionSizeMultiplier() const { return m_confluence.position_size_mult; }
   void              SetStructureFlags(bool has_ob, bool has_fvg, bool has_liquidity=false)
   {
      m_mtf.has_active_ob = has_ob;
      m_mtf.has_active_fvg = has_fvg;
      m_mtf.has_liquidity_pool = has_liquidity;
   }
   
   // Setters
   void              SetMinTrendStrength(double strength) { m_min_trend_strength = strength; }
   void              SetMinConfluence(double conf) { m_min_confluence = conf; }
   
   // Utility
   string            GetAnalysisSummary();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CMTFManager::CMTFManager()
{
   m_symbol = "";
   m_htf.Reset();
   m_mtf.Reset();
   m_ltf.Reset();
   m_confluence.Reset();
   
   m_htf_ma20_handle = INVALID_HANDLE;
   m_htf_ma50_handle = INVALID_HANDLE;
   m_htf_atr_handle = INVALID_HANDLE;
      m_htf_rsi_handle = INVALID_HANDLE;
      m_mtf_atr_handle = INVALID_HANDLE;
      m_ltf_rsi_handle = INVALID_HANDLE;
      m_ltf_atr_handle = INVALID_HANDLE;
   
   m_htf_last_bar = 0;
   m_mtf_last_bar = 0;
   m_ltf_last_bar = 0;
   
   m_min_trend_strength = 30.0;
   m_min_confluence = 60.0;
   m_lookback_bars = 100;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CMTFManager::~CMTFManager()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CMTFManager::Init(string symbol)
{
   m_symbol = symbol;
   
   // Create H1 indicators
   m_htf_ma20_handle = iMA(m_symbol, MTF_HTF, 20, 0, MODE_EMA, PRICE_CLOSE);
   m_htf_ma50_handle = iMA(m_symbol, MTF_HTF, 50, 0, MODE_EMA, PRICE_CLOSE);
   m_htf_atr_handle = iATR(m_symbol, MTF_HTF, 14);
   m_htf_rsi_handle = iRSI(m_symbol, MTF_HTF, 14, PRICE_CLOSE);
   
   // Create M15 indicators
   m_mtf_atr_handle = iATR(m_symbol, MTF_MTF, 14);
   
   // Create M5 indicators
   m_ltf_rsi_handle = iRSI(m_symbol, MTF_LTF, 14, PRICE_CLOSE);
   m_ltf_atr_handle = iATR(m_symbol, MTF_LTF, 14);
   
   // Validate handles
   if(m_htf_ma20_handle == INVALID_HANDLE || 
      m_htf_ma50_handle == INVALID_HANDLE ||
      m_htf_atr_handle == INVALID_HANDLE ||
      m_htf_rsi_handle == INVALID_HANDLE ||
      m_mtf_atr_handle == INVALID_HANDLE ||
      m_ltf_rsi_handle == INVALID_HANDLE ||
      m_ltf_atr_handle == INVALID_HANDLE)
   {
      Print("[MTF] Failed to create indicator handles");
      return false;
   }
   
   Print("[MTF] Manager initialized for ", m_symbol);
   Print("[MTF] HTF=H1, MTF=M15, LTF=M5");
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CMTFManager::Deinit()
{
   if(m_htf_ma20_handle != INVALID_HANDLE) IndicatorRelease(m_htf_ma20_handle);
   if(m_htf_ma50_handle != INVALID_HANDLE) IndicatorRelease(m_htf_ma50_handle);
   if(m_htf_atr_handle != INVALID_HANDLE) IndicatorRelease(m_htf_atr_handle);
   if(m_htf_rsi_handle != INVALID_HANDLE) IndicatorRelease(m_htf_rsi_handle);
   if(m_mtf_atr_handle != INVALID_HANDLE) IndicatorRelease(m_mtf_atr_handle);
   if(m_ltf_rsi_handle != INVALID_HANDLE) IndicatorRelease(m_ltf_rsi_handle);
   if(m_ltf_atr_handle != INVALID_HANDLE) IndicatorRelease(m_ltf_atr_handle);
}

//+------------------------------------------------------------------+
//| Full update                                                       |
//+------------------------------------------------------------------+
void CMTFManager::Update()
{
   datetime htf_bar = iTime(m_symbol, MTF_HTF, 0);
   datetime mtf_bar = iTime(m_symbol, MTF_MTF, 0);
   datetime ltf_bar = iTime(m_symbol, MTF_LTF, 0);
   
   // Update HTF on new H1 bar
   if(htf_bar != m_htf_last_bar)
   {
      AnalyzeHTF();
      m_htf_last_bar = htf_bar;
   }
   
   // Update MTF on new M15 bar
   if(mtf_bar != m_mtf_last_bar)
   {
      AnalyzeMTF();
      m_mtf_last_bar = mtf_bar;
   }
   
   // Update LTF on new M5 bar (or every tick for precision)
   if(ltf_bar != m_ltf_last_bar)
   {
      AnalyzeLTF();
      m_ltf_last_bar = ltf_bar;
   }
}

//+------------------------------------------------------------------+
//| Analyze H1 (HTF)                                                  |
//+------------------------------------------------------------------+
void CMTFManager::AnalyzeHTF()
{
   double ma20[], ma50[], atr[], rsi[];
   ArraySetAsSeries(ma20, true);
   ArraySetAsSeries(ma50, true);
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(rsi, true);
   
   if(CopyBuffer(m_htf_ma20_handle, 0, 0, 3, ma20) < 3) return;
   if(CopyBuffer(m_htf_ma50_handle, 0, 0, 3, ma50) < 3) return;
   if(CopyBuffer(m_htf_atr_handle, 0, 0, 3, atr) < 3) return;
   if(CopyBuffer(m_htf_rsi_handle, 0, 0, 3, rsi) < 3) return;
   
   double close = iClose(m_symbol, MTF_HTF, 0);
   
   // Store values
   m_htf.ma_20 = ma20[0];
   m_htf.ma_50 = ma50[0];
   m_htf.atr = atr[0];
   m_htf.rsi = rsi[0];
   
   // Determine trend
   m_htf.trend = DetermineTrend(ma20[0], ma50[0], close, atr[0]);
   m_htf.trend_strength = CalculateTrendStrength(MTF_HTF);
   
   // Is trending? (based on MA separation)
   double ma_separation = MathAbs(ma20[0] - ma50[0]) / atr[0];
   m_htf.is_trending = (ma_separation > 1.0); // MAs separated by > 1 ATR
   
   // Find swing points
   FindSwingPoints(MTF_HTF, m_htf.swing_high, m_htf.swing_low);
   
   // Daily levels
   m_htf.daily_high = iHigh(m_symbol, PERIOD_D1, 0);
   m_htf.daily_low = iLow(m_symbol, PERIOD_D1, 0);
   
   m_htf.last_update = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Analyze M15 (MTF)                                                 |
//+------------------------------------------------------------------+
void CMTFManager::AnalyzeMTF()
{
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(m_mtf_atr_handle, 0, 0, 3, atr) < 3) return;
   
   double close = iClose(m_symbol, MTF_MTF, 0);
   double high = iHigh(m_symbol, MTF_MTF, 0);
   double low = iLow(m_symbol, MTF_MTF, 0);
   
   // Trend from H1 perspective (inherit direction)
   if(m_htf.trend == MTF_TREND_BULLISH)
      m_mtf.trend = MTF_TREND_BULLISH;
   else if(m_htf.trend == MTF_TREND_BEARISH)
      m_mtf.trend = MTF_TREND_BEARISH;
   else
      m_mtf.trend = MTF_TREND_NEUTRAL;
   
   m_mtf.trend_strength = CalculateTrendStrength(MTF_MTF);
   
   // Find M15 swing points
   FindSwingPoints(MTF_MTF, m_mtf.last_swing_high, m_mtf.last_swing_low);
   
   // Check for break of structure
   double prev_high = iHigh(m_symbol, MTF_MTF, 1);
   double prev_low = iLow(m_symbol, MTF_MTF, 1);
   
   // Simple BOS detection
   if(close > m_mtf.last_swing_high && prev_high <= m_mtf.last_swing_high)
      m_mtf.bos_bullish = true;
   else
      m_mtf.bos_bullish = false;
      
   if(close < m_mtf.last_swing_low && prev_low >= m_mtf.last_swing_low)
      m_mtf.bos_bearish = true;
   else
      m_mtf.bos_bearish = false;
   
   // Note: OB/FVG detection will be done by their respective modules
   // Here we just check if we're in a zone
   m_mtf.has_active_ob = (m_mtf.bos_bullish || m_mtf.bos_bearish);  // heuristic until detectors wire in
   m_mtf.has_active_fvg = false; // Will be set by CFVGDetector
   m_mtf.has_liquidity_pool = false; // Will be set by CLiquiditySweepDetector
   
   m_mtf.last_update = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Analyze M5 (LTF)                                                  |
//+------------------------------------------------------------------+
void CMTFManager::AnalyzeLTF()
{
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(m_ltf_rsi_handle, 0, 0, 3, rsi) < 3) return;
   
   m_ltf.rsi = rsi[0];
   
   // Get recent candles
   double open1 = iOpen(m_symbol, MTF_LTF, 1);
   double close1 = iClose(m_symbol, MTF_LTF, 1);
   double high1 = iHigh(m_symbol, MTF_LTF, 1);
   double low1 = iLow(m_symbol, MTF_LTF, 1);
   
   double open2 = iOpen(m_symbol, MTF_LTF, 2);
   double close2 = iClose(m_symbol, MTF_LTF, 2);
   
   double body1 = MathAbs(close1 - open1);
   double body2 = MathAbs(close2 - open2);
   double range1 = high1 - low1;
   
   // Detect engulfing
   m_ltf.bullish_engulf = (close2 < open2 && close1 > open1 && close1 > open2 && open1 < close2 && body1 > body2);
   m_ltf.bearish_engulf = (close2 > open2 && close1 < open1 && close1 < open2 && open1 > close2 && body1 > body2);
   
   // Detect pin bars
   double upper_wick1 = high1 - MathMax(open1, close1);
   double lower_wick1 = MathMin(open1, close1) - low1;
   
   m_ltf.pin_bar_bullish = (lower_wick1 > body1 * 2 && upper_wick1 < body1 * 0.5);
   m_ltf.pin_bar_bearish = (upper_wick1 > body1 * 2 && lower_wick1 < body1 * 0.5);
   
   // Check confirmation
   m_ltf.has_confirmation = m_ltf.bullish_engulf || m_ltf.bearish_engulf || 
                            m_ltf.pin_bar_bullish || m_ltf.pin_bar_bearish;
   
   // Calculate momentum
   double close_5 = iClose(m_symbol, MTF_LTF, 5);
   m_ltf.momentum = (close1 - close_5) / close_5 * 100;
   
   // Is momentum aligned with HTF trend?
   if(m_htf.trend == MTF_TREND_BULLISH && m_ltf.momentum > 0)
      m_ltf.momentum_aligned = true;
   else if(m_htf.trend == MTF_TREND_BEARISH && m_ltf.momentum < 0)
      m_ltf.momentum_aligned = true;
   else
      m_ltf.momentum_aligned = false;
   
   // Calculate optimal entries
   double atr_m5_buf[];
   ArraySetAsSeries(atr_m5_buf, true);
   if(CopyBuffer(m_ltf_atr_handle, 0, 0, 1, atr_m5_buf) < 1) return;
   double atr_m5 = atr_m5_buf[0];
   double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   
   m_ltf.optimal_entry_long = low1;  // Entry at previous low
   m_ltf.optimal_entry_short = high1; // Entry at previous high
   m_ltf.tight_sl_long = low1 - atr_m5 * 0.5;
   m_ltf.tight_sl_short = high1 + atr_m5 * 0.5;
   
   // Determine LTF trend
   if(rsi[0] > 55 && m_ltf.momentum > 0)
      m_ltf.trend = MTF_TREND_BULLISH;
   else if(rsi[0] < 45 && m_ltf.momentum < 0)
      m_ltf.trend = MTF_TREND_BEARISH;
   else
      m_ltf.trend = MTF_TREND_NEUTRAL;
   
   m_ltf.last_update = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Determine trend direction                                         |
//+------------------------------------------------------------------+
ENUM_MTF_TREND CMTFManager::DetermineTrend(double ma_fast, double ma_slow, double price, double atr)
{
   double threshold = atr * 0.3; // 30% of ATR as threshold
   
   if(price > ma_fast && ma_fast > ma_slow && (ma_fast - ma_slow) > threshold)
      return MTF_TREND_BULLISH;
   else if(price < ma_fast && ma_fast < ma_slow && (ma_slow - ma_fast) > threshold)
      return MTF_TREND_BEARISH;
   else if(MathAbs(ma_fast - ma_slow) < threshold)
      return MTF_TREND_RANGING;
   else
      return MTF_TREND_NEUTRAL;
}

//+------------------------------------------------------------------+
//| Calculate trend strength                                          |
//+------------------------------------------------------------------+
double CMTFManager::CalculateTrendStrength(ENUM_TIMEFRAMES tf)
{
   double closes[];
   ArraySetAsSeries(closes, true);
   if(CopyClose(m_symbol, tf, 0, 20, closes) < 20) return 0;
   
   int up_count = 0;
   int down_count = 0;
   
   for(int i = 0; i < 19; i++)
   {
      if(closes[i] > closes[i+1]) up_count++;
      else if(closes[i] < closes[i+1]) down_count++;
   }
   
   double strength = MathAbs(up_count - down_count) / 19.0 * 100;
   return strength;
}

//+------------------------------------------------------------------+
//| Find swing points                                                 |
//+------------------------------------------------------------------+
void CMTFManager::FindSwingPoints(ENUM_TIMEFRAMES tf, double &swing_high, double &swing_low)
{
   double highs[], lows[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   
   if(CopyHigh(m_symbol, tf, 0, m_lookback_bars, highs) < m_lookback_bars) return;
   if(CopyLow(m_symbol, tf, 0, m_lookback_bars, lows) < m_lookback_bars) return;
   
   swing_high = 0;
   swing_low = DBL_MAX;
   
   // Find swing high (local maximum)
   for(int i = 2; i < m_lookback_bars - 2; i++)
   {
      if(highs[i] > highs[i-1] && highs[i] > highs[i-2] &&
         highs[i] > highs[i+1] && highs[i] > highs[i+2])
      {
         if(highs[i] > swing_high) swing_high = highs[i];
         break;
      }
   }
   
   // Find swing low (local minimum)
   for(int i = 2; i < m_lookback_bars - 2; i++)
   {
      if(lows[i] < lows[i-1] && lows[i] < lows[i-2] &&
         lows[i] < lows[i+1] && lows[i] < lows[i+2])
      {
         if(lows[i] < swing_low) swing_low = lows[i];
         break;
      }
   }
   
   if(swing_low == DBL_MAX) swing_low = lows[ArrayMinimum(lows)];
   if(swing_high == 0) swing_high = highs[ArrayMaximum(highs)];
}

//+------------------------------------------------------------------+
//| Get MTF Confluence                                                |
//+------------------------------------------------------------------+
SMTFConfluence CMTFManager::GetConfluence()
{
   m_confluence.Reset();
   
   m_confluence.htf_trend = m_htf.trend;
   m_confluence.mtf_trend = m_mtf.trend;
   m_confluence.ltf_trend = m_ltf.trend;
   
   int confluence_count = 0;
   
   // Check HTF alignment
   if(m_htf.trend == MTF_TREND_BULLISH || m_htf.trend == MTF_TREND_BEARISH)
   {
      m_confluence.htf_aligned = true;
      confluence_count++;
   }
   
   // Check MTF structure
   if(m_mtf.has_active_ob || m_mtf.has_active_fvg)
   {
      m_confluence.mtf_structure = true;
      confluence_count++;
   }
   
   // Check LTF confirmation
   if(m_ltf.has_confirmation && m_ltf.momentum_aligned)
   {
      m_confluence.ltf_confirmed = true;
      confluence_count++;
   }
   
   m_confluence.confluence_count = confluence_count;
   
   // Determine alignment quality
   if(confluence_count >= 3)
      m_confluence.alignment = MTF_ALIGN_PERFECT;
   else if(confluence_count == 2)
      m_confluence.alignment = MTF_ALIGN_GOOD;
   else if(confluence_count == 1)
      m_confluence.alignment = MTF_ALIGN_WEAK;
   else
      m_confluence.alignment = MTF_ALIGN_NONE;
   
   // Position size multiplier based on alignment
   switch(m_confluence.alignment)
   {
      case MTF_ALIGN_PERFECT: m_confluence.position_size_mult = 1.0; break;
      case MTF_ALIGN_GOOD:    m_confluence.position_size_mult = 0.75; break;
      case MTF_ALIGN_WEAK:    m_confluence.position_size_mult = 0.5; break;
      default:                m_confluence.position_size_mult = 0.0; break;
   }
   
   // Determine signal
   if(m_confluence.alignment >= MTF_ALIGN_GOOD)
   {
      if(m_htf.trend == MTF_TREND_BULLISH && 
         (m_ltf.bullish_engulf || m_ltf.pin_bar_bullish))
      {
         m_confluence.signal = SIGNAL_BUY;
         m_confluence.entry_price = m_ltf.optimal_entry_long;
         m_confluence.stop_loss = m_ltf.tight_sl_long;
         m_confluence.reason = "MTF Bullish: H1 trend + M15 structure + M5 confirmation";
      }
      else if(m_htf.trend == MTF_TREND_BEARISH && 
              (m_ltf.bearish_engulf || m_ltf.pin_bar_bearish))
      {
         m_confluence.signal = SIGNAL_SELL;
         m_confluence.entry_price = m_ltf.optimal_entry_short;
         m_confluence.stop_loss = m_ltf.tight_sl_short;
         m_confluence.reason = "MTF Bearish: H1 trend + M15 structure + M5 confirmation";
      }
   }
   
   // Calculate confidence
   m_confluence.confidence = confluence_count * 25.0 + m_htf.trend_strength * 0.25;
   if(m_confluence.confidence > 100) m_confluence.confidence = 100;
   
   return m_confluence;
}

//+------------------------------------------------------------------+
//| Check if can trade long                                           |
//+------------------------------------------------------------------+
bool CMTFManager::CanTradeLong()
{
   if(m_htf.trend != MTF_TREND_BULLISH) return false;
   if(!m_htf.is_trending) return false;
   if(m_confluence.alignment < MTF_ALIGN_GOOD) return false;
   if(!m_ltf.momentum_aligned) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if can trade short                                          |
//+------------------------------------------------------------------+
bool CMTFManager::CanTradeShort()
{
   if(m_htf.trend != MTF_TREND_BEARISH) return false;
   if(!m_htf.is_trending) return false;
   if(m_confluence.alignment < MTF_ALIGN_GOOD) return false;
   if(!m_ltf.momentum_aligned) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Check LTF confirmation for direction                              |
//+------------------------------------------------------------------+
bool CMTFManager::HasLTFConfirmation(ENUM_SIGNAL_TYPE direction)
{
   if(direction == SIGNAL_BUY)
      return (m_ltf.bullish_engulf || m_ltf.pin_bar_bullish) && m_ltf.momentum > 0;
   else if(direction == SIGNAL_SELL)
      return (m_ltf.bearish_engulf || m_ltf.pin_bar_bearish) && m_ltf.momentum < 0;
   
   return false;
}

//+------------------------------------------------------------------+
//| Get analysis summary                                              |
//+------------------------------------------------------------------+
string CMTFManager::GetAnalysisSummary()
{
   string htf_str = (m_htf.trend == MTF_TREND_BULLISH) ? "BULLISH" :
                    (m_htf.trend == MTF_TREND_BEARISH) ? "BEARISH" : "NEUTRAL";
   string mtf_str = (m_mtf.has_active_ob) ? "OB Active" : 
                    (m_mtf.has_active_fvg) ? "FVG Active" : "No Structure";
   string ltf_str = (m_ltf.has_confirmation) ? "CONFIRMED" : "No Confirmation";
   
   string align_str = (m_confluence.alignment == MTF_ALIGN_PERFECT) ? "PERFECT" :
                      (m_confluence.alignment == MTF_ALIGN_GOOD) ? "GOOD" :
                      (m_confluence.alignment == MTF_ALIGN_WEAK) ? "WEAK" : "NONE";
   
   return StringFormat("[MTF] H1:%s | M15:%s | M5:%s | Align:%s (%.0f%%)",
                       htf_str, mtf_str, ltf_str, align_str, m_confluence.confidence);
}
//+------------------------------------------------------------------+
