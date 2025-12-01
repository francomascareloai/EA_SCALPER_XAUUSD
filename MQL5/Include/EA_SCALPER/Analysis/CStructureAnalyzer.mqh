//+------------------------------------------------------------------+
//|                                            CStructureAnalyzer.mqh |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|              Market Structure Analysis: HH/HL/LH/LL, BOS, CHoCH |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property strict

// === STRUCTURE ENUMERATIONS ===
enum ENUM_STRUCTURE_TYPE
{
   STRUCT_HH = 0,      // Higher High
   STRUCT_HL = 1,      // Higher Low
   STRUCT_LH = 2,      // Lower High
   STRUCT_LL = 3,      // Lower Low
   STRUCT_EQH = 4,     // Equal High
   STRUCT_EQL = 5      // Equal Low
};

enum ENUM_MARKET_BIAS
{
   BIAS_BULLISH = 0,   // Bullish structure (HH + HL)
   BIAS_BEARISH = 1,   // Bearish structure (LH + LL)
   BIAS_RANGING = 2,   // No clear direction
   BIAS_TRANSITION = 3 // Changing direction
};

enum ENUM_STRUCTURE_BREAK
{
   BREAK_NONE = 0,     // No break
   BREAK_BOS = 1,      // Break of Structure (continuation)
   BREAK_CHOCH = 2,    // Change of Character (reversal)
   BREAK_SWEEP = 3     // Liquidity sweep (fake break)
};

// === SWING POINT STRUCTURE ===
struct SSwingPoint
{
   datetime            time;
   double              price;
   int                 bar_index;
   bool                is_high;        // true = swing high, false = swing low
   ENUM_STRUCTURE_TYPE type;
   bool                is_broken;
   datetime            break_time;
   bool                is_valid;
};

// === STRUCTURE BREAK EVENT ===
struct SStructureBreak
{
   datetime            time;
   double              break_price;
   double              swing_price;    // Price of broken swing
   ENUM_STRUCTURE_BREAK type;
   ENUM_MARKET_BIAS    new_bias;
   double              displacement;   // Size of break move
   bool                has_retest;
   double              retest_price;
   int                 strength;       // 0-100 strength score
   bool                is_valid;
};

// === STRUCTURE STATE ===
struct SStructureState
{
   // Current bias
   ENUM_MARKET_BIAS    bias;
   ENUM_MARKET_BIAS    htf_bias;       // Higher timeframe bias
   
   // Last swing points
   SSwingPoint         last_high;
   SSwingPoint         last_low;
   SSwingPoint         prev_high;
   SSwingPoint         prev_low;
   
   // Structure breaks
   SStructureBreak     last_break;
   int                 bos_count;      // Consecutive BOS
   int                 choch_count;    // CHoCH count
   
   // Premium/Discount
   double              equilibrium;    // 50% of range
   bool                in_premium;     // Price above equilibrium
   bool                in_discount;    // Price below equilibrium
   
   // Quality metrics
   double              structure_quality;  // 0-100
   double              trend_strength;     // 0-100
   datetime            last_update;
};

// === MTF TIMEFRAME DEFINITIONS ===
#define STRUCT_TF_HTF    PERIOD_H1    // High Timeframe - Direction
#define STRUCT_TF_MTF    PERIOD_M15   // Medium Timeframe - Structure  
#define STRUCT_TF_LTF    PERIOD_M5    // Low Timeframe - Execution

// === STRUCTURE ANALYZER CLASS ===
class CStructureAnalyzer
{
private:
   // Detection parameters
   int                 m_swing_strength;      // Bars on each side for swing
   double              m_equal_tolerance;     // Tolerance for equal H/L (pips)
   double              m_break_buffer;        // Buffer for valid break (pips)
   int                 m_lookback_bars;       // Bars to analyze
   int                 m_min_swing_distance;  // Min bars between swings
   
   // MTF Support (NEW v3.20)
   ENUM_TIMEFRAMES     m_current_tf;          // Current analysis timeframe
   SStructureState     m_htf_state;           // H1 structure state
   SStructureState     m_mtf_state;           // M15 structure state
   SStructureState     m_ltf_state;           // M5 structure state
   
   // Data storage
   SSwingPoint         m_swing_highs[];
   SSwingPoint         m_swing_lows[];
   SStructureBreak     m_breaks[];
   int                 m_high_count;
   int                 m_low_count;
   int                 m_break_count;
   int                 m_max_swings;
   int                 m_max_breaks;
   
   // Current state
   SStructureState     m_state;
   
public:
   CStructureAnalyzer();
   ~CStructureAnalyzer();
   
   // Configuration
   void SetSwingStrength(int strength) { m_swing_strength = MathMax(2, strength); }
   void SetEqualTolerance(double pips) { m_equal_tolerance = pips; }
   void SetBreakBuffer(double pips) { m_break_buffer = pips; }
   void SetLookback(int bars) { m_lookback_bars = MathMax(50, bars); }
   
   // Main analysis
   SStructureState AnalyzeStructure(string symbol = NULL, int tf = 0);
   ENUM_MARKET_BIAS GetCurrentBias() { return m_state.bias; }
   SStructureState GetState() { return m_state; }
   
   // MTF Analysis (NEW v3.20)
   void AnalyzeMTFStructure(string symbol = NULL);
   SStructureState GetHTFState() { return m_htf_state; }
   SStructureState GetMTFState() { return m_mtf_state; }
   SStructureState GetLTFState() { return m_ltf_state; }
   ENUM_MARKET_BIAS GetHTFBias() { return m_htf_state.bias; }
   ENUM_MARKET_BIAS GetMTFBias() { return m_mtf_state.bias; }
   bool IsMTFAligned();  // Check if all TFs aligned
   
   // Swing detection
   bool DetectSwingPoints(const MqlRates &rates[], int count);
   bool IsSwingHigh(const MqlRates &rates[], int index, int left, int right);
   bool IsSwingLow(const MqlRates &rates[], int index, int left, int right);
   
   // Structure classification
   ENUM_STRUCTURE_TYPE ClassifyHigh(const SSwingPoint &current, const SSwingPoint &previous);
   ENUM_STRUCTURE_TYPE ClassifyLow(const SSwingPoint &current, const SSwingPoint &previous);
   ENUM_MARKET_BIAS DetermineBias();
   
   // Break detection
   bool DetectBreaks(const MqlRates &rates[], int count);
   ENUM_STRUCTURE_BREAK ClassifyBreak(double break_price, const SSwingPoint &broken_swing, bool is_high);
   bool ValidateBreak(const SStructureBreak &brk, const MqlRates &rates[], int break_bar);
   
   // Premium/Discount zones
   void CalculatePremiumDiscount(double current_price);
   bool IsInPremium() { return m_state.in_premium; }
   bool IsInDiscount() { return m_state.in_discount; }
   double GetEquilibrium() { return m_state.equilibrium; }
   
   // Quality scoring
   double CalculateStructureQuality();
   double CalculateTrendStrength();
   
   // Accessors
   int GetSwingHighCount() { return m_high_count; }
   int GetSwingLowCount() { return m_low_count; }
   SSwingPoint GetLastHigh() { return m_state.last_high; }
   SSwingPoint GetLastLow() { return m_state.last_low; }
   SStructureBreak GetLastBreak() { return m_state.last_break; }
   bool HasRecentBOS() { return m_state.last_break.type == BREAK_BOS && m_state.last_break.is_valid; }
   bool HasRecentCHoCH() { return m_state.last_break.type == BREAK_CHOCH && m_state.last_break.is_valid; }
   
   // Utility
   string BiasToString(ENUM_MARKET_BIAS bias);
   string BreakToString(ENUM_STRUCTURE_BREAK brk);
   string StructureTypeToString(ENUM_STRUCTURE_TYPE type);
   
private:
   void AddSwingHigh(const SSwingPoint &swing);
   void AddSwingLow(const SSwingPoint &swing);
   void AddBreak(const SStructureBreak &brk);
   void SortSwingsByTime();
   double GetDisplacement(const MqlRates &rates[], int start_bar, int end_bar);
};

// === CONSTRUCTOR ===
CStructureAnalyzer::CStructureAnalyzer()
{
   // Default parameters for XAUUSD M15
   m_swing_strength = 3;           // 3 bars each side
   m_equal_tolerance = 5.0;        // 5 pips tolerance
   m_break_buffer = 2.0;           // 2 pips buffer for break
   m_lookback_bars = 100;
   m_min_swing_distance = 5;       // Min 5 bars between swings
   
   // MTF default timeframe
   m_current_tf = PERIOD_M15;
   
   // Array limits
   m_max_swings = 50;
   m_max_breaks = 20;
   m_high_count = 0;
   m_low_count = 0;
   m_break_count = 0;
   
   ArrayResize(m_swing_highs, m_max_swings);
   ArrayResize(m_swing_lows, m_max_swings);
   ArrayResize(m_breaks, m_max_breaks);
   
   // Initialize state
   ZeroMemory(m_state);
   ZeroMemory(m_htf_state);
   ZeroMemory(m_mtf_state);
   ZeroMemory(m_ltf_state);
   m_state.bias = BIAS_RANGING;
   m_htf_state.bias = BIAS_RANGING;
   m_mtf_state.bias = BIAS_RANGING;
   m_ltf_state.bias = BIAS_RANGING;
}

CStructureAnalyzer::~CStructureAnalyzer()
{
   ArrayFree(m_swing_highs);
   ArrayFree(m_swing_lows);
   ArrayFree(m_breaks);
}

// === MAIN ANALYSIS ===
SStructureState CStructureAnalyzer::AnalyzeStructure(string symbol = NULL, int tf = 0)
{
   if(symbol == NULL) symbol = _Symbol;
   
   // Get rate data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, PERIOD_CURRENT, 0, m_lookback_bars, rates);
   
   if(copied < m_lookback_bars / 2)
   {
      Print("CStructureAnalyzer: Insufficient data");
      return m_state;
   }
   
   // Reset counts
   m_high_count = 0;
   m_low_count = 0;
   m_break_count = 0;
   
   // Detect swing points
   DetectSwingPoints(rates, copied);
   
   // Determine bias before classifying breaks
   m_state.bias = DetermineBias();
   
   // Detect structure breaks (uses current bias for BOS/CHoCH tagging)
   DetectBreaks(rates, copied);
   
   // Re-evaluate bias after potential breaks
   m_state.bias = DetermineBias();
   
   // Calculate premium/discount
   CalculatePremiumDiscount(rates[0].close);
   
   // Quality metrics
   m_state.structure_quality = CalculateStructureQuality();
   m_state.trend_strength = CalculateTrendStrength();
   m_state.last_update = TimeCurrent();
   
   return m_state;
}

// === SWING DETECTION ===
bool CStructureAnalyzer::DetectSwingPoints(const MqlRates &rates[], int count)
{
   int min_index = m_swing_strength;
   int max_index = count - m_swing_strength - 1;
   
   // Detect swing highs
   for(int i = min_index; i < max_index; i++)
   {
      if(IsSwingHigh(rates, i, m_swing_strength, m_swing_strength))
      {
         SSwingPoint swing;
         ZeroMemory(swing);
         swing.time = rates[i].time;
         swing.price = rates[i].high;
         swing.bar_index = i;
         swing.is_high = true;
         swing.is_valid = true;
         swing.is_broken = false;
         
         // Classify relative to previous high
         if(m_high_count > 0)
            swing.type = ClassifyHigh(swing, m_swing_highs[m_high_count - 1]);
         else
            swing.type = STRUCT_HH; // First high
         
         AddSwingHigh(swing);
      }
   }
   
   // Detect swing lows
   for(int i = min_index; i < max_index; i++)
   {
      if(IsSwingLow(rates, i, m_swing_strength, m_swing_strength))
      {
         SSwingPoint swing;
         ZeroMemory(swing);
         swing.time = rates[i].time;
         swing.price = rates[i].low;
         swing.bar_index = i;
         swing.is_high = false;
         swing.is_valid = true;
         swing.is_broken = false;
         
         // Classify relative to previous low
         if(m_low_count > 0)
            swing.type = ClassifyLow(swing, m_swing_lows[m_low_count - 1]);
         else
            swing.type = STRUCT_HL; // First low
         
         AddSwingLow(swing);
      }
   }
   
   // Store last swings in state
   if(m_high_count >= 2)
   {
      m_state.last_high = m_swing_highs[m_high_count - 1];
      m_state.prev_high = m_swing_highs[m_high_count - 2];
   }
   else if(m_high_count == 1)
   {
      m_state.last_high = m_swing_highs[0];
   }
   
   if(m_low_count >= 2)
   {
      m_state.last_low = m_swing_lows[m_low_count - 1];
      m_state.prev_low = m_swing_lows[m_low_count - 2];
   }
   else if(m_low_count == 1)
   {
      m_state.last_low = m_swing_lows[0];
   }
   
   return (m_high_count > 0 && m_low_count > 0);
}

bool CStructureAnalyzer::IsSwingHigh(const MqlRates &rates[], int index, int left, int right)
{
   double high = rates[index].high;
   
   // Check left side
   for(int i = 1; i <= left; i++)
   {
      if(index + i >= ArraySize(rates)) return false;
      if(rates[index + i].high >= high) return false;
   }
   
   // Check right side
   for(int i = 1; i <= right; i++)
   {
      if(index - i < 0) return false;
      if(rates[index - i].high >= high) return false;
   }
   
   return true;
}

bool CStructureAnalyzer::IsSwingLow(const MqlRates &rates[], int index, int left, int right)
{
   double low = rates[index].low;
   
   // Check left side
   for(int i = 1; i <= left; i++)
   {
      if(index + i >= ArraySize(rates)) return false;
      if(rates[index + i].low <= low) return false;
   }
   
   // Check right side
   for(int i = 1; i <= right; i++)
   {
      if(index - i < 0) return false;
      if(rates[index - i].low <= low) return false;
   }
   
   return true;
}

// === STRUCTURE CLASSIFICATION ===
ENUM_STRUCTURE_TYPE CStructureAnalyzer::ClassifyHigh(const SSwingPoint &current, const SSwingPoint &previous)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tolerance = m_equal_tolerance * point;
   
   double diff = current.price - previous.price;
   
   if(MathAbs(diff) <= tolerance)
      return STRUCT_EQH;
   else if(diff > 0)
      return STRUCT_HH;
   else
      return STRUCT_LH;
}

ENUM_STRUCTURE_TYPE CStructureAnalyzer::ClassifyLow(const SSwingPoint &current, const SSwingPoint &previous)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tolerance = m_equal_tolerance * point;
   
   double diff = current.price - previous.price;
   
   if(MathAbs(diff) <= tolerance)
      return STRUCT_EQL;
   else if(diff > 0)
      return STRUCT_HL;
   else
      return STRUCT_LL;
}

ENUM_MARKET_BIAS CStructureAnalyzer::DetermineBias()
{
   if(m_high_count < 2 || m_low_count < 2)
      return BIAS_RANGING;
   
   // Get recent structure types
   ENUM_STRUCTURE_TYPE last_high_type = m_state.last_high.type;
   ENUM_STRUCTURE_TYPE last_low_type = m_state.last_low.type;
   
   // Classic trend structure
   // Bullish: HH + HL
   if((last_high_type == STRUCT_HH || last_high_type == STRUCT_EQH) &&
      (last_low_type == STRUCT_HL || last_low_type == STRUCT_EQL))
   {
      return BIAS_BULLISH;
   }
   
   // Bearish: LH + LL
   if((last_high_type == STRUCT_LH || last_high_type == STRUCT_EQH) &&
      (last_low_type == STRUCT_LL || last_low_type == STRUCT_EQL))
   {
      return BIAS_BEARISH;
   }
   
   // Check for transition (mixed signals)
   if((last_high_type == STRUCT_LH && last_low_type == STRUCT_HL) ||
      (last_high_type == STRUCT_HH && last_low_type == STRUCT_LL))
   {
      return BIAS_TRANSITION;
   }
   
   return BIAS_RANGING;
}

// === BREAK DETECTION ===
bool CStructureAnalyzer::DetectBreaks(const MqlRates &rates[], int count)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double buffer = m_break_buffer * point;
   
   // Check if recent price broke any swing points
   for(int bar = 0; bar < MathMin(10, count); bar++)
   {
      double high = rates[bar].high;
      double low = rates[bar].low;
      double close = rates[bar].close;
      
      // Check break of swing lows (bearish break)
      for(int i = 0; i < m_low_count; i++)
      {
         if(!m_swing_lows[i].is_broken && m_swing_lows[i].bar_index > bar)
         {
            if(low < m_swing_lows[i].price - buffer)
            {
               // Potential break detected
               SStructureBreak brk;
               ZeroMemory(brk);
               brk.time = rates[bar].time;
               brk.break_price = low;
               brk.swing_price = m_swing_lows[i].price;
               brk.type = ClassifyBreak(low, m_swing_lows[i], false);
               brk.displacement = GetDisplacement(rates, m_swing_lows[i].bar_index, bar);
               brk.is_valid = ValidateBreak(brk, rates, bar);
               
               if(brk.is_valid)
               {
                  brk.new_bias = BIAS_BEARISH;
                  brk.strength = (int)(MathMin(100, brk.displacement / point / 10));
                  AddBreak(brk);
                  m_swing_lows[i].is_broken = true;
                  m_swing_lows[i].break_time = rates[bar].time;
               }
            }
         }
      }
      
      // Check break of swing highs (bullish break)
      for(int i = 0; i < m_high_count; i++)
      {
         if(!m_swing_highs[i].is_broken && m_swing_highs[i].bar_index > bar)
         {
            if(high > m_swing_highs[i].price + buffer)
            {
               SStructureBreak brk;
               ZeroMemory(brk);
               brk.time = rates[bar].time;
               brk.break_price = high;
               brk.swing_price = m_swing_highs[i].price;
               brk.type = ClassifyBreak(high, m_swing_highs[i], true);
               brk.displacement = GetDisplacement(rates, m_swing_highs[i].bar_index, bar);
               brk.is_valid = ValidateBreak(brk, rates, bar);
               
               if(brk.is_valid)
               {
                  brk.new_bias = BIAS_BULLISH;
                  brk.strength = (int)(MathMin(100, brk.displacement / point / 10));
                  AddBreak(brk);
                  m_swing_highs[i].is_broken = true;
                  m_swing_highs[i].break_time = rates[bar].time;
               }
            }
         }
      }
   }
   
   // Store last break in state
   if(m_break_count > 0)
   {
      m_state.last_break = m_breaks[m_break_count - 1];
      
      // Count consecutive BOS
      m_state.bos_count = 0;
      m_state.choch_count = 0;
      for(int i = m_break_count - 1; i >= 0 && i >= m_break_count - 5; i--)
      {
         if(m_breaks[i].type == BREAK_BOS) m_state.bos_count++;
         if(m_breaks[i].type == BREAK_CHOCH) m_state.choch_count++;
      }
   }
   
   return (m_break_count > 0);
}

ENUM_STRUCTURE_BREAK CStructureAnalyzer::ClassifyBreak(double break_price, const SSwingPoint &broken_swing, bool is_high)
{
   // Determine if this is BOS or CHoCH based on current bias
   if(is_high)
   {
      // Breaking a high
      if(m_state.bias == BIAS_BULLISH)
         return BREAK_BOS;    // Continuation
      else
         return BREAK_CHOCH;  // Reversal
   }
   else
   {
      // Breaking a low
      if(m_state.bias == BIAS_BEARISH)
         return BREAK_BOS;    // Continuation
      else
         return BREAK_CHOCH;  // Reversal
   }
}

bool CStructureAnalyzer::ValidateBreak(const SStructureBreak &brk, const MqlRates &rates[], int break_bar)
{
   // Validate with candle close
   if(break_bar >= ArraySize(rates)) return false;
   
   double close = rates[break_bar].close;
   
   // For bearish break, close should be below swing low
   if(brk.new_bias == BIAS_BEARISH)
      return (close < brk.swing_price);
   
   // For bullish break, close should be above swing high
   if(brk.new_bias == BIAS_BULLISH)
      return (close > brk.swing_price);
   
   return false;
}

// === PREMIUM/DISCOUNT ===
void CStructureAnalyzer::CalculatePremiumDiscount(double current_price)
{
   if(!m_state.last_high.is_valid || !m_state.last_low.is_valid)
   {
      m_state.in_premium = false;
      m_state.in_discount = false;
      return;
   }
   
   double range_high = m_state.last_high.price;
   double range_low = m_state.last_low.price;
   
   m_state.equilibrium = (range_high + range_low) / 2;
   m_state.in_premium = (current_price > m_state.equilibrium);
   m_state.in_discount = (current_price < m_state.equilibrium);
}

// === QUALITY SCORING ===
double CStructureAnalyzer::CalculateStructureQuality()
{
   double score = 50.0; // Base score
   
   // More swings = better structure definition
   if(m_high_count >= 3 && m_low_count >= 3) score += 15;
   else if(m_high_count >= 2 && m_low_count >= 2) score += 10;
   
   // Clear bias = higher quality
   if(m_state.bias == BIAS_BULLISH || m_state.bias == BIAS_BEARISH)
      score += 20;
   else if(m_state.bias == BIAS_TRANSITION)
      score -= 10;
   
   // Recent BOS = structure confirmed
   if(m_state.bos_count >= 2) score += 15;
   else if(m_state.bos_count >= 1) score += 10;
   
   // CHoCH reduces quality (uncertainty)
   score -= m_state.choch_count * 5;
   
   return MathMax(0, MathMin(100, score));
}

double CStructureAnalyzer::CalculateTrendStrength()
{
   if(m_state.bias == BIAS_RANGING || m_state.bias == BIAS_TRANSITION)
      return 0;
   
   double strength = 30.0; // Base
   
   // Consecutive BOS increases strength
   strength += m_state.bos_count * 15;
   
   // Higher quality = stronger trend
   strength += m_state.structure_quality * 0.3;
   
   // Premium/discount alignment
   if((m_state.bias == BIAS_BULLISH && m_state.in_discount) ||
      (m_state.bias == BIAS_BEARISH && m_state.in_premium))
   {
      strength += 10; // Good entry zone
   }
   
   return MathMin(100, strength);
}

// === HELPER METHODS ===
void CStructureAnalyzer::AddSwingHigh(const SSwingPoint &swing)
{
   if(m_high_count >= m_max_swings)
   {
      // Shift array
      for(int i = 0; i < m_max_swings - 1; i++)
         m_swing_highs[i] = m_swing_highs[i + 1];
      m_high_count--;
   }
   m_swing_highs[m_high_count] = swing;
   m_high_count++;
}

void CStructureAnalyzer::AddSwingLow(const SSwingPoint &swing)
{
   if(m_low_count >= m_max_swings)
   {
      for(int i = 0; i < m_max_swings - 1; i++)
         m_swing_lows[i] = m_swing_lows[i + 1];
      m_low_count--;
   }
   m_swing_lows[m_low_count] = swing;
   m_low_count++;
}

void CStructureAnalyzer::AddBreak(const SStructureBreak &brk)
{
   if(m_break_count >= m_max_breaks)
   {
      for(int i = 0; i < m_max_breaks - 1; i++)
         m_breaks[i] = m_breaks[i + 1];
      m_break_count--;
   }
   m_breaks[m_break_count] = brk;
   m_break_count++;
}

double CStructureAnalyzer::GetDisplacement(const MqlRates &rates[], int start_bar, int end_bar)
{
   if(start_bar >= ArraySize(rates) || end_bar >= ArraySize(rates))
      return 0;
   
   return MathAbs(rates[end_bar].close - rates[start_bar].close);
}

// === UTILITY ===
string CStructureAnalyzer::BiasToString(ENUM_MARKET_BIAS bias)
{
   switch(bias)
   {
      case BIAS_BULLISH:    return "BULLISH";
      case BIAS_BEARISH:    return "BEARISH";
      case BIAS_RANGING:    return "RANGING";
      case BIAS_TRANSITION: return "TRANSITION";
      default:              return "UNKNOWN";
   }
}

string CStructureAnalyzer::BreakToString(ENUM_STRUCTURE_BREAK brk)
{
   switch(brk)
   {
      case BREAK_NONE:  return "NONE";
      case BREAK_BOS:   return "BOS";
      case BREAK_CHOCH: return "CHOCH";
      case BREAK_SWEEP: return "SWEEP";
      default:          return "UNKNOWN";
   }
}

string CStructureAnalyzer::StructureTypeToString(ENUM_STRUCTURE_TYPE type)
{
   switch(type)
   {
      case STRUCT_HH:  return "HH";
      case STRUCT_HL:  return "HL";
      case STRUCT_LH:  return "LH";
      case STRUCT_LL:  return "LL";
      case STRUCT_EQH: return "EQH";
      case STRUCT_EQL: return "EQL";
      default:         return "UNKNOWN";
   }
}

// === MTF ANALYSIS METHODS (NEW v3.20) ===

//+------------------------------------------------------------------+
//| Analyze structure on all MTF timeframes                          |
//+------------------------------------------------------------------+
void CStructureAnalyzer::AnalyzeMTFStructure(string symbol = NULL)
{
   if(symbol == NULL) symbol = _Symbol;
   
   // Analyze H1 (HTF - Direction)
   MqlRates htf_rates[];
   ArraySetAsSeries(htf_rates, true);
   int htf_copied = CopyRates(symbol, STRUCT_TF_HTF, 0, m_lookback_bars, htf_rates);
   if(htf_copied >= m_lookback_bars / 2)
   {
      ZeroMemory(m_state);
      m_state.bias = BIAS_RANGING;
      // Temporarily store current state
      m_high_count = 0;
      m_low_count = 0;
      m_break_count = 0;
      
      // Analyze H1
      DetectSwingPoints(htf_rates, htf_copied);
      DetectBreaks(htf_rates, htf_copied);
      m_htf_state.bias = DetermineBias();
      m_htf_state.last_high = m_state.last_high;
      m_htf_state.last_low = m_state.last_low;
      m_htf_state.last_break = m_state.last_break;
      m_htf_state.bos_count = m_state.bos_count;
      m_htf_state.structure_quality = CalculateStructureQuality();
      m_htf_state.trend_strength = CalculateTrendStrength();
      CalculatePremiumDiscount(htf_rates[0].close);
      m_htf_state.equilibrium = m_state.equilibrium;
      m_htf_state.in_premium = m_state.in_premium;
      m_htf_state.in_discount = m_state.in_discount;
      m_htf_state.last_update = TimeCurrent();
   }
   
   // Analyze M15 (MTF - Structure)
   MqlRates mtf_rates[];
   ArraySetAsSeries(mtf_rates, true);
   int mtf_copied = CopyRates(symbol, STRUCT_TF_MTF, 0, m_lookback_bars, mtf_rates);
   if(mtf_copied >= m_lookback_bars / 2)
   {
      ZeroMemory(m_state);
      m_state.bias = BIAS_RANGING;
      m_high_count = 0;
      m_low_count = 0;
      m_break_count = 0;
      
      DetectSwingPoints(mtf_rates, mtf_copied);
      DetectBreaks(mtf_rates, mtf_copied);
      m_mtf_state.bias = DetermineBias();
      m_mtf_state.last_high = m_state.last_high;
      m_mtf_state.last_low = m_state.last_low;
      m_mtf_state.last_break = m_state.last_break;
      m_mtf_state.bos_count = m_state.bos_count;
      m_mtf_state.structure_quality = CalculateStructureQuality();
      m_mtf_state.trend_strength = CalculateTrendStrength();
      CalculatePremiumDiscount(mtf_rates[0].close);
      m_mtf_state.equilibrium = m_state.equilibrium;
      m_mtf_state.in_premium = m_state.in_premium;
      m_mtf_state.in_discount = m_state.in_discount;
      m_mtf_state.htf_bias = m_htf_state.bias;  // Store HTF bias for reference
      m_mtf_state.last_update = TimeCurrent();
   }
   
   // Analyze M5 (LTF - Execution)
   MqlRates ltf_rates[];
   ArraySetAsSeries(ltf_rates, true);
   int ltf_copied = CopyRates(symbol, STRUCT_TF_LTF, 0, m_lookback_bars, ltf_rates);
   if(ltf_copied >= m_lookback_bars / 2)
   {
      ZeroMemory(m_state);
      m_state.bias = BIAS_RANGING;
      m_high_count = 0;
      m_low_count = 0;
      m_break_count = 0;
      
      DetectSwingPoints(ltf_rates, ltf_copied);
      DetectBreaks(ltf_rates, ltf_copied);
      m_ltf_state.bias = DetermineBias();
      m_ltf_state.last_high = m_state.last_high;
      m_ltf_state.last_low = m_state.last_low;
      m_ltf_state.last_break = m_state.last_break;
      m_ltf_state.bos_count = m_state.bos_count;
      m_ltf_state.structure_quality = CalculateStructureQuality();
      m_ltf_state.trend_strength = CalculateTrendStrength();
      CalculatePremiumDiscount(ltf_rates[0].close);
      m_ltf_state.equilibrium = m_state.equilibrium;
      m_ltf_state.in_premium = m_state.in_premium;
      m_ltf_state.in_discount = m_state.in_discount;
      m_ltf_state.htf_bias = m_htf_state.bias;
      m_ltf_state.last_update = TimeCurrent();
   }
   
   // Set main state to MTF (M15) for backward compatibility
   m_state = m_mtf_state;
}

//+------------------------------------------------------------------+
//| Check if all MTF timeframes are aligned                          |
//+------------------------------------------------------------------+
bool CStructureAnalyzer::IsMTFAligned()
{
   // All must be bullish or all must be bearish
   if(m_htf_state.bias == BIAS_BULLISH && 
      (m_mtf_state.bias == BIAS_BULLISH || m_mtf_state.bias == BIAS_TRANSITION) &&
      (m_ltf_state.bias == BIAS_BULLISH || m_ltf_state.bias == BIAS_TRANSITION))
   {
      return true;  // Bullish alignment
   }
   
   if(m_htf_state.bias == BIAS_BEARISH && 
      (m_mtf_state.bias == BIAS_BEARISH || m_mtf_state.bias == BIAS_TRANSITION) &&
      (m_ltf_state.bias == BIAS_BEARISH || m_ltf_state.bias == BIAS_TRANSITION))
   {
      return true;  // Bearish alignment
   }
   
   return false;
}
//+------------------------------------------------------------------+
