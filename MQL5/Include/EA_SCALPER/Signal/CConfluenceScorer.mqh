//+------------------------------------------------------------------+
//|                                            CConfluenceScorer.mqh |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|              Unified Confluence Scoring: Integrates All Detectors |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property strict

// Include core definitions first (has ENUM_SIGNAL_TYPE)
#include "../Core/Definitions.mqh"

// Include all analyzers
#include "../Analysis/CRegimeDetector.mqh"
#include "../Analysis/CStructureAnalyzer.mqh"
#include "../Analysis/CLiquiditySweepDetector.mqh"
#include "../Analysis/CAMDCycleTracker.mqh"
#include "../Analysis/EliteOrderBlock.mqh"
#include "../Analysis/EliteFVG.mqh"

// === SIGNAL QUALITY ENUMERATION ===
enum ENUM_SIGNAL_QUALITY
{
   QUALITY_INVALID = 0,    // Do not trade
   QUALITY_LOW = 1,        // Weak setup
   QUALITY_MEDIUM = 2,     // Average setup
   QUALITY_HIGH = 3,       // Strong setup
   QUALITY_ELITE = 4       // Institutional-grade setup
};

// === CONFLUENCE TIER THRESHOLDS ===
// Based on Blueprint v3.0 scoring system
#define TIER_S_MIN      90    // Elite setups (90-100)
#define TIER_A_MIN      80    // High quality (80-89)
#define TIER_B_MIN      70    // Tradeable (70-79)
#define TIER_C_MIN      60    // Marginal (60-69)
#define TIER_INVALID    60    // Below 60 = no trade

// === CONFLUENCE RESULT STRUCTURE ===
struct SConfluenceResult
{
   // Direction
   ENUM_SIGNAL_TYPE direction;
   ENUM_SIGNAL_QUALITY   quality;
   
   // Main score (0-100)
   double                total_score;
   
   // Component scores (0-100 each)
   double                structure_score;    // Market structure (HH/HL, BOS, CHoCH)
   double                regime_score;       // Hurst/Entropy regime
   double                sweep_score;        // Liquidity sweep detection
   double                amd_score;          // AMD cycle phase
   double                ob_score;           // Order block proximity
   double                fvg_score;          // FVG proximity
   double                premium_discount;   // Premium/Discount zone alignment
   
   // Score adjustments
   int                   regime_adjustment;  // From regime detector
   int                   confluence_bonus;   // Multiple confluences bonus
   
   // Counts
   int                   bullish_factors;
   int                   bearish_factors;
   int                   total_confluences;
   
   // Trade setup
   double                entry_price;
   double                stop_loss;
   double                take_profit_1;
   double                take_profit_2;
   double                risk_reward;
   double                position_size_mult; // From regime (0, 0.5, or 1.0)
   
   // Metadata
   string                signal_reason;
   datetime              timestamp;
   bool                  is_valid;
};

// === WEIGHT CONFIGURATION ===
struct SConfluenceWeights
{
   double w_structure;      // Structure weight (default 25%)
   double w_regime;         // Regime weight (default 20%)
   double w_sweep;          // Sweep weight (default 15%)
   double w_amd;            // AMD weight (default 15%)
   double w_ob;             // Order Block weight (default 10%)
   double w_fvg;            // FVG weight (default 10%)
   double w_zone;           // Premium/Discount weight (default 5%)
};

// === CONFLUENCE SCORER CLASS ===
class CConfluenceScorer
{
private:
   // Component analyzers (references, not owned)
   CRegimeDetector*         m_regime;
   CStructureAnalyzer*      m_structure;
   CLiquiditySweepDetector* m_sweep;
   CAMDCycleTracker*        m_amd;
   CEliteOrderBlockDetector* m_ob_detector;
   CEliteFVGDetector*        m_fvg_detector;
   
   // Weights
   SConfluenceWeights       m_weights;
   
   // Thresholds
   int                      m_min_score;         // Minimum score to trade
   int                      m_min_confluences;   // Minimum confluences required
   
   // Last result cache
   SConfluenceResult        m_last_result;
   datetime                 m_last_calculation;
   int                      m_cache_seconds;
   
   // OB/FVG proximity settings
   double                   m_ob_proximity_pips;
   double                   m_fvg_proximity_pips;
   
public:
   CConfluenceScorer();
   ~CConfluenceScorer();
   
   // Attach analyzers
   void AttachRegimeDetector(CRegimeDetector* detector) { m_regime = detector; }
   void AttachStructureAnalyzer(CStructureAnalyzer* analyzer) { m_structure = analyzer; }
   void AttachSweepDetector(CLiquiditySweepDetector* detector) { m_sweep = detector; }
   void AttachAMDTracker(CAMDCycleTracker* tracker) { m_amd = tracker; }
   void AttachOBDetector(CEliteOrderBlockDetector* detector) { m_ob_detector = detector; }
   void AttachFVGDetector(CEliteFVGDetector* detector) { m_fvg_detector = detector; }
   
   // Configuration
   void SetWeights(double structure, double regime, double sweep, double amd, double ob, double fvg, double zone);
   void SetMinScore(int score) { m_min_score = score; }
   void SetMinConfluences(int count) { m_min_confluences = count; }
   void SetProximityPips(double ob_pips, double fvg_pips);
   void SetCacheSeconds(int seconds) { m_cache_seconds = seconds; }
   
   // Main scoring
   SConfluenceResult CalculateConfluence(string symbol = NULL);
   SConfluenceResult GetLastResult() { return m_last_result; }
   
   // Component scoring (can be called individually)
   double ScoreStructure();
   double ScoreRegime();
   double ScoreSweep();
   double ScoreAMD();
   double ScoreOBProximity(double price);
   double ScoreFVGProximity(double price);
   double ScorePremiumDiscount();
   
   // Direction determination
   ENUM_SIGNAL_TYPE DetermineDirection();
   ENUM_SIGNAL_QUALITY ClassifyQuality(double score);
   
   // Trade setup calculation
   void CalculateTradeSetup(SConfluenceResult &result, double current_price);
   
   // Validation
   bool IsValidSetup(const SConfluenceResult &result);
   bool PassesRegimeFilter();
   bool PassesStructureFilter();
   
   // Utility
   string QualityToString(ENUM_SIGNAL_QUALITY quality);
   string DirectionToString(ENUM_SIGNAL_TYPE dir);
   void PrintResult(const SConfluenceResult &result);
   
private:
   double NormalizeScore(double raw_score, double max_value);
   int CountConfluences(const SConfluenceResult &result);
   string BuildSignalReason(const SConfluenceResult &result);
};

// === CONSTRUCTOR ===
CConfluenceScorer::CConfluenceScorer()
{
   // Initialize pointers
   m_regime = NULL;
   m_structure = NULL;
   m_sweep = NULL;
   m_amd = NULL;
   m_ob_detector = NULL;
   m_fvg_detector = NULL;
   
   // Default weights (total = 100%)
   m_weights.w_structure = 0.25;   // 25%
   m_weights.w_regime = 0.20;      // 20%
   m_weights.w_sweep = 0.15;       // 15%
   m_weights.w_amd = 0.15;         // 15%
   m_weights.w_ob = 0.10;          // 10%
   m_weights.w_fvg = 0.10;         // 10%
   m_weights.w_zone = 0.05;        // 5%
   
   // Thresholds
   m_min_score = TIER_B_MIN;       // 70 minimum
   m_min_confluences = 3;          // At least 3 factors
   
   // Cache
   m_cache_seconds = 15;
   m_last_calculation = 0;
   
   // Proximity settings
   m_ob_proximity_pips = 30.0;
   m_fvg_proximity_pips = 20.0;
   
   // Initialize last result
   ZeroMemory(m_last_result);
}

CConfluenceScorer::~CConfluenceScorer()
{
   // Don't delete analyzers - they're owned by the EA
}

// === CONFIGURATION ===
void CConfluenceScorer::SetWeights(double structure, double regime, double sweep, double amd, double ob, double fvg, double zone)
{
   m_weights.w_structure = structure;
   m_weights.w_regime = regime;
   m_weights.w_sweep = sweep;
   m_weights.w_amd = amd;
   m_weights.w_ob = ob;
   m_weights.w_fvg = fvg;
   m_weights.w_zone = zone;
}

void CConfluenceScorer::SetProximityPips(double ob_pips, double fvg_pips)
{
   m_ob_proximity_pips = ob_pips;
   m_fvg_proximity_pips = fvg_pips;
}

// === MAIN SCORING ===
SConfluenceResult CConfluenceScorer::CalculateConfluence(string symbol = NULL)
{
   // Check cache
   if(TimeCurrent() - m_last_calculation < m_cache_seconds && m_last_result.is_valid)
      return m_last_result;
   
   if(symbol == NULL) symbol = _Symbol;
   
   SConfluenceResult result;
   ZeroMemory(result);
   result.timestamp = TimeCurrent();
   
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   
   // === SCORE EACH COMPONENT ===
   
   // 1. Structure score
   result.structure_score = ScoreStructure();
   
   // 2. Regime score
   result.regime_score = ScoreRegime();
   if(m_regime != NULL)
   {
      SRegimeAnalysis regime = m_regime.GetLastAnalysis();
      result.regime_adjustment = regime.score_adjustment;
      result.position_size_mult = regime.size_multiplier;
   }
   
   // 3. Sweep score
   result.sweep_score = ScoreSweep();
   
   // 4. AMD score
   result.amd_score = ScoreAMD();
   
   // 5. OB proximity score
   result.ob_score = ScoreOBProximity(current_price);
   
   // 6. FVG proximity score
   result.fvg_score = ScoreFVGProximity(current_price);
   
   // 7. Premium/Discount score
   result.premium_discount = ScorePremiumDiscount();
   
   // === CALCULATE WEIGHTED TOTAL ===
   result.total_score = 
      result.structure_score * m_weights.w_structure +
      result.regime_score * m_weights.w_regime +
      result.sweep_score * m_weights.w_sweep +
      result.amd_score * m_weights.w_amd +
      result.ob_score * m_weights.w_ob +
      result.fvg_score * m_weights.w_fvg +
      result.premium_discount * m_weights.w_zone;
   
   // Apply regime adjustment
   result.total_score += result.regime_adjustment;
   
   // Count confluences and apply bonus
   result.total_confluences = CountConfluences(result);
   if(result.total_confluences >= 5)
      result.confluence_bonus = 10;
   else if(result.total_confluences >= 4)
      result.confluence_bonus = 5;
   else
      result.confluence_bonus = 0;
   
   result.total_score += result.confluence_bonus;
   
   // Clamp score
   result.total_score = MathMax(0, MathMin(100, result.total_score));
   
   // === DETERMINE DIRECTION ===
   result.direction = DetermineDirection();
   
   // === CLASSIFY QUALITY ===
   result.quality = ClassifyQuality(result.total_score);
   
   // === VALIDATE ===
   result.is_valid = IsValidSetup(result);
   
   // === CALCULATE TRADE SETUP ===
   if(result.is_valid)
   {
      CalculateTradeSetup(result, current_price);
      result.signal_reason = BuildSignalReason(result);
   }
   
   // Cache result
   m_last_result = result;
   m_last_calculation = TimeCurrent();
   
   return result;
}

// === COMPONENT SCORING ===
double CConfluenceScorer::ScoreStructure()
{
   if(m_structure == NULL) return 50.0;
   
   SStructureState state = m_structure.GetState();
   double score = state.structure_quality;
   
   // Bonus for clear bias
   if(state.bias == BIAS_BULLISH || state.bias == BIAS_BEARISH)
      score += 10;
   
   // Bonus for recent BOS
   if(m_structure.HasRecentBOS())
      score += 15;
   
   // Penalty for CHoCH (uncertainty)
   if(m_structure.HasRecentCHoCH())
      score -= 10;
   
   return MathMax(0, MathMin(100, score));
}

double CConfluenceScorer::ScoreRegime()
{
   if(m_regime == NULL) return 50.0;
   
   SRegimeAnalysis regime = m_regime.GetLastAnalysis();
   
   if(!regime.is_valid) return 0;
   
   // Score based on regime type
   double score = 50.0;
   
   switch(regime.regime)
   {
      case REGIME_PRIME_TRENDING:
      case REGIME_PRIME_REVERTING:
         score = 100.0;
         break;
      case REGIME_NOISY_TRENDING:
      case REGIME_NOISY_REVERTING:
         score = 70.0;
         break;
      case REGIME_RANDOM_WALK:
         score = 0.0;  // Block trading
         break;
      default:
         score = 30.0;
   }
   
   // Add confidence factor
   score *= regime.confidence;
   
   return score;
}

double CConfluenceScorer::ScoreSweep()
{
   if(m_sweep == NULL) return 0;
   
   if(!m_sweep.HasRecentSweep(10)) return 0;
   
   SSweepEvent sweep = m_sweep.GetMostRecentSweep();
   // Score based on sweep characteristics
   double score = 50.0;  // Base score
   if(sweep.is_valid_sweep) score += 20;
   if(sweep.has_rejection) score += 15;
   if(sweep.returned_inside) score += 15;
   return MathMin(100, score);
}

double CConfluenceScorer::ScoreAMD()
{
   if(m_amd == NULL) return 0;
   
   SAMDCycle cycle = m_amd.GetCurrentCycle();
   
   if(!cycle.is_active) return 0;
   
   // Score based on quality
   double score = (double)cycle.quality_score;
   
   // Distribution phase is ideal for entry
   if(cycle.current_phase == AMD_PHASE_DISTRIBUTION)
      score += 20;
   
   // Confirmed manipulation is high probability
   if(cycle.manipulation.is_valid)
      score += 15;
   
   return MathMax(0, MathMin(100, score));
}

double CConfluenceScorer::ScoreOBProximity(double price)
{
   if(m_ob_detector == NULL) return 50.0;
   
   // Find nearest active OB (either side)
   SAdvancedOrderBlock best;
   bool found = false;
   double best_dist = DBL_MAX;
   SAdvancedOrderBlock candidate;
   
   if(m_ob_detector->GetNearestOB(OB_BULLISH, candidate))
   {
      double mid = (candidate.high_price + candidate.low_price) / 2.0;
      double dist = MathAbs(price - mid);
      if(dist < best_dist) { best_dist = dist; best = candidate; found = true; }
   }
   if(m_ob_detector->GetNearestOB(OB_BEARISH, candidate))
   {
      double mid = (candidate.high_price + candidate.low_price) / 2.0;
      double dist = MathAbs(price - mid);
      if(dist < best_dist) { best_dist = dist; best = candidate; found = true; }
   }
   
   if(!found) return 30.0; // mild penalty if no OB context
   
   // Normalize distance by ATR(H1)
   double atr[];
   ArrayResize(atr, 1);
   int atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(atr_handle == INVALID_HANDLE || CopyBuffer(atr_handle, 0, 0, 1, atr) <= 0)
   {
      if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
      return 50.0;
   }
   IndicatorRelease(atr_handle);
   
   double distance_atr = (atr[0] > 0) ? best_dist / atr[0] : 5.0;
   
   double score = 0;
   if(distance_atr <= 0.3) score = 100;
   else if(distance_atr <= 0.5) score = 85 + (0.5 - distance_atr) * 75;
   else if(distance_atr <= 1.0) score = 60 + (1.0 - distance_atr) * 50;
   else if(distance_atr <= 2.0) score = (2.0 - distance_atr) * 60;
   else score = 20;
   
   // Weight by OB probability/quality
   score *= (best.probability_score / 100.0);
   return MathMin(100, MathMax(0, score));
}

double CConfluenceScorer::ScoreFVGProximity(double price)
{
   if(m_fvg_detector == NULL) return 50.0;
   
   // Choose nearest bullish/bearish FVG
   SEliteFairValueGap best;
   bool found = false;
   double best_dist = DBL_MAX;
   SEliteFairValueGap candidate;
   
   if(m_fvg_detector->GetNearestFVG(FVG_BULLISH, candidate))
   {
      double mid = (candidate.upper_level + candidate.lower_level) / 2.0;
      double dist = MathAbs(price - mid);
      if(dist < best_dist) { best_dist = dist; best = candidate; found = true; }
   }
   if(m_fvg_detector->GetNearestFVG(FVG_BEARISH, candidate))
   {
      double mid = (candidate.upper_level + candidate.lower_level) / 2.0;
      double dist = MathAbs(price - mid);
      if(dist < best_dist) { best_dist = dist; best = candidate; found = true; }
   }
   
   if(!found) return 30.0; // no FVG nearby
   
   double atr[];
   ArrayResize(atr, 1);
   int atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(atr_handle == INVALID_HANDLE || CopyBuffer(atr_handle, 0, 0, 1, atr) <= 0)
   {
      if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
      return 50.0;
   }
   IndicatorRelease(atr_handle);
   
   double distance_atr = (atr[0] > 0) ? best_dist / atr[0] : 5.0;
   
   double score = 0;
   if(distance_atr <= 0.3) score = 100;
   else if(distance_atr <= 0.5) score = 85 + (0.5 - distance_atr) * 75;
   else if(distance_atr <= 1.0) score = 60 + (1.0 - distance_atr) * 50;
   else if(distance_atr <= 2.0) score = (2.0 - distance_atr) * 60;
   else score = 20;
   
   // Weight by FVG quality/freshness
   score *= (best.quality_score / 100.0);
   if(best.is_fresh) score *= 1.1;
   score *= best.time_decay_factor;
   
   return MathMin(100, MathMax(0, score));
}

double CConfluenceScorer::ScorePremiumDiscount()
{
   if(m_structure == NULL) return 50.0;
   
   SStructureState state = m_structure.GetState();
   
   // Ideal: Buy in discount, Sell in premium
   if(state.bias == BIAS_BULLISH && state.in_discount)
      return 100.0;
   
   if(state.bias == BIAS_BEARISH && state.in_premium)
      return 100.0;
   
   // Suboptimal but acceptable
   if(state.bias == BIAS_BULLISH && !state.in_premium)
      return 70.0;
   
   if(state.bias == BIAS_BEARISH && !state.in_discount)
      return 70.0;
   
   // Counter-trend zones
   if(state.bias == BIAS_BULLISH && state.in_premium)
      return 30.0;
   
   if(state.bias == BIAS_BEARISH && state.in_discount)
      return 30.0;
   
   return 50.0;
}

// === DIRECTION DETERMINATION ===
ENUM_SIGNAL_TYPE CConfluenceScorer::DetermineDirection()
{
   int bullish = 0;
   int bearish = 0;
   
   // Structure bias
   if(m_structure != NULL)
   {
      ENUM_MARKET_BIAS bias = m_structure.GetCurrentBias();
      if(bias == BIAS_BULLISH) bullish += 3;
      else if(bias == BIAS_BEARISH) bearish += 3;
   }
   
   // Regime Kalman trend
   if(m_regime != NULL)
   {
      SRegimeAnalysis regime = m_regime.GetLastAnalysis();
      if(regime.kalman_trend == KALMAN_BULLISH) bullish += 2;
      else if(regime.kalman_trend == KALMAN_BEARISH) bearish += 2;
   }
   
   // Sweep direction
   if(m_sweep != NULL && m_sweep.HasRecentSweep(10))
   {
      SSweepEvent sweep = m_sweep.GetMostRecentSweep();
      // SSL sweep = bullish signal, BSL sweep = bearish signal
      if(sweep.pool.type == LIQUIDITY_SSL) bullish += 2;
      else if(sweep.pool.type == LIQUIDITY_BSL) bearish += 2;
   }
   
   // AMD bias
   if(m_amd != NULL && m_amd.HasValidSetup())
   {
      ENUM_SIGNAL_TYPE amd_dir = m_amd.GetDistributionDirection();
      if(amd_dir == SIGNAL_BUY) bullish += 2;
      else if(amd_dir == SIGNAL_SELL) bearish += 2;
   }
   
   // Store counts
   m_last_result.bullish_factors = bullish;
   m_last_result.bearish_factors = bearish;
   
   // Determine direction (need clear majority)
   if(bullish > bearish && bullish >= 4)
      return SIGNAL_BUY;
   
   if(bearish > bullish && bearish >= 4)
      return SIGNAL_SELL;
   
   return SIGNAL_NONE;
}

ENUM_SIGNAL_QUALITY CConfluenceScorer::ClassifyQuality(double score)
{
   if(score >= TIER_S_MIN) return QUALITY_ELITE;
   if(score >= TIER_A_MIN) return QUALITY_HIGH;
   if(score >= TIER_B_MIN) return QUALITY_MEDIUM;
   if(score >= TIER_C_MIN) return QUALITY_LOW;
   return QUALITY_INVALID;
}

// === TRADE SETUP ===
void CConfluenceScorer::CalculateTradeSetup(SConfluenceResult &result, double current_price)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double atr_buffer = 200 * point; // ~20 pips default
   
   // Get ATR if available
   int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(atr_handle != INVALID_HANDLE)
   {
      double atr[];
      if(CopyBuffer(atr_handle, 0, 0, 1, atr) > 0)
         atr_buffer = atr[0];
      IndicatorRelease(atr_handle);
   }
   
   result.entry_price = current_price;
   
   if(result.direction == SIGNAL_BUY)
   {
      result.stop_loss = current_price - atr_buffer * 1.5;
      result.take_profit_1 = current_price + atr_buffer * 2.0;
      result.take_profit_2 = current_price + atr_buffer * 3.0;
   }
   else if(result.direction == SIGNAL_SELL)
   {
      result.stop_loss = current_price + atr_buffer * 1.5;
      result.take_profit_1 = current_price - atr_buffer * 2.0;
      result.take_profit_2 = current_price - atr_buffer * 3.0;
   }
   
   // Calculate R:R
   double risk = MathAbs(result.entry_price - result.stop_loss);
   double reward = MathAbs(result.take_profit_1 - result.entry_price);
   result.risk_reward = (risk > 0) ? reward / risk : 0;
}

// === VALIDATION ===
bool CConfluenceScorer::IsValidSetup(const SConfluenceResult &result)
{
   // Must have direction
   if(result.direction == SIGNAL_NONE)
      return false;
   
   // Must meet minimum score
   if(result.total_score < m_min_score)
      return false;
   
   // Must have minimum confluences
   if(result.total_confluences < m_min_confluences)
      return false;
   
   // Must pass regime filter (not random walk)
   if(!PassesRegimeFilter())
      return false;
   
   // Must have valid quality
   if(result.quality == QUALITY_INVALID)
      return false;
   
   return true;
}

bool CConfluenceScorer::PassesRegimeFilter()
{
   if(m_regime == NULL) return true;
   
   return m_regime.IsTradingAllowed();
}

bool CConfluenceScorer::PassesStructureFilter()
{
   if(m_structure == NULL) return true;
   
   ENUM_MARKET_BIAS bias = m_structure.GetCurrentBias();
   return (bias == BIAS_BULLISH || bias == BIAS_BEARISH);
}

// === HELPER METHODS ===
int CConfluenceScorer::CountConfluences(const SConfluenceResult &result)
{
   int count = 0;
   
   if(result.structure_score >= 60) count++;
   if(result.regime_score >= 60) count++;
   if(result.sweep_score >= 60) count++;
   if(result.amd_score >= 60) count++;
   if(result.ob_score >= 60) count++;
   if(result.fvg_score >= 60) count++;
   if(result.premium_discount >= 70) count++;
   
   return count;
}

string CConfluenceScorer::BuildSignalReason(const SConfluenceResult &result)
{
   string reason = "";
   
   reason += DirectionToString(result.direction) + " | ";
   reason += "Score: " + DoubleToString(result.total_score, 1) + " | ";
   reason += "Quality: " + QualityToString(result.quality) + " | ";
   reason += "Confluences: " + IntegerToString(result.total_confluences);
   
   return reason;
}

// === UTILITY ===
string CConfluenceScorer::QualityToString(ENUM_SIGNAL_QUALITY quality)
{
   switch(quality)
   {
      case QUALITY_ELITE:   return "ELITE";
      case QUALITY_HIGH:    return "HIGH";
      case QUALITY_MEDIUM:  return "MEDIUM";
      case QUALITY_LOW:     return "LOW";
      case QUALITY_INVALID: return "INVALID";
      default:              return "UNKNOWN";
   }
}

string CConfluenceScorer::DirectionToString(ENUM_SIGNAL_TYPE dir)
{
   switch(dir)
   {
      case SIGNAL_BUY:  return "BUY";
      case SIGNAL_SELL: return "SELL";
      case SIGNAL_NONE: return "NONE";
      default:          return "UNKNOWN";
   }
}

void CConfluenceScorer::PrintResult(const SConfluenceResult &result)
{
   Print("=== CONFLUENCE RESULT ===");
   Print("Direction: ", DirectionToString(result.direction));
   Print("Quality: ", QualityToString(result.quality));
   Print("Total Score: ", DoubleToString(result.total_score, 1));
   Print("  Structure: ", DoubleToString(result.structure_score, 1));
   Print("  Regime: ", DoubleToString(result.regime_score, 1));
   Print("  Sweep: ", DoubleToString(result.sweep_score, 1));
   Print("  AMD: ", DoubleToString(result.amd_score, 1));
   Print("  OB: ", DoubleToString(result.ob_score, 1));
   Print("  FVG: ", DoubleToString(result.fvg_score, 1));
   Print("  Zone: ", DoubleToString(result.premium_discount, 1));
   Print("Confluences: ", result.total_confluences);
   Print("Valid: ", result.is_valid ? "YES" : "NO");
   Print("Position Size Mult: ", DoubleToString(result.position_size_mult, 2));
   if(result.is_valid)
   {
      Print("Entry: ", DoubleToString(result.entry_price, 2));
      Print("SL: ", DoubleToString(result.stop_loss, 2));
      Print("TP1: ", DoubleToString(result.take_profit_1, 2));
      Print("R:R: ", DoubleToString(result.risk_reward, 2));
   }
   Print("=========================");
}
//+------------------------------------------------------------------+
