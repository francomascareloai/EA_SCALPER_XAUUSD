//+------------------------------------------------------------------+
//|                                             CRegimeDetector.mqh |
//|                           EA_SCALPER_XAUUSD - Singularity Edition |
//|                              Regime Detection: Hurst + Entropy + Kalman |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD - Singularity Edition"
#property strict

// === REGIME ENUMERATIONS ===
enum ENUM_MARKET_REGIME
{
   REGIME_PRIME_TRENDING = 0,    // H > 0.55, S < 1.5 - Full confidence trending
   REGIME_NOISY_TRENDING = 1,    // H > 0.55, S >= 1.5 - Trending with noise
   REGIME_PRIME_REVERTING = 2,   // H < 0.45, S < 1.5 - Full confidence mean revert
   REGIME_NOISY_REVERTING = 3,   // H < 0.45, S >= 1.5 - Mean revert with noise
   REGIME_RANDOM_WALK = 4,       // 0.45 <= H <= 0.55 - NO TRADE
   REGIME_UNKNOWN = 5            // Insufficient data
};

enum ENUM_KALMAN_TREND
{
   KALMAN_BULLISH = 0,
   KALMAN_BEARISH = 1,
   KALMAN_NEUTRAL = 2
};

// === REGIME ANALYSIS RESULT STRUCTURE ===
struct SRegimeAnalysis
{
   ENUM_MARKET_REGIME regime;
   double             hurst_exponent;
   double             shannon_entropy;
   double             kalman_trend_velocity;
   ENUM_KALMAN_TREND  kalman_trend;
   double             size_multiplier;     // 0.0, 0.5, or 1.0
   int                score_adjustment;    // -30, 0, or +10
   double             confidence;          // 0.0 to 1.0
   datetime           calculation_time;
   bool               is_valid;
};

// === KALMAN FILTER STATE ===
struct SKalmanState
{
   double x;           // Estimated price
   double P;           // Error covariance
   double velocity;    // Price velocity
};

// === REGIME DETECTOR CLASS ===
class CRegimeDetector
{
private:
   // Calculation parameters
   int                 m_hurst_window;         // Window for Hurst calculation (default 100)
   int                 m_entropy_window;       // Window for Entropy calculation (default 100)
   int                 m_entropy_bins;         // Number of bins for histogram (default 10)
   int                 m_min_samples;          // Minimum samples required
   
   // Thresholds
   double              m_hurst_trending;       // H > this = trending (0.55)
   double              m_hurst_reverting;      // H < this = mean reverting (0.45)
   double              m_entropy_low;          // S < this = low noise (1.5)
   double              m_entropy_high;         // S > this = high noise (2.5)
   
   // Kalman filter parameters
   double              m_kalman_Q;             // Process variance (0.01)
   double              m_kalman_R;             // Measurement variance (1.0)
   double              m_kalman_velocity_threshold; // Trend threshold (0.1)
   SKalmanState        m_kalman_state;
   
   // Cache
   SRegimeAnalysis     m_last_analysis;
   datetime            m_last_calculation;
   int                 m_cache_seconds;        // How long to cache (60s default)
   
   // Data buffers
   double              m_prices[];
   double              m_returns[];
   
public:
   CRegimeDetector();
   ~CRegimeDetector();
   
   // Configuration
   void SetHurstWindow(int window) { m_hurst_window = MathMax(20, window); }
   void SetEntropyWindow(int window) { m_entropy_window = MathMax(20, window); }
   void SetEntropyBins(int bins) { m_entropy_bins = MathMax(5, bins); }
   void SetThresholds(double trending, double reverting, double low_entropy);
   void SetKalmanParams(double Q, double R, double velocity_threshold);
   void SetCacheSeconds(int seconds) { m_cache_seconds = seconds; }
   
   // Main analysis
   SRegimeAnalysis AnalyzeRegime(string symbol = NULL, int tf = 0);
   SRegimeAnalysis GetLastAnalysis() { return m_last_analysis; }
   
   // Individual calculations
   double CalculateHurst(const double &prices[]);
   double CalculateEntropy(const double &returns[]);
   SKalmanState UpdateKalman(double measurement);
   ENUM_KALMAN_TREND GetKalmanTrend(const double &prices[], int lookback = 5);
   
   // Regime classification
   ENUM_MARKET_REGIME ClassifyRegime(double hurst, double entropy);
   double GetSizeMultiplier(ENUM_MARKET_REGIME regime);
   int GetScoreAdjustment(ENUM_MARKET_REGIME regime);
   
   // Utility
   bool IsRandomWalk() { return m_last_analysis.regime == REGIME_RANDOM_WALK; }
   bool IsTradingAllowed() { return m_last_analysis.regime != REGIME_RANDOM_WALK && m_last_analysis.regime != REGIME_UNKNOWN; }
   string RegimeToString(ENUM_MARKET_REGIME regime);
   void ResetKalman() { m_kalman_state.x = 0; m_kalman_state.P = 1.0; m_kalman_state.velocity = 0; }
   
private:
   bool LoadPriceData(string symbol, int tf, int count);
   void CalculateReturns();
   double CalculateConfidence(double hurst, double entropy);
};

// === CONSTRUCTOR ===
CRegimeDetector::CRegimeDetector()
{
   // Default parameters optimized for XAUUSD M15
   m_hurst_window = 100;
   m_entropy_window = 100;
   m_entropy_bins = 10;
   m_min_samples = 50;
   
   // Thresholds from Blueprint
   m_hurst_trending = 0.55;
   m_hurst_reverting = 0.45;
   m_entropy_low = 1.5;
   m_entropy_high = 2.5;
   
   // Kalman parameters
   m_kalman_Q = 0.01;
   m_kalman_R = 1.0;
   m_kalman_velocity_threshold = 0.1;
   m_kalman_state.x = 0;
   m_kalman_state.P = 1.0;
   m_kalman_state.velocity = 0;
   
   // Cache
   m_cache_seconds = 60;
   m_last_calculation = 0;
   
   // Initialize last analysis
   ZeroMemory(m_last_analysis);
   m_last_analysis.regime = REGIME_UNKNOWN;
   m_last_analysis.is_valid = false;
}

CRegimeDetector::~CRegimeDetector()
{
   ArrayFree(m_prices);
   ArrayFree(m_returns);
}

// === CONFIGURATION METHODS ===
void CRegimeDetector::SetThresholds(double trending, double reverting, double low_entropy)
{
   m_hurst_trending = trending;
   m_hurst_reverting = reverting;
   m_entropy_low = low_entropy;
}

void CRegimeDetector::SetKalmanParams(double Q, double R, double velocity_threshold)
{
   m_kalman_Q = Q;
   m_kalman_R = R;
   m_kalman_velocity_threshold = velocity_threshold;
}

// === MAIN ANALYSIS ===
SRegimeAnalysis CRegimeDetector::AnalyzeRegime(string symbol, int tf)
{
   // Check cache
   if(TimeCurrent() - m_last_calculation < m_cache_seconds && m_last_analysis.is_valid)
      return m_last_analysis;
   
   SRegimeAnalysis result;
   ZeroMemory(result);
   result.regime = REGIME_UNKNOWN;
   result.is_valid = false;
   
   // Use current symbol if not specified
   if(symbol == NULL) symbol = _Symbol;
   
   // Load price data
   int required = MathMax(m_hurst_window, m_entropy_window) + 10;
   if(!LoadPriceData(symbol, tf, required))
   {
      Print("CRegimeDetector: Failed to load price data");
      return result;
   }
   
   // Calculate returns
   CalculateReturns();
   
   // Calculate Hurst Exponent
   result.hurst_exponent = CalculateHurst(m_prices);
   if(result.hurst_exponent < 0)
   {
      Print("CRegimeDetector: Hurst calculation failed");
      return result;
   }
   
   // Calculate Shannon Entropy
   result.shannon_entropy = CalculateEntropy(m_returns);
   if(result.shannon_entropy < 0)
   {
      Print("CRegimeDetector: Entropy calculation failed");
      return result;
   }
   
   // Get Kalman trend
   result.kalman_trend = GetKalmanTrend(m_prices, 5);
   result.kalman_trend_velocity = m_kalman_state.velocity;
   
   // Classify regime
   result.regime = ClassifyRegime(result.hurst_exponent, result.shannon_entropy);
   result.size_multiplier = GetSizeMultiplier(result.regime);
   result.score_adjustment = GetScoreAdjustment(result.regime);
   result.confidence = CalculateConfidence(result.hurst_exponent, result.shannon_entropy);
   result.calculation_time = TimeCurrent();
   result.is_valid = true;
   
   // Cache result
   m_last_analysis = result;
   m_last_calculation = TimeCurrent();
   
   return result;
}

// === HURST EXPONENT CALCULATION (R/S Analysis) ===
double CRegimeDetector::CalculateHurst(const double &prices[])
{
   int size = ArraySize(prices);
   if(size < m_min_samples)
      return -1.0;
   
   // Calculate log returns
   double log_returns[];
   ArrayResize(log_returns, size - 1);
   for(int i = 1; i < size; i++)
   {
      if(prices[i-1] <= 0) return -1.0;
      log_returns[i-1] = MathLog(prices[i] / prices[i-1]);
   }
   
   // R/S analysis with multiple window sizes
   int min_k = 10;
   int max_k = MathMin(50, (size - 1) / 2);
   
   if(max_k <= min_k) return -1.0;
   
   double log_n[], log_rs[];
   ArrayResize(log_n, 0);
   ArrayResize(log_rs, 0);
   
   for(int n = min_k; n <= max_k; n++)
   {
      int num_subseries = ArraySize(log_returns) / n;
      if(num_subseries < 1) continue;
      
      double rs_sum = 0;
      int valid_count = 0;
      
      for(int i = 0; i < num_subseries; i++)
      {
         // Extract subseries
         double subseries[];
         ArrayResize(subseries, n);
         for(int j = 0; j < n; j++)
            subseries[j] = log_returns[i * n + j];
         
         // Calculate mean
         double mean = 0;
         for(int j = 0; j < n; j++)
            mean += subseries[j];
         mean /= n;
         
         // Calculate cumulative deviation and R
         double cumdev = 0;
         double max_cumdev = -DBL_MAX;
         double min_cumdev = DBL_MAX;
         for(int j = 0; j < n; j++)
         {
            cumdev += (subseries[j] - mean);
            if(cumdev > max_cumdev) max_cumdev = cumdev;
            if(cumdev < min_cumdev) min_cumdev = cumdev;
         }
         double R = max_cumdev - min_cumdev;
         
         // Calculate S (standard deviation)
         double variance = 0;
         for(int j = 0; j < n; j++)
            variance += MathPow(subseries[j] - mean, 2);
         variance /= (n - 1);
         double S = MathSqrt(variance);
         
         if(S > 0)
         {
            rs_sum += (R / S);
            valid_count++;
         }
         
         ArrayFree(subseries);
      }
      
      if(valid_count > 0)
      {
         double rs_mean = rs_sum / valid_count;
         int idx = ArraySize(log_n);
         ArrayResize(log_n, idx + 1);
         ArrayResize(log_rs, idx + 1);
         log_n[idx] = MathLog((double)n);
         log_rs[idx] = MathLog(rs_mean);
      }
   }
   
   ArrayFree(log_returns);
   
   // Linear regression to get Hurst exponent
   int count = ArraySize(log_n);
   if(count < 3) return -1.0;
   
   double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
   for(int i = 0; i < count; i++)
   {
      sum_x += log_n[i];
      sum_y += log_rs[i];
      sum_xy += log_n[i] * log_rs[i];
      sum_xx += log_n[i] * log_n[i];
   }
   
   double denominator = count * sum_xx - sum_x * sum_x;
   if(MathAbs(denominator) < 1e-10) return -1.0;
   
   double H = (count * sum_xy - sum_x * sum_y) / denominator;
   
   ArrayFree(log_n);
   ArrayFree(log_rs);
   
   // Clip to valid range [0, 1]
   return MathMax(0.0, MathMin(1.0, H));
}

// === SHANNON ENTROPY CALCULATION ===
double CRegimeDetector::CalculateEntropy(const double &returns[])
{
   int size = ArraySize(returns);
   if(size < m_min_samples)
      return -1.0;
   
   // Find min and max for binning
   double min_val = returns[0];
   double max_val = returns[0];
   for(int i = 1; i < size; i++)
   {
      if(returns[i] < min_val) min_val = returns[i];
      if(returns[i] > max_val) max_val = returns[i];
   }
   
   double range = max_val - min_val;
   if(range < 1e-10) return 0.0; // All same values = zero entropy
   
   // Create histogram
   int bins[];
   ArrayResize(bins, m_entropy_bins);
   ArrayInitialize(bins, 0);
   
   double bin_width = range / m_entropy_bins;
   for(int i = 0; i < size; i++)
   {
      int bin_idx = (int)((returns[i] - min_val) / bin_width);
      if(bin_idx >= m_entropy_bins) bin_idx = m_entropy_bins - 1;
      if(bin_idx < 0) bin_idx = 0;
      bins[bin_idx]++;
   }
   
   // Calculate entropy
   double entropy = 0;
   for(int i = 0; i < m_entropy_bins; i++)
   {
      if(bins[i] > 0)
      {
         double p = (double)bins[i] / size;
         entropy -= p * MathLog(p) / MathLog(2.0); // log2
      }
   }
   
   ArrayFree(bins);
   return entropy;
}

// === KALMAN FILTER ===
SKalmanState CRegimeDetector::UpdateKalman(double measurement)
{
   // Initialize if first measurement
   if(m_kalman_state.x == 0)
   {
      m_kalman_state.x = measurement;
      m_kalman_state.P = 1.0;
      m_kalman_state.velocity = 0;
      return m_kalman_state;
   }
   
   // Predict
   double x_pred = m_kalman_state.x;
   double P_pred = m_kalman_state.P + m_kalman_Q;
   
   // Update
   double K = P_pred / (P_pred + m_kalman_R);
   m_kalman_state.x = x_pred + K * (measurement - x_pred);
   m_kalman_state.P = (1 - K) * P_pred;
   
   // Velocity (normalized by price level)
   m_kalman_state.velocity = (measurement - x_pred) / x_pred * 100.0;
   
   return m_kalman_state;
}

ENUM_KALMAN_TREND CRegimeDetector::GetKalmanTrend(const double &prices[], int lookback = 5)
{
   int size = ArraySize(prices);
   if(size < lookback) return KALMAN_NEUTRAL;
   
   // Reset Kalman for fresh calculation
   ResetKalman();
   
   // Process all prices to get filter state
   double velocities[];
   ArrayResize(velocities, size);
   for(int i = 0; i < size; i++)
   {
      UpdateKalman(prices[i]);
      velocities[i] = m_kalman_state.velocity;
   }
   
   // Average velocity over lookback period
   double avg_vel = 0;
   for(int i = size - lookback; i < size; i++)
      avg_vel += velocities[i];
   avg_vel /= lookback;
   
   ArrayFree(velocities);
   
   if(avg_vel > m_kalman_velocity_threshold)
      return KALMAN_BULLISH;
   else if(avg_vel < -m_kalman_velocity_threshold)
      return KALMAN_BEARISH;
   else
      return KALMAN_NEUTRAL;
}

// === REGIME CLASSIFICATION ===
ENUM_MARKET_REGIME CRegimeDetector::ClassifyRegime(double hurst, double entropy)
{
   // Random walk detection (highest priority)
   if(hurst >= m_hurst_reverting && hurst <= m_hurst_trending)
      return REGIME_RANDOM_WALK;
   
   // Trending regimes
   if(hurst > m_hurst_trending)
   {
      if(entropy < m_entropy_low)
         return REGIME_PRIME_TRENDING;
      else
         return REGIME_NOISY_TRENDING;
   }
   
   // Mean reverting regimes
   if(hurst < m_hurst_reverting)
   {
      if(entropy < m_entropy_low)
         return REGIME_PRIME_REVERTING;
      else
         return REGIME_NOISY_REVERTING;
   }
   
   return REGIME_UNKNOWN;
}

double CRegimeDetector::GetSizeMultiplier(ENUM_MARKET_REGIME regime)
{
   switch(regime)
   {
      case REGIME_PRIME_TRENDING:
      case REGIME_PRIME_REVERTING:
         return 1.0;   // Full size
      
      case REGIME_NOISY_TRENDING:
      case REGIME_NOISY_REVERTING:
         return 0.5;   // Half size
      
      case REGIME_RANDOM_WALK:
      case REGIME_UNKNOWN:
      default:
         return 0.0;   // No trade
   }
}

int CRegimeDetector::GetScoreAdjustment(ENUM_MARKET_REGIME regime)
{
   switch(regime)
   {
      case REGIME_PRIME_TRENDING:
      case REGIME_PRIME_REVERTING:
         return 10;    // Boost score
      
      case REGIME_NOISY_TRENDING:
      case REGIME_NOISY_REVERTING:
         return 0;     // Neutral
      
      case REGIME_RANDOM_WALK:
         return -30;   // Block trade
      
      case REGIME_UNKNOWN:
      default:
         return -50;   // Definitely block
   }
}

double CRegimeDetector::CalculateConfidence(double hurst, double entropy)
{
   // Distance from random walk zone
   double hurst_distance = 0;
   if(hurst > m_hurst_trending)
      hurst_distance = (hurst - m_hurst_trending) / (1.0 - m_hurst_trending);
   else if(hurst < m_hurst_reverting)
      hurst_distance = (m_hurst_reverting - hurst) / m_hurst_reverting;
   
   // Lower entropy = higher confidence
   double entropy_score = MathMax(0, 1.0 - entropy / 4.0);
   
   // Combined confidence
   return MathMin(1.0, (hurst_distance + entropy_score) / 2.0);
}

// === UTILITY METHODS ===
bool CRegimeDetector::LoadPriceData(string symbol, int tf, int count)
{
   ArraySetAsSeries(m_prices, true);
   int copied = CopyClose(symbol, PERIOD_CURRENT, 0, count, m_prices);
   if(copied < m_min_samples)
   {
      Print("CRegimeDetector: Only ", copied, " prices available, need ", m_min_samples);
      return false;
   }
   
   // Reverse to chronological order
   ArraySetAsSeries(m_prices, false);
   return true;
}

void CRegimeDetector::CalculateReturns()
{
   int size = ArraySize(m_prices);
   ArrayResize(m_returns, size - 1);
   
   for(int i = 1; i < size; i++)
   {
      if(m_prices[i-1] > 0)
         m_returns[i-1] = (m_prices[i] - m_prices[i-1]) / m_prices[i-1];
      else
         m_returns[i-1] = 0;
   }
}

string CRegimeDetector::RegimeToString(ENUM_MARKET_REGIME regime)
{
   switch(regime)
   {
      case REGIME_PRIME_TRENDING:   return "PRIME_TRENDING";
      case REGIME_NOISY_TRENDING:   return "NOISY_TRENDING";
      case REGIME_PRIME_REVERTING:  return "PRIME_REVERTING";
      case REGIME_NOISY_REVERTING:  return "NOISY_REVERTING";
      case REGIME_RANDOM_WALK:      return "RANDOM_WALK";
      case REGIME_UNKNOWN:          return "UNKNOWN";
      default:                      return "INVALID";
   }
}
//+------------------------------------------------------------------+
