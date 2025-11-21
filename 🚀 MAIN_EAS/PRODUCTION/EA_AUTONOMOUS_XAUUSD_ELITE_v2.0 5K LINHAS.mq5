//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                    EA_AUTONOMOUS_XAUUSD_ELITE_v2.0.mq5 |
//|                           Autonomous Expert Advisor for XAUUSD Trading |
//|                     Advanced ICT/SMC Strategies with FTMO Compliance   |
//+------------------------------------------------------------------+
#property copyright "Developed by Autonomous AI Agent - FTMO Elite Trading System"
#property link      "https://github.com/autonomous-trading"
#property version   "2.00"
#property description "Elite autonomous EA with ICT/SMC strategies, multi-timeframe analysis, and FTMO compliance"
#property strict

// === CORE INCLUDES ===
#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>
#include <Trade/PositionInfo.mqh>
#include <Trade/AccountInfo.mqh>
#include <Trade/DealInfo.mqh>
#include <Trade/HistoryOrderInfo.mqh>
// #include "Include\MCP_Integration_Library.mqh" // Removed - library not available

// === GLOBAL OBJECTS ===
CTrade         trade;
CSymbolInfo    symbol_info;
CPositionInfo  position_info;
CAccountInfo   account_info;

// === ENUMERATIONS ===
enum ENUM_LOT_SIZE_METHOD
{
   LOT_FIXED = 0,           // Fixed lot size
   LOT_PERCENT_RISK = 1,    // Percentage risk-based
   LOT_ADAPTIVE = 2         // Adaptive based on performance
};

enum ENUM_ORDER_BLOCK_TYPE
{
   OB_BULLISH = 0,
   OB_BEARISH = 1,
   OB_BREAKER = 2
};

enum ENUM_ORDER_BLOCK_STATE
{
   OB_STATE_ACTIVE = 0,        // Order block is active
   OB_STATE_TESTED = 1,        // Order block has been tested
   OB_STATE_MITIGATED = 2,     // Order block has been mitigated
   OB_STATE_REFINED = 3,       // Order block has been refined
   OB_STATE_DISABLED = 4       // Order block is disabled
};

enum ENUM_OB_QUALITY
{
   OB_QUALITY_LOW = 0,         // Low quality order block
   OB_QUALITY_MEDIUM = 1,      // Medium quality order block
   OB_QUALITY_HIGH = 2,        // High quality order block
   OB_QUALITY_ELITE = 3        // Elite institutional order block
};

enum ENUM_FVG_TYPE
{
   FVG_BULLISH = 0,
   FVG_BEARISH = 1,
   FVG_BALANCED = 2
};

enum ENUM_FVG_STATE
{
   FVG_STATE_OPEN = 0,         // FVG is open and unfilled
   FVG_STATE_PARTIAL = 1,      // FVG is partially filled
   FVG_STATE_FILLED = 2,       // FVG is completely filled
   FVG_STATE_EXPIRED = 3       // FVG has expired (time-based)
};

enum ENUM_FVG_QUALITY
{
   FVG_QUALITY_LOW = 0,        // Low quality FVG
   FVG_QUALITY_MEDIUM = 1,     // Medium quality FVG
   FVG_QUALITY_HIGH = 2,       // High quality FVG
   FVG_QUALITY_ELITE = 3       // Elite institutional FVG
};

enum ENUM_SIGNAL_TYPE
{
   SIGNAL_NONE = 0,
   SIGNAL_BUY = 1,
   SIGNAL_SELL = 2
};

enum ENUM_LIQUIDITY_TYPE
{
   LIQUIDITY_BSL = 0,          // Buy Side Liquidity
   LIQUIDITY_SSL = 1,          // Sell Side Liquidity
   LIQUIDITY_EQH = 2,          // Equal Highs
   LIQUIDITY_EQL = 3,          // Equal Lows
   LIQUIDITY_POOLS = 4,        // Liquidity Pools
   LIQUIDITY_WEEKLY = 5,       // Weekly Liquidity
   LIQUIDITY_DAILY = 6,        // Daily Liquidity
   LIQUIDITY_SESSION = 7       // Session Liquidity
};

enum ENUM_LIQUIDITY_STATE
{
   LIQUIDITY_UNTAPPED = 0,     // Liquidity not swept
   LIQUIDITY_SWEPT = 1,        // Liquidity has been swept
   LIQUIDITY_PARTIAL = 2,      // Partially swept
   LIQUIDITY_EXPIRED = 3       // Time-based expiry
};

enum ENUM_LIQUIDITY_QUALITY
{
   LIQUIDITY_QUALITY_LOW = 0,     // Low quality liquidity
   LIQUIDITY_QUALITY_MEDIUM = 1,  // Medium quality liquidity
   LIQUIDITY_QUALITY_HIGH = 2,    // High quality liquidity
   LIQUIDITY_QUALITY_ELITE = 3    // Elite institutional liquidity
};

enum ENUM_SESSION_TYPE
{
   SESSION_ASIAN = 0,
   SESSION_LONDON = 1,
   SESSION_NY = 2,
   SESSION_OVERLAP = 3
};

// === INPUT PARAMETERS ===
input group "=== CORE STRATEGY SETTINGS ==="
input int                InpMagicNumber = 20241122;                    // Magic Number
input string             InpComment = "EA_AUTONOMOUS_XAUUSD_v2.0";     // Comment
input ENUM_LOT_SIZE_METHOD InpLotMethod = LOT_PERCENT_RISK;            // Lot Size Method

input group "=== RISK MANAGEMENT (FTMO OPTIMIZED) ==="
input double             InpLotSize = 0.01;                           // Fixed Lot Size
input int                InpStopLoss = 200;                           // Stop Loss (points)
input int                InpTakeProfit = 300;                         // Take Profit (points)
input double             InpRiskPercent = 1.0;                        // Risk Percentage per Trade
input double             InpMaxDailyRisk = 2.0;                       // Maximum Daily Risk (%)
input double             InpMaxDrawdown = 4.0;                        // Emergency Stop Drawdown (%)

input group "=== ICT/SMC STRATEGY PARAMETERS ==="
input double             InpConfluenceThreshold = 85.0;               // Minimum Confluence Score
input bool               InpEnableOrderBlocks = true;                 // Enable Order Block Detection
input bool               InpEnableFVG = true;                         // Enable Fair Value Gap Analysis
input bool               InpEnableLiquidity = true;                   // Enable Liquidity Detection
input bool               InpEnablePriceAction = true;                 // Enable Price Action Patterns

input group "=== MULTI-TIMEFRAME ANALYSIS ==="
input bool               InpUseWeeklyBias = true;                     // Use Weekly Bias (30%)
input bool               InpUseDailyTrend = true;                     // Use Daily Trend (25%)
input bool               InpUseH4Structure = true;                    // Use H4 Structure (20%)
input bool               InpUseH1Setup = true;                       // Use H1 Setup (15%)
input bool               InpUseM15Execution = true;                   // Use M15 Execution (10%)

input group "=== TIME & SESSION FILTERS ==="
input bool               InpTradeLondonSession = true;                // Trade London Session
input bool               InpTradeNYSession = true;                    // Trade NY Session
input bool               InpTradeAsianSession = false;                // Trade Asian Session
input int                InpLondonStart = 8;                         // London Start Hour (GMT)
input int                InpLondonEnd = 12;                          // London End Hour (GMT)
input int                InpNYStart = 13;                            // NY Start Hour (GMT)
input int                InpNYEnd = 17;                              // NY End Hour (GMT)

input group "=== NEWS & RISK FILTERS ==="
input bool               InpEnableNewsFilter = true;                 // Enable News Filter
input int                InpNewsAvoidanceMinutes = 60;               // News Avoidance (minutes)
input int                InpNewsCPIHour = 12;                        // CPI hour (GMT)
input int                InpNewsCPIMinute = 30;                      // CPI minute (GMT)
input int                InpNewsFOMCHour = 18;                       // FOMC hour (GMT)
input int                InpNewsFOMCMinute = 0;                      // FOMC minute (GMT)
input int                InpNewsLondonHour = 8;                      // London news hour (GMT)
input int                InpNewsLondonMinute = 30;                   // London news minute (GMT)
input int                InpNewsBufferMinutes = 35;                  // Buffer around news (minutes)
input double             InpMaxSpread = 25;                          // Maximum Spread (points)
input int                InpMaxTradesPerDay = 3;                     // Maximum Trades per Day
input double             InpServerGMTOffset = 0.0;                   // Server time offset from GMT (hours)

input group "=== LOGGING & DEBUG ==="
input bool               InpVerboseLogging = false;                  // Verbose diagnostic logs

input group "=== AI OPTIMIZATION (MCP INTEGRATION) ==="
input bool               InpEnableMCPIntegration = true;              // Enable MCP AI Integration
input bool               InpUseAIOptimization = true;                // Use AI Parameter Optimization
input bool               InpAITradeValidation = true;                // AI Trade Validation
input int                InpAIOptimizationInterval = 60;             // AI Optimization Interval (minutes)
input double             InpAIConfidenceThreshold = 0.7;             // AI Confidence Threshold
input bool               InpEnableAdaptiveLearning = true;           // Enable Adaptive Learning
input bool               InpEnableEmergencyProtection = true;        // Enable Emergency Protection
input bool               InpEnablePerformanceTracking = true;        // Enable Performance Tracking
input double             InpBreakevenRR = 1.0;                       // Breakeven at R:R Ratio
input double             InpPartialProfitRR = 1.5;                   // Partial Profit at R:R Ratio
input double             InpTrailingStartRR = 2.0;                   // Trailing Start at R:R Ratio

// === AI OPTIMIZATION VARIABLES ===
datetime g_last_ai_optimization = 0;
double g_ai_optimized_confluence = 0.0;
double g_ai_optimized_risk = 0.0;
bool g_ai_optimization_active = false;
string g_ai_recommendations = "";
int g_ai_optimization_count = 0;
datetime g_last_bar_time = 0;
int g_trades_today = 0;
datetime g_today_date = 0;
double g_daily_profit = 0.0;
double g_daily_starting_balance = 0.0;
bool g_emergency_stop = false;
bool g_daily_limit_reached = false;

// === STRUCTURE DEFINITIONS ===
// === ENHANCED STRUCTURE DEFINITIONS ===
struct SAdvancedOrderBlock
{
    datetime            formation_time;         // When order block formed
    double              high_price;             // Order block high
    double              low_price;              // Order block low
    double              refined_entry;          // Optimal entry within OB
    ENUM_ORDER_BLOCK_TYPE type;                // Bullish/Bearish/Breaker
    ENUM_ORDER_BLOCK_STATE state;              // Active/Tested/Mitigated
    ENUM_OB_QUALITY     quality;               // Quality classification
    
    // Advanced Properties
    bool                is_fresh;               // Untested order block
    bool                is_institutional;       // Institutional size
    double              strength;               // OB strength (0-100)
    double              volume_profile;         // Associated volume
    double              displacement_size;      // Size of displacement move
    bool                has_liquidity;          // Contains liquidity pool
    double              reaction_quality;       // Quality of reaction (0-1)
    int                 touch_count;            // Number of times tested
    double              probability_score;      // Success probability (0-100)
    bool                is_premium;             // Premium/Discount zone
    ENUM_TIMEFRAME      origin_timeframe;       // Timeframe where OB formed
    
    // Confluence Factors
    bool                has_fvg_confluence;     // FVG confluence
    bool                has_liquidity_confluence; // Liquidity confluence
    bool                has_structure_confluence; // Structure confluence
    double              confluence_score;       // Total confluence score
};

struct SEliteFairValueGap
{
    datetime            formation_time;         // When FVG formed
    double              upper_level;            // FVG top boundary
    double              lower_level;            // FVG bottom boundary
    double              mid_level;              // FVG middle (50%)
    double              optimal_entry;          // Calculated optimal entry
    ENUM_FVG_TYPE       type;                  // Bullish/Bearish/Balanced
    ENUM_FVG_STATE      state;                 // Open/Partial/Filled/Expired
    ENUM_FVG_QUALITY    quality;               // Quality classification
    
    // Advanced Properties
    double              fill_percentage;        // How much filled (0-100%)
    double              expected_reaction;      // Price reaction probability
    double              displacement_size;      // Size of creating displacement
    bool                has_volume_spike;       // Volume spike confirmation
    bool                is_institutional;       // Institutional FVG
    double              gap_size_points;        // Gap size in points
    ENUM_TIMEFRAME      origin_timeframe;       // Origin timeframe
    double              quality_score;          // FVG quality (0-100)
    bool                is_in_premium;          // Premium/Discount location
    int                 confluence_count;       // Number of confluences
    
    // Confluence Factors
    bool                has_ob_confluence;      // Order Block confluence
    bool                has_liquidity_confluence; // Liquidity confluence
    bool                has_structure_confluence; // Structure confluence
    double              confluence_score;       // Total confluence score
    
    // Reaction Analysis
    double              historical_success_rate; // Historical success rate
    int                 touch_count;            // Times price approached
    double              average_reaction_size;  // Average reaction size
    bool                has_strong_reaction;    // Strong reaction expected
    
    // Timing Analysis
    int                 age_in_bars;           // Age in bars
    datetime            expiry_time;           // Expected expiry time
    bool                is_fresh;              // Fresh unfilled FVG
    double              time_decay_factor;     // Time-based decay factor
};

struct SInstitutionalLiquidityPool
{
    datetime            formation_time;         // When liquidity level identified
    double              price_level;            // Exact liquidity level
    ENUM_LIQUIDITY_TYPE type;                  // Type of liquidity
    ENUM_LIQUIDITY_STATE state;               // Current state
    ENUM_LIQUIDITY_QUALITY quality;           // Quality classification
    
    // Volume and Size Analysis
    double              volume_estimate;       // Estimated volume at level
    double              accumulation_size;     // Size of accumulation
    double              institutional_footprint; // Institution size indicator
    bool                has_orders_confluence; // Multiple order levels
    
    // Sweep Analysis
    double              sweep_probability;     // Likelihood of sweep (0-1)
    bool                is_target;            // Primary liquidity target
    double              sweep_distance;       // Distance for valid sweep
    double              protection_level;     // Stop placement level
    
    // Historical Analysis
    int                 touch_count;          // Times price approached
    double              reaction_strength;    // Historical reaction strength
    double              average_reaction_size; // Average reaction from level
    bool                has_strong_reaction;  // Strong reaction expected
    
    // Timeframe Significance
    ENUM_TIMEFRAME      significance_tf;      // Most significant timeframe
    bool                is_weekly_level;      // Weekly liquidity
    bool                is_daily_level;       // Daily liquidity
    bool                is_session_level;     // Session liquidity
    
    // Confluence Factors
    bool                has_ob_confluence;    // Order Block confluence
    bool                has_fvg_confluence;   // FVG confluence
    bool                has_structure_confluence; // Structure confluence
    double              confluence_score;     // Total confluence score
    
    // Advanced Properties
    double              quality_score;        // Overall quality score
    bool                is_institutional;     // Institutional level
    double              time_decay_factor;    // Time-based decay
    int                 age_in_bars;         // Age in bars
    datetime            expiry_time;         // Expected expiry
    bool                is_fresh;            // Fresh unswept liquidity
};

struct SConfluenceSignal
{
    ENUM_SIGNAL_TYPE    signal_type;
    double              confidence_score;
    double              entry_price;
    double              stop_loss;
    double              take_profit;
    double              risk_reward_ratio;
    
    // Individual scores
    double              orderblock_score;
    double              fvg_score;
    double              liquidity_score;
    double              structure_score;
    double              priceaction_score;
    double              timeframe_score;
    
    // Filters
    bool                session_filter_ok;
    bool                news_filter_ok;
    bool                spread_filter_ok;
    bool                time_filter_ok;
};

struct SPerformanceMetrics
{
    double              total_profit;
    double              total_trades;
    double              win_rate;
    double              profit_factor;
    double              max_drawdown;
    double              sharpe_ratio;
    double              current_drawdown;
    bool                ftmo_compliant;
};

// === ENHANCED GLOBAL ARRAYS ===
SAdvancedOrderBlock        g_elite_order_blocks[50];     // Elite order blocks array
SEliteFairValueGap         g_elite_fair_value_gaps[30];  // Elite FVGs array
SInstitutionalLiquidityPool g_institutional_liquidity[50]; // Institutional liquidity pools
int                        g_elite_ob_count = 0;
int                        g_elite_fvg_count = 0;
int                        g_institutional_liq_count = 0;

// === INSTITUTIONAL LIQUIDITY DETECTOR CLASS ===
class CInstitutionalLiquidityDetector
{
private:
    // Detection parameters
    double              m_min_accumulation_size;    // Minimum accumulation for institutional
    double              m_institutional_threshold;  // Threshold for institutional liquidity
    int                 m_min_touch_count;          // Minimum touches for significance
    double              m_sweep_validation_distance; // Distance for valid sweep
    
    // Multi-timeframe tracking
    SInstitutionalLiquidityPool m_weekly_pools[10];   // Weekly liquidity levels
    SInstitutionalLiquidityPool m_daily_pools[20];    // Daily liquidity levels
    SInstitutionalLiquidityPool m_session_pools[30];  // Session liquidity levels
    int                 m_weekly_count;
    int                 m_daily_count;
    int                 m_session_count;
    
    // Analysis timeframes
    ENUM_TIMEFRAME      m_analysis_timeframes[5];
    int                 m_timeframe_count;
    
public:
    CInstitutionalLiquidityDetector();
    ~CInstitutionalLiquidityDetector();
    
    // Main detection methods
    bool DetectInstitutionalLiquidity();
    bool DetectWeeklyLiquidity();
    bool DetectDailyLiquidity();
    bool DetectSessionLiquidity();
    
    // Analysis methods
    bool ValidateLiquiditySweep(double price);
    double CalculateOptimalSweepEntry();
    bool IsInstitutionalLiquidity(const SInstitutionalLiquidityPool& pool);
    double CalculateLiquidityQuality(const SInstitutionalLiquidityPool& pool);
    
    // Management methods
    void UpdateLiquidityAfterSweep(SInstitutionalLiquidityPool& pool);
    void UpdateLiquidityStatus();
    void RemoveSweptLiquidity();
    int GetActiveLiquidityCount();
    double GetBestLiquidityScore();
    
    // Confluence methods
    void CalculateConfluenceScore(SInstitutionalLiquidityPool& pool);
    bool CheckOrderBlockConfluence(const SInstitutionalLiquidityPool& pool);
    bool CheckFVGConfluence(const SInstitutionalLiquidityPool& pool);
};

// === ELITE FVG DETECTOR CLASS ===
class CEliteFVGDetector
{
private:
    // Detection parameters
    double              m_min_displacement_size;    // Minimum displacement for valid FVG
    double              m_volume_spike_threshold;   // Volume spike threshold
    bool                m_require_structure_break;  // Structure break confirmation
    double              m_min_gap_size;            // Minimum gap size in points
    double              m_max_gap_size;            // Maximum gap size in points
    
    // Quality assessment parameters
    double              m_institutional_threshold;  // Threshold for institutional FVG
    double              m_confluence_weight;        // Weight for confluence scoring
    
    // Multi-timeframe analysis
    ENUM_TIMEFRAME      m_analysis_timeframes[4];   // Analysis timeframes
    int                 m_timeframe_count;          // Number of timeframes
    
public:
    CEliteFVGDetector();
    ~CEliteFVGDetector();
    
    // Main detection methods
    bool DetectEliteFairValueGaps();
    bool DetectInstitutionalFVG();
    bool ValidateWithDisplacement(SEliteFairValueGap& fvg);
    
    // Calculation methods
    double CalculateOptimalFillLevel(const SEliteFairValueGap& fvg);
    double CalculateFVGQualityScore(const SEliteFairValueGap& fvg);
    double CalculateExpectedReaction(const SEliteFairValueGap& fvg);
    ENUM_FVG_QUALITY ClassifyFVGQuality(const SEliteFairValueGap& fvg);
    
    // Analysis methods
    bool IsHighProbabilityFVG(const SEliteFairValueGap& fvg);
    bool IsInstitutionalFVG(const SEliteFairValueGap& fvg);
    void CalculateConfluenceScore(SEliteFairValueGap& fvg);
    
    // Management methods
    void UpdateFVGStatus();
    void RemoveFilledFVGs();
    int GetActiveFVGCount();
    double GetBestFVGScore();
};

// === ELITE ORDER BLOCK DETECTOR CLASS ===
class CEliteOrderBlockDetector
{
private:
    // Detection parameters
    double              m_displacement_threshold;     // Minimum displacement size
    double              m_volume_threshold;           // Volume spike threshold
    bool                m_require_structure_break;    // Structure break confirmation
    bool                m_use_liquidity_confirmation; // Liquidity confirmation
    bool                m_use_volume_confirmation;    // Volume confirmation
    
    // Multi-timeframe analysis
    ENUM_TIMEFRAME      m_analysis_timeframes[5];     // Analysis timeframes
    int                 m_timeframe_count;            // Number of timeframes
    
public:
    CEliteOrderBlockDetector();
    ~CEliteOrderBlockDetector();
    
    // Main detection methods
    bool DetectEliteOrderBlocks();
    bool DetectPremiumOrderBlocks();
    bool DetectDiscountOrderBlocks();
    
    // Validation methods
    bool ValidateWithLiquidity(SAdvancedOrderBlock& ob);
    bool ValidateWithVolume(SAdvancedOrderBlock& ob);
    bool ValidateWithStructure(SAdvancedOrderBlock& ob);
    
    // Calculation methods
    double CalculateOptimalEntry(const SAdvancedOrderBlock& ob);
    double CalculateOrderBlockStrength(const SAdvancedOrderBlock& ob);
    double CalculateProbabilityScore(const SAdvancedOrderBlock& ob);
    ENUM_OB_QUALITY ClassifyOrderBlockQuality(const SAdvancedOrderBlock& ob);
    
    // Analysis methods
    bool IsInstitutionalOrderBlock(const SAdvancedOrderBlock& ob);
    bool IsInPremiumZone(double price);
    bool IsInDiscountZone(double price);
    
    // Management methods
    void UpdateOrderBlockStatus();
    void RemoveInvalidOrderBlocks();
    int GetActiveOrderBlockCount();
};

// === INDICATOR HANDLES ===
int h_atr_h4, h_atr_h1, h_atr_m15;
int h_ema_fast, h_ema_medium, h_ema_slow;
int h_rsi;

// === FUNCTION DECLARATIONS ===
bool InitializeIndicators();
void SearchForTradingOpportunities();
SConfluenceSignal GenerateConfluenceSignal();
double CalculateOrderBlockScore();
double CalculateFVGScore();
double CalculateLiquidityScore();
double CalculateStructureScore();
double CalculatePriceActionScore();
double CalculateTimeframeScore();
bool ValidateAllFilters(const SConfluenceSignal& signal);
void CalculateTradeParameters(SConfluenceSignal& signal, ENUM_SIGNAL_TYPE signal_type);
bool IsInDiscountZone(double price);
bool IsInPremiumZone(double price);
void ExecuteTrade(const SConfluenceSignal& signal);
int CountOpenPositionsByMagic();
double GetTickValuePerPoint();
double CalculateLotSize(const SConfluenceSignal& signal);
double CalculateAdaptiveLotSize(const SConfluenceSignal& signal);
void ManagePositions();
void MoveToBreakeven(ulong ticket);
void TakePartialProfit(ulong ticket);
void UpdateTrailingStop(ulong ticket);
void UpdateOrderBlocks();
void UpdateFairValueGaps();
void UpdateLiquidityZones();
bool CheckEmergencyConditions();
bool IsTradingAllowed();
void CheckNewDay();
void ResetDailyStats();
SPerformanceMetrics CalculatePerformanceMetrics();
bool ValidateSessionFilter();
bool ValidateNewsFilter();
bool ValidateSpreadFilter();
bool ValidateRiskFilter(const SConfluenceSignal& signal);
double CalculatePotentialLoss(const SConfluenceSignal& signal);
bool ValidateTradeLevels(ENUM_SIGNAL_TYPE signal_type, double entry, double sl, double tp);
bool IsBullishEngulfing(const MqlRates& rates[], int index);
bool IsBearishEngulfing(const MqlRates& rates[], int index);
bool IsBullishPinBar(const MqlRates& rates[], int index);
bool IsBearishPinBar(const MqlRates& rates[], int index);
bool IsDoji(const MqlRates& rates[], int index);
bool IsWeeklyBiasAligned();
bool IsDailyTrendValid();
bool IsH4StructureValid();
bool IsH1SetupValid();
bool IsM15ExecutionValid();
bool IsBullishOrderBlock(const MqlRates& rates[], int index);
bool IsBearishOrderBlock(const MqlRates& rates[], int index);
double CalculateOrderBlockStrength(const MqlRates& rates[], int index);
double CalculateFVGStrength(const MqlRates& rates[], int index);
double CalculateLiquidityStrength(const MqlRates& rates[], int index);
bool IsSwingHigh(const MqlRates& rates[], int index, int period);
bool IsSwingLow(const MqlRates& rates[], int index, int period);
bool CheckEnhancedEmergencyConditions();
bool CheckEmergencyConditions();
void CheckAIOptimization();
bool InitializeMCPIntegration();
void CleanupMCPIntegration();
double GetEffectiveConfluenceThreshold();
SConfluenceSignal GenerateAIEnhancedConfluenceSignal();
bool ValidateTradeWithAI(const SConfluenceSignal& signal, bool &approved);

// Elite Order Block Detector helper function declarations
bool DetectBullishOrderBlock(const MqlRates& rates[], int index);
bool DetectBearishOrderBlock(const MqlRates& rates[], int index);
bool CreateOrderBlockStructure(const MqlRates& rates[], int index, ENUM_ORDER_BLOCK_TYPE type, SAdvancedOrderBlock& ob);
bool ValidateOrderBlock(const SAdvancedOrderBlock& ob);
double CalculateAverageBodySize(const MqlRates& rates[], int index, int period);
bool HasVolumeSpike(const MqlRates& rates[], int index);
double CalculateDisplacementSize(const MqlRates& rates[], int index);
double CalculateVolumeProfile(const MqlRates& rates[], int index);
double CalculateReactionQuality(const MqlRates& rates[], int index);
bool CheckFVGConfluence(const SAdvancedOrderBlock& ob);
bool CheckLiquidityConfluence(const SAdvancedOrderBlock& ob);
bool CheckStructureConfluence(const SAdvancedOrderBlock& ob);
double CalculateConfluenceScore(const SAdvancedOrderBlock& ob);
void SortOrderBlocksByQuality();

// Elite FVG Detector helper function declarations
bool DetectBullishFVG(const MqlRates& rates[], int index);
bool DetectBearishFVG(const MqlRates& rates[], int index);
bool CreateFVGStructure(const MqlRates& rates[], int index, ENUM_FVG_TYPE type, SEliteFairValueGap& fvg);
bool ValidateFVG(const SEliteFairValueGap& fvg);
double CalculateDisplacementAfterFVG(const MqlRates& rates[], int index, bool is_bullish);
bool HasVolumeConfirmation(const MqlRates& rates[], int index);
bool CheckOrderBlockConfluence(const SEliteFairValueGap& fvg);
void SortFVGsByQuality();

// Institutional Liquidity Detector helper function declarations
void CreateSimpleLiquidityPool(double price_level, ENUM_LIQUIDITY_TYPE type);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize trade object
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_FOK);
    
    // Initialize symbol
    if(!symbol_info.Name(_Symbol))
    {
        Print("ERROR: Failed to initialize symbol info");
        return INIT_FAILED;
    }
    
    // Initialize indicators
    if(!InitializeIndicators())
    {
        Print("ERROR: Failed to initialize indicators");
        return INIT_FAILED;
    }
    
    // Initialize Elite Order Block Detector
    g_elite_ob_detector = new CEliteOrderBlockDetector();
    if(g_elite_ob_detector == NULL)
    {
        Print("ERROR: Failed to initialize Elite Order Block Detector");
        return INIT_FAILED;
    }
    
    // Initialize Elite FVG Detector
    g_elite_fvg_detector = new CEliteFVGDetector();
    if(g_elite_fvg_detector == NULL)
    {
        Print("ERROR: Failed to initialize Elite FVG Detector");
        if(g_elite_ob_detector != NULL) delete g_elite_ob_detector;
        return INIT_FAILED;
    }
    
    // Initialize Institutional Liquidity Detector
    g_institutional_liq_detector = new CInstitutionalLiquidityDetector();
    if(g_institutional_liq_detector == NULL)
    {
        Print("ERROR: Failed to initialize Institutional Liquidity Detector");
        if(g_elite_ob_detector != NULL) delete g_elite_ob_detector;
        if(g_elite_fvg_detector != NULL) delete g_elite_fvg_detector;
        return INIT_FAILED;
    }
    
    // Initialize arrays
    for(int i = 0; i < 50; i++)
    {
        g_elite_order_blocks[i].formation_time = 0;
        g_elite_order_blocks[i].high_price = 0.0;
        g_elite_order_blocks[i].low_price = 0.0;
        g_elite_order_blocks[i].state = OB_STATE_DISABLED;
        g_elite_order_blocks[i].strength = 0.0;
        g_elite_order_blocks[i].probability_score = 0.0;
    }
    
    for(int i = 0; i < 30; i++)
    {
        g_elite_fair_value_gaps[i].formation_time = 0;
        g_elite_fair_value_gaps[i].upper_level = 0.0;
        g_elite_fair_value_gaps[i].lower_level = 0.0;
        g_elite_fair_value_gaps[i].state = FVG_STATE_EXPIRED;
        g_elite_fair_value_gaps[i].quality_score = 0.0;
        g_elite_fair_value_gaps[i].expected_reaction = 0.0;
    }
    
    for(int i = 0; i < 50; i++)
    {
        g_institutional_liquidity[i].formation_time = 0;
        g_institutional_liquidity[i].price_level = 0.0;
        g_institutional_liquidity[i].state = LIQUIDITY_EXPIRED;
        g_institutional_liquidity[i].quality_score = 0.0;
        g_institutional_liquidity[i].sweep_probability = 0.0;
    }
    
    // Initialize daily tracking
    ResetDailyStats();
    
    // Initialize enhanced confluence weights
    InitializeEliteConfluenceWeights();
    
    // Initialize FTMO compliance system
    InitializeFTMOCompliance();
    
    // Initialize MCP AI Integration
    if(InpEnableMCPIntegration)
    {
        if(InitializeMCPIntegration())
        {
            Print("✅ MCP AI Integration initialized successfully");
            g_ai_optimization_active = true;
        }
        else
        {
            Print("⚠️ MCP AI Integration failed - continuing without AI optimization");
            g_ai_optimization_active = false;
        }
    }
    
    // Print initialization success
    Print("=== EA_AUTONOMOUS_XAUUSD_ELITE_v2.0 INITIALIZED ===");
    Print("Symbol: ", _Symbol);
    Print("Timeframe: ", EnumToString(PERIOD_CURRENT));
    Print("Magic Number: ", InpMagicNumber);
    Print("Risk per Trade: ", InpRiskPercent, "%");
    Print("Confluence Threshold: ", InpConfluenceThreshold, "%");
    Print("Elite Order Block Detector: ACTIVE");
    Print("Elite FVG Detector: ACTIVE");
    Print("Institutional Liquidity Detector: ACTIVE");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles
    if(h_atr_h4 != INVALID_HANDLE) IndicatorRelease(h_atr_h4);
    if(h_atr_h1 != INVALID_HANDLE) IndicatorRelease(h_atr_h1);
    if(h_atr_m15 != INVALID_HANDLE) IndicatorRelease(h_atr_m15);
    if(h_ema_fast != INVALID_HANDLE) IndicatorRelease(h_ema_fast);
    if(h_ema_medium != INVALID_HANDLE) IndicatorRelease(h_ema_medium);
    if(h_ema_slow != INVALID_HANDLE) IndicatorRelease(h_ema_slow);
    if(h_rsi != INVALID_HANDLE) IndicatorRelease(h_rsi);
    
    // Clean up Elite Order Block Detector
    if(g_elite_ob_detector != NULL)
    {
        delete g_elite_ob_detector;
        g_elite_ob_detector = NULL;
    }
    
    // Clean up Elite FVG Detector
    if(g_elite_fvg_detector != NULL)
    {
        delete g_elite_fvg_detector;
        g_elite_fvg_detector = NULL;
    }
    
    // Clean up Institutional Liquidity Detector
    if(g_institutional_liq_detector != NULL)
    {
        delete g_institutional_liq_detector;
        g_institutional_liq_detector = NULL;
    }
    
    // Clean up MCP Integration
    if(InpEnableMCPIntegration)
    {
        CleanupMCPIntegration();
        Print("✅ MCP AI Integration cleaned up");
    }
    
    Print("=== EA_AUTONOMOUS_XAUUSD_ELITE_v2.0 DEINITIALIZED ===");
    Print("Cleanup completed successfully");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // --- CRITICAL: EVERY TICK OPERATIONS ---
    
    // 1. FTMO Compliance Check (Must run every tick to catch violations immediately)
    if(!CheckFTMOCompliance())
    {
        // Print compliance report every hour when non-compliant
        static datetime last_report_time = 0;
        if(TimeCurrent() - last_report_time > 3600) // 1 hour
        {
            Print(GetFTMOComplianceReport());
            last_report_time = TimeCurrent();
        }
        return; // Stop all trading if not compliant
    }
    
    // 2. Emergency protection check (enhanced with FTMO)
    if(InpEnableEmergencyProtection && CheckEnhancedEmergencyConditions()) return;
    
    // 3. Check if trading is allowed
    if(!IsTradingAllowed()) return;
    
    // 4. Manage existing positions (Trailing stops, BE, etc.)
    ManagePositions();
    
    // --- NEW BAR OPERATIONS ---
    
    // Check if new bar
    datetime current_bar_time = iTime(_Symbol, PERIOD_M15, 0);
    if(current_bar_time == g_last_bar_time) return;
    g_last_bar_time = current_bar_time;
    
    // 5. Reset daily stats if new day
    CheckNewDay();
    
    // 6. Check for AI parameter optimization
    if(InpEnableMCPIntegration && InpUseAIOptimization && g_ai_optimization_active)
    {
        CheckAIOptimization();
    }
    
    // 7. Check for new trading opportunities (only if FTMO compliant)
    int active_positions = CountOpenPositionsByMagic();
    if(active_positions < 2) // Max 2 positions for FTMO and only this EA
    {
        SearchForTradingOpportunities();
    }
}

//+------------------------------------------------------------------+
//| Trade transaction handler                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
    // Keep daily stats synchronized even if no new tick
    CheckNewDay();
    
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal <= 0)
        return;
    
    long deal_magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
    string deal_symbol = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
    
    if(deal_magic != InpMagicNumber || deal_symbol != _Symbol)
        return;
    
    ENUM_DEAL_ENTRY entry_type = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
    double deal_profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) +
                         HistoryDealGetDouble(trans.deal, DEAL_COMMISSION) +
                         HistoryDealGetDouble(trans.deal, DEAL_SWAP);
    
    // Count trades on entry
    if(entry_type == DEAL_ENTRY_IN)
        g_trades_today++;
    
    // Track P/L on exit legs
    if(entry_type == DEAL_ENTRY_OUT || entry_type == DEAL_ENTRY_OUT_BY || entry_type == DEAL_ENTRY_INOUT)
        g_daily_profit += deal_profit;
    
    double daily_risk_amount = g_daily_starting_balance * (InpMaxDailyRisk / 100.0);
    if(daily_risk_amount > 0 && g_daily_profit <= -daily_risk_amount)
    {
        g_daily_limit_reached = true;
        g_emergency_stop = true;
        Print("🛑 Daily loss limit reached via trade transaction. Trading halted for the day.");
    }
}

//+------------------------------------------------------------------+
//| Initialize all indicators                                        |
//+------------------------------------------------------------------+
bool InitializeIndicators()
{
    h_atr_h4 = iATR(_Symbol, PERIOD_H4, 14);
    h_atr_h1 = iATR(_Symbol, PERIOD_H1, 14);
    h_atr_m15 = iATR(_Symbol, PERIOD_M15, 14);
    
    h_ema_fast = iMA(_Symbol, PERIOD_M15, 8, 0, MODE_EMA, PRICE_CLOSE);
    h_ema_medium = iMA(_Symbol, PERIOD_M15, 21, 0, MODE_EMA, PRICE_CLOSE);
    h_ema_slow = iMA(_Symbol, PERIOD_M15, 55, 0, MODE_EMA, PRICE_CLOSE);
    
    h_rsi = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
    
    // Verify all handles
    if(h_atr_h4 == INVALID_HANDLE || h_atr_h1 == INVALID_HANDLE || 
       h_atr_m15 == INVALID_HANDLE || h_ema_fast == INVALID_HANDLE ||
       h_ema_medium == INVALID_HANDLE || h_ema_slow == INVALID_HANDLE ||
       h_rsi == INVALID_HANDLE)
    {
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check for new trading opportunities                             |
//+------------------------------------------------------------------+
void SearchForTradingOpportunities()
{
    // Update market analysis
    UpdateOrderBlocks();
    UpdateFairValueGaps();
    UpdateLiquidityZones();
    
    // Generate AI-enhanced confluence signal
    SConfluenceSignal signal;
    
    if(InpEnableMCPIntegration && g_ai_optimization_active)
    {
        signal = GenerateAIEnhancedConfluenceSignal();
        Print("🤖 Using AI-enhanced signal generation");
    }
    else
    {
        signal = GenerateConfluenceSignal();
    }
    
    // Get effective thresholds (AI-optimized or default)
    double effective_threshold = GetEffectiveConfluenceThreshold();
    
    // Validate signal quality
    if(signal.signal_type == SIGNAL_NONE) return;
    if(signal.confidence_score < effective_threshold) 
    {
        Print("📉 Signal rejected - Confidence: ", DoubleToString(signal.confidence_score, 1), 
              "% < Threshold: ", DoubleToString(effective_threshold, 1), "%");
        return;
    }
    
    // Check all filters
    if(!ValidateAllFilters(signal)) return;
    
    // Execute trade
    ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
//| Elite Confluence Scoring System - Enhanced Multi-Layer Analysis |
//+------------------------------------------------------------------+

// Enhanced confluence scoring structure
struct SEliteConfluenceAnalysis
{
    // Individual Component Scores (0-100)
    double order_block_score;
    double fvg_score;
    double liquidity_score;
    double structure_score;
    double priceaction_score;
    double timeframe_score;
    
    // Final Scores
    double total_confluence_score;
    double signal_strength_multiplier;
    
    // Quality Metrics
    ENUM_SIGNAL_TYPE dominant_signal;
    int confluence_component_count;
    double institutional_alignment;
    
    // Context Analysis
    bool is_premium_zone;
    bool is_discount_zone;
};

// Enhanced confluence weights
struct SEliteConfluenceWeights
{
    double order_block_weight;      // 25%
    double fvg_weight;             // 20% 
    double liquidity_weight;       // 20%
    double structure_weight;       // 15%
    double priceaction_weight;     // 10%
    double timeframe_weight;       // 10%
};

// Global enhanced confluence weights
SEliteConfluenceWeights g_confluence_weights;

//+------------------------------------------------------------------+
//| Initialize Enhanced Confluence Weights                          |
//+------------------------------------------------------------------+
void InitializeEliteConfluenceWeights()
{
    g_confluence_weights.order_block_weight = 0.25;     // 25%
    g_confluence_weights.fvg_weight = 0.20;            // 20%
    g_confluence_weights.liquidity_weight = 0.20;      // 20%
    g_confluence_weights.structure_weight = 0.15;      // 15%
    g_confluence_weights.priceaction_weight = 0.10;    // 10%
    g_confluence_weights.timeframe_weight = 0.10;      // 10%
}

//+------------------------------------------------------------------+
//| Enhanced Confluence Signal Generation                           |
//+------------------------------------------------------------------+
SConfluenceSignal GenerateConfluenceSignal()
{
    SConfluenceSignal signal;
    SEliteConfluenceAnalysis analysis;
    
    // Initialize signal structure
    signal.signal_type = SIGNAL_NONE;
    signal.confidence_score = 0.0;
    signal.entry_price = 0.0;
    signal.stop_loss = 0.0;
    signal.take_profit = 0.0;
    signal.risk_reward_ratio = 0.0;
    signal.session_filter_ok = false;
    signal.news_filter_ok = false;
    signal.spread_filter_ok = false;
    signal.time_filter_ok = false;
    
    // Calculate individual component scores with enhanced analysis
    analysis.order_block_score = CalculateEnhancedOrderBlockScore();
    analysis.fvg_score = CalculateEnhancedFVGScore();
    analysis.liquidity_score = CalculateEnhancedLiquidityScore();
    analysis.structure_score = CalculateEnhancedStructureScore();
    analysis.priceaction_score = CalculateEnhancedPriceActionScore();
    analysis.timeframe_score = CalculateEnhancedTimeframeScore();
    
    // Calculate weighted confluence score
    analysis.total_confluence_score = 
        analysis.order_block_score * g_confluence_weights.order_block_weight +
        analysis.fvg_score * g_confluence_weights.fvg_weight +
        analysis.liquidity_score * g_confluence_weights.liquidity_weight +
        analysis.structure_score * g_confluence_weights.structure_weight +
        analysis.priceaction_score * g_confluence_weights.priceaction_weight +
        analysis.timeframe_score * g_confluence_weights.timeframe_weight;
    
    // Context analysis
    double current_price = symbol_info.Ask();
    analysis.is_premium_zone = IsInPremiumZone(current_price);
    analysis.is_discount_zone = IsInDiscountZone(current_price);
    analysis.institutional_alignment = CalculateInstitutionalAlignment();
    
    // Determine signal direction with enhanced logic
    signal.signal_type = DetermineSignalDirection(analysis);
    signal.confidence_score = analysis.total_confluence_score;
    
    // Store component scores in signal for reference
    signal.orderblock_score = analysis.order_block_score;
    signal.fvg_score = analysis.fvg_score;
    signal.liquidity_score = analysis.liquidity_score;
    signal.structure_score = analysis.structure_score;
    signal.priceaction_score = analysis.priceaction_score;
    signal.timeframe_score = analysis.timeframe_score;
    
    // Calculate trade parameters if signal is valid
    if(signal.signal_type != SIGNAL_NONE && signal.confidence_score >= InpConfluenceThreshold)
    {
        CalculateTradeParameters(signal, signal.signal_type);
    }
    
    return signal;
}

//+------------------------------------------------------------------+
//| Enhanced Scoring Functions                                      |
//+------------------------------------------------------------------+

// Enhanced Order Block Score (replaces original)
double CalculateEnhancedOrderBlockScore()
{
    if(!InpEnableOrderBlocks) return 0.0;
    
    double total_score = 0.0;
    double current_price = symbol_info.Bid();
    int valid_obs = 0;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        const SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        if(ob.state == OB_STATE_DISABLED || ob.state == OB_STATE_MITIGATED) continue;
            
        double ob_score = 0.0;
        
        // Proximity Analysis (40 points max)
        bool price_in_ob = (current_price >= ob.low_price && current_price <= ob.high_price);
        
        if(price_in_ob)
        {
            ob_score = ob.probability_score * 0.9;
            if(ob.quality == OB_QUALITY_ELITE) ob_score += 15.0;
            else if(ob.quality == OB_QUALITY_HIGH) ob_score += 10.0;
            if(ob.is_institutional) ob_score += 10.0;
            if(ob.is_fresh) ob_score += 8.0;
            ob_score += ob.confluence_score * 0.2;
        }
        else
        {
            double distance = MathMin(MathAbs(current_price - ob.high_price), MathAbs(current_price - ob.low_price));
            double max_distance = 100 * _Point;
            
            if(distance <= max_distance)
            {
                double proximity_factor = 1.0 - (distance / max_distance);
                ob_score = ob.probability_score * proximity_factor * 0.7;
                if(ob.quality == OB_QUALITY_ELITE) ob_score += 10.0;
                if(ob.is_institutional) ob_score += 5.0;
            }
        }
        
        // Time-based decay
        int age_hours = (int)((TimeCurrent() - ob.formation_time) / 3600);
        double decay_factor = MathMax(0.5, 1.0 - (age_hours / 48.0));
        ob_score *= decay_factor;
        
        total_score += ob_score;
        valid_obs++;
    }
    
    double final_score = (valid_obs > 0) ? total_score / valid_obs : 0.0;
    if(valid_obs >= 3) final_score *= 1.25;
    else if(valid_obs >= 2) final_score *= 1.15;
    
    return MathMin(final_score, 100.0);
}

// Enhanced FVG Score
double CalculateEnhancedFVGScore()
{
    if(!InpEnableFVG) return 0.0;
    
    double total_score = 0.0;
    double current_price = symbol_info.Bid();
    int valid_fvgs = 0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        const SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        if(fvg.state == FVG_STATE_FILLED || fvg.state == FVG_STATE_EXPIRED) continue;
            
        double fvg_score = 0.0;
        
        bool price_in_fvg = (current_price >= fvg.lower_level && current_price <= fvg.upper_level);
        
        if(price_in_fvg)
        {
            double fill_factor = 1.0 - (fvg.fill_percentage / 100.0);
            fvg_score = fvg.quality_score * fill_factor * 0.9;
            
            if(fvg.quality == FVG_QUALITY_ELITE) fvg_score += 20.0;
            else if(fvg.quality == FVG_QUALITY_HIGH) fvg_score += 15.0;
            if(fvg.is_institutional) fvg_score += 12.0;
            if(fvg.is_fresh) fvg_score += 10.0;
            fvg_score += fvg.expected_reaction * 15.0;
            fvg_score += fvg.confluence_score * 0.25;
        }
        else
        {
            double distance = MathMin(MathAbs(current_price - fvg.upper_level), MathAbs(current_price - fvg.lower_level));
            double max_distance = 50 * _Point;
            
            if(distance <= max_distance)
            {
                double proximity_factor = 1.0 - (distance / max_distance);
                fvg_score = fvg.quality_score * proximity_factor * 0.7;
                if(fvg.quality == FVG_QUALITY_ELITE) fvg_score += 12.0;
                if(fvg.is_institutional) fvg_score += 6.0;
                fvg_score += fvg.expected_reaction * 8.0;
            }
        }
        
        fvg_score *= fvg.time_decay_factor;
        
        if(fvg.state == FVG_STATE_PARTIAL)
        {
            double penalty = fvg.fill_percentage * 0.005;
            fvg_score *= (1.0 - penalty);
        }
        
        total_score += fvg_score;
        valid_fvgs++;
    }
    
    double final_score = (valid_fvgs > 0) ? total_score / valid_fvgs : 0.0;
    if(valid_fvgs >= 2) final_score *= 1.1;
    
    return MathMin(final_score, 100.0);
}

// Enhanced Liquidity Score
double CalculateEnhancedLiquidityScore()
{
    if(!InpEnableLiquidity) return 0.0;
    
    double total_score = 0.0;
    double current_price = symbol_info.Bid();
    int valid_liquidity = 0;
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        const SInstitutionalLiquidityPool& pool = g_institutional_liquidity[i];
        if(pool.state == LIQUIDITY_SWEPT || pool.state == LIQUIDITY_EXPIRED) continue;
            
        double liq_score = 0.0;
        
        double distance = MathAbs(current_price - pool.price_level);
        double max_distance = 150 * _Point;
        
        if(distance <= max_distance)
        {
            double proximity_factor = 1.0 - (distance / max_distance);
            liq_score += 30.0 * proximity_factor;
        }
        
        liq_score += pool.quality_score * 0.25;
        liq_score += pool.sweep_probability * 20.0;
        
        switch(pool.type)
        {
            case LIQUIDITY_WEEKLY: liq_score += 15.0; break;
            case LIQUIDITY_DAILY: liq_score += 12.0; break;
            case LIQUIDITY_SESSION: liq_score += 8.0; break;
            default: liq_score += 5.0; break;
        }
        
        if(pool.is_institutional) liq_score += 10.0;
        if(pool.is_fresh) liq_score += 8.0;
        liq_score += pool.confluence_score * 0.1;
        liq_score *= pool.time_decay_factor;
        
        total_score += liq_score;
        valid_liquidity++;
    }
    
    double final_score = (valid_liquidity > 0) ? total_score / valid_liquidity : 0.0;
    if(valid_liquidity >= 3) final_score *= 1.2;
    else if(valid_liquidity >= 2) final_score *= 1.1;
    
    return MathMin(final_score, 100.0);
}

// Enhanced Structure Score
double CalculateEnhancedStructureScore()
{
    double score = 0.0;
    
    double ema_fast[3], ema_medium[3], ema_slow[3];
    
    if(CopyBuffer(h_ema_fast, 0, 0, 3, ema_fast) <= 0 ||
       CopyBuffer(h_ema_medium, 0, 0, 3, ema_medium) <= 0 ||
       CopyBuffer(h_ema_slow, 0, 0, 3, ema_slow) <= 0)
        return 0.0;
    
    ArraySetAsSeries(ema_fast, true);
    ArraySetAsSeries(ema_medium, true);
    ArraySetAsSeries(ema_slow, true);
    
    // EMA alignment (40 points)
    if(ema_fast[0] > ema_medium[0] && ema_medium[0] > ema_slow[0])
        score += 40.0;
    else if(ema_fast[0] < ema_medium[0] && ema_medium[0] < ema_slow[0])
        score += 40.0;
    
    // Fresh crossover (30 points)
    if((ema_fast[1] <= ema_medium[1] && ema_fast[0] > ema_medium[0]) ||
       (ema_fast[1] >= ema_medium[1] && ema_fast[0] < ema_medium[0]))
        score += 30.0;
    
    // Price position (30 points)
    double current_price = symbol_info.Bid();
    if((current_price > ema_fast[0] && ema_fast[0] > ema_medium[0]) ||
       (current_price < ema_fast[0] && ema_fast[0] < ema_medium[0]))
        score += 30.0;
    
    return MathMin(score, 100.0);
}

// Enhanced Price Action Score
double CalculateEnhancedPriceActionScore()
{
    if(!InpEnablePriceAction) return 0.0;
    
    double score = 0.0;
    
    MqlRates rates[5];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 5, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    // Reversal patterns (60 points max)
    if(IsBullishEngulfing(rates, 1))
    {
        score += 50.0;
        if(rates[1].tick_volume > rates[2].tick_volume * 1.5) score += 10.0;
    }
    else if(IsBearishEngulfing(rates, 1))
    {
        score += 50.0;
        if(rates[1].tick_volume > rates[2].tick_volume * 1.5) score += 10.0;
    }
    else if(IsBullishPinBar(rates, 1))
    {
        score += 35.0;
    }
    else if(IsBearishPinBar(rates, 1))
    {
        score += 35.0;
    }
    else if(IsDoji(rates, 1))
    {
        score += 20.0;
    }
    
    // Momentum analysis (40 points max)
    double momentum = CalculateCurrentMomentum(rates);
    score += momentum * 40.0;
    
    return MathMin(score, 100.0);
}

// Enhanced Timeframe Score
double CalculateEnhancedTimeframeScore()
{
    double score = 0.0;
    
    // Weekly bias (30%)
    if(InpUseWeeklyBias && IsWeeklyBiasAligned()) score += 30.0;
    
    // Daily trend (25%)
    if(InpUseDailyTrend && IsDailyTrendValid()) score += 25.0;
    
    // H4 structure (20%)
    if(InpUseH4Structure && IsH4StructureValid()) score += 20.0;
    
    // H1 setup (15%)
    if(InpUseH1Setup && IsH1SetupValid()) score += 15.0;
    
    // M15 execution (10%)
    if(InpUseM15Execution && IsM15ExecutionValid()) score += 10.0;
    
    return MathMin(score, 100.0);
}

// Missing utility functions
double CalculateInstitutionalAlignment()
{
    double alignment = 0.5; // Base 50%
    
    // Check institutional order blocks
    int institutional_obs = 0;
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        if(g_elite_order_blocks[i].is_institutional && g_elite_order_blocks[i].state == OB_STATE_ACTIVE)
            institutional_obs++;
    }
    
    // Check institutional FVGs
    int institutional_fvgs = 0;
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].is_institutional && g_elite_fair_value_gaps[i].state == FVG_STATE_OPEN)
            institutional_fvgs++;
    }
    
    // Check institutional liquidity
    int institutional_liq = 0;
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        if(g_institutional_liquidity[i].is_institutional && g_institutional_liquidity[i].state == LIQUIDITY_UNTAPPED)
            institutional_liq++;
    }
    
    // Calculate alignment score
    int total_institutional = institutional_obs + institutional_fvgs + institutional_liq;
    
    if(total_institutional >= 3) alignment = 0.9;
    else if(total_institutional >= 2) alignment = 0.8;
    else if(total_institutional >= 1) alignment = 0.7;
    
    return alignment;
}

ENUM_SIGNAL_TYPE DetermineSignalDirection(const SEliteConfluenceAnalysis& analysis)
{
    double current_price = symbol_info.Ask();
    
    // Direction scoring based on component analysis
    double bullish_score = 0.0;
    double bearish_score = 0.0;
    
    // Order Block directional bias
    if(analysis.order_block_score > 70)
    {
        if(analysis.is_discount_zone) bullish_score += 25.0;
        if(analysis.is_premium_zone) bearish_score += 25.0;
    }
    
    // FVG directional bias
    if(analysis.fvg_score > 70)
    {
        // Check FVG types for direction
        for(int i = 0; i < g_elite_fvg_count; i++)
        {
            const SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
            if(fvg.state == FVG_STATE_OPEN || fvg.state == FVG_STATE_PARTIAL)
            {
                if(fvg.type == FVG_BULLISH) bullish_score += 15.0;
                else if(fvg.type == FVG_BEARISH) bearish_score += 15.0;
            }
        }
    }
    
    // Structure bias
    if(analysis.structure_score > 70)
    {
        double ema_fast[1], ema_medium[1];
        if(CopyBuffer(h_ema_fast, 0, 0, 1, ema_fast) > 0 &&
           CopyBuffer(h_ema_medium, 0, 0, 1, ema_medium) > 0)
        {
            if(ema_fast[0] > ema_medium[0]) bullish_score += 15.0;
            else bearish_score += 15.0;
        }
    }
    
    // Premium/Discount context
    if(analysis.is_discount_zone) bullish_score += 10.0;
    if(analysis.is_premium_zone) bearish_score += 10.0;
    
    // Institutional alignment bonus
    if(analysis.institutional_alignment > 0.8)
    {
        bullish_score += 10.0;
        bearish_score += 10.0;
    }
    
    // Determine final signal
    if(bullish_score > bearish_score && bullish_score > 50.0)
        return SIGNAL_BUY;
    else if(bearish_score > bullish_score && bearish_score > 50.0)
        return SIGNAL_SELL;
    else
        return SIGNAL_NONE;
}

double CalculateCurrentMomentum(const MqlRates& rates[])
{
    if(ArraySize(rates) < 3) return 0.0;
    
    // Simple momentum calculation
    double recent_move = MathAbs(rates[0].close - rates[2].close);
    double average_range = 0.0;
    
    for(int i = 0; i < 3; i++)
    {
        average_range += (rates[i].high - rates[i].low);
    }
    average_range /= 3.0;
    
    return (average_range > 0) ? MathMin(recent_move / average_range, 1.0) : 0.0;
}

bool DetectFlagPattern(const MqlRates& rates[])
{
    // Simplified flag pattern detection
    if(ArraySize(rates) < 5) return false;
    
    // Check for consolidation after strong move
    double strong_move = MathAbs(rates[4].close - rates[0].close);
    double consolidation_range = 0.0;
    
    for(int i = 1; i < 4; i++)
    {
        consolidation_range += (rates[i].high - rates[i].low);
    }
    consolidation_range /= 3.0;
    
    return (strong_move > consolidation_range * 3.0);
}

bool DetectPennantPattern(const MqlRates& rates[])
{
    // Simplified pennant pattern detection
    if(ArraySize(rates) < 5) return false;
    
    // Check for converging ranges
    double early_range = rates[3].high - rates[3].low;
    double recent_range = rates[1].high - rates[1].low;
    
    return (early_range > recent_range * 1.5);
}
double CalculateOrderBlockScore()
{
    if(!InpEnableOrderBlocks) return 0.0;
    
    double score = 0.0;
    double current_price = symbol_info.Bid();
    double max_score = 0.0;
    
    // Check elite order blocks for confluence
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        const SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        
        // Skip inactive or mitigated order blocks
        if(ob.state == OB_STATE_DISABLED || ob.state == OB_STATE_MITIGATED) 
            continue;
        
        double ob_score = 0.0;
        
        // Calculate proximity score
        bool price_in_ob = (current_price >= ob.low_price && current_price <= ob.high_price);
        
        if(price_in_ob)
        {
            // Price is within order block - excellent confluence
            ob_score = ob.probability_score * 0.9; // Use 90% of probability score
            
            // Bonus for high-quality order blocks
            if(ob.quality == OB_QUALITY_ELITE) ob_score += 15.0;
            else if(ob.quality == OB_QUALITY_HIGH) ob_score += 10.0;
            else if(ob.quality == OB_QUALITY_MEDIUM) ob_score += 5.0;
            
            // Bonus for institutional order blocks
            if(ob.is_institutional) ob_score += 10.0;
            
            // Bonus for fresh order blocks
            if(ob.is_fresh) ob_score += 8.0;
            
            // Confluence bonus
            ob_score += ob.confluence_score * 0.2;
        }
        else
        {
            // Calculate distance-based score
            double distance = MathMin(
                MathAbs(current_price - ob.high_price),
                MathAbs(current_price - ob.low_price)
            );
            
            double max_distance = 100 * _Point; // 10 pips maximum consideration
            
            if(distance <= max_distance)
            {
                double proximity_factor = 1.0 - (distance / max_distance);
                ob_score = ob.probability_score * proximity_factor * 0.7; // Reduce score for distance
                
                // Quality bonuses
                if(ob.quality == OB_QUALITY_ELITE) ob_score += 10.0;
                else if(ob.quality == OB_QUALITY_HIGH) ob_score += 7.0;
                else if(ob.quality == OB_QUALITY_MEDIUM) ob_score += 3.0;
                
                // Institutional bonus
                if(ob.is_institutional) ob_score += 5.0;
            }
        }
        
        // Direction alignment bonus
        double signal_direction = 0;
        if(current_price < ob.refined_entry && ob.type == OB_BULLISH) signal_direction = 1;
        else if(current_price > ob.refined_entry && ob.type == OB_BEARISH) signal_direction = 1;
        
        if(signal_direction > 0) ob_score += 10.0;
        
        // Keep track of the best order block score
        if(ob_score > max_score) max_score = ob_score;
    }
    
    score = max_score;
    
    // Ensure score doesn't exceed 100
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Calculate Fair Value Gap Score - Enhanced Elite Version         |
//+------------------------------------------------------------------+
double CalculateFVGScore()
{
    if(!InpEnableFVG) return 0.0;
    
    double score = 0.0;
    double current_price = symbol_info.Bid();
    double max_score = 0.0;
    
    // Check elite Fair Value Gaps for confluence
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        const SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        
        // Skip filled or expired FVGs
        if(fvg.state == FVG_STATE_FILLED || fvg.state == FVG_STATE_EXPIRED) 
            continue;
        
        double fvg_score = 0.0;
        
        // Calculate proximity and interaction score
        bool price_in_fvg = (current_price >= fvg.lower_level && current_price <= fvg.upper_level);
        
        if(price_in_fvg)
        {
            // Price is within FVG - excellent confluence
            double fill_factor = 1.0 - (fvg.fill_percentage / 100.0);
            fvg_score = fvg.quality_score * fill_factor * 0.9;
            
            // Quality bonuses
            if(fvg.quality == FVG_QUALITY_ELITE) fvg_score += 20.0;
            else if(fvg.quality == FVG_QUALITY_HIGH) fvg_score += 15.0;
            else if(fvg.quality == FVG_QUALITY_MEDIUM) fvg_score += 8.0;
            
            // Institutional bonus
            if(fvg.is_institutional) fvg_score += 12.0;
            
            // Fresh FVG bonus
            if(fvg.is_fresh) fvg_score += 10.0;
            
            // Expected reaction bonus
            fvg_score += fvg.expected_reaction * 15.0;
            
            // Confluence bonus
            fvg_score += fvg.confluence_score * 0.25;
        }
        else
        {
            // Calculate proximity score
            double distance = MathMin(
                MathAbs(current_price - fvg.upper_level),
                MathAbs(current_price - fvg.lower_level)
            );
            
            double max_distance = 50 * _Point; // 5 pips maximum consideration
            
            if(distance <= max_distance)
            {
                double proximity_factor = 1.0 - (distance / max_distance);
                fvg_score = fvg.quality_score * proximity_factor * 0.7;
                
                // Quality bonuses (reduced for distance)
                if(fvg.quality == FVG_QUALITY_ELITE) fvg_score += 12.0;
                else if(fvg.quality == FVG_QUALITY_HIGH) fvg_score += 8.0;
                else if(fvg.quality == FVG_QUALITY_MEDIUM) fvg_score += 4.0;
                
                // Institutional bonus
                if(fvg.is_institutional) fvg_score += 6.0;
                
                // Expected reaction bonus
                fvg_score += fvg.expected_reaction * 8.0;
            }
        }
        
        // Direction alignment bonus
        double signal_direction = 0;
        if(current_price <= fvg.optimal_entry && fvg.type == FVG_BULLISH) signal_direction = 1;
        else if(current_price >= fvg.optimal_entry && fvg.type == FVG_BEARISH) signal_direction = 1;
        
        if(signal_direction > 0) fvg_score += 8.0;
        
        // Time decay factor
        fvg_score *= fvg.time_decay_factor;
        
        // Partial fill penalty
        if(fvg.state == FVG_STATE_PARTIAL)
        {
            double penalty = fvg.fill_percentage * 0.005; // Up to 50% penalty
            fvg_score *= (1.0 - penalty);
        }
        
        // Keep track of the best FVG score
        if(fvg_score > max_score) max_score = fvg_score;
    }
    
    score = max_score;
    
    // Ensure score doesn't exceed 100
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Calculate Liquidity Score                                       |
//+------------------------------------------------------------------+
double CalculateLiquidityScore()
{
    if(!InpEnableLiquidity) return 0.0;
    
    double score = 0.0;
    double current_price = symbol_info.Bid();
    
    // Check for liquidity zones
    for(int i = 0; i < g_liq_count; i++)
    {
        if(g_liquidity_zones[i].is_swept) continue;
        
        double distance = MathAbs(current_price - g_liquidity_zones[i].price_level);
        double max_distance = 80 * _Point; // 8 pips
        
        if(distance <= max_distance)
        {
            double proximity_score = (1.0 - distance / max_distance) * 75.0;
            score += proximity_score * g_liquidity_zones[i].strength;
        }
    }
    
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Calculate Structure Score                                        |
//+------------------------------------------------------------------+
double CalculateStructureScore()
{
    double score = 0.0;
    
    // Get EMA values
    double ema_fast[3], ema_medium[3], ema_slow[3];
    
    if(CopyBuffer(h_ema_fast, 0, 0, 3, ema_fast) <= 0 ||
       CopyBuffer(h_ema_medium, 0, 0, 3, ema_medium) <= 0 ||
       CopyBuffer(h_ema_slow, 0, 0, 3, ema_slow) <= 0)
        return 0.0;
    
    ArraySetAsSeries(ema_fast, true);
    ArraySetAsSeries(ema_medium, true);
    ArraySetAsSeries(ema_slow, true);
    
    // Check EMA alignment
    if(ema_fast[0] > ema_medium[0] && ema_medium[0] > ema_slow[0])
    {
        score += 40.0; // Bullish alignment
    }
    else if(ema_fast[0] < ema_medium[0] && ema_medium[0] < ema_slow[0])
    {
        score += 40.0; // Bearish alignment
    }
    
    // Check for recent EMA crossover
    if((ema_fast[1] <= ema_medium[1] && ema_fast[0] > ema_medium[0]) ||
       (ema_fast[1] >= ema_medium[1] && ema_fast[0] < ema_medium[0]))
    {
        score += 30.0; // Fresh crossover
    }
    
    // Check price position relative to EMAs
    double current_price = symbol_info.Bid();
    if((current_price > ema_fast[0] && ema_fast[0] > ema_medium[0]) ||
       (current_price < ema_fast[0] && ema_fast[0] < ema_medium[0]))
    {
        score += 30.0; // Price aligns with structure
    }
    
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Calculate Price Action Score                                    |
//+------------------------------------------------------------------+
double CalculatePriceActionScore()
{
    if(!InpEnablePriceAction) return 0.0;
    
    double score = 0.0;
    
    // Get recent price data
    MqlRates rates[5];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 5, rates) <= 0) return 0.0;
    ArraySetAsSeries(rates, true);
    
    // Check for bullish engulfing
    if(IsBullishEngulfing(rates, 1))
    {
        score += 35.0;
    }
    
    // Check for bearish engulfing
    if(IsBearishEngulfing(rates, 1))
    {
        score += 35.0;
    }
    
    // Check for pin bars
    if(IsBullishPinBar(rates, 1))
    {
        score += 30.0;
    }
    
    if(IsBearishPinBar(rates, 1))
    {
        score += 30.0;
    }
    
    // Check for doji
    if(IsDoji(rates, 1))
    {
        score += 20.0;
    }
    
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Calculate Timeframe Score                                       |
//+------------------------------------------------------------------+
double CalculateTimeframeScore()
{
    double score = 0.0;
    
    // Weekly bias (30% weight)
    if(InpUseWeeklyBias && IsWeeklyBiasAligned()) score += 30.0;
    
    // Daily trend (25% weight)
    if(InpUseDailyTrend && IsDailyTrendValid()) score += 25.0;
    
    // H4 structure (20% weight)
    if(InpUseH4Structure && IsH4StructureValid()) score += 20.0;
    
    // H1 setup (15% weight)
    if(InpUseH1Setup && IsH1SetupValid()) score += 15.0;
    
    // M15 execution (10% weight)
    if(InpUseM15Execution && IsM15ExecutionValid()) score += 10.0;
    
    return MathMin(score, 100.0);
}

//+------------------------------------------------------------------+
//| Emergency protection and utility functions                       |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| FTMO Ultra-Conservative Compliance Monitoring System            |
//+------------------------------------------------------------------+

// FTMO compliance structure
struct SFTMOComplianceData
{
    // Daily limits and tracking
    double daily_loss_limit;           // 5% max daily loss
    double daily_loss_current;         // Current daily loss
    double daily_starting_balance;     // Balance at day start
    datetime daily_reset_time;         // Last daily reset
    
    // Overall limits
    double max_drawdown_limit;         // 10% max total drawdown
    double max_drawdown_current;       // Current drawdown
    double account_high_water_mark;    // Highest balance reached
    
    // Trading limits
    int max_trades_per_day;            // Maximum trades per day
    int trades_today_count;            // Current trades today
    double max_lot_size_allowed;       // Maximum lot size
    double max_risk_per_trade;         // Maximum risk per trade (1%)
    
    // Compliance status
    bool is_compliant;                 // Overall compliance status
    bool daily_limit_breached;        // Daily loss limit breach
    bool drawdown_limit_breached;     // Max drawdown breach
    bool trading_halted;               // Emergency trading halt
    
    // Risk monitoring
    double total_open_risk;            // Total risk from open positions
    double correlation_risk;           // Risk from correlated positions
    datetime last_check_time;         // Last compliance check
    
    // Violation tracking
    int violation_count;               // Number of violations today
    string last_violation_reason;     // Last violation reason
    datetime last_violation_time;     // Last violation time
    
    // Conservative factors
    double safety_buffer;              // Safety buffer (default 20%)
    bool weekend_gap_protection;      // Weekend gap protection
    bool news_trading_halt;           // News-based trading halt
};

// Global FTMO compliance data
SFTMOComplianceData g_ftmo_compliance;

//+------------------------------------------------------------------+
//| Initialize FTMO Compliance System                              |
//+------------------------------------------------------------------+
void InitializeFTMOCompliance()
{
    // Set FTMO limits (ultra-conservative)
    g_ftmo_compliance.daily_loss_limit = 4.0;        // 4% instead of 5% (buffer)
    g_ftmo_compliance.max_drawdown_limit = 8.0;      // 8% instead of 10% (buffer)
    g_ftmo_compliance.max_trades_per_day = 3;        // Conservative limit
    g_ftmo_compliance.max_risk_per_trade = 0.8;      // 0.8% instead of 1% (buffer)
    g_ftmo_compliance.safety_buffer = 0.2;           // 20% safety buffer
    
    // Initialize tracking
    g_ftmo_compliance.daily_starting_balance = account_info.Balance();
    g_ftmo_compliance.account_high_water_mark = account_info.Balance();
    g_ftmo_compliance.daily_reset_time = TimeCurrent();
    
    // Initialize status
    g_ftmo_compliance.is_compliant = true;
    g_ftmo_compliance.daily_limit_breached = false;
    g_ftmo_compliance.drawdown_limit_breached = false;
    g_ftmo_compliance.trading_halted = false;
    
    // Initialize risk tracking
    g_ftmo_compliance.total_open_risk = 0.0;
    g_ftmo_compliance.correlation_risk = 0.0;
    g_ftmo_compliance.trades_today_count = 0;
    g_ftmo_compliance.violation_count = 0;
    
    // Conservative settings
    g_ftmo_compliance.weekend_gap_protection = true;
    g_ftmo_compliance.news_trading_halt = true;
    
    g_ftmo_compliance.last_check_time = TimeCurrent();
    
    Print("FTMO Compliance System Initialized:");
    Print("- Daily Loss Limit: ", g_ftmo_compliance.daily_loss_limit, "%");
    Print("- Max Drawdown Limit: ", g_ftmo_compliance.max_drawdown_limit, "%");
    Print("- Max Trades Per Day: ", g_ftmo_compliance.max_trades_per_day);
    Print("- Max Risk Per Trade: ", g_ftmo_compliance.max_risk_per_trade, "%");
}

//+------------------------------------------------------------------+
//| Real-time FTMO Compliance Check                                |
//+------------------------------------------------------------------+
bool CheckFTMOCompliance()
{
    // Update compliance data
    UpdateFTMOComplianceData();
    
    // Check if trading should be halted
    if(g_ftmo_compliance.trading_halted)
    {
        Print("⛔ FTMO COMPLIANCE: Trading halted due to rule violation");
        return false;
    }
    
    // 1. Daily Loss Limit Check (Critical)
    if(CheckDailyLossLimit() == false)
    {
        g_ftmo_compliance.daily_limit_breached = true;
        g_ftmo_compliance.trading_halted = true;
        HaltTradingEmergency("Daily loss limit exceeded");
        return false;
    }
    
    // 2. Maximum Drawdown Check (Critical)
    if(CheckMaxDrawdownLimit() == false)
    {
        g_ftmo_compliance.drawdown_limit_breached = true;
        g_ftmo_compliance.trading_halted = true;
        HaltTradingEmergency("Maximum drawdown limit exceeded");
        return false;
    }
    
    // 3. Daily Trade Count Check
    if(g_ftmo_compliance.trades_today_count >= g_ftmo_compliance.max_trades_per_day)
    {
        Print("⚠️ FTMO COMPLIANCE: Daily trade limit reached (", g_ftmo_compliance.trades_today_count, "/", g_ftmo_compliance.max_trades_per_day, ")");
        return false;
    }
    
    // 4. Weekend Gap Protection
    if(g_ftmo_compliance.weekend_gap_protection && IsWeekendGapRisk())
    {
        Print("⚠️ FTMO COMPLIANCE: Weekend gap protection active");
        return false;
    }
    
    // 5. News Trading Halt
    if(g_ftmo_compliance.news_trading_halt && IsHighImpactNewsTime())
    {
        Print("⚠️ FTMO COMPLIANCE: News trading halt active");
        return false;
    }
    
    // 6. Total Open Risk Check
    if(CheckTotalOpenRisk() == false)
    {
        Print("⚠️ FTMO COMPLIANCE: Total open risk exceeds limits");
        return false;
    }
    
    // 7. Correlation Risk Check
    if(CheckCorrelationRisk() == false)
    {
        Print("⚠️ FTMO COMPLIANCE: Correlation risk too high");
        return false;
    }
    
    // All checks passed
    g_ftmo_compliance.is_compliant = true;
    return true;
}

//+------------------------------------------------------------------+
//| Update FTMO Compliance Data                                    |
//+------------------------------------------------------------------+
void UpdateFTMOComplianceData()
{
    double current_balance = account_info.Balance();
    double current_equity = account_info.Equity();
    
    // Check for new day reset
    datetime current_time = TimeCurrent();
    MqlDateTime dt_current, dt_last;
    TimeToStruct(current_time, dt_current);
    TimeToStruct(g_ftmo_compliance.daily_reset_time, dt_last);
    
    if(dt_current.day != dt_last.day)
    {
        ResetDailyFTMOTracking();
    }
    
    // Update daily loss calculation
    g_ftmo_compliance.daily_loss_current = 
        ((g_ftmo_compliance.daily_starting_balance - current_equity) / g_ftmo_compliance.daily_starting_balance) * 100.0;
    
    // Update high water mark
    if(current_balance > g_ftmo_compliance.account_high_water_mark)
    {
        g_ftmo_compliance.account_high_water_mark = current_balance;
    }
    
    // Update current drawdown
    g_ftmo_compliance.max_drawdown_current = 
        ((g_ftmo_compliance.account_high_water_mark - current_equity) / g_ftmo_compliance.account_high_water_mark) * 100.0;
    
    // Update total open risk
    g_ftmo_compliance.total_open_risk = CalculateTotalOpenRisk();
    
    g_ftmo_compliance.last_check_time = current_time;
}

//+------------------------------------------------------------------+
//| Check Daily Loss Limit                                         |
//+------------------------------------------------------------------+
bool CheckDailyLossLimit()
{
    double effective_limit = g_ftmo_compliance.daily_loss_limit * (1.0 - g_ftmo_compliance.safety_buffer);
    
    if(g_ftmo_compliance.daily_loss_current >= effective_limit)
    {
        LogFTMOViolation("Daily loss limit approached: " + 
                        DoubleToString(g_ftmo_compliance.daily_loss_current, 2) + "% >= " + 
                        DoubleToString(effective_limit, 2) + "%");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check Maximum Drawdown Limit                                   |
//+------------------------------------------------------------------+
bool CheckMaxDrawdownLimit()
{
    double effective_limit = g_ftmo_compliance.max_drawdown_limit * (1.0 - g_ftmo_compliance.safety_buffer);
    
    if(g_ftmo_compliance.max_drawdown_current >= effective_limit)
    {
        LogFTMOViolation("Maximum drawdown limit approached: " + 
                        DoubleToString(g_ftmo_compliance.max_drawdown_current, 2) + "% >= " + 
                        DoubleToString(effective_limit, 2) + "%");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check Total Open Risk                                          |
//+------------------------------------------------------------------+
bool CheckTotalOpenRisk()
{
    double max_total_risk = account_info.Balance() * (g_ftmo_compliance.max_risk_per_trade / 100.0) * 2.0; // Max 2 positions
    
    if(g_ftmo_compliance.total_open_risk >= max_total_risk)
    {
        LogFTMOViolation("Total open risk too high: " + 
                        DoubleToString(g_ftmo_compliance.total_open_risk, 2) + " >= " + 
                        DoubleToString(max_total_risk, 2));
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check Correlation Risk                                         |
//+------------------------------------------------------------------+
bool CheckCorrelationRisk()
{
    // For XAUUSD EA, this would check if multiple XAUUSD positions exist
    int xauusd_positions = 0;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(position_info.SelectByIndex(i))
        {
            if(position_info.Symbol() == _Symbol && position_info.Magic() == InpMagicNumber)
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

//+------------------------------------------------------------------+
//| Calculate Total Open Risk                                      |
//+------------------------------------------------------------------+
double CalculateTotalOpenRisk()
{
    double total_risk = 0.0;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(position_info.SelectByIndex(i))
        {
            if(position_info.Magic() == InpMagicNumber)
            {
                double position_risk = CalculatePositionRisk(position_info.Ticket());
                total_risk += position_risk;
            }
        }
    }
    
    return total_risk;
}

//+------------------------------------------------------------------+
//| Calculate Position Risk                                        |
//+------------------------------------------------------------------+
double CalculatePositionRisk(ulong ticket)
{
    if(!position_info.SelectByTicket(ticket)) return 0.0;
    
    double entry_price = position_info.PriceOpen();
    double sl_price = position_info.StopLoss();
    double volume = position_info.Volume();
    
    if(sl_price == 0.0) return 0.0; // No SL set
    
    double sl_distance = MathAbs(entry_price - sl_price);
    double tick_value = symbol_info.TickValue();
    
    return (sl_distance / _Point) * tick_value * volume;
}

//+------------------------------------------------------------------+
//| Validate Trade Before Execution                                |
//+------------------------------------------------------------------+
bool ValidateFTMOTradeCompliance(const SConfluenceSignal& signal)
{
    // 1. Check if trading is halted
    if(g_ftmo_compliance.trading_halted)
    {
        return false;
    }
    
    // 2. Check daily trade limit
    if(g_ftmo_compliance.trades_today_count >= g_ftmo_compliance.max_trades_per_day)
    {
        return false;
    }
    
    // 3. Calculate potential trade risk
    double lot_size = CalculateLotSize(signal);
    double potential_risk = CalculatePotentialLoss(signal);
    double risk_percentage = (potential_risk / account_info.Balance()) * 100.0;
    
    // 4. Check individual trade risk
    if(risk_percentage > g_ftmo_compliance.max_risk_per_trade)
    {
        LogFTMOViolation("Trade risk too high: " + DoubleToString(risk_percentage, 2) + "% > " + 
                        DoubleToString(g_ftmo_compliance.max_risk_per_trade, 2) + "%");
        return false;
    }
    
    // 5. Check combined risk after this trade
    double total_risk_after = g_ftmo_compliance.total_open_risk + potential_risk;
    double max_combined_risk = account_info.Balance() * (g_ftmo_compliance.max_risk_per_trade / 100.0) * 2.0;
    
    if(total_risk_after > max_combined_risk)
    {
        LogFTMOViolation("Combined risk would be too high: " + DoubleToString(total_risk_after, 2));
        return false;
    }
    
    // 6. Check if trade would cause daily loss limit approach
    double current_equity = account_info.Equity();
    double potential_daily_loss = ((g_ftmo_compliance.daily_starting_balance - (current_equity - potential_risk)) / 
                                   g_ftmo_compliance.daily_starting_balance) * 100.0;
    
    double safe_daily_limit = g_ftmo_compliance.daily_loss_limit * (1.0 - g_ftmo_compliance.safety_buffer);
    
    if(potential_daily_loss > safe_daily_limit)
    {
        LogFTMOViolation("Trade could cause daily limit breach: " + DoubleToString(potential_daily_loss, 2) + "%");
        return false;
    }
    
    // 7. Check lot size limits
    double max_lot = symbol_info.LotsMax();
    double conservative_max_lot = MathMin(max_lot, 1.0); // Conservative 1 lot max
    
    if(lot_size > conservative_max_lot)
    {
        LogFTMOViolation("Lot size too large: " + DoubleToString(lot_size, 2) + " > " + 
                        DoubleToString(conservative_max_lot, 2));
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Reset Daily FTMO Tracking                                      |
//+------------------------------------------------------------------+
void ResetDailyFTMOTracking()
{
    g_ftmo_compliance.daily_starting_balance = account_info.Balance();
    g_ftmo_compliance.daily_loss_current = 0.0;
    g_ftmo_compliance.trades_today_count = 0;
    g_ftmo_compliance.violation_count = 0;
    g_ftmo_compliance.daily_reset_time = TimeCurrent();
    
    // Reset daily flags but keep overall compliance status
    g_ftmo_compliance.daily_limit_breached = false;
    
    Print("📅 FTMO Daily Tracking Reset | Starting Balance: ", g_ftmo_compliance.daily_starting_balance);
}

//+------------------------------------------------------------------+
//| Log FTMO Violation                                             |
//+------------------------------------------------------------------+
void LogFTMOViolation(string reason)
{
    g_ftmo_compliance.violation_count++;
    g_ftmo_compliance.last_violation_reason = reason;
    g_ftmo_compliance.last_violation_time = TimeCurrent();
    
    Print("🚨 FTMO VIOLATION #", g_ftmo_compliance.violation_count, ": ", reason);
    
    // Log to file for audit trail
    int file_handle = FileOpen("FTMO_Violations_" + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) + ".log", 
                               FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_APPEND, "\t");
    if(file_handle != INVALID_HANDLE)
    {
        FileWrite(file_handle, TimeToString(TimeCurrent()), reason, 
                  DoubleToString(account_info.Balance(), 2), 
                  DoubleToString(account_info.Equity(), 2));
        FileClose(file_handle);
    }
}

//+------------------------------------------------------------------+
//| Emergency Trading Halt                                         |
//+------------------------------------------------------------------+
void HaltTradingEmergency(string reason)
{
    g_ftmo_compliance.trading_halted = true;
    g_ftmo_compliance.is_compliant = false;
    
    Print("🛑 EMERGENCY TRADING HALT: ", reason);
    Print("🛑 All trading suspended to protect FTMO account");
    
    // Close all open positions immediately
    CloseAllPositionsEmergency(reason);
    
    // Log emergency halt
    int file_handle = FileOpen("FTMO_Emergency_Halts_" + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) + ".log", 
                               FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_APPEND, "\t");
    if(file_handle != INVALID_HANDLE)
    {
        FileWrite(file_handle, TimeToString(TimeCurrent()), "EMERGENCY HALT", reason, 
                  DoubleToString(account_info.Balance(), 2), 
                  DoubleToString(account_info.Equity(), 2));
        FileClose(file_handle);
    }
    
    // Send alert if possible
    Alert("FTMO EMERGENCY HALT: ", reason);
}

//+------------------------------------------------------------------+
//| Close All Positions Emergency                                  |
//+------------------------------------------------------------------+
void CloseAllPositionsEmergency(string reason)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(position_info.SelectByIndex(i))
        {
            if(position_info.Magic() == InpMagicNumber)
            {
                ulong ticket = position_info.Ticket();
                if(trade.PositionClose(ticket))
                {
                    Print("🛑 Emergency close: Ticket ", ticket, " - Reason: ", reason);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check Weekend Gap Risk                                         |
//+------------------------------------------------------------------+
bool IsWeekendGapRisk()
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

//+------------------------------------------------------------------+
//| Track Trade Execution for FTMO                                |
//+------------------------------------------------------------------+
void TrackFTMOTradeExecution(ulong ticket, double lot_size, double risk)
{
    g_ftmo_compliance.trades_today_count++;
    
    Print("📊 FTMO Trade Executed: Ticket ", ticket, 
          " | Trades Today: ", g_ftmo_compliance.trades_today_count, "/", g_ftmo_compliance.max_trades_per_day,
          " | Risk: ", DoubleToString(risk, 2), " (", DoubleToString((risk/account_info.Balance())*100, 2), "%)");
}

//+------------------------------------------------------------------+
//| Get FTMO Compliance Status Report                              |
//+------------------------------------------------------------------+
string GetFTMOComplianceReport()
{
    string report = "\n=== FTMO COMPLIANCE STATUS ===";
    report += "\nOverall Status: " + (g_ftmo_compliance.is_compliant ? "✅ COMPLIANT" : "❌ NON-COMPLIANT");
    report += "\nTrading Status: " + (g_ftmo_compliance.trading_halted ? "🛑 HALTED" : "✅ ACTIVE");
    
    report += "\n\n--- Daily Limits ---";
    report += "\nDaily Loss: " + DoubleToString(g_ftmo_compliance.daily_loss_current, 2) + "% / " + 
              DoubleToString(g_ftmo_compliance.daily_loss_limit, 2) + "%";
    report += "\nTrades Today: " + IntegerToString(g_ftmo_compliance.trades_today_count) + " / " + 
              IntegerToString(g_ftmo_compliance.max_trades_per_day);
    
    report += "\n\n--- Overall Limits ---";
    report += "\nMax Drawdown: " + DoubleToString(g_ftmo_compliance.max_drawdown_current, 2) + "% / " + 
              DoubleToString(g_ftmo_compliance.max_drawdown_limit, 2) + "%";
    report += "\nTotal Open Risk: $" + DoubleToString(g_ftmo_compliance.total_open_risk, 2);
    
    if(g_ftmo_compliance.violation_count > 0)
    {
        report += "\n\n--- Violations ---";
        report += "\nViolations Today: " + IntegerToString(g_ftmo_compliance.violation_count);
        report += "\nLast Violation: " + g_ftmo_compliance.last_violation_reason;
    }
    
    report += "\n========================\n";
    
    return report;
}

//+------------------------------------------------------------------+
//| Enhanced CheckEmergencyConditions with FTMO Integration       |
//+------------------------------------------------------------------+

bool IsTradingAllowed()
{
    if(g_emergency_stop) return false;
    if(g_daily_limit_reached) return false;
    if(!ValidateSessionFilter()) return false;
    
    return true;
}

void CheckNewDay()
{
    datetime today = (datetime)StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    if(g_today_date != today)
    {
        ResetDailyStats();
        g_today_date = today;
    }
}

void ResetDailyStats()
{
    g_trades_today = 0;
    g_daily_profit = 0.0;
    g_daily_starting_balance = account_info.Balance();
    g_daily_limit_reached = false;
    g_emergency_stop = false;
}

SPerformanceMetrics CalculatePerformanceMetrics()
{
    SPerformanceMetrics metrics;
    // Initialize all members individually to avoid deprecation warning
    metrics.total_profit = 0.0;
    metrics.total_trades = 0.0;
    metrics.win_rate = 0.0;
    metrics.profit_factor = 0.0;
    metrics.max_drawdown = 0.0;
    metrics.sharpe_ratio = 0.0;
    metrics.current_drawdown = 0.0;
    metrics.ftmo_compliant = false;
    
    // Calculate basic metrics from account history
    HistorySelect(0, TimeCurrent());
    int total_deals = HistoryDealsTotal();
    
    double total_profit = 0;
    int winning_trades = 0;
    int total_trades = 0;
    
    for(int i = 0; i < total_deals; i++)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == InpMagicNumber)
        {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            if(profit != 0) // Only count closed trades
            {
                total_profit += profit;
                total_trades++;
                if(profit > 0) winning_trades++;
            }
        }
    }
    
    metrics.total_profit = total_profit;
    metrics.total_trades = total_trades;
    metrics.win_rate = (total_trades > 0) ? (double)winning_trades / total_trades : 0;
    metrics.ftmo_compliant = (metrics.win_rate > 0.7 && total_profit > 0);
    
    return metrics;
}

// Global Detector Instances
CEliteOrderBlockDetector* g_elite_ob_detector = NULL;
CEliteFVGDetector* g_elite_fvg_detector = NULL;
CInstitutionalLiquidityDetector* g_institutional_liq_detector = NULL;

// Legacy arrays for compatibility
struct SLiquidityZone {
    double price_level;
    bool is_swept;
    double strength;
};
SLiquidityZone g_liquidity_zones[50];
int g_liq_count = 0;

struct SFairValueGap {
    double upper_level;
    double lower_level;
};
SFairValueGap g_fair_value_gaps[30];
int g_fvg_count = 0;

//+------------------------------------------------------------------+
//| Institutional Liquidity Detector Implementation                  |
//+------------------------------------------------------------------+

// Constructor
CInstitutionalLiquidityDetector::CInstitutionalLiquidityDetector()
{
    // Initialize detection parameters for XAUUSD
    m_min_accumulation_size = 500.0 * _Point;   // 50 pips minimum accumulation
    m_institutional_threshold = 1000.0 * _Point; // 100 pips for institutional
    m_min_touch_count = 2;                      // Minimum 2 touches for significance
    m_sweep_validation_distance = 20.0 * _Point; // 2 pips for valid sweep
    
    // Initialize counts
    m_weekly_count = 0;
    m_daily_count = 0;
    m_session_count = 0;
    
    // Set analysis timeframes
    m_analysis_timeframes[0] = PERIOD_W1;
    m_analysis_timeframes[1] = PERIOD_D1;
    m_analysis_timeframes[2] = PERIOD_H4;
    m_analysis_timeframes[3] = PERIOD_H1;
    m_analysis_timeframes[4] = PERIOD_M15;
    m_timeframe_count = 5;
}

// Destructor
CInstitutionalLiquidityDetector::~CInstitutionalLiquidityDetector()
{
    // Cleanup if needed
}

// Main detection method for institutional liquidity
bool CInstitutionalLiquidityDetector::DetectInstitutionalLiquidity()
{
    // Reset current liquidity count
    g_institutional_liq_count = 0;
    
    // Detect liquidity across different timeframes
    bool weekly_detected = DetectWeeklyLiquidity();
    bool daily_detected = DetectDailyLiquidity();
    bool session_detected = DetectSessionLiquidity();
    
    return g_institutional_liq_count > 0;
}

// Detect weekly liquidity levels
bool CInstitutionalLiquidityDetector::DetectWeeklyLiquidity()
{
    MqlRates weekly_rates[20];
    if(CopyRates(_Symbol, PERIOD_W1, 0, 20, weekly_rates) <= 0)
        return false;
        
    ArraySetAsSeries(weekly_rates, true);
    
    // Simple weekly high/low detection
    for(int i = 1; i < 15; i++)
    {
        if(weekly_rates[i].high > weekly_rates[i-1].high && weekly_rates[i].high > weekly_rates[i+1].high)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(weekly_rates[i].high, LIQUIDITY_WEEKLY);
            }
        }
        if(weekly_rates[i].low < weekly_rates[i-1].low && weekly_rates[i].low < weekly_rates[i+1].low)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(weekly_rates[i].low, LIQUIDITY_WEEKLY);
            }
        }
    }
    
    return true;
}

// Detect daily liquidity levels
bool CInstitutionalLiquidityDetector::DetectDailyLiquidity()
{
    MqlRates daily_rates[30];
    if(CopyRates(_Symbol, PERIOD_D1, 0, 30, daily_rates) <= 0)
        return false;
        
    ArraySetAsSeries(daily_rates, true);
    
    // Simple daily high/low detection
    for(int i = 1; i < 25; i++)
    {
        if(daily_rates[i].high > daily_rates[i-1].high && daily_rates[i].high > daily_rates[i+1].high)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(daily_rates[i].high, LIQUIDITY_DAILY);
            }
        }
        if(daily_rates[i].low < daily_rates[i-1].low && daily_rates[i].low < daily_rates[i+1].low)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(daily_rates[i].low, LIQUIDITY_DAILY);
            }
        }
    }
    
    return true;
}

// Detect session liquidity levels
bool CInstitutionalLiquidityDetector::DetectSessionLiquidity()
{
    MqlRates h1_rates[50];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, h1_rates) <= 0)
        return false;
        
    ArraySetAsSeries(h1_rates, true);
    
    // Session high/low detection
    for(int i = 2; i < 45; i++)
    {
        if(h1_rates[i].high > h1_rates[i-1].high && h1_rates[i].high > h1_rates[i+1].high &&
           h1_rates[i].high > h1_rates[i-2].high && h1_rates[i].high > h1_rates[i+2].high)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(h1_rates[i].high, LIQUIDITY_SESSION);
            }
        }
        if(h1_rates[i].low < h1_rates[i-1].low && h1_rates[i].low < h1_rates[i+1].low &&
           h1_rates[i].low < h1_rates[i-2].low && h1_rates[i].low < h1_rates[i+2].low)
        {
            if(g_institutional_liq_count < 50)
            {
                CreateSimpleLiquidityPool(h1_rates[i].low, LIQUIDITY_SESSION);
            }
        }
    }
    
    return true;
}

// Create simple liquidity pool
void CInstitutionalLiquidityDetector::CreateSimpleLiquidityPool(double price_level, ENUM_LIQUIDITY_TYPE type)
{
    SInstitutionalLiquidityPool pool;
    
    // Initialize basic properties
    pool.formation_time = TimeCurrent();
    pool.price_level = price_level;
    pool.type = type;
    pool.state = LIQUIDITY_UNTAPPED;
    pool.touch_count = 0;
    pool.is_fresh = true;
    
    // Set quality based on type
    switch(type)
    {
        case LIQUIDITY_WEEKLY: 
            pool.quality = LIQUIDITY_QUALITY_HIGH;
            pool.quality_score = 85.0;
            pool.is_institutional = true;
            break;
        case LIQUIDITY_DAILY:
            pool.quality = LIQUIDITY_QUALITY_MEDIUM;
            pool.quality_score = 70.0;
            pool.is_institutional = true;
            break;
        default:
            pool.quality = LIQUIDITY_QUALITY_MEDIUM;
            pool.quality_score = 60.0;
            pool.is_institutional = false;
            break;
    }
    
    // Set other properties
    pool.sweep_probability = 0.7;
    pool.confluence_score = 0.0;
    pool.time_decay_factor = 1.0;
    
    // Add to array
    g_institutional_liquidity[g_institutional_liq_count] = pool;
    g_institutional_liq_count++;
}

// Update liquidity status
void CInstitutionalLiquidityDetector::UpdateLiquidityStatus()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        SInstitutionalLiquidityPool& pool = g_institutional_liquidity[i];
        
        // Check if price has approached the liquidity level
        double distance = MathAbs(current_price - pool.price_level);
        
        if(distance <= 30 * _Point) // Within 3 pips
        {
            pool.touch_count++;
            
            // Check if swept
            if(distance <= 10 * _Point) // Within 1 pip
            {
                pool.state = LIQUIDITY_SWEPT;
                pool.is_fresh = false;
            }
        }
        
        // Age-based expiry
        if(TimeCurrent() - pool.formation_time > 7 * 24 * 3600) // 7 days
        {
            if(pool.state == LIQUIDITY_UNTAPPED)
                pool.state = LIQUIDITY_EXPIRED;
        }
    }
}

// Remove swept liquidity
void CInstitutionalLiquidityDetector::RemoveSweptLiquidity()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        if(g_institutional_liquidity[i].state != LIQUIDITY_SWEPT &&
           g_institutional_liquidity[i].state != LIQUIDITY_EXPIRED)
        {
            if(valid_count != i)
            {
                g_institutional_liquidity[valid_count] = g_institutional_liquidity[i];
            }
            valid_count++;
        }
    }
    
    g_institutional_liq_count = valid_count;
}

// Get active liquidity count
int CInstitutionalLiquidityDetector::GetActiveLiquidityCount()
{
    int active_count = 0;
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        if(g_institutional_liquidity[i].state == LIQUIDITY_UNTAPPED)
        {
            active_count++;
        }
    }
    
    return active_count;
}

// Get best liquidity score
double CInstitutionalLiquidityDetector::GetBestLiquidityScore()
{
    double best_score = 0.0;
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        if(g_institutional_liquidity[i].state == LIQUIDITY_UNTAPPED)
        {
            if(g_institutional_liquidity[i].quality_score > best_score)
                best_score = g_institutional_liquidity[i].quality_score;
        }
    }
    
    return best_score;
}

//+------------------------------------------------------------------+
//| Elite FVG Detector Implementation                                |
//+------------------------------------------------------------------+

// Constructor
CEliteFVGDetector::CEliteFVGDetector()
{
    // Initialize detection parameters for XAUUSD
    m_min_displacement_size = 150.0 * _Point;   // 15 pips minimum displacement
    m_volume_spike_threshold = 1.8;             // 1.8x average volume
    m_require_structure_break = true;           // Require structure break
    m_min_gap_size = 20.0 * _Point;            // 2 pips minimum gap
    m_max_gap_size = 500.0 * _Point;           // 50 pips maximum gap
    
    // Quality assessment
    m_institutional_threshold = 300.0 * _Point; // 30 pips for institutional
    m_confluence_weight = 0.3;                  // 30% weight for confluence
    
    // Set analysis timeframes
    m_analysis_timeframes[0] = PERIOD_H1;
    m_analysis_timeframes[1] = PERIOD_M15;
    m_analysis_timeframes[2] = PERIOD_M5;
    m_timeframe_count = 3;
}

// Destructor
CEliteFVGDetector::~CEliteFVGDetector()
{
    // Cleanup if needed
}

// Main detection method for elite FVGs
bool CEliteFVGDetector::DetectEliteFairValueGaps()
{
    // Reset current FVG count
    g_elite_fvg_count = 0;
    
    // Get market data for analysis
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 100, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Analyze each potential FVG formation (need 3 consecutive candles)
    for(int i = 2; i < 97; i++)
    {
        // Check for bullish FVG pattern
        if(DetectBullishFVG(rates, i))
        {
            SEliteFairValueGap fvg;
            if(CreateFVGStructure(rates, i, FVG_BULLISH, fvg))
            {
                if(ValidateFVG(fvg))
                {
                    g_elite_fair_value_gaps[g_elite_fvg_count] = fvg;
                    g_elite_fvg_count++;
                    if(g_elite_fvg_count >= 30) break;
                }
            }
        }
        
        // Check for bearish FVG pattern
        if(DetectBearishFVG(rates, i))
        {
            SEliteFairValueGap fvg;
            if(CreateFVGStructure(rates, i, FVG_BEARISH, fvg))
            {
                if(ValidateFVG(fvg))
                {
                    g_elite_fair_value_gaps[g_elite_fvg_count] = fvg;
                    g_elite_fvg_count++;
                    if(g_elite_fvg_count >= 30) break;
                }
            }
        }
    }
    
    // Sort FVGs by quality and proximity
    SortFVGsByQuality();
    
    return g_elite_fvg_count > 0;
}

// Detect bullish Fair Value Gap
bool CEliteFVGDetector::DetectBullishFVG(const MqlRates& rates[], int index)
{
    if(index < 2 || index >= ArraySize(rates) - 1) return false;
    
    // Bullish FVG: gap between candle[i+1].high and candle[i-1].low
    // Where candle[i] doesn't fill the gap
    
    double gap_top = rates[index+1].high;      // Previous candle high
    double gap_bottom = rates[index-1].low;    // Next candle low
    double middle_high = rates[index].high;    // Middle candle high
    double middle_low = rates[index].low;      // Middle candle low
    
    // Check if there's a valid gap
    if(gap_bottom >= gap_top) return false;    // No gap exists
    
    // Check if middle candle doesn't fill the gap
    if(middle_high >= gap_top || middle_low <= gap_bottom) return false;
    
    // Calculate gap size
    double gap_size = gap_top - gap_bottom;
    
    // Validate gap size
    if(gap_size < m_min_gap_size || gap_size > m_max_gap_size) return false;
    
    // Check for displacement after FVG formation
    double displacement = CalculateDisplacementAfterFVG(rates, index, true);
    if(displacement < m_min_displacement_size) return false;
    
    // Volume confirmation
    if(!HasVolumeConfirmation(rates, index)) return false;
    
    return true;
}

// Detect bearish Fair Value Gap
bool CEliteFVGDetector::DetectBearishFVG(const MqlRates& rates[], int index)
{
    if(index < 2 || index >= ArraySize(rates) - 1) return false;
    
    // Bearish FVG: gap between candle[i+1].low and candle[i-1].high
    // Where candle[i] doesn't fill the gap
    
    double gap_top = rates[index-1].high;      // Next candle high
    double gap_bottom = rates[index+1].low;    // Previous candle low
    double middle_high = rates[index].high;    // Middle candle high
    double middle_low = rates[index].low;      // Middle candle low
    
    // Check if there's a valid gap
    if(gap_bottom >= gap_top) return false;    // No gap exists
    
    // Check if middle candle doesn't fill the gap
    if(middle_high >= gap_top || middle_low <= gap_bottom) return false;
    
    // Calculate gap size
    double gap_size = gap_top - gap_bottom;
    
    // Validate gap size
    if(gap_size < m_min_gap_size || gap_size > m_max_gap_size) return false;
    
    // Check for displacement after FVG formation
    double displacement = CalculateDisplacementAfterFVG(rates, index, false);
    if(displacement < m_min_displacement_size) return false;
    
    // Volume confirmation
    if(!HasVolumeConfirmation(rates, index)) return false;
    
    return true;
}

// Create FVG structure
bool CEliteFVGDetector::CreateFVGStructure(const MqlRates& rates[], int index, ENUM_FVG_TYPE type, SEliteFairValueGap& fvg)
{
    // Initialize FVG structure
    fvg.formation_time = rates[index].time;
    fvg.type = type;
    fvg.state = FVG_STATE_OPEN;
    fvg.fill_percentage = 0.0;
    fvg.touch_count = 0;
    fvg.is_fresh = true;
    fvg.origin_timeframe = PERIOD_M15;
    
    // Set gap boundaries based on type
    if(type == FVG_BULLISH)
    {
        fvg.upper_level = rates[index+1].high;
        fvg.lower_level = rates[index-1].low;
    }
    else
    {
        fvg.upper_level = rates[index-1].high;
        fvg.lower_level = rates[index+1].low;
    }
    
    // Calculate middle level and optimal entry
    fvg.mid_level = (fvg.upper_level + fvg.lower_level) / 2.0;
    
    if(type == FVG_BULLISH)
        fvg.optimal_entry = fvg.lower_level + (fvg.upper_level - fvg.lower_level) * 0.618; // 61.8% level
    else
        fvg.optimal_entry = fvg.upper_level - (fvg.upper_level - fvg.lower_level) * 0.618; // 61.8% level
    
    // Calculate gap size
    fvg.gap_size_points = (fvg.upper_level - fvg.lower_level) / _Point;
    
    // Calculate advanced properties
    fvg.displacement_size = CalculateDisplacementAfterFVG(rates, index, type == FVG_BULLISH);
    fvg.has_volume_spike = HasVolumeConfirmation(rates, index);
    fvg.quality_score = CalculateFVGQualityScore(fvg);
    fvg.expected_reaction = CalculateExpectedReaction(fvg);
    fvg.quality = ClassifyFVGQuality(fvg);
    
    // Determine institutional characteristics
    fvg.is_institutional = IsInstitutionalFVG(fvg);
    
    // Calculate confluence factors
    CalculateConfluenceScore(fvg);
    
    // Set timing properties
    fvg.age_in_bars = 0;
    fvg.expiry_time = fvg.formation_time + 24*3600; // 24 hours expiry
    fvg.time_decay_factor = 1.0;
    
    // Determine premium/discount location
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    fvg.is_in_premium = IsInPremiumZone(current_price);
    
    return true;
}

// Validate FVG quality and requirements
bool CEliteFVGDetector::ValidateFVG(const SEliteFairValueGap& fvg)
{
    // Minimum quality score requirement
    if(fvg.quality_score < 60.0) return false;
    
    // Must be at least medium quality
    if(fvg.quality < FVG_QUALITY_MEDIUM) return false;
    
    // Minimum expected reaction
    if(fvg.expected_reaction < 0.6) return false;
    
    // Size requirements
    if(fvg.gap_size_points < 20 || fvg.gap_size_points > 500) return false;
    
    return true;
}

// Calculate FVG quality score
double CEliteFVGDetector::CalculateFVGQualityScore(const SEliteFairValueGap& fvg)
{
    double score = 50.0; // Base score
    
    // Gap size factor (optimal around 30-100 points for XAUUSD)
    if(fvg.gap_size_points >= 30 && fvg.gap_size_points <= 100)
        score += 20.0;
    else if(fvg.gap_size_points >= 20 && fvg.gap_size_points <= 150)
        score += 15.0;
    else if(fvg.gap_size_points >= 15 && fvg.gap_size_points <= 200)
        score += 10.0;
    
    // Displacement factor
    double displacement_points = fvg.displacement_size / _Point;
    if(displacement_points >= 200) score += 20.0;
    else if(displacement_points >= 150) score += 15.0;
    else if(displacement_points >= 100) score += 10.0;
    
    // Volume confirmation
    if(fvg.has_volume_spike) score += 15.0;
    
    // Institutional characteristics
    if(fvg.is_institutional) score += 10.0;
    
    // Confluence score
    score += fvg.confluence_score * 0.15;
    
    return MathMin(score, 100.0);
}

// Calculate expected reaction probability
double CEliteFVGDetector::CalculateExpectedReaction(const SEliteFairValueGap& fvg)
{
    double reaction = 0.5; // Base 50% probability
    
    // Quality bonus
    reaction += fvg.quality_score * 0.005; // Up to 50% bonus from quality
    
    // Gap size factor
    if(fvg.gap_size_points >= 30 && fvg.gap_size_points <= 80)
        reaction += 0.2; // Optimal size range
    
    // Institutional bonus
    if(fvg.is_institutional) reaction += 0.15;
    
    // Fresh FVG bonus
    if(fvg.is_fresh) reaction += 0.1;
    
    // Volume confirmation bonus
    if(fvg.has_volume_spike) reaction += 0.1;
    
    // Premium/discount consideration
    if((fvg.type == FVG_BULLISH && !fvg.is_in_premium) ||
       (fvg.type == FVG_BEARISH && fvg.is_in_premium))
    {
        reaction += 0.15; // Trading from appropriate zones
    }
    
    return MathMin(reaction, 1.0);
}

// Classify FVG quality
ENUM_FVG_QUALITY CEliteFVGDetector::ClassifyFVGQuality(const SEliteFairValueGap& fvg)
{
    if(fvg.quality_score >= 90.0 && fvg.is_institutional) return FVG_QUALITY_ELITE;
    if(fvg.quality_score >= 80.0) return FVG_QUALITY_HIGH;
    if(fvg.quality_score >= 65.0) return FVG_QUALITY_MEDIUM;
    return FVG_QUALITY_LOW;
}

// Check if FVG has institutional characteristics
bool CEliteFVGDetector::IsInstitutionalFVG(const SEliteFairValueGap& fvg)
{
    // Large gap size
    if(fvg.gap_size_points < 200) return false; // 20 pips minimum for institutional
    
    // Strong displacement
    if(fvg.displacement_size < m_institutional_threshold) return false;
    
    // Volume spike confirmation
    if(!fvg.has_volume_spike) return false;
    
    // High quality score
    if(fvg.quality_score < 80.0) return false;
    
    return true;
}

// Calculate confluence score
void CEliteFVGDetector::CalculateConfluenceScore(SEliteFairValueGap& fvg)
{
    double score = 0.0;
    
    // Check Order Block confluence
    fvg.has_ob_confluence = CheckOrderBlockConfluence(fvg);
    if(fvg.has_ob_confluence) score += 35.0;
    
    // Check Liquidity confluence
    fvg.has_liquidity_confluence = CheckLiquidityConfluence(fvg);
    if(fvg.has_liquidity_confluence) score += 30.0;
    
    // Check Structure confluence
    fvg.has_structure_confluence = CheckStructureConfluence(fvg);
    if(fvg.has_structure_confluence) score += 35.0;
    
    fvg.confluence_score = score;
    fvg.confluence_count = (fvg.has_ob_confluence ? 1 : 0) +
                          (fvg.has_liquidity_confluence ? 1 : 0) +
                          (fvg.has_structure_confluence ? 1 : 0);
}

// Helper methods implementation
double CEliteFVGDetector::CalculateDisplacementAfterFVG(const MqlRates& rates[], int index, bool is_bullish)
{
    if(index < 5) return 0.0;
    
    double max_displacement = 0.0;
    double reference_price = is_bullish ? rates[index].close : rates[index].close;
    
    // Check displacement in next 5 candles
    for(int i = index - 1; i >= MathMax(0, index - 5); i--)
    {
        double displacement = 0;
        if(is_bullish)
            displacement = rates[i].high - reference_price;
        else
            displacement = reference_price - rates[i].low;
            
        if(displacement > max_displacement)
            max_displacement = displacement;
    }
    
    return max_displacement;
}

bool CEliteFVGDetector::HasVolumeConfirmation(const MqlRates& rates[], int index)
{
    if(index >= ArraySize(rates) - 2) return true; // Default if insufficient data
    
    // Check volume spike in the FVG formation period
    long total_volume = rates[index-1].tick_volume + rates[index].tick_volume + rates[index+1].tick_volume;
    long avg_volume = 0;
    
    // Calculate average volume of previous 10 bars
    for(int i = index + 2; i < index + 12 && i < ArraySize(rates); i++)
    {
        avg_volume += rates[i].tick_volume;
    }
    avg_volume /= 10;
    
    return total_volume > avg_volume * m_volume_spike_threshold;
}

bool CEliteFVGDetector::CheckOrderBlockConfluence(const SEliteFairValueGap& fvg)
{
    // Check if there's an Order Block near the FVG
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        const SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        if(ob.state == OB_STATE_DISABLED || ob.state == OB_STATE_MITIGATED) continue;
        
        // Check if FVG overlaps or is near Order Block
        double distance = MathMin(
            MathAbs(fvg.upper_level - ob.high_price),
            MathAbs(fvg.lower_level - ob.low_price)
        );
        
        if(distance <= 30 * _Point) // Within 3 pips
            return true;
    }
    return false;
}

bool CEliteFVGDetector::CheckLiquidityConfluence(const SEliteFairValueGap& fvg)
{
    // Check if there's liquidity near the FVG
    for(int i = 0; i < g_liq_count; i++)
    {
        double distance = MathMin(
            MathAbs(fvg.upper_level - g_liquidity_zones[i].price_level),
            MathAbs(fvg.lower_level - g_liquidity_zones[i].price_level)
        );
        
        if(distance <= 25 * _Point) // Within 2.5 pips
            return true;
    }
    return false;
}

bool CEliteFVGDetector::CheckStructureConfluence(const SEliteFairValueGap& fvg)
{
    // Check if FVG aligns with market structure
    MqlRates rates[50];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Simple trend alignment check
    bool uptrend = rates[0].close > rates[10].close;
    
    if(fvg.type == FVG_BULLISH && uptrend) return true;
    if(fvg.type == FVG_BEARISH && !uptrend) return true;
    
    return false;
}

void CEliteFVGDetector::SortFVGsByQuality()
{
    // Simple bubble sort by quality score
    for(int i = 0; i < g_elite_fvg_count - 1; i++)
    {
        for(int j = 0; j < g_elite_fvg_count - i - 1; j++)
        {
            if(g_elite_fair_value_gaps[j].quality_score < g_elite_fair_value_gaps[j + 1].quality_score)
            {
                SEliteFairValueGap temp = g_elite_fair_value_gaps[j];
                g_elite_fair_value_gaps[j] = g_elite_fair_value_gaps[j + 1];
                g_elite_fair_value_gaps[j + 1] = temp;
            }
        }
    }
}

// Update FVG status
void CEliteFVGDetector::UpdateFVGStatus()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        
        // Check if price has entered the FVG
        if(current_price >= fvg.lower_level && current_price <= fvg.upper_level)
        {
            fvg.touch_count++;
            fvg.is_fresh = false;
            
            // Calculate fill percentage
            if(fvg.type == FVG_BULLISH)
            {
                fvg.fill_percentage = (current_price - fvg.lower_level) / (fvg.upper_level - fvg.lower_level) * 100.0;
            }
            else
            {
                fvg.fill_percentage = (fvg.upper_level - current_price) / (fvg.upper_level - fvg.lower_level) * 100.0;
            }
            
            // Update state based on fill percentage
            if(fvg.fill_percentage >= 100.0)
                fvg.state = FVG_STATE_FILLED;
            else if(fvg.fill_percentage >= 50.0)
                fvg.state = FVG_STATE_PARTIAL;
        }
        
        // Age-based expiry (24 hours for M15 timeframe)
        fvg.age_in_bars = (int)((TimeCurrent() - fvg.formation_time) / (15 * 60)); // 15 minutes per bar
        
        if(TimeCurrent() > fvg.expiry_time && fvg.touch_count == 0)
        {
            fvg.state = FVG_STATE_EXPIRED;
        }
        
        // Calculate time decay factor
        double age_hours = (TimeCurrent() - fvg.formation_time) / 3600.0;
        fvg.time_decay_factor = MathMax(0.1, 1.0 - (age_hours / 24.0)); // Decay over 24 hours
    }
}

// Remove filled/expired FVGs
void CEliteFVGDetector::RemoveFilledFVGs()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state != FVG_STATE_FILLED &&
           g_elite_fair_value_gaps[i].state != FVG_STATE_EXPIRED)
        {
            if(valid_count != i)
            {
                g_elite_fair_value_gaps[valid_count] = g_elite_fair_value_gaps[i];
            }
            valid_count++;
        }
    }
    
    g_elite_fvg_count = valid_count;
}

// Get active FVG count
int CEliteFVGDetector::GetActiveFVGCount()
{
    int active_count = 0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state == FVG_STATE_OPEN ||
           g_elite_fair_value_gaps[i].state == FVG_STATE_PARTIAL)
        {
            active_count++;
        }
    }
    
    return active_count;
}

// Get best FVG score
double CEliteFVGDetector::GetBestFVGScore()
{
    double best_score = 0.0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state == FVG_STATE_OPEN ||
           g_elite_fair_value_gaps[i].state == FVG_STATE_PARTIAL)
        {
            if(g_elite_fair_value_gaps[i].quality_score > best_score)
                best_score = g_elite_fair_value_gaps[i].quality_score;
        }
    }
    
    return best_score;
}

//+------------------------------------------------------------------+
//| Elite Order Block Detector Implementation                        |
//+------------------------------------------------------------------+

// Constructor
CEliteOrderBlockDetector::CEliteOrderBlockDetector()
{
    // Initialize detection parameters for XAUUSD
    m_displacement_threshold = 200.0 * _Point;  // 20 pips displacement minimum
    m_volume_threshold = 1.5;                   // 1.5x average volume
    m_require_structure_break = true;           // Require structure break
    m_use_liquidity_confirmation = true;        // Use liquidity confirmation
    m_use_volume_confirmation = true;           // Use volume confirmation
    
    // Set analysis timeframes
    m_analysis_timeframes[0] = PERIOD_H4;
    m_analysis_timeframes[1] = PERIOD_H1;
    m_analysis_timeframes[2] = PERIOD_M15;
    m_analysis_timeframes[3] = PERIOD_M5;
    m_timeframe_count = 4;
}

// Destructor
CEliteOrderBlockDetector::~CEliteOrderBlockDetector()
{
    // Cleanup if needed
}

// Main detection method for elite order blocks
bool CEliteOrderBlockDetector::DetectEliteOrderBlocks()
{
    // Reset current order blocks count
    g_elite_ob_count = 0;
    
    // Get market data for analysis
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 100, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Analyze each potential order block formation
    for(int i = 5; i < 95; i++)
    {
        // Check for bullish order block pattern
        if(DetectBullishOrderBlock(rates, i))
        {
            SAdvancedOrderBlock ob;
            if(CreateOrderBlockStructure(rates, i, OB_BULLISH, ob))
            {
                if(ValidateOrderBlock(ob))
                {
                    g_elite_order_blocks[g_elite_ob_count] = ob;
                    g_elite_ob_count++;
                    if(g_elite_ob_count >= 50) break;
                }
            }
        }
        
        // Check for bearish order block pattern
        if(DetectBearishOrderBlock(rates, i))
        {
            SAdvancedOrderBlock ob;
            if(CreateOrderBlockStructure(rates, i, OB_BEARISH, ob))
            {
                if(ValidateOrderBlock(ob))
                {
                    g_elite_order_blocks[g_elite_ob_count] = ob;
                    g_elite_ob_count++;
                    if(g_elite_ob_count >= 50) break;
                }
            }
        }
    }
    
    // Sort order blocks by quality and proximity
    SortOrderBlocksByQuality();
    
    return g_elite_ob_count > 0;
}

// Detect bullish order block formation
bool CEliteOrderBlockDetector::DetectBullishOrderBlock(const MqlRates& rates[], int index)
{
    if(index < 3 || index >= ArraySize(rates) - 2) return false;
    
    // Look for bullish order block pattern:
    // 1. Strong bullish candle (displacement)
    // 2. Small retracement/consolidation 
    // 3. Strong move away (imbalance creation)
    
    double current_body = rates[index].close - rates[index].open;
    double prev_body = rates[index+1].close - rates[index+1].open;
    double next_displacement = 0;
    
    // Calculate displacement after the order block
    for(int j = index-1; j >= MathMax(0, index-5); j--)
    {
        if(rates[j].close > rates[index].high)
        {
            next_displacement = rates[j].close - rates[index].high;
            break;
        }
    }
    
    // Bullish order block criteria:
    // 1. Current candle is bullish
    if(current_body <= 0) return false;
    
    // 2. Strong bullish body (at least 50% of range)
    double total_range = rates[index].high - rates[index].low;
    if(total_range > 0 && current_body < total_range * 0.5) return false;
    
    // 3. Significant size relative to recent candles
    double avg_body = CalculateAverageBodySize(rates, index, 10);
    if(current_body < avg_body * 1.5) return false;
    
    // 4. Must have displacement after formation
    if(next_displacement < m_displacement_threshold) return false;
    
    // 5. Volume confirmation (if available)
    if(m_use_volume_confirmation)
    {
        if(!HasVolumeSpike(rates, index)) return false;
    }
    
    return true;
}

// Detect bearish order block formation  
bool CEliteOrderBlockDetector::DetectBearishOrderBlock(const MqlRates& rates[], int index)
{
    if(index < 3 || index >= ArraySize(rates) - 2) return false;
    
    double current_body = rates[index].open - rates[index].close;
    double next_displacement = 0;
    
    // Calculate displacement after the order block
    for(int j = index-1; j >= MathMax(0, index-5); j--)
    {
        if(rates[j].close < rates[index].low)
        {
            next_displacement = rates[index].low - rates[j].close;
            break;
        }
    }
    
    // Bearish order block criteria:
    // 1. Current candle is bearish
    if(current_body <= 0) return false;
    
    // 2. Strong bearish body (at least 50% of range)
    double total_range = rates[index].high - rates[index].low;
    if(total_range > 0 && current_body < total_range * 0.5) return false;
    
    // 3. Significant size relative to recent candles
    double avg_body = CalculateAverageBodySize(rates, index, 10);
    if(current_body < avg_body * 1.5) return false;
    
    // 4. Must have displacement after formation
    if(next_displacement < m_displacement_threshold) return false;
    
    // 5. Volume confirmation (if available)
    if(m_use_volume_confirmation)
    {
        if(!HasVolumeSpike(rates, index)) return false;
    }
    
    return true;
}

// Create order block structure
bool CEliteOrderBlockDetector::CreateOrderBlockStructure(const MqlRates& rates[], int index, ENUM_ORDER_BLOCK_TYPE type, SAdvancedOrderBlock& ob)
{
    // Initialize order block structure
    ob.formation_time = rates[index].time;
    ob.high_price = rates[index].high;
    ob.low_price = rates[index].low;
    ob.type = type;
    ob.state = OB_STATE_ACTIVE;
    ob.is_fresh = true;
    ob.touch_count = 0;
    ob.origin_timeframe = PERIOD_M15;
    
    // Calculate refined entry
    if(type == OB_BULLISH)
    {
        ob.refined_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    else
    {
        ob.refined_entry = ob.high_price - (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    
    // Calculate advanced properties
    ob.displacement_size = CalculateDisplacementSize(rates, index);
    ob.strength = CalculateOrderBlockStrength(ob);
    ob.volume_profile = CalculateVolumeProfile(rates, index);
    ob.reaction_quality = CalculateReactionQuality(rates, index);
    ob.probability_score = CalculateProbabilityScore(ob);
    ob.quality = ClassifyOrderBlockQuality(ob);
    
    // Determine premium/discount zone
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    ob.is_premium = IsInPremiumZone(current_price);
    
    // Check institutional characteristics
    ob.is_institutional = IsInstitutionalOrderBlock(ob);
    
    // Calculate confluence factors
    ob.has_fvg_confluence = CheckFVGConfluence(ob);
    ob.has_liquidity_confluence = CheckLiquidityConfluence(ob);
    ob.has_structure_confluence = CheckStructureConfluence(ob);
    ob.confluence_score = CalculateConfluenceScore(ob);
    
    return true;
}

// Validate order block quality and requirements
bool CEliteOrderBlockDetector::ValidateOrderBlock(const SAdvancedOrderBlock& ob)
{
    // Minimum strength requirement
    if(ob.strength < 60.0) return false;
    
    // Minimum probability score
    if(ob.probability_score < 70.0) return false;
    
    // Must be at least medium quality
    if(ob.quality < OB_QUALITY_MEDIUM) return false;
    
    // Must have minimum confluence score
    if(ob.confluence_score < 65.0) return false;
    
    // Additional validations
    if(m_use_liquidity_confirmation && !ob.has_liquidity) return false;
    
    return true;
}

// Calculate order block strength
double CEliteOrderBlockDetector::CalculateOrderBlockStrength(const SAdvancedOrderBlock& ob)
{
    double strength = 0.0;
    
    // Base strength from displacement size
    strength += MathMin(ob.displacement_size / (100 * _Point), 30.0); // Max 30 points
    
    // Volume strength (if available)
    strength += MathMin(ob.volume_profile * 20.0, 20.0); // Max 20 points
    
    // Reaction quality
    strength += ob.reaction_quality * 25.0; // Max 25 points
    
    // Institutional characteristics
    if(ob.is_institutional) strength += 15.0;
    
    // Confluence bonus
    if(ob.has_fvg_confluence) strength += 5.0;
    if(ob.has_liquidity_confluence) strength += 5.0;
    if(ob.has_structure_confluence) strength += 5.0;
    
    return MathMin(strength, 100.0);
}

// Calculate probability score
double CEliteOrderBlockDetector::CalculateProbabilityScore(const SAdvancedOrderBlock& ob)
{
    double probability = 50.0; // Base probability
    
    // Quality bonus
    switch(ob.quality)
    {
        case OB_QUALITY_HIGH: probability += 20.0; break;
        case OB_QUALITY_ELITE: probability += 30.0; break;
        case OB_QUALITY_MEDIUM: probability += 10.0; break;
        default: break;
    }
    
    // Institutional bonus
    if(ob.is_institutional) probability += 15.0;
    
    // Fresh order block bonus
    if(ob.is_fresh) probability += 10.0;
    
    // Premium/discount zone consideration
    if((ob.type == OB_BULLISH && !ob.is_premium) || 
       (ob.type == OB_BEARISH && ob.is_premium))
    {
        probability += 15.0; // Trading from discount/premium appropriately
    }
    
    // Confluence bonus
    probability += ob.confluence_score * 0.2;
    
    return MathMin(probability, 100.0);
}

// Classify order block quality
ENUM_OB_QUALITY CEliteOrderBlockDetector::ClassifyOrderBlockQuality(const SAdvancedOrderBlock& ob)
{
    double quality_score = 0.0;
    
    // Size factor
    if(ob.displacement_size >= 300 * _Point) quality_score += 25.0;
    else if(ob.displacement_size >= 200 * _Point) quality_score += 15.0;
    else if(ob.displacement_size >= 100 * _Point) quality_score += 10.0;
    
    // Volume factor
    if(ob.volume_profile > 1.8) quality_score += 25.0;
    else if(ob.volume_profile > 1.5) quality_score += 15.0;
    else if(ob.volume_profile > 1.2) quality_score += 10.0;
    
    // Reaction quality
    quality_score += ob.reaction_quality * 25.0;
    
    // Institutional characteristics
    if(ob.is_institutional) quality_score += 15.0;
    
    // Confluence score
    quality_score += ob.confluence_score * 0.1;
    
    // Classify based on total score
    if(quality_score >= 85.0) return OB_QUALITY_ELITE;
    if(quality_score >= 70.0) return OB_QUALITY_HIGH;
    if(quality_score >= 55.0) return OB_QUALITY_MEDIUM;
    return OB_QUALITY_LOW;
}

// Helper methods implementation
double CEliteOrderBlockDetector::CalculateAverageBodySize(const MqlRates& rates[], int index, int period)
{
    double total_body = 0.0;
    int count = 0;
    
    for(int i = index; i < index + period && i < ArraySize(rates); i++)
    {
        total_body += MathAbs(rates[i].close - rates[i].open);
        count++;
    }
    
    return (count > 0) ? total_body / count : 0.0;
}

bool CEliteOrderBlockDetector::HasVolumeSpike(const MqlRates& rates[], int index)
{
    // Simple volume spike detection based on tick volume
    if(index >= ArraySize(rates) - 5) return true; // Default to true if insufficient data
    
    long current_volume = rates[index].tick_volume;
    long avg_volume = 0;
    
    // Calculate average volume of previous 10 bars
    for(int i = index + 1; i < index + 11 && i < ArraySize(rates); i++)
    {
        avg_volume += rates[i].tick_volume;
    }
    avg_volume /= 10;
    
    return current_volume > avg_volume * m_volume_threshold;
}

double CEliteOrderBlockDetector::CalculateDisplacementSize(const MqlRates& rates[], int index)
{
    if(index < 5) return 0.0;
    
    double max_displacement = 0.0;
    
    // Check displacement in next 5 candles
    for(int i = index - 1; i >= MathMax(0, index - 5); i--)
    {
        double displacement = MathAbs(rates[i].close - rates[index].close);
        if(displacement > max_displacement)
            max_displacement = displacement;
    }
    
    return max_displacement;
}

double CEliteOrderBlockDetector::CalculateVolumeProfile(const MqlRates& rates[], int index)
{
    if(index >= ArraySize(rates) - 1) return 1.0;
    
    long current_volume = rates[index].tick_volume;
    long prev_volume = rates[index + 1].tick_volume;
    
    return (prev_volume > 0) ? (double)current_volume / prev_volume : 1.0;
}

double CEliteOrderBlockDetector::CalculateReactionQuality(const MqlRates& rates[], int index)
{
    // Measure the quality of market reaction after order block formation
    if(index < 3) return 0.5;
    
    double initial_price = rates[index].close;
    double max_move = 0.0;
    bool found_reaction = false;
    
    // Check next 3 candles for reaction
    for(int i = index - 1; i >= MathMax(0, index - 3); i--)
    {
        double move = MathAbs(rates[i].close - initial_price);
        if(move > max_move)
        {
            max_move = move;
            found_reaction = true;
        }
    }
    
    if(!found_reaction) return 0.0;
    
    // Normalize reaction quality (0-1)
    double quality = MathMin(max_move / (200 * _Point), 1.0);
    return quality;
}

bool CEliteOrderBlockDetector::CheckFVGConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if there's a Fair Value Gap near the order block
    for(int i = 0; i < g_fvg_count; i++)
    {
        double distance_to_fvg = MathMin(
            MathAbs(ob.high_price - g_fair_value_gaps[i].upper_level),
            MathAbs(ob.low_price - g_fair_value_gaps[i].lower_level)
        );
        
        if(distance_to_fvg <= 50 * _Point) // Within 5 pips
            return true;
    }
    return false;
}

bool CEliteOrderBlockDetector::CheckLiquidityConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if there's liquidity near the order block
    for(int i = 0; i < g_liq_count; i++)
    {
        double distance_to_liquidity = MathMin(
            MathAbs(ob.high_price - g_liquidity_zones[i].price_level),
            MathAbs(ob.low_price - g_liquidity_zones[i].price_level)
        );
        
        if(distance_to_liquidity <= 30 * _Point) // Within 3 pips
            return true;
    }
    return false;
}

bool CEliteOrderBlockDetector::CheckStructureConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if order block aligns with market structure
    // This is a simplified implementation
    MqlRates rates[50];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Check if we're in an appropriate market structure for this OB type
    bool uptrend = rates[0].close > rates[10].close;
    
    if(ob.type == OB_BULLISH && uptrend) return true;
    if(ob.type == OB_BEARISH && !uptrend) return true;
    
    return false;
}

double CEliteOrderBlockDetector::CalculateConfluenceScore(const SAdvancedOrderBlock& ob)
{
    double score = 0.0;
    
    if(ob.has_fvg_confluence) score += 30.0;
    if(ob.has_liquidity_confluence) score += 35.0;
    if(ob.has_structure_confluence) score += 35.0;
    
    return score;
}

void CEliteOrderBlockDetector::SortOrderBlocksByQuality()
{
    // Simple bubble sort by probability score (can be optimized)
    for(int i = 0; i < g_elite_ob_count - 1; i++)
    {
        for(int j = 0; j < g_elite_ob_count - i - 1; j++)
        {
            if(g_elite_order_blocks[j].probability_score < g_elite_order_blocks[j + 1].probability_score)
            {
                SAdvancedOrderBlock temp = g_elite_order_blocks[j];
                g_elite_order_blocks[j] = g_elite_order_blocks[j + 1];
                g_elite_order_blocks[j + 1] = temp;
            }
        }
    }
}

// Update order block status based on current market conditions
void CEliteOrderBlockDetector::UpdateOrderBlockStatus()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        
        // Check if price has interacted with the order block
        if(current_price >= ob.low_price && current_price <= ob.high_price)
        {
            ob.touch_count++;
            if(ob.state == OB_STATE_ACTIVE)
            {
                ob.state = OB_STATE_TESTED;
                ob.is_fresh = false;
            }
        }
        
        // Check if order block has been mitigated
        if((ob.type == OB_BULLISH && current_price < ob.low_price) ||
           (ob.type == OB_BEARISH && current_price > ob.high_price))
        {
            ob.state = OB_STATE_MITIGATED;
        }
        
        // Age-based invalidation (24 hours for M15 timeframe)
        if(TimeCurrent() - ob.formation_time > 24 * 3600)
        {
            if(ob.touch_count == 0)
                ob.state = OB_STATE_DISABLED;
        }
    }
}

// Remove invalid order blocks
void CEliteOrderBlockDetector::RemoveInvalidOrderBlocks()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        if(g_elite_order_blocks[i].state != OB_STATE_DISABLED &&
           g_elite_order_blocks[i].state != OB_STATE_MITIGATED)
        {
            if(valid_count != i)
            {
                g_elite_order_blocks[valid_count] = g_elite_order_blocks[i];
            }
            valid_count++;
        }
    }
    
    g_elite_ob_count = valid_count;
}

// Get count of active order blocks
int CEliteOrderBlockDetector::GetActiveOrderBlockCount()
{
    int active_count = 0;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        if(g_elite_order_blocks[i].state == OB_STATE_ACTIVE ||
           g_elite_order_blocks[i].state == OB_STATE_TESTED)
        {
            active_count++;
        }
    }
    
    return active_count;
}

// Essential helper functions
bool IsWeeklyBiasAligned() 
{ 
    // Simplified implementation - can be enhanced
    MqlRates weekly_rates[3];
    if(CopyRates(_Symbol, PERIOD_W1, 0, 3, weekly_rates) <= 0) return true;
    ArraySetAsSeries(weekly_rates, true);
    
    // Basic weekly bias check - bullish if close > open
    return weekly_rates[1].close > weekly_rates[1].open;
}

bool IsDailyTrendValid() 
{ 
    // Simplified implementation - can be enhanced
    MqlRates daily_rates[5];
    if(CopyRates(_Symbol, PERIOD_D1, 0, 5, daily_rates) <= 0) return true;
    ArraySetAsSeries(daily_rates, true);
    
    // Basic trend check - compare recent closes
    return daily_rates[0].close > daily_rates[2].close;
}

bool IsH4StructureValid() 
{ 
    // Simplified implementation - can be enhanced
    MqlRates h4_rates[10];
    if(CopyRates(_Symbol, PERIOD_H4, 0, 10, h4_rates) <= 0) return true;
    ArraySetAsSeries(h4_rates, true);
    
    // Basic structure check
    return h4_rates[0].high > h4_rates[5].high || h4_rates[0].low < h4_rates[5].low;
}

bool IsH1SetupValid() 
{ 
    // Simplified implementation - can be enhanced
    MqlRates h1_rates[5];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 5, h1_rates) <= 0) return true;
    ArraySetAsSeries(h1_rates, true);
    
    // Basic setup validation
    return true; // Always valid for now
}

bool IsM15ExecutionValid() 
{ 
    // Simplified implementation - can be enhanced
    MqlRates m15_rates[3];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 3, m15_rates) <= 0) return true;
    ArraySetAsSeries(m15_rates, true);
    
    // Basic execution timing check
    return true; // Always valid for now
}

void CalculateTradeParameters(SConfluenceSignal& signal, ENUM_SIGNAL_TYPE signal_type)
{
    double current_price = (signal_type == SIGNAL_BUY) ? symbol_info.Ask() : symbol_info.Bid();
    signal.entry_price = current_price;
    
    // Dynamic stop based on ATR and recent structure
    double atr_buffer[1];
    double sl_distance_price = InpStopLoss * _Point;
    if(CopyBuffer(h_atr_m15, 0, 0, 1, atr_buffer) > 0)
    {
        sl_distance_price = MathMax(sl_distance_price, atr_buffer[0] * 1.5);
    }
    
    // Use recent swing structure to avoid stopping inside noise
    MqlRates m15_rates[20];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 20, m15_rates) > 0)
    {
        ArraySetAsSeries(m15_rates, true);
        double recent_swing_low = m15_rates[1].low;
        double recent_swing_high = m15_rates[1].high;
        
        for(int i = 1; i < 10; i++)
        {
            recent_swing_low = MathMin(recent_swing_low, m15_rates[i].low);
            recent_swing_high = MathMax(recent_swing_high, m15_rates[i].high);
        }
        
        if(signal_type == SIGNAL_BUY)
            sl_distance_price = MathMax(sl_distance_price, MathAbs(current_price - recent_swing_low));
        else
            sl_distance_price = MathMax(sl_distance_price, MathAbs(recent_swing_high - current_price));
    }
    
    // Calculate TP with adaptive RR based on inputs (fallback >=1.5R)
    double base_rr = (InpStopLoss > 0) ? ((double)InpTakeProfit / (double)InpStopLoss) : 1.5;
    base_rr = MathMax(1.5, base_rr);
    double tp_distance_price = sl_distance_price * base_rr;
    
    if(signal_type == SIGNAL_BUY)
    {
        signal.stop_loss = current_price - sl_distance_price;
        signal.take_profit = current_price + tp_distance_price;
    }
    else
    {
        signal.stop_loss = current_price + sl_distance_price;
        signal.take_profit = current_price - tp_distance_price;
    }
    
    signal.risk_reward_ratio = (sl_distance_price > 0) ? tp_distance_price / sl_distance_price : base_rr;
}

bool IsInDiscountZone(double price) 
{ 
    // Simplified implementation - check if price is in lower 40% of recent range
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 100, rates) <= 0) return true;
    
    // Find highest and lowest manually
    double highest = rates[0].high;
    double lowest = rates[0].low;
    for(int i = 1; i < 100; i++)
    {
        if(rates[i].high > highest) highest = rates[i].high;
        if(rates[i].low < lowest) lowest = rates[i].low;
    }
    
    double range = highest - lowest;
    if(range <= 0) return true;
    
    return price <= (lowest + range * 0.4);
}

bool IsInPremiumZone(double price) 
{ 
    // Simplified implementation - check if price is in upper 40% of recent range
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 100, rates) <= 0) return true;
    
    // Find highest and lowest manually
    double highest = rates[0].high;
    double lowest = rates[0].low;
    for(int i = 1; i < 100; i++)
    {
        if(rates[i].high > highest) highest = rates[i].high;
        if(rates[i].low < lowest) lowest = rates[i].low;
    }
    
    double range = highest - lowest;
    if(range <= 0) return true;
    
    return price >= (highest - range * 0.4);
}

// Validation functions
bool ValidateAllFilters(const SConfluenceSignal& signal)
{
    if(!ValidateSessionFilter()) return false;
    if(InpEnableNewsFilter && !ValidateNewsFilter()) return false;
    if(!ValidateSpreadFilter()) return false;
    if(g_trades_today >= InpMaxTradesPerDay) return false;
    if(!ValidateRiskFilter(signal)) return false;
    return true;
}

bool ValidateSessionFilter()
{
    MqlDateTime dt;
    
    // Convert server time to GMT using configurable offset to align session windows
    datetime adjusted_time = TimeCurrent() - (int)MathRound(InpServerGMTOffset * 3600.0);
    TimeToStruct(adjusted_time, dt);
    int current_hour = dt.hour;
    
    if(InpTradeLondonSession && current_hour >= InpLondonStart && current_hour < InpLondonEnd)
        return true;
    if(InpTradeNYSession && current_hour >= InpNYStart && current_hour < InpNYEnd)
        return true;
    if(InpTradeAsianSession && (current_hour >= 22 || current_hour < 6))
        return true;
        
    return false;
}

bool ValidateNewsFilter()
{
    MqlDateTime dt;
    datetime adjusted_time = TimeCurrent() - (int)MathRound(InpServerGMTOffset * 3600.0);
    TimeToStruct(adjusted_time, dt);
    
    int total_minutes_now = dt.hour * 60 + dt.min;
    int total_minutes_cpi = InpNewsCPIHour * 60 + InpNewsCPIMinute;
    int total_minutes_fomc = InpNewsFOMCHour * 60 + InpNewsFOMCMinute;
    int total_minutes_london = InpNewsLondonHour * 60 + InpNewsLondonMinute;
    
    if(MathAbs(total_minutes_now - total_minutes_cpi) <= InpNewsBufferMinutes) return false;
    if(MathAbs(total_minutes_now - total_minutes_fomc) <= InpNewsBufferMinutes) return false;
    if(MathAbs(total_minutes_now - total_minutes_london) <= InpNewsBufferMinutes) return false;
    
    return true;
}

bool ValidateSpreadFilter()
{
    double spread = symbol_info.Spread() * _Point / _Point;
    return spread <= InpMaxSpread;
}

bool ValidateRiskFilter(const SConfluenceSignal& signal)
{
    double potential_loss = CalculatePotentialLoss(signal);
    double account_balance = account_info.Balance();
    double daily_risk_amount = account_balance * (InpMaxDailyRisk / 100.0);
    
    if(g_daily_profit - potential_loss < -daily_risk_amount)
        return false;
        
    double current_equity = account_info.Equity();
    double drawdown = (account_balance - current_equity) / account_balance * 100.0;
    
    return drawdown < InpMaxDrawdown;
}

double CalculatePotentialLoss(const SConfluenceSignal& signal)
{
    double lot_size = CalculateLotSize(signal);
    double sl_distance = MathAbs(signal.entry_price - signal.stop_loss);
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tick_size <= 0 || tick_value <= 0 || sl_distance <= 0) return 0.0;
    
    double ticks = sl_distance / tick_size;
    return ticks * tick_value * lot_size;
}

bool ValidateTradeLevels(ENUM_SIGNAL_TYPE signal_type, double entry, double sl, double tp)
{
    double min_stop_level = symbol_info.StopsLevel() * _Point;
    
    if(signal_type == SIGNAL_BUY)
    {
        if(sl >= entry - min_stop_level) return false;
        if(tp <= entry + min_stop_level) return false;
    }
    else
    {
        if(sl <= entry + min_stop_level) return false;
        if(tp >= entry - min_stop_level) return false;
    }
    
    return true;
}

int CountOpenPositionsByMagic()
{
    int count = 0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(position_info.SelectByIndex(i))
        {
            if(position_info.Magic() == InpMagicNumber && position_info.Symbol() == _Symbol)
                count++;
        }
    }
    return count;
}

double GetTickValuePerPoint()
{
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tick_size > 0.0)
        return tick_value / tick_size;
    
    return (tick_value > 0 && _Point > 0) ? tick_value / _Point : tick_value;
}

// Placeholder MCP/AI integrations to keep EA functional when external services are unavailable
bool InitializeMCPIntegration()
{
    // Stub: integrate external AI here. Return false to fall back to native logic.
    return false;
}

void CleanupMCPIntegration()
{
    // Stub for MCP cleanup
}

void CheckAIOptimization()
{
    // Stub: mark last optimization time to avoid tight loops
    g_last_ai_optimization = TimeCurrent();
}

double GetEffectiveConfluenceThreshold()
{
    if(g_ai_optimization_active && g_ai_optimized_confluence > 0.0)
        return g_ai_optimized_confluence;
    
    return InpConfluenceThreshold;
}

SConfluenceSignal GenerateAIEnhancedConfluenceSignal()
{
    // Fallback: reuse native confluence; hook AI adjustments here
    return GenerateConfluenceSignal();
}

bool ValidateTradeWithAI(const SConfluenceSignal& signal, bool &approved)
{
    // Fallback approval model based on confidence threshold multiplier
    double effective_threshold = InpConfluenceThreshold * InpAIConfidenceThreshold;
    approved = (signal.confidence_score >= effective_threshold);
    return true; // AI call succeeded (stub)
}

bool CheckEmergencyConditions()
{
    return false; // No additional checks in stub
}

bool CheckEnhancedEmergencyConditions()
{
    // Extend here with latency/volatility guards if needed
    return CheckEmergencyConditions();
}

// Trading execution functions
void ExecuteTrade(const SConfluenceSignal& signal)
{
    // Enhanced signal with AI validation
    if(InpEnableMCPIntegration && InpAITradeValidation && g_ai_optimization_active)
    {
        bool ai_validation = false;
        if(ValidateTradeWithAI(signal, ai_validation))
        {
            if(!ai_validation)
            {
                Print("🤖 AI Trade Validation: REJECTED - Trade does not meet AI criteria");
                return;
            }
            else
            {
                Print("🤖 AI Trade Validation: APPROVED - Trade meets AI criteria");
            }
        }
    }
    
    // FTMO Pre-trade validation
    if(!ValidateFTMOTradeCompliance(signal))
    {
        Print("⛔ Trade rejected - FTMO compliance check failed");
        return;
    }
    
    double lot_size = CalculateLotSize(signal);
    if(lot_size <= 0) return;
    
    // Calculate trade risk for FTMO tracking
    double trade_risk = CalculatePotentialLoss(signal);
    
    // Validate broker stop level constraints
    if(!ValidateTradeLevels(signal.signal_type, signal.entry_price, signal.stop_loss, signal.take_profit))
    {
        Print("❌ Trade levels invalid for broker constraints");
        return;
    }
    
    bool success = false;
    ulong ticket = 0;
    
    if(signal.signal_type == SIGNAL_BUY)
    {
        success = trade.Buy(lot_size, _Symbol, signal.entry_price, signal.stop_loss, signal.take_profit, InpComment);
        ticket = trade.ResultOrder();
    }
    else if(signal.signal_type == SIGNAL_SELL)
    {
        success = trade.Sell(lot_size, _Symbol, signal.entry_price, signal.stop_loss, signal.take_profit, InpComment);
        ticket = trade.ResultOrder();
    }
    
    if(success && ticket > 0)
    {
        // Track trade for FTMO compliance
        TrackFTMOTradeExecution(ticket, lot_size, trade_risk);
        
        Print("✅ Trade executed successfully: ", EnumToString(signal.signal_type), 
              " | Ticket: ", ticket,
              " | Confidence: ", DoubleToString(signal.confidence_score, 1), "%",
              " | Risk: $", DoubleToString(trade_risk, 2));
              
        // Print FTMO status after trade
        Print(GetFTMOComplianceReport());
    }
    else
    {
        Print("❌ Trade execution failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
    }
}

// Price action detection (enhanced implementations)
bool IsBullishEngulfing(const MqlRates& rates[], int index) 
{ 
    if(index < 1) return false;
    
    // Previous candle must be bearish
    if(rates[index+1].close >= rates[index+1].open) return false;
    
    // Current candle must be bullish
    if(rates[index].close <= rates[index].open) return false;
    
    // Current body must engulf previous body
    if(rates[index].open >= rates[index+1].close) return false;
    if(rates[index].close <= rates[index+1].open) return false;
    
    return true;
}

bool IsBearishEngulfing(const MqlRates& rates[], int index) 
{ 
    if(index < 1) return false;
    
    // Previous candle must be bullish
    if(rates[index+1].close <= rates[index+1].open) return false;
    
    // Current candle must be bearish
    if(rates[index].close >= rates[index].open) return false;
    
    // Current body must engulf previous body
    if(rates[index].open <= rates[index+1].close) return false;
    if(rates[index].close >= rates[index+1].open) return false;
    
    return true;
}

bool IsBullishPinBar(const MqlRates& rates[], int index) 
{ 
    double body = MathAbs(rates[index].close - rates[index].open);
    double total_range = rates[index].high - rates[index].low;
    double lower_shadow = MathMin(rates[index].open, rates[index].close) - rates[index].low;
    double upper_shadow = rates[index].high - MathMax(rates[index].open, rates[index].close);
    
    // Pin bar requirements
    if(total_range == 0) return false;
    if(lower_shadow < total_range * 0.6) return false;  // Long lower shadow
    if(body > total_range * 0.25) return false;         // Small body
    if(upper_shadow > total_range * 0.20) return false; // Small upper shadow
    
    return true;
}

bool IsBearishPinBar(const MqlRates& rates[], int index) 
{ 
    double body = MathAbs(rates[index].close - rates[index].open);
    double total_range = rates[index].high - rates[index].low;
    double lower_shadow = MathMin(rates[index].open, rates[index].close) - rates[index].low;
    double upper_shadow = rates[index].high - MathMax(rates[index].open, rates[index].close);
    
    // Pin bar requirements (inverted)
    if(total_range == 0) return false;
    if(upper_shadow < total_range * 0.6) return false;  // Long upper shadow
    if(body > total_range * 0.25) return false;         // Small body
    if(lower_shadow > total_range * 0.20) return false; // Small lower shadow
    
    return true;
}

bool IsDoji(const MqlRates& rates[], int index) 
{ 
    double body = MathAbs(rates[index].close - rates[index].open);
    double total_range = rates[index].high - rates[index].low;
    
    if(total_range == 0) return false;
    
    // Doji: very small body relative to total range
    return body <= total_range * 0.1;
}

// Order block functions (enhanced implementations)
bool IsBullishOrderBlock(const MqlRates& rates[], int index) 
{ 
    if(index < 2) return false;
    
    // Look for bullish order block pattern
    // Strong bullish candle followed by pullback
    double body = rates[index].close - rates[index].open;
    double prev_body = rates[index+1].close - rates[index+1].open;
    
    return (body > 0 && body > prev_body * 1.5);
}

bool IsBearishOrderBlock(const MqlRates& rates[], int index) 
{ 
    if(index < 2) return false;
    
    // Look for bearish order block pattern
    double body = rates[index].open - rates[index].close;
    double prev_body = rates[index+1].open - rates[index+1].close;
    
    return (body > 0 && body > prev_body * 1.5);
}

double CalculateOrderBlockStrength(const MqlRates& rates[], int index) 
{ 
    if(index < 1) return 1.0;
    
    double body = MathAbs(rates[index].close - rates[index].open);
    double avg_body = 0;
    
    // Calculate average body size for strength comparison
    for(int i = index; i < index + 10 && i < ArraySize(rates); i++)
    {
        avg_body += MathAbs(rates[i].close - rates[i].open);
    }
    avg_body /= 10.0;
    
    return (avg_body > 0) ? MathMin(body / avg_body, 3.0) : 1.0;
}

double CalculateFVGStrength(const MqlRates& rates[], int index) 
{ 
    if(index < 2) return 1.0;
    
    // Calculate FVG strength based on gap size
    double gap_size = MathAbs(rates[index-1].high - rates[index+1].low);
    double avg_range = (rates[index].high - rates[index].low + 
                       rates[index+1].high - rates[index+1].low) / 2.0;
    
    return (avg_range > 0) ? MathMin(gap_size / avg_range, 2.0) : 1.0;
}

double CalculateLiquidityStrength(const MqlRates& rates[], int index) 
{ 
    if(index < 5) return 1.0;
    
    // Calculate liquidity strength based on touch count and volume
    double level = rates[index].high; // or low for support
    int touch_count = 0;
    
    for(int i = index + 1; i < index + 20 && i < ArraySize(rates); i++)
    {
        if(MathAbs(rates[i].high - level) < 10 * _Point ||
           MathAbs(rates[i].low - level) < 10 * _Point)
        {
            touch_count++;
        }
    }
    
    return MathMin(touch_count / 5.0, 2.0);
}

bool IsSwingHigh(const MqlRates& rates[], int index, int period) 
{ 
    if(index < period || index + period >= ArraySize(rates)) return false;
    
    double current_high = rates[index].high;
    
    // Check if current high is higher than surrounding highs
    for(int i = index - period; i <= index + period; i++)
    {
        if(i != index && rates[i].high >= current_high)
            return false;
    }
    
    return true;
}

bool IsSwingLow(const MqlRates& rates[], int index, int period) 
{ 
    if(index < period || index + period >= ArraySize(rates)) return false;
    
    double current_low = rates[index].low;
    
    // Check if current low is lower than surrounding lows
    for(int i = index - period; i <= index + period; i++)
    {
        if(i != index && rates[i].low <= current_low)
            return false;
    }
    
    return true;
}

// Essential helper functions


double CalculateLotSize(const SConfluenceSignal& signal)
{
    double lot_size = InpLotSize;
    
    if(InpLotMethod == LOT_PERCENT_RISK)
    {
        double account_balance = account_info.Balance();
        double risk_amount = account_balance * (InpRiskPercent / 100.0);
        double sl_distance = MathAbs(signal.entry_price - signal.stop_loss);
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        
        if(sl_distance > 0 && tick_value > 0 && tick_size > 0)
        {
            double ticks = sl_distance / tick_size;
            lot_size = (ticks > 0.0) ? risk_amount / (ticks * tick_value) : lot_size;
        }
    }
    
    // Normalize lot size
    double min_lot = symbol_info.LotsMin();
    double max_lot = symbol_info.LotsMax();
    double lot_step = symbol_info.LotsStep();
    
    lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
    return MathFloor(lot_size / lot_step) * lot_step;
}

double CalculateAdaptiveLotSize(const SConfluenceSignal& signal)
{
    double base_lot = InpLotSize;
    double confidence_multiplier = signal.confidence_score / 100.0;
    
    SPerformanceMetrics metrics = CalculatePerformanceMetrics();
    double performance_multiplier = 1.0;
    
    if(metrics.win_rate > 0.8) performance_multiplier = 1.2;
    else if(metrics.win_rate < 0.6) performance_multiplier = 0.8;
    
    return base_lot * confidence_multiplier * performance_multiplier;
}

//+------------------------------------------------------------------+
//| Autonomous Position Management System - AI-Driven Dynamic Exits |
//+------------------------------------------------------------------+

// Enhanced position management structure
struct SAutonomousPositionData
{
    ulong ticket;
    datetime entry_time;
    double entry_price;
    double initial_sl;
    double initial_tp;
    double current_sl;
    double current_tp;
    ENUM_POSITION_TYPE position_type;
    
    // Dynamic management data
    double max_favorable_excursion;    // Best price reached
    double max_adverse_excursion;      // Worst price reached
    double current_r_multiple;         // Current R:R ratio
    double peak_r_multiple;            // Peak R:R achieved
    
    // AI decision factors
    double market_sentiment_score;     // Market sentiment at entry
    double confluence_score_at_entry;  // Original confluence score
    double current_confluence_score;   // Current confluence
    bool has_moved_to_breakeven;
    bool has_taken_partial_profit;
    int partial_profit_count;
    
    // Dynamic exit triggers
    bool momentum_reversal_detected;
    bool structure_break_detected;
    bool volume_exhaustion_detected;
    bool time_based_exit_triggered;
    
    // Performance tracking
    double unrealized_pnl;
    double unrealized_pnl_percent;
    datetime last_update_time;
};

// Global position tracking
SAutonomousPositionData g_position_data[100];
int g_position_count = 0;

//+------------------------------------------------------------------+
//| Enhanced Position Management                                     |
//+------------------------------------------------------------------+
void ManagePositions()
{
    // Update position data first
    UpdatePositionData();
    
    // Process each active position
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!position_info.SelectByIndex(i)) continue;
        if(position_info.Magic() != InpMagicNumber) continue;
        if(position_info.Symbol() != _Symbol) continue;
        
        ulong ticket = position_info.Ticket();
        SAutonomousPositionData* pos_data = GetPositionData(ticket);
        
        if(pos_data == NULL) continue;
        
        // Update real-time metrics
        UpdatePositionMetrics(pos_data);
        
        // Execute autonomous management strategies
        ExecuteBreakevenStrategy(pos_data);
        ExecutePartialProfitStrategy(pos_data);
        ExecuteTrailingStopStrategy(pos_data);
        ExecuteDynamicExitStrategy(pos_data);
        ExecuteEmergencyExitStrategy(pos_data);
        
        // Update position state
        UpdatePositionState(pos_data);
    }
    
    // Clean up closed positions
    CleanupClosedPositions();
}

//+------------------------------------------------------------------+
//| Update Position Data                                            |
//+------------------------------------------------------------------+
void UpdatePositionData()
{
    // Refresh position count
    int current_positions = PositionsTotal();
    
    // Add new positions to tracking
    for(int i = 0; i < current_positions; i++)
    {
        if(!position_info.SelectByIndex(i)) continue;
        if(position_info.Magic() != InpMagicNumber) continue;
        if(position_info.Symbol() != _Symbol) continue;
        
        ulong ticket = position_info.Ticket();
        
        // Check if position already tracked
        bool already_tracked = false;
        for(int j = 0; j < g_position_count; j++)
        {
            if(g_position_data[j].ticket == ticket)
            {
                already_tracked = true;
                break;
            }
        }
        
        // Add new position to tracking
        if(!already_tracked && g_position_count < 100)
        {
            InitializePositionData(ticket);
        }
    }
}

//+------------------------------------------------------------------+
//| Initialize Position Data                                        |
//+------------------------------------------------------------------+
void InitializePositionData(ulong ticket)
{
    if(!position_info.SelectByTicket(ticket)) return;
    
    SAutonomousPositionData* pos_data = &g_position_data[g_position_count];
    
    // Basic position info
    pos_data.ticket = ticket;
    pos_data.entry_time = (datetime)position_info.Time();
    pos_data.entry_price = position_info.PriceOpen();
    pos_data.initial_sl = position_info.StopLoss();
    pos_data.initial_tp = position_info.TakeProfit();
    pos_data.current_sl = pos_data.initial_sl;
    pos_data.current_tp = pos_data.initial_tp;
    pos_data.position_type = (ENUM_POSITION_TYPE)position_info.PositionType();
    
    // Initialize tracking metrics
    pos_data.max_favorable_excursion = 0.0;
    pos_data.max_adverse_excursion = 0.0;
    pos_data.current_r_multiple = 0.0;
    pos_data.peak_r_multiple = 0.0;
    
    // AI factors
    pos_data.confluence_score_at_entry = CalculateCurrentConfluenceScore();
    pos_data.market_sentiment_score = CalculateMarketSentiment();
    
    // Management flags
    pos_data.has_moved_to_breakeven = false;
    pos_data.has_taken_partial_profit = false;
    pos_data.partial_profit_count = 0;
    
    // Exit triggers
    pos_data.momentum_reversal_detected = false;
    pos_data.structure_break_detected = false;
    pos_data.volume_exhaustion_detected = false;
    pos_data.time_based_exit_triggered = false;
    
    pos_data.last_update_time = TimeCurrent();
    
    g_position_count++;
    
    Print("Initialized position tracking for ticket: ", ticket, 
          " | Entry Confluence: ", DoubleToString(pos_data.confluence_score_at_entry, 1));
}

//+------------------------------------------------------------------+
//| Update Position Metrics                                        |
//+------------------------------------------------------------------+
void UpdatePositionMetrics(SAutonomousPositionData* pos_data)
{
    if(!position_info.SelectByTicket(pos_data.ticket)) return;
    
    double current_price = (pos_data.position_type == POSITION_TYPE_BUY) ? 
                          symbol_info.Bid() : symbol_info.Ask();
    
    // Calculate current P&L metrics
    double sl_distance = MathAbs(pos_data.entry_price - pos_data.initial_sl);
    double current_profit = 0.0;
    
    if(pos_data.position_type == POSITION_TYPE_BUY)
    {
        current_profit = current_price - pos_data.entry_price;
    }
    else
    {
        current_profit = pos_data.entry_price - current_price;
    }
    
    // Update R-multiple
    pos_data.current_r_multiple = (sl_distance > 0) ? current_profit / sl_distance : 0.0;
    
    // Update peak R-multiple
    if(pos_data.current_r_multiple > pos_data.peak_r_multiple)
    {
        pos_data.peak_r_multiple = pos_data.current_r_multiple;
    }
    
    // Update excursion metrics
    if(current_profit > pos_data.max_favorable_excursion)
    {
        pos_data.max_favorable_excursion = current_profit;
    }
    
    if(current_profit < pos_data.max_adverse_excursion)
    {
        pos_data.max_adverse_excursion = current_profit;
    }
    
    // Update P&L
    pos_data.unrealized_pnl = position_info.Profit();
    pos_data.unrealized_pnl_percent = (pos_data.entry_price > 0) ? 
        (current_profit / pos_data.entry_price) * 100.0 : 0.0;
    
    // Update current confluence score
    pos_data.current_confluence_score = CalculateCurrentConfluenceScore();
    
    pos_data.last_update_time = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Execute Breakeven Strategy                                      |
//+------------------------------------------------------------------+
void ExecuteBreakevenStrategy(SAutonomousPositionData* pos_data)
{
    if(pos_data.has_moved_to_breakeven) return;
    if(pos_data.current_r_multiple < InpBreakevenRR) return;
    
    // Enhanced breakeven logic with buffer
    double buffer = 10 * _Point; // 1 pip buffer
    double new_sl = 0.0;
    
    if(pos_data.position_type == POSITION_TYPE_BUY)
    {
        new_sl = pos_data.entry_price + buffer;
    }
    else
    {
        new_sl = pos_data.entry_price - buffer;
    }
    
    // Additional conditions for breakeven
    bool confluence_still_valid = (pos_data.current_confluence_score >= pos_data.confluence_score_at_entry * 0.8);
    bool no_major_reversal = !pos_data.momentum_reversal_detected;
    
    if(confluence_still_valid && no_major_reversal)
    {
        if(trade.PositionModify(pos_data.ticket, new_sl, pos_data.current_tp))
        {
            pos_data.current_sl = new_sl;
            pos_data.has_moved_to_breakeven = true;
            
            Print("Moved to breakeven - Ticket: ", pos_data.ticket, 
                  " | R-Multiple: ", DoubleToString(pos_data.current_r_multiple, 2));
        }
    }
}

//+------------------------------------------------------------------+
//| Execute Partial Profit Strategy                                |
//+------------------------------------------------------------------+
void ExecutePartialProfitStrategy(SAutonomousPositionData* pos_data)
{
    if(pos_data.partial_profit_count >= 2) return; // Max 2 partial profits
    if(pos_data.current_r_multiple < InpPartialProfitRR) return;
    
    // Dynamic partial profit levels
    double required_r = InpPartialProfitRR + (pos_data.partial_profit_count * 0.5);
    
    if(pos_data.current_r_multiple >= required_r)
    {
        // Calculate partial size based on position performance
        double partial_percentage = 0.33; // Default 33%
        
        // Increase partial if confluence is declining
        if(pos_data.current_confluence_score < pos_data.confluence_score_at_entry * 0.7)
        {
            partial_percentage = 0.50; // Take 50% if confluence weakening
        }
        
        // Increase partial if momentum is slowing
        if(pos_data.momentum_reversal_detected)
        {
            partial_percentage = 0.67; // Take 67% if momentum reversing
        }
        
        double current_volume = position_info.Volume();
        double partial_volume = current_volume * partial_percentage;
        partial_volume = MathMax(symbol_info.LotsMin(), partial_volume);
        
        // Execute partial profit using partial close to avoid hedging
        bool success = trade.PositionClosePartial(pos_data.ticket, partial_volume, 0, 0, 0, "Partial TP");
        
        if(success)
        {
            pos_data.partial_profit_count++;
            pos_data.has_taken_partial_profit = true;
            
            Print("Partial profit taken - Ticket: ", pos_data.ticket, 
                  " | Volume: ", DoubleToString(partial_volume, 2),
                  " | R-Multiple: ", DoubleToString(pos_data.current_r_multiple, 2));
        }
    }
}

//+------------------------------------------------------------------+
//| Execute Trailing Stop Strategy                                 |
//+------------------------------------------------------------------+
void ExecuteTrailingStopStrategy(SAutonomousPositionData* pos_data)
{
    if(pos_data.current_r_multiple < InpTrailingStartRR) return;
    
    // Dynamic trailing distance based on volatility and confluence
    double atr_buffer[1];
    if(CopyBuffer(h_atr_m15, 0, 0, 1, atr_buffer) <= 0) return;
    
    double base_trail_distance = atr_buffer[0] * 1.5;
    
    // Adjust trailing distance based on confluence
    double confluence_factor = pos_data.current_confluence_score / 100.0;
    double trail_distance = base_trail_distance * (2.0 - confluence_factor); // Tighter trail with higher confluence
    
    // Calculate new trailing stop
    double new_sl = 0.0;
    double current_price = (pos_data.position_type == POSITION_TYPE_BUY) ? 
                          symbol_info.Bid() : symbol_info.Ask();
    
    if(pos_data.position_type == POSITION_TYPE_BUY)
    {
        new_sl = current_price - trail_distance;
        // Only move SL up
        if(new_sl > pos_data.current_sl + 10 * _Point)
        {
            if(trade.PositionModify(pos_data.ticket, new_sl, pos_data.current_tp))
            {
                pos_data.current_sl = new_sl;
                Print("Trailing stop updated - Ticket: ", pos_data.ticket, " | New SL: ", new_sl);
            }
        }
    }
    else
    {
        new_sl = current_price + trail_distance;
        // Only move SL down
        if(new_sl < pos_data.current_sl - 10 * _Point)
        {
            if(trade.PositionModify(pos_data.ticket, new_sl, pos_data.current_tp))
            {
                pos_data.current_sl = new_sl;
                Print("Trailing stop updated - Ticket: ", pos_data.ticket, " | New SL: ", new_sl);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Execute Dynamic Exit Strategy                                  |
//+------------------------------------------------------------------+
void ExecuteDynamicExitStrategy(SAutonomousPositionData* pos_data)
{
    // Check for various exit conditions
    
    // 1. Confluence Deterioration Exit
    if(pos_data.current_confluence_score < pos_data.confluence_score_at_entry * 0.5 &&
       pos_data.current_r_multiple > 0.5) // Only if in profit
    {
        ClosePosition(pos_data.ticket, "Confluence deteriorated");
        return;
    }
    
    // 2. Time-based Exit (avoid holding too long)
    int position_age_hours = (int)((TimeCurrent() - pos_data.entry_time) / 3600);
    if(position_age_hours > 24 && pos_data.current_r_multiple < 1.0) // Exit if held >24h with low profit
    {
        ClosePosition(pos_data.ticket, "Time-based exit");
        return;
    }
    
    // 3. Structure Break Exit
    if(DetectStructureBreak(pos_data))
    {
        pos_data.structure_break_detected = true;
        if(pos_data.current_r_multiple > 0.2) // Exit if in profit
        {
            ClosePosition(pos_data.ticket, "Structure break detected");
            return;
        }
    }
    
    // 4. Momentum Reversal Exit
    if(DetectMomentumReversal(pos_data))
    {
        pos_data.momentum_reversal_detected = true;
        if(pos_data.current_r_multiple > 0.8) // Exit if significant profit
        {
            ClosePosition(pos_data.ticket, "Momentum reversal detected");
            return;
        }
    }
    
    // 5. Volume Exhaustion Exit
    if(DetectVolumeExhaustion(pos_data))
    {
        pos_data.volume_exhaustion_detected = true;
        if(pos_data.current_r_multiple > 1.0) // Exit if good profit
        {
            ClosePosition(pos_data.ticket, "Volume exhaustion detected");
            return;
        }
    }
}

//+------------------------------------------------------------------+
//| Execute Emergency Exit Strategy                                |
//+------------------------------------------------------------------+
void ExecuteEmergencyExitStrategy(SAutonomousPositionData* pos_data)
{
    // Emergency conditions for immediate exit
    
    // 1. Severe drawdown from peak
    double drawdown_from_peak = pos_data.peak_r_multiple - pos_data.current_r_multiple;
    if(drawdown_from_peak > 1.5 && pos_data.peak_r_multiple > 2.0) // 1.5R drawdown from 2R+ peak
    {
        ClosePosition(pos_data.ticket, "Emergency: Severe drawdown from peak");
        return;
    }
    
    // 2. News event emergency exit
    if(InpEnableNewsFilter && IsHighImpactNewsTime())
    {
        if(pos_data.current_r_multiple > 0.5) // Exit profitable trades before news
        {
            ClosePosition(pos_data.ticket, "Emergency: High impact news approaching");
            return;
        }
    }
    
    // 3. Multiple exit signals combined
    int exit_signal_count = 0;
    if(pos_data.momentum_reversal_detected) exit_signal_count++;
    if(pos_data.structure_break_detected) exit_signal_count++;
    if(pos_data.volume_exhaustion_detected) exit_signal_count++;
    
    if(exit_signal_count >= 2 && pos_data.current_r_multiple > 0.3)
    {
        ClosePosition(pos_data.ticket, "Emergency: Multiple exit signals");
        return;
    }
}

//+------------------------------------------------------------------+
//| Utility Functions for Position Management                      |
//+------------------------------------------------------------------+

SAutonomousPositionData* GetPositionData(ulong ticket)
{
    for(int i = 0; i < g_position_count; i++)
    {
        if(g_position_data[i].ticket == ticket)
        {
            return &g_position_data[i];
        }
    }
    return NULL;
}

void UpdatePositionState(SAutonomousPositionData* pos_data)
{
    // Update any additional state information
    pos_data.last_update_time = TimeCurrent();
}

void CleanupClosedPositions()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_position_count; i++)
    {
        if(position_info.SelectByTicket(g_position_data[i].ticket))
        {
            // Position still open, keep it
            if(valid_count != i)
            {
                g_position_data[valid_count] = g_position_data[i];
            }
            valid_count++;
        }
    }
    
    g_position_count = valid_count;
}

void ClosePosition(ulong ticket, string reason)
{
    if(trade.PositionClose(ticket))
    {
        Print("Position closed - Ticket: ", ticket, " | Reason: ", reason);
    }
}

double CalculateCurrentConfluenceScore()
{
    SConfluenceSignal signal = GenerateConfluenceSignal();
    return signal.confidence_score;
}

double CalculateMarketSentiment()
{
    // Simplified market sentiment calculation
    double sentiment = 0.5; // Neutral
    
    // Factor in various indicators
    double ema_fast[1], ema_slow[1];
    if(CopyBuffer(h_ema_fast, 0, 0, 1, ema_fast) > 0 &&
       CopyBuffer(h_ema_slow, 0, 0, 1, ema_slow) > 0)
    {
        if(ema_fast[0] > ema_slow[0]) sentiment += 0.2;
        else sentiment -= 0.2;
    }
    
    // Factor in RSI
    double rsi[1];
    if(CopyBuffer(h_rsi, 0, 0, 1, rsi) > 0)
    {
        if(rsi[0] > 50) sentiment += 0.1;
        else sentiment -= 0.1;
    }
    
    return MathMax(0.0, MathMin(1.0, sentiment));
}

bool DetectStructureBreak(SAutonomousPositionData* pos_data)
{
    // Simplified structure break detection
    MqlRates rates[10];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 10, rates) <= 0) return false;
    
    ArraySetAsSeries(rates, true);
    
    // Look for significant structure changes
    if(pos_data.position_type == POSITION_TYPE_BUY)
    {
        // Look for lower lows formation
        for(int i = 1; i < 5; i++)
        {
            if(rates[i].low < rates[i+1].low && rates[i].low < rates[i-1].low)
                return true;
        }
    }
    else
    {
        // Look for higher highs formation
        for(int i = 1; i < 5; i++)
        {
            if(rates[i].high > rates[i+1].high && rates[i].high > rates[i-1].high)
                return true;
        }
    }
    
    return false;
}

bool DetectMomentumReversal(SAutonomousPositionData* pos_data)
{
    // Check for momentum divergence or reversal patterns
    MqlRates rates[5];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 5, rates) <= 0) return false;
    
    ArraySetAsSeries(rates, true);
    
    // Simple momentum reversal check
    if(pos_data.position_type == POSITION_TYPE_BUY)
    {
        // Look for bearish reversal patterns
        return IsBearishEngulfing(rates, 1) || IsBearishPinBar(rates, 1);
    }
    else
    {
        // Look for bullish reversal patterns
        return IsBullishEngulfing(rates, 1) || IsBullishPinBar(rates, 1);
    }
}

bool DetectVolumeExhaustion(SAutonomousPositionData* pos_data)
{
    // Check for volume exhaustion
    MqlRates rates[5];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 5, rates) <= 0) return false;
    
    ArraySetAsSeries(rates, true);
    
    // Compare recent volume to average
    long recent_volume = rates[0].tick_volume + rates[1].tick_volume;
    long average_volume = 0;
    
    for(int i = 2; i < 5; i++)
    {
        average_volume += rates[i].tick_volume;
    }
    average_volume /= 3;
    
    // Volume exhaustion if recent volume is significantly lower
    return (recent_volume < average_volume * 0.6);
}

bool IsHighImpactNewsTime()
{
    // Simplified high impact news detection
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // Major news times (GMT)
    if(dt.hour == 12 && dt.min >= 30 && dt.min <= 35) return true; // US CPI
    if(dt.hour == 18 && dt.min >= 0 && dt.min <= 30) return true;  // FOMC
    if(dt.hour == 8 && dt.min >= 30 && dt.min <= 35) return true;  // London open news
    
    return false;
}

// Market analysis functions - Enhanced with Elite Detection
void UpdateOrderBlocks() 
{ 
    if(g_elite_ob_detector != NULL)
    {
        // Update existing order blocks status
        g_elite_ob_detector.UpdateOrderBlockStatus();
        
        // Remove invalid order blocks
        g_elite_ob_detector.RemoveInvalidOrderBlocks();
        
        // Detect new elite order blocks
        g_elite_ob_detector.DetectEliteOrderBlocks();
        
        if(InpVerboseLogging)
            Print("Elite Order Blocks Updated - Active Count: ", g_elite_ob_detector.GetActiveOrderBlockCount());
    }
    else
    {
        g_elite_ob_count = 0; // Fallback
    }
}
void UpdateFairValueGaps() 
{ 
    if(g_elite_fvg_detector != NULL)
    {
        // Update existing FVG status
        g_elite_fvg_detector.UpdateFVGStatus();
        
        // Remove filled/expired FVGs
        g_elite_fvg_detector.RemoveFilledFVGs();
        
        // Detect new elite FVGs
        g_elite_fvg_detector.DetectEliteFairValueGaps();
        
        if(InpVerboseLogging)
            Print("Elite Fair Value Gaps Updated - Active Count: ", g_elite_fvg_detector.GetActiveFVGCount(),
                  " | Best Score: ", DoubleToString(g_elite_fvg_detector.GetBestFVGScore(), 1));
    }
    else
    {
        g_elite_fvg_count = 0; // Fallback
    }
}
void UpdateLiquidityZones() 
{ 
    if(g_institutional_liq_detector != NULL)
    {
        // Update existing liquidity status
        g_institutional_liq_detector.UpdateLiquidityStatus();
        
        // Remove swept liquidity
        g_institutional_liq_detector.RemoveSweptLiquidity();
        
        // Detect new institutional liquidity
        g_institutional_liq_detector.DetectInstitutionalLiquidity();
        
        if(InpVerboseLogging)
            Print("Institutional Liquidity Updated - Active Count: ", g_institutional_liq_detector.GetActiveLiquidityCount(),
                  " | Best Score: ", DoubleToString(g_institutional_liq_detector.GetBestLiquidityScore(), 1));
    }
    else
    {
        g_liq_count = 0; // Fallback
    }
}

// Additional missing utility methods and validations
bool ValidateOrderBlockConfluence(const SInstitutionalLiquidityPool& pool)
{
    // Check if there's an Order Block near the liquidity pool
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        const SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        if(ob.state == OB_STATE_DISABLED || ob.state == OB_STATE_MITIGATED) continue;
        
        // Check if liquidity pool overlaps or is near Order Block
        double distance = MathMin(
            MathAbs(pool.price_level - ob.high_price),
            MathAbs(pool.price_level - ob.low_price)
        );
        
        if(distance <= 40 * _Point) // Within 4 pips
            return true;
    }
    return false;
}

bool ValidateFVGConfluence(const SInstitutionalLiquidityPool& pool)
{
    // Check if there's a Fair Value Gap near the liquidity pool
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        const SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        if(fvg.state == FVG_STATE_FILLED || fvg.state == FVG_STATE_EXPIRED) continue;
        
        // Check if liquidity pool overlaps with FVG
        double distance = MathMin(
            MathAbs(pool.price_level - fvg.upper_level),
            MathAbs(pool.price_level - fvg.lower_level)
        );
        
        if(distance <= 30 * _Point) // Within 3 pips
            return true;
    }
    return false;
}

bool ValidateStructureConfluence(const SInstitutionalLiquidityPool& pool)
{
    // Check if liquidity pool aligns with market structure
    MqlRates rates[50];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Simple trend alignment check
    bool uptrend = rates[0].close > rates[10].close;
    
    if(pool.type == LIQUIDITY_BSL && !uptrend) return true;  // BSL in downtrend
    if(pool.type == LIQUIDITY_SSL && uptrend) return true;   // SSL in uptrend
    
    return false;
}

// Complete missing institutional detection methods
void CInstitutionalLiquidityDetector::CalculateConfluenceScore(SInstitutionalLiquidityPool& pool)
{
    double score = 0.0;
    
    // Check Order Block confluence
    pool.has_ob_confluence = ValidateOrderBlockConfluence(pool);
    if(pool.has_ob_confluence) score += 35.0;
    
    // Check FVG confluence
    pool.has_fvg_confluence = ValidateFVGConfluence(pool);
    if(pool.has_fvg_confluence) score += 30.0;
    
    // Check Structure confluence
    pool.has_structure_confluence = ValidateStructureConfluence(pool);
    if(pool.has_structure_confluence) score += 35.0;
    
    pool.confluence_score = score;
}

bool CInstitutionalLiquidityDetector::CheckOrderBlockConfluence(const SInstitutionalLiquidityPool& pool)
{
    return ValidateOrderBlockConfluence(pool);
}

bool CInstitutionalLiquidityDetector::CheckFVGConfluence(const SInstitutionalLiquidityPool& pool)
{
    return ValidateFVGConfluence(pool);
}

// Validation methods
bool CInstitutionalLiquidityDetector::ValidateLiquiditySweep(double price)
{
    // Check if current price has swept any liquidity pools
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        SInstitutionalLiquidityPool& pool = g_institutional_liquidity[i];
        if(pool.state != LIQUIDITY_UNTAPPED) continue;
        
        double distance = MathAbs(price - pool.price_level);
        
        if(distance <= m_sweep_validation_distance)
        {
            pool.state = LIQUIDITY_SWEPT;
            pool.is_fresh = false;
            return true;
        }
    }
    return false;
}

double CInstitutionalLiquidityDetector::CalculateOptimalSweepEntry()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double optimal_entry = current_price;
    double best_distance = DBL_MAX;
    
    // Find the nearest high-quality liquidity pool
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        const SInstitutionalLiquidityPool& pool = g_institutional_liquidity[i];
        if(pool.state != LIQUIDITY_UNTAPPED) continue;
        if(pool.quality_score < 70.0) continue;
        
        double distance = MathAbs(current_price - pool.price_level);
        
        if(distance < best_distance)
        {
            best_distance = distance;
            optimal_entry = pool.price_level;
        }
    }
    
    return optimal_entry;
}

bool CInstitutionalLiquidityDetector::IsInstitutionalLiquidity(const SInstitutionalLiquidityPool& pool)
{
    // Criteria for institutional liquidity
    if(pool.quality_score < 80.0) return false;
    if(pool.confluence_score < 60.0) return false;
    if(pool.type != LIQUIDITY_WEEKLY && pool.type != LIQUIDITY_DAILY) return false;
    
    return true;
}

double CInstitutionalLiquidityDetector::CalculateLiquidityQuality(const SInstitutionalLiquidityPool& pool)
{
    double quality = 50.0; // Base quality
    
    // Type-based quality
    switch(pool.type)
    {
        case LIQUIDITY_WEEKLY: quality += 25.0; break;
        case LIQUIDITY_DAILY: quality += 20.0; break;
        case LIQUIDITY_SESSION: quality += 15.0; break;
        default: quality += 10.0; break;
    }
    
    // Touch count factor
    if(pool.touch_count >= 3) quality += 15.0;
    else if(pool.touch_count >= 2) quality += 10.0;
    
    // Confluence bonus
    quality += pool.confluence_score * 0.2;
    
    // Fresh liquidity bonus
    if(pool.is_fresh) quality += 10.0;
    
    // Institutional characteristics
    if(pool.is_institutional) quality += 15.0;
    
    return MathMin(quality, 100.0);
}

void CInstitutionalLiquidityDetector::UpdateLiquidityAfterSweep(SInstitutionalLiquidityPool& pool)
{
    pool.state = LIQUIDITY_SWEPT;
    pool.is_fresh = false;
    pool.time_decay_factor = 0.1; // Heavily decay swept liquidity
    
    // Reduce quality after sweep
    pool.quality_score *= 0.3;
    pool.sweep_probability = 0.0;
}

// Complete missing Elite FVG detector methods
double CEliteFVGDetector::CalculateOptimalFillLevel(const SEliteFairValueGap& fvg)
{
    // Calculate optimal fill level based on FVG type and market conditions
    double optimal_level = fvg.mid_level; // Default to middle
    
    if(fvg.type == FVG_BULLISH)
    {
        // For bullish FVG, optimal fill is at 61.8% level from bottom
        optimal_level = fvg.lower_level + (fvg.upper_level - fvg.lower_level) * 0.618;
    }
    else if(fvg.type == FVG_BEARISH)
    {
        // For bearish FVG, optimal fill is at 61.8% level from top
        optimal_level = fvg.upper_level - (fvg.upper_level - fvg.lower_level) * 0.618;
    }
    
    return optimal_level;
}

bool CEliteFVGDetector::IsHighProbabilityFVG(const SEliteFairValueGap& fvg)
{
    // High probability criteria
    if(fvg.quality_score < 75.0) return false;
    if(fvg.expected_reaction < 0.7) return false;
    if(!fvg.has_volume_spike) return false;
    if(fvg.fill_percentage > 30.0) return false; // Not too filled
    
    // Confluence requirements
    if(fvg.confluence_count < 2) return false;
    
    return true;
}

bool CEliteFVGDetector::DetectInstitutionalFVG()
{
    // Enhanced institutional FVG detection
    bool found_institutional = false;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        
        // Check institutional characteristics
        if(fvg.gap_size_points >= 300 &&  // Large gap (30+ pips)
           fvg.displacement_size >= 500 * _Point && // Strong displacement
           fvg.has_volume_spike &&         // Volume confirmation
           fvg.quality_score >= 85.0)     // High quality
        {
            fvg.is_institutional = true;
            fvg.quality = FVG_QUALITY_ELITE;
            found_institutional = true;
        }
    }
    
    return found_institutional;
}

bool CEliteFVGDetector::ValidateWithDisplacement(SEliteFairValueGap& fvg)
{
    // Validate FVG with displacement analysis
    if(fvg.displacement_size < m_min_displacement_size) return false;
    
    // Check displacement quality
    double displacement_quality = fvg.displacement_size / (200.0 * _Point);
    if(displacement_quality < 1.5) return false;
    
    // Update FVG strength based on displacement
    fvg.quality_score += MathMin(displacement_quality * 10.0, 25.0);
    
    return true;
}

// Complete missing Elite Order Block detector methods
double CEliteOrderBlockDetector::CalculateOptimalEntry(const SAdvancedOrderBlock& ob)
{
    double optimal_entry = ob.refined_entry;
    
    // Adjust based on order block type and market conditions
    if(ob.type == OB_BULLISH)
    {
        // For bullish OB, optimal entry is at 70% level from bottom
        optimal_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.7;
    }
    else if(ob.type == OB_BEARISH)
    {
        // For bearish OB, optimal entry is at 30% level from bottom
        optimal_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.3;
    }
    
    // Adjust for premium/discount zones
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    if(ob.is_premium && ob.type == OB_BEARISH)
    {
        optimal_entry = ob.high_price - (ob.high_price - ob.low_price) * 0.2;
    }
    else if(!ob.is_premium && ob.type == OB_BULLISH)
    {
        optimal_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.2;
    }
    
    return optimal_entry;
}

bool CEliteOrderBlockDetector::DetectPremiumOrderBlocks()
{
    // Detect order blocks in premium zones
    bool found_premium = false;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        
        if(IsInPremiumZone(ob.refined_entry) && ob.type == OB_BEARISH)
        {
            ob.is_premium = true;
            ob.probability_score += 15.0; // Bonus for premium bearish OB
            found_premium = true;
        }
    }
    
    return found_premium;
}

bool CEliteOrderBlockDetector::DetectDiscountOrderBlocks()
{
    // Detect order blocks in discount zones
    bool found_discount = false;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        
        if(IsInDiscountZone(ob.refined_entry) && ob.type == OB_BULLISH)
        {
            ob.is_premium = false;
            ob.probability_score += 15.0; // Bonus for discount bullish OB
            found_discount = true;
        }
    }
    
    return found_discount;
}

bool CEliteOrderBlockDetector::ValidateWithLiquidity(SAdvancedOrderBlock& ob)
{
    // Check if order block has liquidity confluence
    ob.has_liquidity = false;
    
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        const SInstitutionalLiquidityPool& pool = g_institutional_liquidity[i];
        if(pool.state != LIQUIDITY_UNTAPPED) continue;
        
        double distance = MathMin(
            MathAbs(ob.high_price - pool.price_level),
            MathAbs(ob.low_price - pool.price_level)
        );
        
        if(distance <= 50 * _Point) // Within 5 pips
        {
            ob.has_liquidity = true;
            ob.strength += 10.0;
            break;
        }
    }
    
    return ob.has_liquidity;
}

bool CEliteOrderBlockDetector::ValidateWithVolume(SAdvancedOrderBlock& ob)
{
    // Validate order block with volume analysis
    if(ob.volume_profile < m_volume_threshold) return false;
    
    // Bonus for strong volume
    if(ob.volume_profile > 2.0)
    {
        ob.strength += 15.0;
        ob.is_institutional = true;
    }
    
    return true;
}

bool CEliteOrderBlockDetector::ValidateWithStructure(SAdvancedOrderBlock& ob)
{
    // Validate with market structure
    MqlRates rates[30];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 30, rates) <= 0) return true;
    
    ArraySetAsSeries(rates, true);
    
    // Check structure alignment
    bool structure_valid = false;
    
    if(ob.type == OB_BULLISH)
    {
        // Look for higher lows formation
        for(int i = 5; i < 25; i++)
        {
            if(rates[i].low < rates[i-5].low && rates[i].low > rates[i+5].low)
            {
                structure_valid = true;
                break;
            }
        }
    }
    else if(ob.type == OB_BEARISH)
    {
        // Look for lower highs formation
        for(int i = 5; i < 25; i++)
        {
            if(rates[i].high > rates[i-5].high && rates[i].high < rates[i+5].high)
            {
                structure_valid = true;
                break;
            }
        }
    }
    
    if(structure_valid)
    {
        ob.has_structure_confluence = true;
        ob.strength += 12.0;
    }
    
    return structure_valid;
}

bool CEliteOrderBlockDetector::IsInstitutionalOrderBlock(const SAdvancedOrderBlock& ob)
{
    // Institutional order block criteria
    if(ob.strength < 80.0) return false;
    if(ob.displacement_size < 300 * _Point) return false;
    if(ob.volume_profile < 1.8) return false;
    if(ob.quality < OB_QUALITY_HIGH) return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| End of EA Implementation                                         |
//+------------------------------------------------------------------+
