
//+------------------------------------------------------------------+
//|                                           DataStructures.mqh |
//|                        TradeDev_Master Elite Trading System     |
//|                                   Advanced Market Analysis      |
//+------------------------------------------------------------------+

#ifndef DATA_STRUCTURES_MQH
#define DATA_STRUCTURES_MQH

// Enumerações para Log System
enum ENUM_LOG_LEVEL
{
   LOG_DEBUG = 0,
   LOG_INFO = 1,
   LOG_WARNING = 2,
   LOG_ERROR = 3,
   LOG_CRITICAL = 4
};

// Enumerações para SL/TP Calculation
enum ENUM_SL_CALCULATION_METHOD
{
   SL_FIXED = 0,
   SL_ATR = 1,
   SL_STRUCTURE = 2,
   SL_HYBRID = 3,
   SL_DYNAMIC = 4
};

enum ENUM_TP_CALCULATION_METHOD
{
   TP_FIXED = 0,
   TP_ATR = 1,
   TP_STRUCTURE = 2,
   TP_RR_RATIO = 3,
   TP_FIBONACCI = 4
};

// Enumerações para Trailing Stop
enum ENUM_TRAILING_METHOD
{
   TRAILING_NONE = 0,
   TRAILING_FIXED = 1,
   TRAILING_ATR = 2,
   TRAILING_STRUCTURE_BREAKS = 3,
   TRAILING_SMART = 4,
   TRAILING_PARABOLIC = 5
};

// Enumerações ICT/SMC
enum ENUM_ORDER_BLOCK_TYPE
{
   OB_BULLISH = 0,
   OB_BEARISH = 1,
   OB_MITIGATION = 2
};

enum ENUM_FVG_TYPE
{
   FVG_BULLISH = 0,
   FVG_BEARISH = 1,
   FVG_BALANCED = 2
};

enum ENUM_LIQUIDITY_TYPE
{
   LIQ_EQUAL_HIGHS = 0,
   LIQ_EQUAL_LOWS = 1,
   LIQ_SWEEP_HIGH = 2,
   LIQ_SWEEP_LOW = 3,
   LIQ_INSTITUTIONAL = 4
};

// Estruturas ICT/SMC
struct SOrderBlock
{
   datetime time_start;
   datetime time_end;
   double high;
   double low;
   double open;
   double close;
   double volume;
   ENUM_ORDER_BLOCK_TYPE type;
   bool is_valid;
   bool is_mitigated;
   int strength;
   double mitigation_percentage;
   datetime last_test_time;
   int test_count;
};

struct SFVG
{
   datetime time;
   double high;
   double low;
   double gap_size;
   ENUM_FVG_TYPE type;
   bool is_filled;
   double fill_percentage;
   bool is_valid;
   int timeframe;
   double volume_imbalance;
};

struct SLiquidityLevel
{
   double price;
   datetime time_created;
   datetime time_last_test;
   ENUM_LIQUIDITY_TYPE type;
   int touches;
   bool is_broken;
   bool is_swept;
   double volume;
   double strength;
   bool is_institutional;
};

struct SMarketStructure
{
   bool is_bullish_structure;
   bool is_bearish_structure;
   bool is_ranging;
   double structure_high;
   double structure_low;
   datetime last_structure_break;
   bool choch_detected;
   bool bos_detected;
   double momentum_strength;
};

// Estruturas de Performance
struct SPerformanceMetrics
{
   double total_profit;
   double total_loss;
   double profit_factor;
   double sharpe_ratio;
   double max_drawdown;
   double max_drawdown_percent;
   int total_trades;
   int winning_trades;
   int losing_trades;
   double win_rate;
   double avg_win;
   double avg_loss;
   double largest_win;
   double largest_loss;
   double recovery_factor;
   datetime last_update;
};

// Estruturas de Confluência
struct SSignalConfluence
{
   double score;
   bool order_block_confluence;
   bool fvg_confluence;
   bool liquidity_confluence;
   bool structure_confluence;
   bool volume_confluence;
   bool time_confluence;
   bool fibonacci_confluence;
   bool trend_confluence;
   datetime signal_time;
   bool is_valid;
};

// Estruturas de Risk Management
struct SRiskParameters
{
   double max_risk_per_trade;
   double max_daily_risk;
   double max_weekly_risk;
   double max_monthly_risk;
   double current_daily_risk;
   double current_weekly_risk;
   double current_monthly_risk;
   bool risk_limit_reached;
   datetime last_reset_time;
};

// Estruturas de Cache
struct SCacheEntry
{
   string key;
   string value;
   datetime timestamp;
   datetime expiry;
   bool is_valid;
};

// Estruturas de Configuração
struct SConfigData
{
   string section;
   string key;
   string value;
   string description;
   bool is_encrypted;
};

// Estruturas de Estatísticas do EA
struct SEAStatistics
{
   datetime start_time;
   datetime last_update;
   int total_signals;
   int signals_taken;
   int signals_filtered;
   double signal_accuracy;
   double avg_signal_strength;
   SPerformanceMetrics performance;
   SRiskParameters risk_status;
};

#endif // DATA_STRUCTURES_MQH
