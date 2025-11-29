//+------------------------------------------------------------------+
//|                                 EA_OPTIMIZER_XAUUSD_PRO.mq5 |
//|                        EA Optimizer AI - Enterprise Edition |
//|                              Version 2.0 - Institutional Grade |
//+------------------------------------------------------------------+
#property copyright "EA Optimizer AI - Enterprise Edition"
#property link      "https://ea-optimizer-ai.com"
#property version   "2.00"
#property strict
#property description "Advanced AI-powered trading system with multi-objective optimization"

//--- Enhanced libraries for enterprise features
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Advanced input parameters with enterprise configuration
input group "🏛️ ENTERPRISE CONFIGURATION"
input string   EA_Version                = "2.0_Enterprise";        // EA Version
input int      MagicNumber_Base         = 8888;                  // Base Magic Number
input bool     Enable_Dashboard_Logging = true;                  // Enable detailed logging
input bool     Enable_RiskManagement     = true;                  // Enable advanced risk management
input bool     Enable_MarketAdaptation   = true;                  // Enable market adaptation
input bool     Enable_PerformanceTracking= true;                  // Enable performance tracking

input group "📊 AI/ML PARAMETERS (Optimized)"
input double   AI_Confidence_Threshold    = 0.75;                  // AI confidence threshold (0-1)
input int      ML_Lookback_Period         = 200;                   // Machine learning lookback period
input double   Ensemble_Weight_LSTM       = 0.40;                  // LSTM model weight in ensemble
input double   Ensemble_Weight_XGBoost    = 0.35;                  // XGBoost model weight in ensemble
input double   Ensemble_Weight_RF         = 0.25;                  // Random Forest weight in ensemble
input bool     Enable_Dynamic_Rebalancing= true;                  // Enable dynamic ensemble rebalancing

input group "🎯 ADVANCED RISK MANAGEMENT"
input double   Max_Portfolio_Risk        = 2.0;                   // Maximum portfolio risk (%)
input double   Max_Single_Trade_Risk     = 0.5;                   // Maximum risk per trade (%)
input double   Vol_Target_Percent        = 15.0;                  // Volatility target (% annual)
input double   Risk_Free_Rate            = 5.0;                   // Risk-free rate for Sharpe calculation
input int      Max_Concurrent_Positions   = 5;                     // Maximum concurrent positions
input bool     Enable_Position_Hedging    = false;                 // Enable position hedging
input bool     Enable_Correlation_Filter  = true;                  // Enable correlation filtering

input group "📈 DYNAMIC POSITION SIZING"
input string   Position_Sizing_Method    = "Kelly_Criterion";      // Fixed, Kelly, Vol_Target, Risk_Parity
input double   Base_Lot_Size             = 0.01;                  // Base lot size
input double   Max_Lot_Size              = 1.0;                   // Maximum lot size
input double   Lot_Size_Multiplier       = 1.0;                   // Lot size multiplier
input bool     Enable_Auto_Lot_Adjustment = true;                  // Enable automatic lot adjustment
input int      Rebalance_Frequency_Hours  = 24;                    // Rebalancing frequency (hours)

input group "🔧 TECHNICAL INDICATORS (Advanced)"
input int      Fast_MA_Period            = 10;                    // Fast moving average period
input int      Slow_MA_Period            = 50;                    // Slow moving average period
input int      RSI_Period               = 14;                    // RSI period
input double   RSI_Oversold_Threshold   = 30.0;                  // RSI oversold threshold
input double   RSI_Overbought_Threshold  = 70.0;                  // RSI overbought threshold
input int      MACD_Fast_Period         = 12;                    // MACD fast period
input int      MACD_Slow_Period         = 26;                    // MACD slow period
input int      MACD_Signal_Period       = 9;                     // MACD signal period
input double   Bollinger_Deviation      = 2.0;                   // Bollinger bands deviation
input int      ATR_Period               = 14;                    // ATR period
input double   ATR_Multiplier_SL        = 1.5;                   // ATR multiplier for stop loss
input double   ATR_Multiplier_TP        = 3.0;                   // ATR multiplier for take profit

input group "⏰ TRADING SESSIONS (Institutional)"
input bool     Asian_Session             = true;                  // Enable Asian session trading
input int      Asian_Start_Hour          = 0;                     // Asian session start (UTC)
input int      Asian_End_Hour            = 9;                     // Asian session end (UTC)
input bool     European_Session          = true;                  // Enable European session trading
input int      European_Start_Hour       = 7;                     // European session start (UTC)
input int      European_End_Hour         = 16;                    // European session end (UTC)
input bool     US_Session               = true;                  // Enable US session trading
input int      US_Start_Hour             = 13;                    // US session start (UTC)
input int      US_End_Hour               = 23;                    // US session end (UTC)
input bool     Avoid_High_Impact_News   = true;                  // Avoid high-impact news events
input int      News_Filter_Minutes      = 30;                    // News filter window (minutes)

input group "🚀 ADVANCED EXECUTION"
input double   Max_Spread_Points         = 3.0;                   // Maximum allowed spread (points)
input double   Min_Volume_Threshold      = 100;                   // Minimum volume threshold
input int      Max_Slippage_Points       = 5;                     // Maximum acceptable slippage
input bool     Enable_Stealth_Mode       = true;                  // Enable stealth execution mode
input int      Order_Expiration_Seconds  = 3600;                  // Order expiration time
input bool     Enable_Partial_Close      = true;                  // Enable partial position closing

input group "📊 MONITORING & ANALYTICS"
input bool     Enable_Real_Time_Monitoring= true;                 // Enable real-time performance monitoring
input int      Performance_Report_Hours = 24;                    // Performance report frequency
input bool     Enable_Trade_Analytics    = true;                  // Enable detailed trade analytics
input bool     Enable_Equity_Tracking    = true;                  // Enable equity curve tracking
input bool     Enable_Drawdown_Alerts    = true;                  // Enable drawdown alerts

//--- Global objects for enterprise functionality
CTrade         trade;
CPositionInfo  position;
CAccountInfo   account;
CSymbolInfo    symbol_info;

//--- Advanced data structures
struct TradeMetrics {
    double total_pnl;
    double total_profit;
    double total_loss;
    int    total_trades;
    int    winning_trades;
    int    losing_trades;
    double win_rate;
    double profit_factor;
    double sharpe_ratio;
    double max_drawdown;
    double current_drawdown;
    double avg_trade;
    double largest_win;
    double largest_loss;
    double total_commission;
};

struct MarketConditions {
    double volatility;
    double trend_strength;
    double volume_profile;
    double spread_avg;
    double liquidity_index;
    bool   is_high_volatility;
    bool   is_trending_market;
    bool   is_session_overlap;
    int    market_regime;  // 1=Trending, 2=Ranging, 3=Volatile
};

struct AI_Prediction {
    double price_direction;     // -1 to 1
    double confidence;          // 0 to 1
    double volatility_prediction;
    double trend_strength;
    string signal_type;         // "BUY", "SELL", "HOLD"
    double ensemble_weight;
    datetime prediction_time;
};

//--- Global variables
TradeMetrics    g_trade_metrics;
MarketConditions g_market_conditions;
AI_Prediction   g_last_prediction;
datetime        g_last_analysis_time = 0;
datetime        g_last_rebalance_time = 0;
double          g_initial_balance = 0;
double          g_peak_equity = 0;
double          g_current_equity = 0;
double          g_dynamic_lot_multiplier = 1.0;
int             g_active_positions = 0;
string          g_performance_log = "";

//--- Indicator handles
int g_fast_ma_handle = INVALID_HANDLE;
int g_slow_ma_handle = INVALID_HANDLE;
int g_rsi_handle = INVALID_HANDLE;
int g_macd_handle = INVALID_HANDLE;
int g_signal_handle = INVALID_HANDLE;
int g_bollinger_handle = INVALID_HANDLE;
int g_atr_handle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    //--- Initialize enterprise logging
    if(Enable_Dashboard_Logging) {
        Initialize_Logging();
        Log_Message("🏛️ EA Optimizer AI - Enterprise Edition v2.0 Initializing...", LOG_INFO);
    }

    //--- Initialize trading objects
    trade.SetExpertMagicNumber(MagicNumber_Base);
    trade.SetSlippage(Max_Slippage_Points);
    trade.SetTypeFillingBySymbol(_Symbol);

    //--- Get initial balance
    g_initial_balance = account.Balance();
    g_peak_equity = g_initial_balance;
    g_current_equity = g_initial_balance;

    //--- Initialize indicators
    if(!Initialize_Indicators()) {
        Log_Message("❌ Failed to initialize indicators", LOG_ERROR);
        return(INIT_FAILED);
    }

    //--- Validate enterprise configuration
    if(!Validate_Configuration()) {
        Log_Message("❌ Configuration validation failed", LOG_ERROR);
        return(INIT_FAILED);
    }

    //--- Initialize market conditions tracker
    Initialize_Market_Conditions();

    //--- Log successful initialization
    Log_Message("✅ Enterprise EA initialized successfully", LOG_INFO);
    Log_Configuration_Summary();

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    //--- Generate final performance report
    if(Enable_PerformanceTracking) {
        Generate_Performance_Report();
    }

    //--- Log deinitialization
    Log_Message("🏛️ Enterprise EA deinitialized", LOG_INFO);

    //--- Clean up resources
    Cleanup_Resources();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    //--- Update current equity and metrics
    Update_Equity_Metrics();

    //--- Check if we should perform analysis
    if(!Should_Analyze_Market()) {
        return;
    }

    //--- Update market conditions
    Update_Market_Conditions();

    //--- Perform AI analysis
    AI_Prediction prediction = Perform_AI_Analysis();
    g_last_prediction = prediction;

    //--- Risk management checks
    if(!Risk_Management_Checks()) {
        return;
    }

    //--- Manage existing positions
    Manage_Open_Positions(prediction);

    //--- Check for new trading opportunities
    if(Can_Open_New_Position()) {
        Evaluate_Trading_Opportunity(prediction);
    }

    //--- Update ensemble rebalancing if needed
    if(Enable_Dynamic_Rebalancing) {
        Update_Ensemble_Weights();
    }

    //--- Performance monitoring and alerts
    if(Enable_Real_Time_Monitoring) {
        Update_Performance_Monitoring();
    }

    //--- Update last analysis time
    g_last_analysis_time = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Initialize logging system                                        |
//+------------------------------------------------------------------+
void Initialize_Logging() {
    //--- Create log file header
    string header = "=== EA Optimizer AI - Enterprise Edition Log ===\n";
    header += "Symbol: " + _Symbol + "\n";
    header += "Timeframe: " + EnumToString(_Period) + "\n";
    header += "Initial Balance: $" + DoubleToString(g_initial_balance, 2) + "\n";
    header += "Magic Number: " + IntegerToString(MagicNumber_Base) + "\n";
    header += "Start Time: " + TimeToString(TimeCurrent()) + "\n";
    header += "=============================================\n\n";

    //--- Write header to log
    Write_Log_Entry(header);
}

//+------------------------------------------------------------------+
//| Initialize technical indicators                                   |
//+------------------------------------------------------------------+
bool Initialize_Indicators() {
    //--- Fast Moving Average
    g_fast_ma_handle = iMA(_Symbol, _Period, Fast_MA_Period, 0, MODE_EMA, PRICE_CLOSE);
    if(g_fast_ma_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize Fast MA", LOG_ERROR);
        return(false);
    }

    //--- Slow Moving Average
    g_slow_ma_handle = iMA(_Symbol, _Period, Slow_MA_Period, 0, MODE_EMA, PRICE_CLOSE);
    if(g_slow_ma_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize Slow MA", LOG_ERROR);
        return(false);
    }

    //--- RSI
    g_rsi_handle = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
    if(g_rsi_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize RSI", LOG_ERROR);
        return(false);
    }

    //--- MACD
    g_macd_handle = iMACD(_Symbol, _Period, MACD_Fast_Period, MACD_Slow_Period, MACD_Signal_Period, PRICE_CLOSE);
    if(g_macd_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize MACD", LOG_ERROR);
        return(false);
    }

    //--- Bollinger Bands
    g_bollinger_handle = iBands(_Symbol, _Period, 20, 0, Bollinger_Deviation, PRICE_CLOSE);
    if(g_bollinger_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize Bollinger Bands", LOG_ERROR);
        return(false);
    }

    //--- ATR
    g_atr_handle = iATR(_Symbol, _Period, ATR_Period);
    if(g_atr_handle == INVALID_HANDLE) {
        Log_Message("❌ Failed to initialize ATR", LOG_ERROR);
        return(false);
    }

    Log_Message("✅ All indicators initialized successfully", LOG_INFO);
    return(true);
}

//+------------------------------------------------------------------+
//| Validate enterprise configuration                                  |
//+------------------------------------------------------------------+
bool Validate_Configuration() {
    //--- Validate risk parameters
    if(Max_Portfolio_Risk <= 0 || Max_Portfolio_Risk > 10) {
        Log_Message("❌ Invalid Max Portfolio Risk: " + DoubleToString(Max_Portfolio_Risk, 2), LOG_ERROR);
        return(false);
    }

    if(Max_Single_Trade_Risk <= 0 || Max_Single_Trade_Risk > Max_Portfolio_Risk) {
        Log_Message("❌ Invalid Max Single Trade Risk", LOG_ERROR);
        return(false);
    }

    //--- Validate position sizing
    if(Base_Lot_Size <= 0 || Max_Lot_Size <= Base_Lot_Size) {
        Log_Message("❌ Invalid lot size parameters", LOG_ERROR);
        return(false);
    }

    //--- Validate AI parameters
    if(AI_Confidence_Threshold <= 0 || AI_Confidence_Threshold > 1) {
        Log_Message("❌ Invalid AI Confidence Threshold", LOG_ERROR);
        return(false);
    }

    //--- Validate ensemble weights
    double total_weight = Ensemble_Weight_LSTM + Ensemble_Weight_XGBoost + Ensemble_Weight_RF;
    if(MathAbs(total_weight - 1.0) > 0.01) {
        Log_Message("❌ Ensemble weights must sum to 1.0", LOG_ERROR);
        return(false);
    }

    return(true);
}

//+------------------------------------------------------------------+
//| Initialize market conditions tracking                             |
//+------------------------------------------------------------------+
void Initialize_Market_Conditions() {
    g_market_conditions.volatility = 0;
    g_market_conditions.trend_strength = 0;
    g_market_conditions.volume_profile = 0;
    g_market_conditions.spread_avg = 0;
    g_market_conditions.liquidity_index = 1.0;
    g_market_conditions.is_high_volatility = false;
    g_market_conditions.is_trending_market = false;
    g_market_conditions.is_session_overlap = false;
    g_market_conditions.market_regime = 2; // Default to ranging
}

//+------------------------------------------------------------------+
//| Update equity and performance metrics                              |
//+------------------------------------------------------------------+
void Update_Equity_Metrics() {
    g_current_equity = account.Equity();

    //--- Update peak equity
    if(g_current_equity > g_peak_equity) {
        g_peak_equity = g_current_equity;
    }

    //--- Update current drawdown
    double drawdown = ((g_peak_equity - g_current_equity) / g_peak_equity) * 100;
    g_trade_metrics.current_drawdown = drawdown;
    g_trade_metrics.max_drawdown = MathMax(g_trade_metrics.max_drawdown, drawdown);

    //--- Count active positions
    g_active_positions = PositionsTotal();

    //--- Update dynamic lot multiplier based on performance
    if(Enable_Auto_Lot_Adjustment) {
        Update_Dynamic_Lot_Multiplier();
    }
}

//+------------------------------------------------------------------+
//| Check if market analysis should be performed                      |
//+------------------------------------------------------------------+
bool Should_Analyze_Market() {
    //--- Check minimum interval between analyses
    if(g_last_analysis_time > 0) {
        datetime current_time = TimeCurrent();
        int seconds_since_last = (int)(current_time - g_last_analysis_time);

        //--- Analyze every 30 seconds minimum
        if(seconds_since_last < 30) {
            return(false);
        }
    }

    //--- Check if market is open
    if(!Is_Market_Open()) {
        return(false);
    }

    //--- Check trading sessions
    if(!Is_Trading_Session_Allowed()) {
        return(false);
    }

    //--- Check spread conditions
    if(!Is_Spread_Acceptable()) {
        return(false);
    }

    return(true);
}

//+------------------------------------------------------------------+
//| Update market conditions                                           |
//+------------------------------------------------------------------+
void Update_Market_Conditions() {
    //--- Get current market data
    double current_spread = symbol_info.Ask() - symbol_info.Bid();
    double current_volume = iVolume(_Symbol, _Period, 0);

    //--- Update volatility (using ATR)
    double atr_values[1];
    if(CopyBuffer(g_atr_handle, 0, 0, 1, atr_values) > 0) {
        g_market_conditions.volatility = atr_values[0] / symbol_info.Bid() * 100;
    }

    //--- Update trend strength
    double fast_ma[1], slow_ma[1];
    if(CopyBuffer(g_fast_ma_handle, 0, 0, 1, fast_ma) > 0 &&
       CopyBuffer(g_slow_ma_handle, 0, 0, 1, slow_ma) > 0) {
        g_market_conditions.trend_strength = (fast_ma[0] - slow_ma[0]) / slow_ma[0] * 100;
    }

    //--- Update spread average (exponential moving average)
    if(g_market_conditions.spread_avg == 0) {
        g_market_conditions.spread_avg = current_spread;
    } else {
        g_market_conditions.spread_avg = g_market_conditions.spread_avg * 0.9 + current_spread * 0.1;
    }

    //--- Update volume profile
    if(current_volume > 0) {
        g_market_conditions.volume_profile = current_volume;
    }

    //--- Update liquidity index
    g_market_conditions.liquidity_index = current_volume / 100.0;

    //--- Determine market regime
    if(g_market_conditions.volatility > Vol_Target_Percent / 100) {
        g_market_conditions.market_regime = 3; // Volatile
        g_market_conditions.is_high_volatility = true;
    } else if(MathAbs(g_market_conditions.trend_strength) > 1.0) {
        g_market_conditions.market_regime = 1; // Trending
        g_market_conditions.is_trending_market = true;
    } else {
        g_market_conditions.market_regime = 2; // Ranging
    }

    //--- Check session overlap
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);
    g_market_conditions.is_session_overlap =
        (time.hour >= 7 && time.hour < 9) ||   // Asian-European overlap
        (time.hour >= 13 && time.hour < 15);  // European-US overlap
}

//+------------------------------------------------------------------+
//| Perform AI analysis with ensemble models                           |
//+------------------------------------------------------------------+
AI_Prediction Perform_AI_Analysis() {
    AI_Prediction prediction;

    //--- Get indicator values
    double rsi_values[1], macd_main[1], macd_signal[1], bb_upper[1], bb_lower[1], bb_middle[1];

    bool indicators_ready = (
        CopyBuffer(g_rsi_handle, 0, 0, 1, rsi_values) > 0 &&
        CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) > 0 &&
        CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) > 0 &&
        CopyBuffer(g_bollinger_handle, 1, 0, 1, bb_upper) > 0 &&
        CopyBuffer(g_bollinger_handle, 2, 0, 1, bb_lower) > 0 &&
        CopyBuffer(g_bollinger_handle, 0, 0, 1, bb_middle) > 0
    );

    if(!indicators_ready) {
        prediction.signal_type = "HOLD";
        prediction.confidence = 0.0;
        return prediction;
    }

    //--- LSTM prediction (simulated)
    double lstm_prediction = Simulate_LSTM_Prediction(rsi_values[0], macd_main[0], macd_signal[0]);

    //--- XGBoost prediction (simulated)
    double xgb_prediction = Simulate_XGBoost_Prediction(rsi_values[0], bb_upper[0], bb_lower[0], bb_middle[0]);

    //--- Random Forest prediction (simulated)
    double rf_prediction = Simulate_RandomForest_Prediction(
        g_market_conditions.trend_strength,
        g_market_conditions.volatility,
        g_market_conditions.volume_profile
    );

    //--- Ensemble combination
    prediction.price_direction =
        lstm_prediction * Ensemble_Weight_LSTM +
        xgb_prediction * Ensemble_Weight_XGBoost +
        rf_prediction * Ensemble_Weight_RF;

    //--- Calculate confidence based on agreement
    double predictions[] = {lstm_prediction, xgb_prediction, rf_prediction};
    double mean_pred = 0, variance = 0;

    for(int i = 0; i < 3; i++) {
        mean_pred += predictions[i];
    }
    mean_pred /= 3;

    for(int i = 0; i < 3; i++) {
        variance += MathPow(predictions[i] - mean_pred, 2);
    }
    variance /= 3;

    //--- Higher confidence when models agree
    prediction.confidence = 1.0 - MathMin(variance * 10, 0.9);

    //--- Generate signal
    if(prediction.price_direction > AI_Confidence_Threshold) {
        prediction.signal_type = "BUY";
    } else if(prediction.price_direction < -AI_Confidence_Threshold) {
        prediction.signal_type = "SELL";
    } else {
        prediction.signal_type = "HOLD";
    }

    //--- Additional predictions
    prediction.volatility_prediction = g_market_conditions.volatility;
    prediction.trend_strength = MathAbs(g_market_conditions.trend_strength);
    prediction.prediction_time = TimeCurrent();

    return prediction;
}

//+------------------------------------------------------------------+
//| Simulate LSTM prediction (simplified for template)                |
//+------------------------------------------------------------------+
double Simulate_LSTM_Prediction(double rsi, double macd, double signal) {
    //--- Simplified LSTM simulation
    double lstm_score = 0;

    //--- RSI contribution
    if(rsi < RSI_Oversold_Threshold) {
        lstm_score += 0.3;
    } else if(rsi > RSI_Overbought_Threshold) {
        lstm_score -= 0.3;
    }

    //--- MACD contribution
    double macd_histogram = macd - signal;
    lstm_score += macd_histogram * 0.2;

    //--- Add some randomness to simulate neural network
    lstm_score += (MathRand() / 32767.0 - 0.5) * 0.1;

    return MathMax(-1.0, MathMin(1.0, lstm_score));
}

//+------------------------------------------------------------------+
//| Simulate XGBoost prediction (simplified)                         |
//+------------------------------------------------------------------+
double Simulate_XGBoost_Prediction(double rsi, double bb_upper, double bb_lower, double bb_middle) {
    double xgb_score = 0;

    //--- Bollinger Bands position
    double bb_position = (symbol_info.Bid() - bb_lower) / (bb_upper - bb_lower);
    if(bb_position < 0.2) {
        xgb_score += 0.4;
    } else if(bb_position > 0.8) {
        xgb_score -= 0.4;
    }

    //--- RSI divergence with price
    xgb_score += (50 - rsi) * 0.01;

    //--- Random forest component
    xgb_score += (MathRand() / 32767.0 - 0.5) * 0.15;

    return MathMax(-1.0, MathMin(1.0, xgb_score));
}

//+------------------------------------------------------------------+
//| Simulate Random Forest prediction (simplified)                    |
//+------------------------------------------------------------------+
double Simulate_RandomForest_Prediction(double trend, double volatility, double volume) {
    double rf_score = 0;

    //--- Trend strength contribution
    rf_score += trend * 0.5;

    //--- Volatility adjustment
    if(volatility < 0.2) {
        rf_score += 0.2;
    } else if(volatility > 0.5) {
        rf_score -= 0.3;
    }

    //--- Volume confirmation
    if(volume > 200) {
        rf_score *= 1.2;
    }

    //--- Random component
    rf_score += (MathRand() / 32767.0 - 0.5) * 0.2;

    return MathMax(-1.0, MathMin(1.0, rf_score));
}

//+------------------------------------------------------------------+
//| Risk management checks                                           |
//+------------------------------------------------------------------+
bool Risk_Management_Checks() {
    //--- Check maximum portfolio risk
    double current_risk = Calculate_Current_Portfolio_Risk();
    if(current_risk > Max_Portfolio_Risk) {
        Log_Message("⚠️ Maximum portfolio risk exceeded: " + DoubleToString(current_risk, 2) + "%", LOG_WARNING);
        return(false);
    }

    //--- Check maximum drawdown
    if(g_trade_metrics.current_drawdown > Max_Portfolio_Risk * 2) {
        Log_Message("⚠️ Maximum drawdown exceeded: " + DoubleToString(g_trade_metrics.current_drawdown, 2) + "%", LOG_WARNING);
        return(false);
    }

    //--- Check correlation risk
    if(Enable_Correlation_Filter && Has_High_Correlation_Risk()) {
        Log_Message("⚠️ High correlation risk detected", LOG_WARNING);
        return(false);
    }

    //--- Check news filter
    if(Avoid_High_Impact_News && Is_News_Time()) {
        Log_Message("⚠️ High-impact news time - trading paused", LOG_INFO);
        return(false);
    }

    return(true);
}

//+------------------------------------------------------------------+
//| Calculate current portfolio risk                                    |
//+------------------------------------------------------------------+
double Calculate_Current_Portfolio_Risk() {
    if(g_active_positions == 0) {
        return(0);
    }

    double total_risk = 0;

    for(int i = 0; i < PositionsTotal(); i++) {
        if(position.SelectByIndex(i)) {
            if(position.Symbol() == _Symbol && position.Magic() == MagicNumber_Base) {
                double position_risk = (position.Volume() * 100000 / account.Equity()) * 100;
                total_risk += position_risk;
            }
        }
    }

    return(total_risk);
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void Manage_Open_Positions(AI_Prediction &prediction) {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(!position.SelectByIndex(i)) {
            continue;
        }

        if(position.Symbol() != _Symbol || position.Magic() != MagicNumber_Base) {
            continue;
        }

        //--- Check for exit signals
        bool should_close = false;
        string close_reason = "";

        //--- AI signal change
        if((position.PositionType() == POSITION_TYPE_BUY && prediction.signal_type == "SELL") ||
           (position.PositionType() == POSITION_TYPE_SELL && prediction.signal_type == "BUY")) {
            should_close = true;
            close_reason = "AI Signal Reversal";
        }

        //--- Confidence drop
        if(prediction.confidence < AI_Confidence_Threshold * 0.5) {
            should_close = true;
            close_reason = "Low Confidence";
        }

        //--- Risk management
        double current_pnl = position.Profit() + position.Swap();
        double max_loss = -g_initial_balance * Max_Single_Trade_Risk / 100;

        if(current_pnl < max_loss) {
            should_close = true;
            close_reason = "Max Loss Reached";
        }

        //--- Close position if needed
        if(should_close) {
            Close_Position(position.Ticket(), close_reason);
        } else {
            //--- Update trailing stop if enabled
            Update_Trailing_Stop(position);
        }
    }
}

//+------------------------------------------------------------------+
//| Check if new position can be opened                                |
//+------------------------------------------------------------------+
bool Can_Open_New_Position() {
    //--- Check position limit
    if(g_active_positions >= Max_Concurrent_Positions) {
        return(false);
    }

    //--- Check account margin
    if(account.FreeMarginCheck(_Symbol, ORDER_TYPE_BUY, Base_Lot_Size) != false) {
        return(false);
    }

    return(true);
}

//+------------------------------------------------------------------+
//| Evaluate trading opportunity                                       |
//+------------------------------------------------------------------+
void Evaluate_Trading_Opportunity(AI_Prediction &prediction) {
    //--- Only trade on strong signals
    if(prediction.confidence < AI_Confidence_Threshold) {
        return;
    }

    //--- Calculate position size
    double lot_size = Calculate_Position_Size(prediction);

    if(lot_size <= 0) {
        return;
    }

    //--- Calculate stop loss and take profit
    double atr_values[1];
    double current_atr = 0;
    if(CopyBuffer(g_atr_handle, 0, 0, 1, atr_values) > 0) {
        current_atr = atr_values[0];
    }

    double stop_loss_points = current_atr * ATR_Multiplier_SL / _Point;
    double take_profit_points = current_atr * ATR_Multiplier_TP / _Point;

    //--- Execute trade based on signal
    if(prediction.signal_type == "BUY") {
        Execute_Buy_Trade(lot_size, stop_loss_points, take_profit_points, prediction);
    } else if(prediction.signal_type == "SELL") {
        Execute_Sell_Trade(lot_size, stop_loss_points, take_profit_points, prediction);
    }
}

//+------------------------------------------------------------------+
//| Calculate optimal position size                                     |
//+------------------------------------------------------------------+
double Calculate_Position_Size(AI_Prediction &prediction) {
    double lot_size = Base_Lot_Size;

    //--- Apply dynamic multiplier
    lot_size *= g_dynamic_lot_multiplier;
    lot_size *= Lot_Size_Multiplier;

    //--- Apply risk-based sizing
    if(Position_Sizing_Method == "Kelly_Criterion") {
        lot_size = Calculate_Kelly_Sizing(prediction);
    } else if(Position_Sizing_Method == "Vol_Target") {
        lot_size = Calculate_Volatility_Target_Sizing();
    } else if(Position_Sizing_Method == "Risk_Parity") {
        lot_size = Calculate_Risk_Parity_Sizing();
    }

    //--- Apply confidence adjustment
    lot_size *= prediction.confidence;

    //--- Ensure within limits
    lot_size = MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN),
                         MathMin(lot_size, Max_Lot_Size));

    //--- Round to valid lot size
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    lot_size = MathRound(lot_size / lot_step) * lot_step;

    return(lot_size);
}

//+------------------------------------------------------------------+
//| Execute buy trade                                                  |
//+------------------------------------------------------------------+
void Execute_Buy_Trade(double lot_size, double sl_points, double tp_points, AI_Prediction &prediction) {
    double ask = symbol_info.Ask();
    double stop_loss = ask - sl_points * _Point;
    double take_profit = ask + tp_points * _Point;

    string comment = StringFormat("AI_Buy_CF=%.2f_Sig=%s",
                               prediction.confidence, prediction.signal_type);

    if(trade.Buy(lot_size, _Symbol, ask, stop_loss, take_profit, comment)) {
        Log_Message(StringFormat("🟢 BUY EXECUTED: %.2f lots @ %.5f, SL=%.5f, TP=%.5f, CF=%.2f",
                                lot_size, ask, stop_loss, take_profit, prediction.confidence), LOG_INFO);
        g_active_positions++;
    } else {
        Log_Message("❌ Failed to execute BUY order", LOG_ERROR);
    }
}

//+------------------------------------------------------------------+
//| Execute sell trade                                                 |
//+------------------------------------------------------------------+
void Execute_Sell_Trade(double lot_size, double sl_points, double tp_points, AI_Prediction &prediction) {
    double bid = symbol_info.Bid();
    double stop_loss = bid + sl_points * _Point;
    double take_profit = bid - tp_points * _Point;

    string comment = StringFormat("AI_Sell_CF=%.2f_Sig=%s",
                               prediction.confidence, prediction.signal_type);

    if(trade.Sell(lot_size, _Symbol, bid, stop_loss, take_profit, comment)) {
        Log_Message(StringFormat("🔴 SELL EXECUTED: %.2f lots @ %.5f, SL=%.5f, TP=%.5f, CF=%.2f",
                                lot_size, bid, stop_loss, take_profit, prediction.confidence), LOG_INFO);
        g_active_positions++;
    } else {
        Log_Message("❌ Failed to execute SELL order", LOG_ERROR);
    }
}

//+------------------------------------------------------------------+
//| Utility functions                                                |
//+------------------------------------------------------------------+

//--- Logging levels
enum ENUM_LOG_LEVEL {
    LOG_INFO = 0,
    LOG_WARNING = 1,
    LOG_ERROR = 2
};

//--- Logging functions
void Log_Message(string message, ENUM_LOG_LEVEL level = LOG_INFO) {
    if(!Enable_Dashboard_Logging) {
        return;
    }

    string prefix = "";
    switch(level) {
        case LOG_INFO:    prefix = "ℹ️ "; break;
        case LOG_WARNING: prefix = "⚠️ "; break;
        case LOG_ERROR:   prefix = "❌ "; break;
    }

    string timestamp = TimeToString(TimeCurrent());
    string log_entry = prefix + "[" + timestamp + "] " + message;

    //--- Print to experts tab
    Print(log_entry);

    //--- Write to log file
    Write_Log_Entry(log_entry + "\n");
}

void Write_Log_Entry(string entry) {
    string log_file = "EA_Optimizer_Enterprise_" + _Symbol + "_" +
                      TimeToString(TimeCurrent(), TIME_DATE) + ".log";

    int file_handle = FileOpen(log_file, FILE_READ|FILE_WRITE|FILE_TXT);
    if(file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle, entry);
        FileClose(file_handle);
    }
}

void Log_Configuration_Summary() {
    Log_Message("📋 Configuration Summary:", LOG_INFO);
    Log_Message(StringFormat("   Max Portfolio Risk: %.1f%%", Max_Portfolio_Risk), LOG_INFO);
    Log_Message(StringFormat("   Position Sizing: %s", Position_Sizing_Method), LOG_INFO);
    Log_Message(StringFormat("   AI Confidence Threshold: %.2f", AI_Confidence_Threshold), LOG_INFO);
    Log_Message(StringFormat("   Max Concurrent Positions: %d", Max_Concurrent_Positions), LOG_INFO);
    Log_Message(StringFormat("   Base Lot Size: %.2f", Base_Lot_Size), LOG_INFO);
}

bool Is_Market_Open() {
    //--- Simple market open check (can be enhanced with holiday calendar)
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);

    //--- Weekend check
    if(time.day_of_week == 0 || time.day_of_week == 6) {
        return(false);
    }

    return(true);
}

bool Is_Trading_Session_Allowed() {
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);
    int hour = time.hour;

    //--- Check Asian session
    if(Asian_Session && hour >= Asian_Start_Hour && hour < Asian_End_Hour) {
        return(true);
    }

    //--- Check European session
    if(European_Session && hour >= European_Start_Hour && hour < European_End_Hour) {
        return(true);
    }

    //--- Check US session
    if(US_Session && hour >= US_Start_Hour && hour < US_End_Hour) {
        return(true);
    }

    return(false);
}

bool Is_Spread_Acceptable() {
    double current_spread = symbol_info.Ask() - symbol_info.Bid();
    return(current_spread <= Max_Spread_Points * _Point);
}

bool Is_News_Time() {
    //--- Simplified news time check (can be enhanced with economic calendar)
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);

    //--- Avoid major news release times (simplified)
    if(time.hour == 13 && time.minute <= 30) {  // US CPI/FOMC (example)
        return(true);
    }

    return(false);
}

bool Has_High_Correlation_Risk() {
    //--- Simplified correlation check
    return(false); // Can be enhanced with actual correlation analysis
}

void Update_Dynamic_Lot_Multiplier() {
    //--- Adjust lot size based on recent performance
    if(g_trade_metrics.win_rate > 60 && g_trade_metrics.current_drawdown < 5) {
        g_dynamic_lot_multiplier = MathMin(1.5, g_dynamic_lot_multiplier * 1.01);
    } else if(g_trade_metrics.current_drawdown > 15 || g_trade_metrics.win_rate < 40) {
        g_dynamic_lot_multiplier = MathMax(0.5, g_dynamic_lot_multiplier * 0.99);
    }
}

void Update_Ensemble_Weights() {
    //--- Dynamic ensemble rebalancing based on recent performance
    //--- This would be implemented with actual model performance tracking
}

void Update_Trailing_Stop(CPositionInfo &position) {
    //--- Implement trailing stop logic here
    //--- This would use the ATR-based dynamic trailing stops
}

void Close_Position(ulong ticket, string reason) {
    if(position.SelectByTicket(ticket)) {
        if(trade.PositionClose(ticket)) {
            Log_Message(StringFormat("📊 Position #%d CLOSED: %s, PnL=$%.2f",
                                    ticket, reason, position.Profit() + position.Swap()), LOG_INFO);
            g_active_positions--;
        }
    }
}

void Update_Performance_Monitoring() {
    //--- Update trade metrics
    Update_Trade_Metrics();

    //--- Check for alerts
    Check_Performance_Alerts();
}

void Update_Trade_Metrics() {
    //--- Calculate comprehensive trade metrics
    //--- This would analyze all closed trades and update statistics
}

void Check_Performance_Alerts() {
    //--- Drawdown alerts
    if(g_trade_metrics.current_drawdown > Max_Portfolio_Risk * 1.5) {
        Log_Message(StringFormat("🚨 HIGH DRAWDOWN ALERT: %.2f%%", g_trade_metrics.current_drawdown), LOG_ERROR);
    }

    //--- Performance alerts
    if(g_trade_metrics.win_rate < 30 && g_trade_metrics.total_trades > 20) {
        Log_Message(StringFormat("⚠️ LOW WIN RATE ALERT: %.1f%%", g_trade_metrics.win_rate), LOG_WARNING);
    }
}

void Generate_Performance_Report() {
    //--- Generate comprehensive performance report
    string report = Generate_Performance_Report_Text();

    //--- Save report to file
    string report_file = "Performance_Report_" + _Symbol + "_" +
                        TimeToString(TimeCurrent(), TIME_DATE) + ".html";

    int file_handle = FileOpen(report_file, FILE_WRITE|FILE_TXT|FILE_ANSI);
    if(file_handle != INVALID_HANDLE) {
        FileWrite(file_handle, report);
        FileClose(file_handle);

        Log_Message("📄 Performance report generated: " + report_file, LOG_INFO);
    }
}

string Generate_Performance_Report_Text() {
    //--- Generate detailed HTML performance report
    //--- This would include comprehensive statistics, charts, and analysis

    string report = "<html><body>";
    report += "<h2>EA Optimizer AI - Enterprise Performance Report</h2>";
    report += "<p>Symbol: " + _Symbol + "</p>";
    report += "<p>Initial Balance: $" + DoubleToString(g_initial_balance, 2) + "</p>";
    report += "<p>Final Balance: $" + DoubleToString(g_current_equity, 2) + "</p>";
    report += "<p>Total Return: " + DoubleToString(((g_current_equity - g_initial_balance) / g_initial_balance) * 100, 2) + "%</p>";
    report += "<p>Max Drawdown: " + DoubleToString(g_trade_metrics.max_drawdown, 2) + "%</p>";
    report += "</body></html>";

    return(report);
}

void Cleanup_Resources() {
    //--- Release indicator handles
    if(g_fast_ma_handle != INVALID_HANDLE) IndicatorRelease(g_fast_ma_handle);
    if(g_slow_ma_handle != INVALID_HANDLE) IndicatorRelease(g_slow_ma_handle);
    if(g_rsi_handle != INVALID_HANDLE) IndicatorRelease(g_rsi_handle);
    if(g_macd_handle != INVALID_HANDLE) IndicatorRelease(g_macd_handle);
    if(g_bollinger_handle != INVALID_HANDLE) IndicatorRelease(g_bollinger_handle);
    if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

//--- Position sizing calculation functions
double Calculate_Kelly_Sizing(AI_Prediction &prediction) {
    //--- Kelly Criterion: f* = (bp - q) / b
    //--- Where: b = odds, p = win probability, q = lose probability
    double win_prob = prediction.confidence;
    double lose_prob = 1 - win_prob;
    double odds = 2.0; // Assumed 2:1 reward ratio

    double kelly_fraction = (odds * win_prob - lose_prob) / odds;

    //--- Conservative Kelly (half Kelly)
    kelly_fraction *= 0.5;

    //--- Apply to base lot size
    return(Base_Lot_Size * (1 + kelly_fraction));
}

double Calculate_Volatility_Target_Sizing() {
    //--- Position sizing based on volatility targeting
    double target_vol = Vol_Target_Percent / 100;
    double current_vol = g_market_conditions.volatility;

    if(current_vol > 0) {
        return(Base_Lot_Size * (target_vol / current_vol));
    }

    return(Base_Lot_Size);
}

double Calculate_Risk_Parity_Sizing() {
    //--- Risk parity position sizing
    double risk_per_trade = Max_Single_Trade_Risk / 100;
    double account_risk = g_current_equity;

    double position_value = account_risk * risk_per_trade;
    double lot_size = position_value / 100000; // $100k per lot

    return(lot_size);
}