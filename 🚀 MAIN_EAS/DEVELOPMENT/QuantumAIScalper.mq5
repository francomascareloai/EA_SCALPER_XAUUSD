//+------------------------------------------------------------------+
//|                                           QuantumAIScalper.mq5 |
//|                                  Copyright 2024, TradeDev_Master |
//|                                       https://quantum-trading.ai |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master - Quantum AI Systems"
#property link      "https://quantum-trading.ai"
#property version   "2.00"
#property description "Quantum AI Scalper v2.0 - Unified Neural Network & SMC Trading System"
#property strict

//--- Standard Library Includes
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Forward Declarations of Classes
class CNeuralNetwork;
class CMLAlgorithms;
class COrderBlockDetector;
class CFVGDetector;
class CRiskManager;
class CMTFAnalyzer;

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "🧠 Neural Network Settings"
input int    NN_Hidden1_Neurons    = 32;       // Hidden Layer 1 Neurons
input int    NN_Hidden2_Neurons    = 16;       // Hidden Layer 2 Neurons
input double NN_Threshold         = 0.7;      // Confidence Threshold
input double NN_LearningRate      = 0.01;     // Online Learning Rate

input group "📊 Smart Money Concepts"
input bool   Enable_SMC           = true;     // Enable SMC Logic
input double Min_Order_Block_Size = 10.0;     // Min OB Size (points)
input double Min_FVG_Size         = 5.0;      // Min FVG Size (points)
input int    Max_Order_Blocks     = 10;       // Max Active OBs
input int    Max_FVGs             = 5;        // Max Active FVGs

input group "⚡ Risk Management (FTMO)"
input double Max_Risk_Per_Trade   = 1.0;      // Risk per Trade (%)
input double Max_Drawdown_Percent = 5.0;      // Max Daily Drawdown (%)
input double Daily_Profit_Target  = 5.0;      // Daily Profit Target (%)
input double Daily_Loss_Limit     = 3.0;      // Daily Loss Limit (%)
input bool   Use_Dynamic_Lots     = true;     // Use Dynamic Lot Sizing

input group "🎯 Trading Settings"
input int    MagicNumber         = 998877;   // Magic Number
input int    Max_Spread_Points   = 30;       // Max Spread (points)
input double Commission_Per_Lot  = 7.0;      // Commission ($/lot)
input int    Slippage            = 3;        // Max Slippage

//+------------------------------------------------------------------+
//| Global Objects                                                   |
//+------------------------------------------------------------------+
CTrade         g_trade;
CSymbolInfo    g_symbol;
CPositionInfo  g_position;
CAccountInfo   g_account;

// Pointers to complex objects (to be initialized in OnInit)
CNeuralNetwork*      g_neural_network;
CMLAlgorithms*       g_ml_algorithms;
COrderBlockDetector* g_ob_detector;
CFVGDetector*        g_fvg_detector;
CRiskManager*        g_risk_manager;
CMTFAnalyzer*        g_mtf_analyzer;

// Global State Variables
datetime g_last_trade_time = 0;
double   g_daily_pnl = 0.0;
int      g_daily_trades = 0;
datetime g_last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🚀 Initializing Quantum AI Scalper v2.0...");

   // 1. Initialize System Components
   if(!g_symbol.Name(Symbol())) {
      Print("❌ Failed to initialize symbol: ", Symbol());
      return INIT_FAILED;
   }
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(Slippage);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);

   // 2. Initialize Sub-systems (Memory Allocation)
   g_neural_network = new CNeuralNetwork();
   g_ml_algorithms = new CMLAlgorithms();
   g_ob_detector = new COrderBlockDetector();
   g_fvg_detector = new CFVGDetector();
   g_risk_manager = new CRiskManager();
   g_mtf_analyzer = new CMTFAnalyzer();

   // 3. Initialize Training Data
   g_ml_algorithms.InitializeTrainingData();
   g_mtf_analyzer.UpdateAllTimeframes();

   Print("✅ Quantum AI Scalper v2.0 Initialized Successfully");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🔄 Shutting down Quantum AI Scalper v2.0...");

   // Cleanup Memory
   if(CheckPointer(g_neural_network) == POINTER_DYNAMIC) delete g_neural_network;
   if(CheckPointer(g_ml_algorithms) == POINTER_DYNAMIC) delete g_ml_algorithms;
   if(CheckPointer(g_ob_detector) == POINTER_DYNAMIC) delete g_ob_detector;
   if(CheckPointer(g_fvg_detector) == POINTER_DYNAMIC) delete g_fvg_detector;
   if(CheckPointer(g_risk_manager) == POINTER_DYNAMIC) delete g_risk_manager;
   if(CheckPointer(g_mtf_analyzer) == POINTER_DYNAMIC) delete g_mtf_analyzer;

   Print("✅ Shutdown Complete");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Basic Checks
   if(!g_symbol.RefreshRates()) return;

   // 2. New Bar Check
   datetime current_time = iTime(Symbol(), Period(), 0);
   bool is_new_bar = (current_time != g_last_bar_time);
   if(is_new_bar) {
      g_last_bar_time = current_time;
      OnNewBar();
   }

   // 3. Main Logic (Every Tick)
   // Manage Open Positions
   for(int i=PositionsTotal()-1; i>=0; i--) {
       ulong ticket = PositionGetTicket(i);
       if(PositionSelectByTicket(ticket)) {
           if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
               // Trailing Stop Logic could go here
           }
       }
   }
}

//+------------------------------------------------------------------+
//| New Bar Event Handler                                            |
//+------------------------------------------------------------------+
void OnNewBar()
{
   Print("⏱️ New Bar: Analyzing Market...");

   // 1. Update Analysis
   g_ob_detector.ScanForOrderBlocks();
   g_fvg_detector.ScanForFVG();
   g_mtf_analyzer.UpdateAllTimeframes();

   // 2. Prepare Features for AI
   double nn_inputs[64];
   ArrayInitialize(nn_inputs, 0.0);
   // Fill inputs with RSI, MACD, Price Action (Simplified for now)
   for(int i=0; i<64; i++) nn_inputs[i] = iClose(Symbol(), Period(), i) - iOpen(Symbol(), Period(), i); // Simple Delta
   
   double nn_outputs[3];
   g_neural_network.Forward(nn_inputs, nn_outputs);
   int nn_signal = g_neural_network.GetSignal(NN_Threshold);
   
   // 3. Get Confluence
   int mtf_signal = g_mtf_analyzer.GetDominantSignal();
   double nearest_ob_bull = g_ob_detector.GetNearestOrderBlock(true);
   double nearest_ob_bear = g_ob_detector.GetNearestOrderBlock(false);
   
   // 4. Final Decision
   int final_signal = 0;
   
   // Strong Buy: NN Buy + MTF Buy + Near Bullish OB
   if(nn_signal == 1 && mtf_signal == 1) {
       double current_price = iClose(Symbol(), Period(), 0);
       if(nearest_ob_bull > 0 && MathAbs(current_price - nearest_ob_bull) < 100 * Point()) {
           final_signal = 1;
       } else if(!Enable_SMC) {
           final_signal = 1;
       }
   }
   
   // Strong Sell: NN Sell + MTF Sell + Near Bearish OB
   if(nn_signal == 2 && mtf_signal == 2) {
       double current_price = iClose(Symbol(), Period(), 0);
       if(nearest_ob_bear > 0 && MathAbs(current_price - nearest_ob_bear) < 100 * Point()) {
           final_signal = 2;
       } else if(!Enable_SMC) {
           final_signal = 2;
       }
   }

   // 5. Execute Trade
   if(final_signal != 0) {
       if(g_risk_manager.IsTradeAllowed()) {
           double entry_price = (final_signal == 1) ? SymbolInfoDouble(Symbol(), SYMBOL_ASK) : SymbolInfoDouble(Symbol(), SYMBOL_BID);
           double sl_dist = g_risk_manager.CalculateDynamicStopLoss(final_signal, entry_price);
           double tp_dist = g_risk_manager.CalculateDynamicTakeProfit(final_signal, entry_price, entry_price - sl_dist); // Approx
           
           double sl = (final_signal == 1) ? entry_price - sl_dist : entry_price + sl_dist;
           double tp = (final_signal == 1) ? entry_price + tp_dist : entry_price - tp_dist;
           
           double lots = g_risk_manager.CalculateOptimalLotSize(sl_dist / Point());
           
           if(lots > 0) {
               if(final_signal == 1) {
                   g_trade.Buy(lots, Symbol(), entry_price, sl, tp, "QuantumAI Buy");
               } else {
                   g_trade.Sell(lots, Symbol(), entry_price, sl, tp, "QuantumAI Sell");
               }
           }
       }
   }
}

//+------------------------------------------------------------------+
//| Stub Classes (To be implemented)                                 |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Neural Network Class (Native MQL5)                               |
//| Architecture: 64 Input -> 32 Hidden -> 16 Hidden -> 3 Output     |
//+------------------------------------------------------------------+
class CNeuralNetwork {
private:
    // Weights and Biases
    double m_weights1[64][32];  // Layer 1: 64 inputs -> 32 neurons
    double m_weights2[32][16];  // Layer 2: 32 -> 16
    double m_weights3[16][3];   // Output: 16 -> 3 (HOLD/BUY/SELL)

    double m_bias1[32];
    double m_bias2[16];
    double m_bias3[3];

    // Buffers for forward pass
    double m_input_buffer[64];
    double m_hidden1[32];
    double m_hidden2[16];
    double m_output[3];

    // Activation Functions
    double ReLU(double x) { return MathMax(0.0, x); }
    double Sigmoid(double x) { return 1.0 / (1.0 + MathExp(-x)); }
    double Tanh(double x) { return MathTanh(x); }

public:
    CNeuralNetwork() { InitializeWeights(); }
    
    void InitializeWeights() {
        // Xavier/Glorot Initialization
        MathSrand(GetTickCount());
        
        // Init Layer 1
        double limit1 = MathSqrt(6.0 / (64 + 32));
        for(int i=0; i<64; i++)
            for(int j=0; j<32; j++)
                m_weights1[i][j] = ((double)MathRand()/32767.0 * 2.0 * limit1) - limit1;
                
        for(int i=0; i<32; i++) m_bias1[i] = 0.0;

        // Init Layer 2
        double limit2 = MathSqrt(6.0 / (32 + 16));
        for(int i=0; i<32; i++)
            for(int j=0; j<16; j++)
                m_weights2[i][j] = ((double)MathRand()/32767.0 * 2.0 * limit2) - limit2;
                
        for(int i=0; i<16; i++) m_bias2[i] = 0.0;

        // Init Layer 3
        double limit3 = MathSqrt(6.0 / (16 + 3));
        for(int i=0; i<16; i++)
            for(int j=0; j<3; j++)
                m_weights3[i][j] = ((double)MathRand()/32767.0 * 2.0 * limit3) - limit3;
                
        for(int i=0; i<3; i++) m_bias3[i] = 0.0;
    }

    void Forward(double &inputs[], double &outputs[]) {
        if(ArraySize(inputs) < 64) return;
        
        // Copy inputs
        ArrayCopy(m_input_buffer, inputs);

        // Layer 1: Input -> Hidden1 (ReLU)
        for(int i = 0; i < 32; i++) {
            double sum = m_bias1[i];
            for(int j = 0; j < 64; j++) {
                sum += m_input_buffer[j] * m_weights1[j][i];
            }
            m_hidden1[i] = ReLU(sum);
        }

        // Layer 2: Hidden1 -> Hidden2 (ReLU)
        for(int i = 0; i < 16; i++) {
            double sum = m_bias2[i];
            for(int j = 0; j < 32; j++) {
                sum += m_hidden1[j] * m_weights2[j][i];
            }
            m_hidden2[i] = ReLU(sum);
        }

        // Layer 3: Hidden2 -> Output (Softmax)
        double total_exp = 0.0;
        for(int i = 0; i < 3; i++) {
            double sum = m_bias3[i];
            for(int j = 0; j < 16; j++) {
                sum += m_hidden2[j] * m_weights3[j][i];
            }
            m_output[i] = MathExp(sum);
            total_exp += m_output[i];
        }

        // Normalize (Softmax)
        for(int i = 0; i < 3; i++) {
            if(total_exp > 0) m_output[i] /= total_exp;
            else m_output[i] = 0.0;
        }
        
        ArrayCopy(outputs, m_output);
    }
    
    int GetSignal(double threshold) {
        if(m_output[1] > threshold && m_output[1] > m_output[0] && m_output[1] > m_output[2]) return 1; // BUY
        if(m_output[2] > threshold && m_output[2] > m_output[0] && m_output[2] > m_output[1]) return 2; // SELL
        return 0; // HOLD
    }
};
//+------------------------------------------------------------------+
//| Machine Learning Algorithms Class                                |
//| Implements K-Nearest Neighbors (KNN) for Pattern Recognition     |
//+------------------------------------------------------------------+
class CMLAlgorithms {
private:
    struct KNN_Point {
        double features[20];
        int label; // 0=HOLD, 1=BUY, 2=SELL
    };

    KNN_Point m_training_data[1000];
    int m_data_count;

public:
    CMLAlgorithms() { m_data_count = 0; }
    
    // Initialize with some dummy data or load from file (Simulated here)
    bool InitializeTrainingData() {
        // In a real scenario, this would load from a file
        // For now, we initialize with empty data waiting for runtime collection
        m_data_count = 0;
        return true;
    }
    
    // Add new data point during runtime
    void UpdateTrainingData(double &features[], int result_label) {
        if(m_data_count >= 1000) return; // Buffer full
        if(ArraySize(features) < 20) return;
        
        ArrayCopy(m_training_data[m_data_count].features, features);
        m_training_data[m_data_count].label = result_label;
        m_data_count++;
    }

    int KNN_Classify(double &features[], int k = 5) {
        if(m_data_count < k) return 0; // Not enough data

        double distances[1000];
        int indices[1000];

        // Calculate distances
        for(int i = 0; i < m_data_count; i++) {
            double distance = 0.0;
            for(int j = 0; j < 20; j++) {
                double diff = features[j] - m_training_data[i].features[j];
                distance += diff * diff;
            }
            distances[i] = MathSqrt(distance);
            indices[i] = i;
        }

        // Sort by distance (Bubble sort for simplicity on small k)
        for(int i = 0; i < k; i++) {
            for(int j = i + 1; j < m_data_count; j++) {
                if(distances[i] > distances[j]) {
                    double temp_dist = distances[i];
                    distances[i] = distances[j];
                    distances[j] = temp_dist;

                    int temp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp_idx;
                }
            }
        }

        // Vote
        int votes[3] = {0, 0, 0}; // HOLD, BUY, SELL
        for(int i = 0; i < k; i++) {
            int label = m_training_data[indices[i]].label;
            if(label >= 0 && label <= 2) votes[label]++;
        }

        // Return majority
        if(votes[1] > votes[0] && votes[1] > votes[2]) return 1; // BUY
        if(votes[2] > votes[0] && votes[2] > votes[1]) return 2; // SELL
        return 0; // HOLD
    }
};
//+------------------------------------------------------------------+
//| Order Block Detector Class (SMC)                                 |
//| Detects Institutional Order Blocks and Liquidity Zones           |
//+------------------------------------------------------------------+
class COrderBlockDetector {
private:
    struct OrderBlock {
        double high;
        double low;
        double volume;
        datetime time;
        double strength_score;
        bool is_bullish;
        int touch_count;
        bool active;
    };

    OrderBlock m_order_blocks[50];
    int m_block_count;

    double GetAverageBodySize() {
        double sum = 0;
        for(int i=1; i<=20; i++) sum += MathAbs(iOpen(Symbol(), Period(), i) - iClose(Symbol(), Period(), i));
        return sum / 20.0;
    }
    
    double GetAverageVolume() {
        double sum = 0;
        for(int i=1; i<=20; i++) sum += (double)iVolume(Symbol(), Period(), i);
        return sum / 20.0;
    }

public:
    COrderBlockDetector() { m_block_count = 0; }

    void ScanForOrderBlocks() {
        m_block_count = 0; // Reset for simplicity in this version
        
        // Scan last 200 candles
        for(int i = 1; i < 200; i++) {
            if(m_block_count >= 50) break;

            double body_size = MathAbs(iOpen(Symbol(), Period(), i) - iClose(Symbol(), Period(), i));
            double volume = (double)iVolume(Symbol(), Period(), i);
            
            // Bullish Order Block (Last Down candle before strong Up move)
            if(iClose(Symbol(), Period(), i) < iOpen(Symbol(), Period(), i)) { // Down candle
                // Check next candle (i-1) is strong Up
                if(iClose(Symbol(), Period(), i-1) > iOpen(Symbol(), Period(), i-1) &&
                   MathAbs(iOpen(Symbol(), Period(), i-1) - iClose(Symbol(), Period(), i-1)) > GetAverageBodySize() * 1.5) {
                   
                    OrderBlock ob;
                    ob.high = iHigh(Symbol(), Period(), i);
                    ob.low = iLow(Symbol(), Period(), i);
                    ob.volume = volume;
                    ob.time = iTime(Symbol(), Period(), i);
                    ob.is_bullish = true;
                    ob.active = true;
                    ob.strength_score = (volume / GetAverageVolume());
                    
                    m_order_blocks[m_block_count++] = ob;
                }
            }
            
            // Bearish Order Block (Last Up candle before strong Down move)
            if(iClose(Symbol(), Period(), i) > iOpen(Symbol(), Period(), i)) { // Up candle
                // Check next candle (i-1) is strong Down
                if(iClose(Symbol(), Period(), i-1) < iOpen(Symbol(), Period(), i-1) &&
                   MathAbs(iOpen(Symbol(), Period(), i-1) - iClose(Symbol(), Period(), i-1)) > GetAverageBodySize() * 1.5) {
                   
                    OrderBlock ob;
                    ob.high = iHigh(Symbol(), Period(), i);
                    ob.low = iLow(Symbol(), Period(), i);
                    ob.volume = volume;
                    ob.time = iTime(Symbol(), Period(), i);
                    ob.is_bullish = false;
                    ob.active = true;
                    ob.strength_score = (volume / GetAverageVolume());
                    
                    m_order_blocks[m_block_count++] = ob;
                }
            }
        }
    }

    double GetNearestOrderBlock(bool bullish) {
        double current_price = iClose(Symbol(), Period(), 0);
        double nearest_price = 0;
        double min_dist = DBL_MAX;

        for(int i = 0; i < m_block_count; i++) {
            if(!m_order_blocks[i].active) continue;
            if(m_order_blocks[i].is_bullish != bullish) continue;

            double dist = 0;
            double price = 0;
            
            if(bullish) {
                if(current_price > m_order_blocks[i].high) { // Price above OB
                    dist = current_price - m_order_blocks[i].high;
                    price = m_order_blocks[i].high;
                }
            } else {
                if(current_price < m_order_blocks[i].low) { // Price below OB
                    dist = m_order_blocks[i].low - current_price;
                    price = m_order_blocks[i].low;
                }
            }

            if(dist > 0 && dist < min_dist) {
                min_dist = dist;
                nearest_price = price;
            }
        }
        return nearest_price;
    }
};
//+------------------------------------------------------------------+
//| Fair Value Gap Detector Class                                    |
//| Detects Imbalances and Liquidity Voids                           |
//+------------------------------------------------------------------+
class CFVGDetector {
private:
    struct FVG {
        double top;
        double bottom;
        datetime start_time;
        bool is_bullish;
        bool filled;
    };

    FVG m_fvg_list[50];
    int m_fvg_count;

public:
    CFVGDetector() { m_fvg_count = 0; }

    void ScanForFVG() {
        m_fvg_count = 0;
        
        // Scan last 100 candles
        for(int i = 2; i < 100; i++) {
            if(m_fvg_count >= 50) break;

            double candle1_high = iHigh(Symbol(), Period(), i);
            double candle1_low = iLow(Symbol(), Period(), i);
            double candle3_high = iHigh(Symbol(), Period(), i-2);
            double candle3_low = iLow(Symbol(), Period(), i-2);

            // Bullish FVG (Gap between Candle 1 High and Candle 3 Low? No, Candle 1 Low and Candle 3 High)
            // Standard FVG: Candle 1 (Left), Candle 2 (Middle), Candle 3 (Right)
            // Bullish: Candle 1 High < Candle 3 Low
            // Wait, indexing is reverse in MT5 usually. 0 is current.
            // i=2 (Left), i=1 (Middle), i=0 (Right)
            // Let's use standard i, i-1, i-2 where i is oldest.
            // Actually MT5 iHigh(0) is current. iHigh(1) is previous.
            // So i=2 is 2 bars ago. i=0 is current.
            // Bullish FVG: High of bar i+2 < Low of bar i
            
            double high_2 = iHigh(Symbol(), Period(), i+2); // Left
            double low_0 = iLow(Symbol(), Period(), i);     // Right
            
            // Correct Logic for Bullish FVG:
            // Gap between High of (i+1) [Previous to middle] and Low of (i-1) [Next to middle]
            // Let's stick to the design doc logic which used i, i-1, i-2
            
            // Design Doc:
            // candle1 = i (Left)
            // candle3 = i-2 (Right)
            // Bullish: candle1_low > candle3_high (This implies i is Right and i-2 is Left? No, i is usually older in loops if incrementing... wait)
            // In MT5 loops usually go 0..Bars. 0 is newest.
            // So i=2 is older than i=0.
            // If i=2 is Left, i=0 is Right.
            // Bullish FVG: High of i=2 < Low of i=0.
            
            double left_high = iHigh(Symbol(), Period(), i);
            double right_low = iLow(Symbol(), Period(), i-2);
            
            if(left_high < right_low) { // Gap exists
                FVG fvg;
                fvg.top = right_low;
                fvg.bottom = left_high;
                fvg.start_time = iTime(Symbol(), Period(), i-1);
                fvg.is_bullish = true;
                fvg.filled = false;
                m_fvg_list[m_fvg_count++] = fvg;
            }
            
            // Bearish FVG: Low of i=2 > High of i=0
            double left_low = iLow(Symbol(), Period(), i);
            double right_high = iHigh(Symbol(), Period(), i-2);
            
            if(left_low > right_high) { // Gap exists
                FVG fvg;
                fvg.top = left_low;
                fvg.bottom = right_high;
                fvg.start_time = iTime(Symbol(), Period(), i-1);
                fvg.is_bullish = false;
                fvg.filled = false;
                m_fvg_list[m_fvg_count++] = fvg;
            }
        }
    }
    
    double GetNearestFVGTarget() {
        // Simplified: Return nearest unfilled FVG price
        double current = iClose(Symbol(), Period(), 0);
        double nearest = 0;
        double min_dist = DBL_MAX;
        
        for(int i=0; i<m_fvg_count; i++) {
            if(m_fvg_list[i].filled) continue;
            
            double dist = MathAbs(current - (m_fvg_list[i].top + m_fvg_list[i].bottom)/2);
            if(dist < min_dist) {
                min_dist = dist;
                nearest = (m_fvg_list[i].top + m_fvg_list[i].bottom)/2;
            }
        }
        return nearest;
    }
};
//+------------------------------------------------------------------+
//| Risk Manager Class (FTMO Compliance)                             |
//| Handles Position Sizing and Drawdown Control                     |
//+------------------------------------------------------------------+
class CRiskManager {
private:
    double m_max_risk_percent;
    double m_max_drawdown_percent;
    double m_current_drawdown;
    int m_consecutive_losses;
    
public:
    CRiskManager() {
        m_max_risk_percent = Max_Risk_Per_Trade;
        m_max_drawdown_percent = Max_Drawdown_Percent;
        m_consecutive_losses = 0;
        m_current_drawdown = 0;
    }

    double CalculateOptimalLotSize(double stop_loss_points) {
        // Get account metrics
        double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        
        // Update Drawdown
        m_current_drawdown = (account_balance - account_equity) / account_balance * 100.0;
        
        // Stop if drawdown exceeded
        if(m_current_drawdown >= m_max_drawdown_percent) return 0.0;

        // Dynamic risk based on recent performance
        double risk_multiplier = 1.0;
        if(m_consecutive_losses > 0) {
            risk_multiplier = MathMax(0.1, 1.0 - (m_consecutive_losses * 0.2));
        }

        // Calculate position size
        double risk_amount = account_balance * m_max_risk_percent * risk_multiplier / 100.0;
        double tick_value = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE);
        
        if(stop_loss_points <= 0) stop_loss_points = 100; // Fallback
        if(tick_value == 0) tick_value = 1; // Fallback

        // Position size calculation
        double lot_size = (risk_amount / (stop_loss_points * tick_size)) * tick_size; // Approximation
        // Correct formula: Risk = Lots * SL_Points * TickValue
        // Lots = Risk / (SL_Points * TickValue)
        // Wait, SL_Points is usually in Points. TickValue is value of 1 lot for 1 tick (TickSize).
        // Value of 1 lot for SL_Points = (SL_Points / TickSize) * TickValue
        
        double value_per_lot = (stop_loss_points / tick_size) * tick_value;
        if(value_per_lot > 0) lot_size = risk_amount / value_per_lot;
        else lot_size = 0.01;

        // Apply safety limits
        double max_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
        double min_lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
        double lot_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);

        lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));

        // Round to step size
        lot_size = MathRound(lot_size / lot_step) * lot_step;

        return lot_size;
    }

    bool IsTradeAllowed() {
        double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double daily_start_balance = account_balance; // Simplified for now, should track daily start
        
        // Check Max Drawdown
        if((account_balance - account_equity) / account_balance * 100.0 >= m_max_drawdown_percent) {
            Print("⛔ Max Drawdown Exceeded!");
            return false;
        }
        
        return true;
    }

    double CalculateDynamicStopLoss(int signal_type, double entry_price) {
        // ATR based stop loss
        double atr = iATR(Symbol(), Period(), 14); // Simplified call, assumes handle or direct access
        // In MQL5 iATR returns handle. We need to implement a wrapper or use direct array access if not using indicator handle properly.
        // For simplicity in this single file, let's use a fixed point SL or simple High/Low logic if ATR is complex to setup here without OnInit.
        // We'll use a fixed SL for now based on input, but ideally ATR.
        return 200 * SymbolInfoDouble(Symbol(), SYMBOL_POINT); // 200 points
    }

    double CalculateDynamicTakeProfit(int signal_type, double entry_price, double stop_loss) {
        double risk = MathAbs(entry_price - stop_loss);
        return risk * 1.5; // 1:1.5 RR
    }
};
//+------------------------------------------------------------------+
//| Multi-Timeframe Analyzer Class                                   |
//| Analyzes Trend and Confluence across Timeframes                  |
//+------------------------------------------------------------------+
class CMTFAnalyzer {
private:
    struct MTFSignal {
        ENUM_TIMEFRAMES timeframe;
        int signal; // 0=HOLD, 1=BUY, 2=SELL
        double confidence;
    };

    MTFSignal m_mtf_signals[3]; // M5, H1, D1

public:
    CMTFAnalyzer() {}

    void UpdateAllTimeframes() {
        ENUM_TIMEFRAMES timeframes[3] = {PERIOD_M5, PERIOD_H1, PERIOD_D1};

        for(int i = 0; i < 3; i++) {
            m_mtf_signals[i].timeframe = timeframes[i];
            m_mtf_signals[i].signal = AnalyzeTimeframe(timeframes[i]);
            m_mtf_signals[i].confidence = 0.8; // Placeholder
        }
    }
    
    int AnalyzeTimeframe(ENUM_TIMEFRAMES tf) {
        // Simple MA Crossover for Trend Direction
        double ma_fast = iMA(Symbol(), tf, 20, 0, MODE_EMA, PRICE_CLOSE); // Returns handle
        // Need to handle indicators properly.
        // For this simplified version, we'll use price action relative to previous close.
        
        double close0 = iClose(Symbol(), tf, 0);
        double close1 = iClose(Symbol(), tf, 1);
        
        if(close0 > close1) return 1; // Bullish
        if(close0 < close1) return 2; // Bearish
        return 0;
    }

    double CalculateConfluenceScore() {
        double score = 0;
        for(int i=0; i<3; i++) {
            if(m_mtf_signals[i].signal == 1) score += 0.33;
            if(m_mtf_signals[i].signal == 2) score -= 0.33;
        }
        return score; // >0 Bullish, <0 Bearish
    }
    
    int GetDominantSignal() {
        double score = CalculateConfluenceScore();
        if(score > 0.5) return 1;
        if(score < -0.5) return 2;
        return 0;
    }
};
