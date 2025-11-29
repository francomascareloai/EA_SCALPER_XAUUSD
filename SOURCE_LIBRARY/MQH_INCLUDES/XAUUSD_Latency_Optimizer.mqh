//+------------------------------------------------------------------+
//|                                 XAUUSD_Latency_Optimizer.mqh    |
//|                                     Copyright 2024, Elite Trading |
//+------------------------------------------------------------------+

#include "XAUUSD_ML_Core.mqh"

//+------------------------------------------------------------------+
//| Advanced Latency Optimization for 120ms+ Environments          |
//+------------------------------------------------------------------+
class CXAUUSD_LatencyOptimizer
{
private:
    // Latency Configuration
    int               m_averageLatencyMS;
    int               m_maxAcceptableLatencyMS;
    double            m_slippageTolerancePips;
    bool              m_predictiveExecutionEnabled;
    
    // Order Buffer System
    struct SBufferedOrder
    {
        STradeSignal  signal;
        datetime      submitTime;
        datetime      validUntil;
        bool          isValid;
        bool          executed;
        double        predictedPrice;
        int           bufferID;
    };
    
    SBufferedOrder    m_orderBuffer[20];
    int               m_bufferIndex;
    int               m_nextBufferID;
    
    // Predictive Price System
    struct SPricePredictor
    {
        double        velocityBuffer[10];
        double        accelerationBuffer[5];
        double        lastPrediction;
        datetime      lastPredictionTime;
        double        predictionAccuracy;
        int           correctPredictions;
        int           totalPredictions;
    };
    
    SPricePredictor   m_pricePredictor;
    
    // Execution Timing
    struct SExecutionTiming
    {
        datetime      orderSubmitTime;
        datetime      orderExecuteTime;
        datetime      confirmationTime;
        int           actualLatencyMS;
        double        priceSlippage;
        bool          wasSuccessful;
    };
    
    SExecutionTiming  m_executionHistory[100];
    int               m_executionHistoryIndex;
    
    // Market Condition Monitor
    struct SMarketConditions
    {
        double        currentVolatility;
        double        spreadLevel;
        double        priceVelocity;
        bool          isVolatileMarket;
        bool          isNewsTime;
        bool          isLiquidSession;
    };
    
    SMarketConditions m_marketConditions;
    
    // Performance Metrics
    struct SLatencyMetrics
    {
        double        averageLatency;
        double        averageSlippage;
        double        executionSuccessRate;
        double        predictionAccuracy;
        int           totalExecutions;
        int           successfulExecutions;
        datetime      lastUpdate;
    };
    
    SLatencyMetrics   m_metrics;

public:
                     CXAUUSD_LatencyOptimizer();
    
    // Configuration
    void             SetLatencyParameters(int avgLatency, int maxLatency, double slippageTolerance);
    void             EnablePredictiveExecution(bool enabled);
    
    // Order Processing
    bool             BufferOrder(const STradeSignal &signal, int validitySeconds = 3);
    bool             ExecuteBufferedOrders();
    bool             ValidateOrderBeforeExecution(const SBufferedOrder &order);
    bool             CancelInvalidOrders();
    
    // Predictive Execution
    double           PredictPriceAtExecution(double currentPrice, int latencyMS);
    bool             ShouldExecuteWithPrediction(const STradeSignal &signal);
    void             UpdatePricePredictor();
    
    // Market Condition Analysis
    void             UpdateMarketConditions();
    bool             IsOptimalExecutionTime();
    double           CalculateOptimalTiming(const STradeSignal &signal);
    
    // Latency Mitigation Strategies
    bool             PreValidateMarketConditions(const STradeSignal &signal);
    bool             UseAggressiveExecution(const STradeSignal &signal);
    bool             ImplementDelayedExecution(const STradeSignal &signal);
    
    // Performance Monitoring
    void             RecordExecutionMetrics(const SBufferedOrder &order, bool success, double slippage);
    void             UpdateLatencyMetrics();
    SLatencyMetrics  GetPerformanceMetrics();
    
    // Emergency Protocols
    bool             IsLatencyTooHigh();
    void             ActivateEmergencyMode();
    void             OptimizeForHighLatency();

private:
    // Prediction Algorithms
    double           CalculateLinearPrediction(double currentPrice);
    double           CalculateVelocityBasedPrediction(double currentPrice);
    double           CalculateAccelerationPrediction(double currentPrice);
    double           CalculateVolatilityAdjustedPrediction(double currentPrice);
    
    // Buffer Management
    void             InitializeOrderBuffer();
    int              FindAvailableBufferSlot();
    void             CleanupExpiredOrders();
    
    // Timing Analysis
    double           CalculateOptimalEntryTiming(const STradeSignal &signal);
    bool             IsTimingOptimal(const STradeSignal &signal);
    void             RecordTimingMetrics(const SExecutionTiming &timing);
    
    // Risk Assessment for High Latency
    bool             AssessLatencyRisk(const STradeSignal &signal);
    double           CalculateLatencyRiskMultiplier();
    bool             ShouldSkipTradeForLatency(const STradeSignal &signal);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CXAUUSD_LatencyOptimizer::CXAUUSD_LatencyOptimizer()
{
    m_averageLatencyMS = 120;
    m_maxAcceptableLatencyMS = 200;
    m_slippageTolerancePips = 2.0;
    m_predictiveExecutionEnabled = true;
    
    m_bufferIndex = 0;
    m_nextBufferID = 1;
    m_executionHistoryIndex = 0;
    
    // Initialize buffers
    InitializeOrderBuffer();
    ZeroMemory(m_executionHistory);
    ZeroMemory(m_pricePredictor);
    ZeroMemory(m_marketConditions);
    ZeroMemory(m_metrics);
    
    m_metrics.lastUpdate = TimeCurrent();
    
    Print("⚡ Latency Optimizer initialized for 120ms+ environment");
}

//+------------------------------------------------------------------+
//| Buffer Order for Delayed Execution                              |
//+------------------------------------------------------------------+
bool CXAUUSD_LatencyOptimizer::BufferOrder(const STradeSignal &signal, int validitySeconds)
{
    // Update market conditions before buffering
    UpdateMarketConditions();
    
    // Check if conditions are suitable for high-latency execution
    if(!PreValidateMarketConditions(signal))
    {
        Print("🚫 Order rejected: Market conditions unsuitable for high latency");
        return false;
    }
    
    // Find available buffer slot
    int slotIndex = FindAvailableBufferSlot();
    if(slotIndex < 0)
    {
        Print("⚠️ Order buffer full, cleaning up expired orders");
        CleanupExpiredOrders();
        slotIndex = FindAvailableBufferSlot();
        if(slotIndex < 0)
        {
            Print("❌ Cannot buffer order: No available slots");
            return false;
        }
    }
    
    // Calculate predicted execution price
    double predictedPrice = PredictPriceAtExecution(signal.entryPrice, m_averageLatencyMS);
    
    // Buffer the order
    m_orderBuffer[slotIndex].signal = signal;
    m_orderBuffer[slotIndex].submitTime = TimeCurrent();
    m_orderBuffer[slotIndex].validUntil = TimeCurrent() + validitySeconds;
    m_orderBuffer[slotIndex].isValid = true;
    m_orderBuffer[slotIndex].executed = false;
    m_orderBuffer[slotIndex].predictedPrice = predictedPrice;
    m_orderBuffer[slotIndex].bufferID = m_nextBufferID++;
    
    Print("📦 Order buffered ID:", m_orderBuffer[slotIndex].bufferID, 
          " Predicted price:", predictedPrice, 
          " Valid until:", TimeToString(m_orderBuffer[slotIndex].validUntil));
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute Buffered Orders                                          |
//+------------------------------------------------------------------+
bool CXAUUSD_LatencyOptimizer::ExecuteBufferedOrders()
{
    bool anyExecuted = false;
    datetime currentTime = TimeCurrent();
    
    for(int i = 0; i < 20; i++)
    {
        if(m_orderBuffer[i].isValid && !m_orderBuffer[i].executed)
        {
            // Check if order is still within validity period
            if(currentTime > m_orderBuffer[i].validUntil)
            {
                Print("⏰ Order expired ID:", m_orderBuffer[i].bufferID);
                m_orderBuffer[i].isValid = false;
                continue;
            }
            
            // Validate order before execution
            if(ValidateOrderBeforeExecution(m_orderBuffer[i]))
            {
                // Execute the order
                bool executionSuccess = ExecuteOrder(m_orderBuffer[i]);
                
                // Record execution metrics
                double slippage = CalculateExecutionSlippage(m_orderBuffer[i]);
                RecordExecutionMetrics(m_orderBuffer[i], executionSuccess, slippage);
                
                m_orderBuffer[i].executed = true;
                m_orderBuffer[i].isValid = false;
                anyExecuted = true;
                
                Print(executionSuccess ? "✅" : "❌", " Order executed ID:", m_orderBuffer[i].bufferID,
                      " Slippage:", slippage, " pips");
            }
            else
            {
                Print("🚫 Order validation failed ID:", m_orderBuffer[i].bufferID);
                m_orderBuffer[i].isValid = false;
            }
        }
    }
    
    return anyExecuted;
}

//+------------------------------------------------------------------+
//| Predict Price at Execution Time                                 |
//+------------------------------------------------------------------+
double CXAUUSD_LatencyOptimizer::PredictPriceAtExecution(double currentPrice, int latencyMS)
{
    if(!m_predictiveExecutionEnabled)
        return currentPrice;
    
    // Update price velocity and acceleration
    UpdatePricePredictor();
    
    // Calculate time-adjusted prediction
    double timeFactorSeconds = latencyMS / 1000.0;
    
    // Use multiple prediction methods
    double linearPrediction = CalculateLinearPrediction(currentPrice);
    double velocityPrediction = CalculateVelocityBasedPrediction(currentPrice);
    double accelerationPrediction = CalculateAccelerationPrediction(currentPrice);
    double volatilityPrediction = CalculateVolatilityAdjustedPrediction(currentPrice);
    
    // Weighted ensemble prediction
    double weights[4] = {0.2, 0.3, 0.3, 0.2}; // Linear, Velocity, Acceleration, Volatility
    double predictions[4] = {linearPrediction, velocityPrediction, 
                            accelerationPrediction, volatilityPrediction};
    
    double ensemblePrediction = 0.0;
    for(int i = 0; i < 4; i++)
    {
        ensemblePrediction += predictions[i] * weights[i];
    }
    
    // Apply latency adjustment
    double latencyAdjustment = (ensemblePrediction - currentPrice) * timeFactorSeconds;
    double finalPrediction = currentPrice + latencyAdjustment;
    
    // Store prediction for accuracy tracking
    m_pricePredictor.lastPrediction = finalPrediction;
    m_pricePredictor.lastPredictionTime = TimeCurrent();
    
    return finalPrediction;
}

//+------------------------------------------------------------------+
//| Validate Order Before Execution                                 |
//+------------------------------------------------------------------+
bool CXAUUSD_LatencyOptimizer::ValidateOrderBeforeExecution(const SBufferedOrder &order)
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Check price deviation from predicted
    double priceDifference = MathAbs(currentPrice - order.predictedPrice) / Point;
    if(priceDifference > m_slippageTolerancePips)
    {
        Print("🚫 Price moved too much: ", priceDifference, " pips vs tolerance ", m_slippageTolerancePips);
        return false;
    }
    
    // Check if market conditions are still favorable
    UpdateMarketConditions();
    
    // Reject if volatility spiked
    if(m_marketConditions.isVolatileMarket && order.signal.strategy != "High Volatility Momentum")
    {
        Print("🚫 Volatility spike detected during execution window");
        return false;
    }
    
    // Reject if spread widened significantly
    double currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * Point;
    if(currentSpread > m_marketConditions.spreadLevel * 2.0)
    {
        Print("🚫 Spread widened significantly");
        return false;
    }
    
    // Check if we're in news time
    if(m_marketConditions.isNewsTime)
    {
        Print("🚫 News event detected, canceling execution");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Pre-Validate Market Conditions                                  |
//+------------------------------------------------------------------+
bool CXAUUSD_LatencyOptimizer::PreValidateMarketConditions(const STradeSignal &signal)
{
    // Check current market conditions
    double currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * Point;
    double atr = iATR(_Symbol, PERIOD_M15, 14);
    
    // Reject if spread is too wide for the strategy
    double maxAcceptableSpread = atr * 0.3; // 30% of ATR
    if(currentSpread > maxAcceptableSpread)
    {
        Print("🚫 Spread too wide for high latency: ", currentSpread, " vs max ", maxAcceptableSpread);
        return false;
    }
    
    // Check price velocity
    double priceVelocity = CalculateCurrentVelocity();
    double maxVelocity = atr / Point * 0.5; // 50% of ATR per minute
    
    if(MathAbs(priceVelocity) > maxVelocity)
    {
        Print("🚫 Price moving too fast for high latency: ", priceVelocity, " pips/min");
        return false;
    }
    
    // Check if it's a liquid session
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    bool isLiquidSession = (dt.hour >= 8 && dt.hour <= 17) || (dt.hour >= 13 && dt.hour <= 22);
    
    if(!isLiquidSession)
    {
        Print("🚫 Illiquid session detected, not suitable for high latency");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate Velocity-Based Prediction                             |
//+------------------------------------------------------------------+
double CXAUUSD_LatencyOptimizer::CalculateVelocityBasedPrediction(double currentPrice)
{
    // Calculate recent price velocity
    double price1min = iClose(_Symbol, PERIOD_M1, 1);
    double price2min = iClose(_Symbol, PERIOD_M1, 2);
    double price3min = iClose(_Symbol, PERIOD_M1, 3);
    
    // Calculate velocities
    double velocity1 = (currentPrice - price1min) / Point;
    double velocity2 = (price1min - price2min) / Point;
    double velocity3 = (price2min - price3min) / Point;
    
    // Average velocity with decay
    double avgVelocity = (velocity1 * 0.5) + (velocity2 * 0.3) + (velocity3 * 0.2);
    
    // Project velocity forward by latency time
    double timeFactorMinutes = m_averageLatencyMS / 60000.0; // Convert to minutes
    double velocityProjection = avgVelocity * timeFactorMinutes;
    
    return currentPrice + (velocityProjection * Point);
}

//+------------------------------------------------------------------+
//| Calculate Acceleration Prediction                               |
//+------------------------------------------------------------------+
double CXAUUSD_LatencyOptimizer::CalculateAccelerationPrediction(double currentPrice)
{
    // Calculate price acceleration (change in velocity)
    double velocities[3];
    for(int i = 0; i < 3; i++)
    {
        double price_now = iClose(_Symbol, PERIOD_M1, i);
        double price_prev = iClose(_Symbol, PERIOD_M1, i + 1);
        velocities[i] = (price_now - price_prev) / Point;
    }
    
    // Calculate acceleration
    double acceleration = (velocities[0] - velocities[1]) - (velocities[1] - velocities[2]);
    
    // Project with acceleration
    double timeFactorMinutes = m_averageLatencyMS / 60000.0;
    double velocityProjection = velocities[0] * timeFactorMinutes;
    double accelerationProjection = 0.5 * acceleration * timeFactorMinutes * timeFactorMinutes;
    
    return currentPrice + (velocityProjection + accelerationProjection) * Point;
}

//+------------------------------------------------------------------+
//| Update Market Conditions                                        |
//+------------------------------------------------------------------+
void CXAUUSD_LatencyOptimizer::UpdateMarketConditions()
{
    // Update volatility
    m_marketConditions.currentVolatility = iATR(_Symbol, PERIOD_M15, 14);
    m_marketConditions.isVolatileMarket = (m_marketConditions.currentVolatility / Point) > 25.0;
    
    // Update spread
    m_marketConditions.spreadLevel = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * Point;
    
    // Update price velocity
    m_marketConditions.priceVelocity = CalculateCurrentVelocity();
    
    // Check session liquidity
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    m_marketConditions.isLiquidSession = (dt.hour >= 8 && dt.hour <= 17) || (dt.hour >= 13 && dt.hour <= 22);
    
    // Simple news detection (would integrate with economic calendar)
    m_marketConditions.isNewsTime = (dt.min >= 28 && dt.min <= 32) && 
                                   (dt.hour == 8 || dt.hour == 10 || dt.hour == 14);
}

//+------------------------------------------------------------------+
//| Record Execution Metrics                                        |
//+------------------------------------------------------------------+
void CXAUUSD_LatencyOptimizer::RecordExecutionMetrics(const SBufferedOrder &order, bool success, double slippage)
{
    // Record in execution history
    int idx = m_executionHistoryIndex % 100;
    
    m_executionHistory[idx].orderSubmitTime = order.submitTime;
    m_executionHistory[idx].orderExecuteTime = TimeCurrent();
    m_executionHistory[idx].actualLatencyMS = (int)((TimeCurrent() - order.submitTime) * 1000);
    m_executionHistory[idx].priceSlippage = slippage;
    m_executionHistory[idx].wasSuccessful = success;
    
    m_executionHistoryIndex++;
    
    // Update aggregate metrics
    m_metrics.totalExecutions++;
    if(success) m_metrics.successfulExecutions++;
    
    m_metrics.executionSuccessRate = (double)m_metrics.successfulExecutions / m_metrics.totalExecutions;
    
    // Update prediction accuracy if applicable
    if(m_pricePredictor.lastPredictionTime == order.submitTime)
    {
        double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double predictionError = MathAbs(currentPrice - m_pricePredictor.lastPrediction) / Point;
        
        m_pricePredictor.totalPredictions++;
        if(predictionError <= m_slippageTolerancePips)
        {
            m_pricePredictor.correctPredictions++;
        }
        
        m_pricePredictor.predictionAccuracy = 
            (double)m_pricePredictor.correctPredictions / m_pricePredictor.totalPredictions;
    }
    
    UpdateLatencyMetrics();
}

//+------------------------------------------------------------------+
//| Helper Methods                                                   |
//+------------------------------------------------------------------+
void CXAUUSD_LatencyOptimizer::InitializeOrderBuffer()
{
    for(int i = 0; i < 20; i++)
    {
        ZeroMemory(m_orderBuffer[i]);
        m_orderBuffer[i].isValid = false;
        m_orderBuffer[i].executed = false;
    }
}

int CXAUUSD_LatencyOptimizer::FindAvailableBufferSlot()
{
    for(int i = 0; i < 20; i++)
    {
        if(!m_orderBuffer[i].isValid && !m_orderBuffer[i].executed)
        {
            return i;
        }
    }
    return -1; // No available slot
}

double CXAUUSD_LatencyOptimizer::CalculateCurrentVelocity()
{
    double currentPrice = iClose(_Symbol, PERIOD_M1, 0);
    double price1minAgo = iClose(_Symbol, PERIOD_M1, 1);
    return (currentPrice - price1minAgo) / Point; // Pips per minute
}

bool CXAUUSD_LatencyOptimizer::ExecuteOrder(const SBufferedOrder &order)
{
    CTrade trade;
    trade.SetExpertMagicNumber(12345);
    trade.SetDeviationInPoints((int)m_slippageTolerancePips);
    
    double lotSize = 0.01; // Would calculate optimal lot size
    
    if(order.signal.type == SIGNAL_BUY)
    {
        return trade.Buy(lotSize, _Symbol, 0, order.signal.stopLoss, order.signal.takeProfit, 
                        "Latency Optimized");
    }
    else if(order.signal.type == SIGNAL_SELL)
    {
        return trade.Sell(lotSize, _Symbol, 0, order.signal.stopLoss, order.signal.takeProfit, 
                         "Latency Optimized");
    }
    
    return false;
}

double CXAUUSD_LatencyOptimizer::CalculateExecutionSlippage(const SBufferedOrder &order)
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    return MathAbs(currentPrice - order.signal.entryPrice) / Point;
}

// Simplified implementations for remaining methods
double CXAUUSD_LatencyOptimizer::CalculateLinearPrediction(double currentPrice) { return currentPrice; }
double CXAUUSD_LatencyOptimizer::CalculateVolatilityAdjustedPrediction(double currentPrice) { return currentPrice; }
void CXAUUSD_LatencyOptimizer::UpdatePricePredictor() { }
void CXAUUSD_LatencyOptimizer::CleanupExpiredOrders() { }
void CXAUUSD_LatencyOptimizer::UpdateLatencyMetrics() { }