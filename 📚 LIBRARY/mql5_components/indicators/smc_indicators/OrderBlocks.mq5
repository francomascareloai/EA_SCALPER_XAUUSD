//+------------------------------------------------------------------+
//|                                                   OrderBlocks.mq5 |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 clrBlue
#parameter int OBPeriod = 50;

// Indicator buffers
double BullishOBBuffer[];
double BearishOBBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {
    // Indicator buffers mapping
    SetIndexBuffer(0, BullishOBBuffer, INDICATOR_DATA);
    SetIndexBuffer(1, BearishOBBuffer, INDICATOR_DATA);
    
    // Indicator lines
    SetIndexStyle(0, DRAW_ARROW);
    SetIndexArrow(0, 233);
    SetIndexStyle(1, DRAW_ARROW);
    SetIndexArrow(1, 234);
    
    // Indicator short name
    IndicatorShortName("Order Blocks");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[]) {
    
    // Calculate order blocks
    CalculateOrderBlocks(rates_total, prev_calculated, open, high, low, close);
    
    return(rates_total);
}

//+------------------------------------------------------------------+
//| Calculate Order Blocks                                           |
//+------------------------------------------------------------------+
void CalculateOrderBlocks(int rates_total, int prev_calculated, const double &open[], const double &high[], const double &low[], const double &close[]) {
    // Reset buffers
    for (int i = 0; i < rates_total; i++) {
        BullishOBBuffer[i] = 0;
        BearishOBBuffer[i] = 0;
    }
    
    // Simple order block detection logic
    for (int i = 1; i < rates_total - 1 && i < OBPeriod; i++) {
        // Bullish order block - previous bearish candle followed by bullish candle
        if (close[i] > open[i] && close[i-1] < open[i-1]) {
            // Check if there's a significant difference
            if ((open[i] - close[i-1]) > (high[i-1] - low[i-1]) * 2) {
                BullishOBBuffer[i] = low[i-1];
            }
        }
        
        // Bearish order block - previous bullish candle followed by bearish candle
        if (close[i] < open[i] && close[i-1] > open[i-1]) {
            // Check if there's a significant difference
            if ((close[i-1] - open[i]) > (high[i-1] - low[i-1]) * 2) {
                BearishOBBuffer[i] = high[i-1];
            }
        }
    }
}