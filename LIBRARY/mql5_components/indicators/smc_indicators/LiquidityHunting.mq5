//+------------------------------------------------------------------+
//|                                              LiquidityHunting.mq5 |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 clrYellow
#property indicator_color2 clrPurple
#parameter int LiquidityPeriod = 50;

// Indicator buffers
double LiquidityHighBuffer[];
double LiquidityLowBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {
    // Indicator buffers mapping
    SetIndexBuffer(0, LiquidityHighBuffer, INDICATOR_DATA);
    SetIndexBuffer(1, LiquidityLowBuffer, INDICATOR_DATA);
    
    // Indicator lines
    SetIndexStyle(0, DRAW_ARROW);
    SetIndexArrow(0, 159);
    SetIndexStyle(1, DRAW_ARROW);
    SetIndexArrow(1, 159);
    
    // Indicator short name
    IndicatorShortName("Liquidity Hunting");
    
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
    
    // Calculate liquidity zones
    CalculateLiquidity(rates_total, prev_calculated, high, low, close);
    
    return(rates_total);
}

//+------------------------------------------------------------------+
//| Calculate Liquidity Zones                                        |
//+------------------------------------------------------------------+
void CalculateLiquidity(int rates_total, int prev_calculated, const double &high[], const double &low[], const double &close[]) {
    // Reset buffers
    for (int i = 0; i < rates_total; i++) {
        LiquidityHighBuffer[i] = 0;
        LiquidityLowBuffer[i] = 0;
    }
    
    // Simple liquidity detection logic
    for (int i = 10; i < rates_total && i < LiquidityPeriod; i++) {
        // Find recent swing highs and lows
        double highestHigh = high[i];
        double lowestLow = low[i];
        
        for (int j = 1; j <= 10; j++) {
            if (high[i-j] > highestHigh) {
                highestHigh = high[i-j];
            }
            if (low[i-j] < lowestLow) {
                lowestLow = low[i-j];
            }
        }
        
        // Mark potential liquidity zones
        if (close[i] > highestHigh * 0.99 && close[i] < highestHigh * 1.01) {
            LiquidityHighBuffer[i] = high[i];
        }
        
        if (close[i] > lowestLow * 0.99 && close[i] < lowestLow * 1.01) {
            LiquidityLowBuffer[i] = low[i];
        }
    }
}