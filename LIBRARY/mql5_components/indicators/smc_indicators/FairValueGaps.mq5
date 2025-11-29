//+------------------------------------------------------------------+
//|                                                FairValueGaps.mq5 |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 clrGreen
#property indicator_color2 clrRed
#parameter int FVGPeriod = 50;

// Indicator buffers
double BullishFVGBuffer[];
double BearishFVGBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {
    // Indicator buffers mapping
    SetIndexBuffer(0, BullishFVGBuffer, INDICATOR_DATA);
    SetIndexBuffer(1, BearishFVGBuffer, INDICATOR_DATA);
    
    // Indicator lines
    SetIndexStyle(0, DRAW_SECTION);
    SetIndexStyle(1, DRAW_SECTION);
    
    // Indicator short name
    IndicatorShortName("Fair Value Gaps");
    
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
    
    // Calculate fair value gaps
    CalculateFVG(rates_total, prev_calculated, open, high, low, close);
    
    return(rates_total);
}

//+------------------------------------------------------------------+
//| Calculate Fair Value Gaps                                        |
//+------------------------------------------------------------------+
void CalculateFVG(int rates_total, int prev_calculated, const double &open[], const double &high[], const double &low[], const double &close[]) {
    // Reset buffers
    for (int i = 0; i < rates_total; i++) {
        BullishFVGBuffer[i] = 0;
        BearishFVGBuffer[i] = 0;
    }
    
    // Simple FVG detection logic
    for (int i = 2; i < rates_total && i < FVGPeriod; i++) {
        // Bullish FVG - gap down
        if (low[i] > high[i-2]) {
            BullishFVGBuffer[i-1] = (low[i] + high[i-2]) / 2;
        }
        
        // Bearish FVG - gap up
        if (high[i] < low[i-2]) {
            BearishFVGBuffer[i-1] = (high[i] + low[i-2]) / 2;
        }
    }
}