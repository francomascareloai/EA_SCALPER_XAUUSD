#property copyright "TradeDev_Master 2024"
#property version   "1.00"
#property description "Volume Weighted RSI Indicator"
#property indicator_separate_window
#property indicator_minimum 0
#property indicator_maximum 100
#property indicator_buffers 2
#property indicator_plots   1
#property indicator_label1  "VWRSI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2
#property indicator_label2  "Signal"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrOrange
#property indicator_width2  1

input int Period = 14; // RSI Period
input int SignalPeriod = 9; // Signal Period

double vwrsiBuffer[];
double signalBuffer[];

int OnInit()
{
    SetIndexBuffer(0, vwrsiBuffer);
    SetIndexBuffer(1, signalBuffer);
    
    IndicatorSetString(INDICATOR_SHORTNAME, "VWRSI(" + IntegerToString(Period) + ")");
    return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                const prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    int limit = rates_total - prev_calculated;
    if(prev_calculated > 0) limit++;
    
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    ArraySetAsSeries(close, true);
    ArraySetAsSeries(volume, true);
    ArraySetAsSeries(vwrsiBuffer, true);
    ArraySetAsSeries(signalBuffer, true);
    
    // Calcular VWRSI
    for(int i = limit-1; i >= 0; i--)
    {
        if(i >= rates_total - Period) continue;
        
        double typicalPrice = (high[i] + low[i] + close[i]) / 3.0;
        double prevTypical = (high[i+1] + low[i+1] + close[i+1]) / 3.0;
        double change = typicalPrice - prevTypical;
        
        double gain = (change > 0) ? change * volume[i] : 0;
        double loss = (change < 0) ? MathAbs(change) * volume[i] : 0;
        
        // Calcular médias móveis exponenciais para ganhos e perdas
        static double avgGain[], avgLoss[];
        if(ArraySize(avgGain) < rates_total) ArrayResize(avgGain, rates_total);
        if(ArraySize(avgLoss) < rates_total) ArrayResize(avgLoss, rates_total);
        
        if(i == rates_total - 1)
        {
            avgGain[i] = gain;
            avgLoss[i] = loss;
        }
        else
        {
            avgGain[i] = (avgGain[i+1] * (Period-1) + gain) / Period;
            avgLoss[i] = (avgLoss[i+1] * (Period-1) + loss) / Period;
        }
        
        // Calcular VWRSI
        if(avgLoss[i] == 0)
            vwrsiBuffer[i] = 100;
        else
            vwrsiBuffer[i] = 100 - (100 / (1 + (avgGain[i] / avgLoss[i])));
    }
    
    // Calcular linha de sinal (média móvel do VWRSI)
    for(int i = limit-1; i >= 0; i--)
    {
        double sum = 0;
        int count = 0;
        for(int j = 0; j < SignalPeriod; j++)
        {
            if(i+j >= rates_total) continue;
            sum += vwrsiBuffer[i+j];
            count++;
        }
        signalBuffer[i] = (count > 0) ? sum / count : EMPTY_VALUE;
    }
    
    return(rates_total);
}