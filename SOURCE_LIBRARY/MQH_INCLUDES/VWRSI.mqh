#property copyright "TradeDev_Master 2024"
#property version   "1.00"
#property strict

class CVWRSI
{
private:
    int m_handle;
    string m_symbol;
    ENUM_TIMEFRAMES m_timeframe;
    int m_period;
    
public:
    CVWRSI(string symbol, ENUM_TIMEFRAMES timeframe, int period);
    ~CVWRSI();
    
    bool Initialize();
    double GetValue(int shift);
    double GetSignal(int shift);
};

CVWRSI::CVWRSI(string symbol, ENUM_TIMEFRAMES timeframe, int period)
{
    m_symbol = symbol;
    m_timeframe = timeframe;
    m_period = period;
    m_handle = INVALID_HANDLE;
}

CVWRSI::~CVWRSI()
{
    if(m_handle != INVALID_HANDLE) 
    {
        IndicatorRelease(m_handle);
    }
}

bool CVWRSI::Initialize()
{
    m_handle = iCustom(m_symbol, m_timeframe, "Custom\\VWRSI", m_period);
    return (m_handle != INVALID_HANDLE);
}

double CVWRSI::GetValue(int shift)
{
    double buffer[];
    ArraySetAsSeries(buffer, true);
    
    if(CopyBuffer(m_handle, 0, shift, 1, buffer) == 1)
    {
        return buffer[0];
    }
    
    return EMPTY_VALUE;
}

double CVWRSI::GetSignal(int shift)
{
    double buffer[];
    ArraySetAsSeries(buffer, true);
    
    if(CopyBuffer(m_handle, 1, shift, 1, buffer) == 1)
    {
        return buffer[0];
    }
    
    return EMPTY_VALUE;
}