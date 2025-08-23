//+------------------------------------------------------------------+
//|                                   MarketStructureAnalyzer.mqh |
//|                        Copyright 2024, TradeDev_Master Team |
//|                                   https://github.com/tradedev |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, TradeDev_Master Team"
#property link      "https://github.com/tradedev"
#property version   "1.00"
#property strict

#include "../Core/DataStructures.mqh"
#include "../Core/Interfaces.mqh"
#include "../Core/Logger.mqh"

//+------------------------------------------------------------------+
//| Market Structure Analyzer Class                                 |
//+------------------------------------------------------------------+
class CMarketStructureAnalyzer : public IAnalyzer
{
private:
    // Configurações
    int               m_structure_period;     // Período para análise
    int               m_swing_strength;       // Força dos swings
    bool              m_use_multi_timeframe;  // Usar múltiplos timeframes
    
    // Estado da estrutura
    ENUM_MARKET_STRUCTURE m_current_structure;
    ENUM_TREND_DIRECTION  m_trend_direction;
    datetime              m_last_structure_change;
    
    // Arrays de dados
    SSwingPoint       m_swing_highs[];
    SSwingPoint       m_swing_lows[];
    int               m_highs_count;
    int               m_lows_count;
    
    // Cache
    datetime          m_last_analysis;
    bool              m_cache_valid;
    
public:
    // Construtor/Destrutor
                     CMarketStructureAnalyzer(void);
                    ~CMarketStructureAnalyzer(void);
    
    // Métodos principais
    virtual bool      Initialize(void) override;
    virtual bool      Update(void) override;
    virtual void      Reset(void) override;
    
    // Configuração
    void              SetStructurePeriod(int period) { m_structure_period = period; }
    void              SetSwingStrength(int strength) { m_swing_strength = strength; }
    void              SetMultiTimeframe(bool enable) { m_use_multi_timeframe = enable; }
    
    // Análise de estrutura
    bool              AnalyzeMarketStructure(void);
    bool              DetectStructureBreak(void);
    bool              DetectTrendChange(void);
    
    // Análise de swings
    bool              FindSwingPoints(void);
    bool              ValidateSwingPoint(const SSwingPoint &swing);
    double            CalculateSwingStrength(const SSwingPoint &swing);
    
    // Getters
    ENUM_MARKET_STRUCTURE GetCurrentStructure(void) const { return m_current_structure; }
    ENUM_TREND_DIRECTION  GetTrendDirection(void) const { return m_trend_direction; }
    datetime              GetLastStructureChange(void) const { return m_last_structure_change; }
    
    // Análise de níveis
    double            GetLastHigherHigh(void);
    double            GetLastLowerLow(void);
    double            GetLastHigherLow(void);
    double            GetLastLowerHigh(void);
    
    // Confirmações
    bool              IsUptrend(void);
    bool              IsDowntrend(void);
    bool              IsRanging(void);
    bool              IsStructureIntact(void);
    
private:
    // Métodos auxiliares
    bool              IsSwingHigh(int bar_index);
    bool              IsSwingLow(int bar_index);
    ENUM_MARKET_STRUCTURE DetermineStructure(void);
    ENUM_TREND_DIRECTION  DetermineTrend(void);
    void              UpdateSwingArrays(void);
    void              CleanupOldSwings(void);
};

//+------------------------------------------------------------------+
//| Construtor                                                       |
//+------------------------------------------------------------------+
CMarketStructureAnalyzer::CMarketStructureAnalyzer(void) :
    m_structure_period(50),
    m_swing_strength(3),
    m_use_multi_timeframe(false),
    m_current_structure(MARKET_STRUCTURE_RANGING),
    m_trend_direction(TREND_DIRECTION_SIDEWAYS),
    m_last_structure_change(0),
    m_highs_count(0),
    m_lows_count(0),
    m_last_analysis(0),
    m_cache_valid(false)
{
    ArrayResize(m_swing_highs, 50);
    ArrayResize(m_swing_lows, 50);
    ArrayInitialize(m_swing_highs, 0);
    ArrayInitialize(m_swing_lows, 0);
}

//+------------------------------------------------------------------+
//| Destrutor                                                        |
//+------------------------------------------------------------------+
CMarketStructureAnalyzer::~CMarketStructureAnalyzer(void)
{
    ArrayFree(m_swing_highs);
    ArrayFree(m_swing_lows);
}

//+------------------------------------------------------------------+
//| Inicialização                                                    |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::Initialize(void)
{
    CLogger::Log(LOG_INFO, "CMarketStructureAnalyzer", "Inicializando analisador de estrutura...");
    
    if(Bars(_Symbol, _Period) < m_structure_period)
    {
        CLogger::Log(LOG_ERROR, "CMarketStructureAnalyzer", "Dados insuficientes");
        return false;
    }
    
    Reset();
    
    CLogger::Log(LOG_INFO, "CMarketStructureAnalyzer", "Analisador inicializado");
    return true;
}

//+------------------------------------------------------------------+
//| Atualização                                                      |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::Update(void)
{
    datetime current_time = TimeCurrent();
    
    if(m_cache_valid && current_time == m_last_analysis)
        return true;
    
    bool result = AnalyzeMarketStructure();
    
    if(result)
    {
        m_last_analysis = current_time;
        m_cache_valid = true;
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Reset                                                            |
//+------------------------------------------------------------------+
void CMarketStructureAnalyzer::Reset(void)
{
    m_current_structure = MARKET_STRUCTURE_RANGING;
    m_trend_direction = TREND_DIRECTION_SIDEWAYS;
    m_last_structure_change = 0;
    m_highs_count = 0;
    m_lows_count = 0;
    m_last_analysis = 0;
    m_cache_valid = false;
    
    ArrayInitialize(m_swing_highs, 0);
    ArrayInitialize(m_swing_lows, 0);
    
    CLogger::Log(LOG_INFO, "CMarketStructureAnalyzer", "Analisador resetado");
}

//+------------------------------------------------------------------+
//| Analisar estrutura do mercado                                   |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::AnalyzeMarketStructure(void)
{
    // Encontrar pontos de swing
    if(!FindSwingPoints())
        return false;
    
    // Determinar estrutura atual
    ENUM_MARKET_STRUCTURE new_structure = DetermineStructure();
    ENUM_TREND_DIRECTION new_trend = DetermineTrend();
    
    // Verificar mudança de estrutura
    if(new_structure != m_current_structure)
    {
        m_current_structure = new_structure;
        m_last_structure_change = TimeCurrent();
        
        CLogger::Log(LOG_INFO, "CMarketStructureAnalyzer", 
                    StringFormat("Mudança de estrutura detectada: %d", (int)new_structure));
    }
    
    m_trend_direction = new_trend;
    
    // Limpar swings antigos
    CleanupOldSwings();
    
    return true;
}

//+------------------------------------------------------------------+
//| Encontrar pontos de swing                                       |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::FindSwingPoints(void)
{
    m_highs_count = 0;
    m_lows_count = 0;
    
    for(int i = m_swing_strength; i < m_structure_period - m_swing_strength; i++)
    {
        // Verificar swing high
        if(IsSwingHigh(i))
        {
            if(m_highs_count < ArraySize(m_swing_highs))
            {
                SSwingPoint swing;
                swing.price = iHigh(_Symbol, _Period, i);
                swing.time = iTime(_Symbol, _Period, i);
                swing.bar_index = i;
                swing.type = SWING_TYPE_HIGH;
                swing.strength = CalculateSwingStrength(swing);
                swing.confirmed = true;
                
                if(ValidateSwingPoint(swing))
                {
                    m_swing_highs[m_highs_count] = swing;
                    m_highs_count++;
                }
            }
        }
        
        // Verificar swing low
        if(IsSwingLow(i))
        {
            if(m_lows_count < ArraySize(m_swing_lows))
            {
                SSwingPoint swing;
                swing.price = iLow(_Symbol, _Period, i);
                swing.time = iTime(_Symbol, _Period, i);
                swing.bar_index = i;
                swing.type = SWING_TYPE_LOW;
                swing.strength = CalculateSwingStrength(swing);
                swing.confirmed = true;
                
                if(ValidateSwingPoint(swing))
                {
                    m_swing_lows[m_lows_count] = swing;
                    m_lows_count++;
                }
            }
        }
    }
    
    return (m_highs_count > 0 || m_lows_count > 0);
}

//+------------------------------------------------------------------+
//| Verificar se é swing high                                       |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::IsSwingHigh(int bar_index)
{
    double current_high = iHigh(_Symbol, _Period, bar_index);
    
    // Verificar barras à esquerda e direita
    for(int i = 1; i <= m_swing_strength; i++)
    {
        if(iHigh(_Symbol, _Period, bar_index - i) >= current_high ||
           iHigh(_Symbol, _Period, bar_index + i) >= current_high)
        {
            return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Verificar se é swing low                                        |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::IsSwingLow(int bar_index)
{
    double current_low = iLow(_Symbol, _Period, bar_index);
    
    // Verificar barras à esquerda e direita
    for(int i = 1; i <= m_swing_strength; i++)
    {
        if(iLow(_Symbol, _Period, bar_index - i) <= current_low ||
           iLow(_Symbol, _Period, bar_index + i) <= current_low)
        {
            return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Determinar estrutura do mercado                                 |
//+------------------------------------------------------------------+
ENUM_MARKET_STRUCTURE CMarketStructureAnalyzer::DetermineStructure(void)
{
    if(m_highs_count < 2 || m_lows_count < 2)
        return MARKET_STRUCTURE_RANGING;
    
    // Analisar últimos 2 highs e lows
    bool higher_highs = (m_swing_highs[0].price > m_swing_highs[1].price);
    bool higher_lows = (m_swing_lows[0].price > m_swing_lows[1].price);
    bool lower_highs = (m_swing_highs[0].price < m_swing_highs[1].price);
    bool lower_lows = (m_swing_lows[0].price < m_swing_lows[1].price);
    
    if(higher_highs && higher_lows)
        return MARKET_STRUCTURE_UPTREND;
    else if(lower_highs && lower_lows)
        return MARKET_STRUCTURE_DOWNTREND;
    else
        return MARKET_STRUCTURE_RANGING;
}

//+------------------------------------------------------------------+
//| Determinar direção da tendência                                 |
//+------------------------------------------------------------------+
ENUM_TREND_DIRECTION CMarketStructureAnalyzer::DetermineTrend(void)
{
    switch(m_current_structure)
    {
        case MARKET_STRUCTURE_UPTREND:
            return TREND_DIRECTION_UP;
        case MARKET_STRUCTURE_DOWNTREND:
            return TREND_DIRECTION_DOWN;
        default:
            return TREND_DIRECTION_SIDEWAYS;
    }
}

//+------------------------------------------------------------------+
//| Validar ponto de swing                                          |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::ValidateSwingPoint(const SSwingPoint &swing)
{
    // Verificar força mínima
    if(swing.strength < 0.5)
        return false;
    
    // Verificar se não é muito próximo de outro swing
    if(swing.type == SWING_TYPE_HIGH)
    {
        for(int i = 0; i < m_highs_count; i++)
        {
            if(MathAbs(swing.price - m_swing_highs[i].price) < 10 * _Point)
                return false;
        }
    }
    else
    {
        for(int i = 0; i < m_lows_count; i++)
        {
            if(MathAbs(swing.price - m_swing_lows[i].price) < 10 * _Point)
                return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Calcular força do swing                                         |
//+------------------------------------------------------------------+
double CMarketStructureAnalyzer::CalculateSwingStrength(const SSwingPoint &swing)
{
    double atr = iATR(_Symbol, _Period, 14, swing.bar_index);
    if(atr <= 0) return 1.0;
    
    double range = 0.0;
    
    if(swing.type == SWING_TYPE_HIGH)
    {
        // Calcular range do swing high
        double lowest = DBL_MAX;
        for(int i = swing.bar_index - m_swing_strength; i <= swing.bar_index + m_swing_strength; i++)
        {
            double low = iLow(_Symbol, _Period, i);
            if(low < lowest) lowest = low;
        }
        range = swing.price - lowest;
    }
    else
    {
        // Calcular range do swing low
        double highest = 0.0;
        for(int i = swing.bar_index - m_swing_strength; i <= swing.bar_index + m_swing_strength; i++)
        {
            double high = iHigh(_Symbol, _Period, i);
            if(high > highest) highest = high;
        }
        range = highest - swing.price;
    }
    
    return range / atr;
}

//+------------------------------------------------------------------+
//| Limpar swings antigos                                           |
//+------------------------------------------------------------------+
void CMarketStructureAnalyzer::CleanupOldSwings(void)
{
    // Manter apenas os últimos 20 swings de cada tipo
    if(m_highs_count > 20)
        m_highs_count = 20;
    
    if(m_lows_count > 20)
        m_lows_count = 20;
}

//+------------------------------------------------------------------+
//| Obter último higher high                                        |
//+------------------------------------------------------------------+
double CMarketStructureAnalyzer::GetLastHigherHigh(void)
{
    if(m_highs_count < 2) return 0.0;
    
    for(int i = 0; i < m_highs_count - 1; i++)
    {
        if(m_swing_highs[i].price > m_swing_highs[i + 1].price)
            return m_swing_highs[i].price;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| Obter último lower low                                          |
//+------------------------------------------------------------------+
double CMarketStructureAnalyzer::GetLastLowerLow(void)
{
    if(m_lows_count < 2) return 0.0;
    
    for(int i = 0; i < m_lows_count - 1; i++)
    {
        if(m_swing_lows[i].price < m_swing_lows[i + 1].price)
            return m_swing_lows[i].price;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| Verificar se está em uptrend                                    |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::IsUptrend(void)
{
    return (m_current_structure == MARKET_STRUCTURE_UPTREND);
}

//+------------------------------------------------------------------+
//| Verificar se está em downtrend                                  |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::IsDowntrend(void)
{
    return (m_current_structure == MARKET_STRUCTURE_DOWNTREND);
}

//+------------------------------------------------------------------+
//| Verificar se está em range                                      |
//+------------------------------------------------------------------+
bool CMarketStructureAnalyzer::IsRanging(void)
{
    return (m_current_structure == MARKET_STRUCTURE_RANGING);
}
