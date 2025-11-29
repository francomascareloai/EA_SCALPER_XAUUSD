//+------------------------------------------------------------------+
//|                                         LiquidityDetector.mqh |
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
//| Liquidity Detector Class                                        |
//+------------------------------------------------------------------+
class CLiquidityDetector : public IDetector
{
private:
    // Configurações
    int               m_lookback_bars;       // Barras para análise
    double            m_min_liquidity_size;  // Tamanho mínimo de liquidez
    int               m_min_touches;         // Mínimo de toques no nível
    bool              m_use_volume_filter;   // Usar filtro de volume
    
    // Arrays de dados
    SLiquidityZone    m_liquidity_zones[];   // Zonas de liquidez
    int               m_zone_count;          // Contador de zonas
    
    // Cache
    datetime          m_last_update;
    bool              m_cache_valid;
    
public:
    // Construtor/Destrutor
                     CLiquidityDetector(void);
                    ~CLiquidityDetector(void);
    
    // Métodos principais
    virtual bool      Initialize(void) override;
    virtual bool      Update(void) override;
    virtual void      Reset(void) override;
    
    // Configuração
    void              SetLookbackBars(int bars) { m_lookback_bars = bars; }
    void              SetMinLiquiditySize(double size) { m_min_liquidity_size = size; }
    void              SetMinTouches(int touches) { m_min_touches = touches; }
    void              SetVolumeFilter(bool enable) { m_use_volume_filter = enable; }
    
    // Detecção de liquidez
    bool              DetectLiquidityZones(void);
    bool              DetectBuySideLiquidity(void);
    bool              DetectSellSideLiquidity(void);
    
    // Análise de zonas
    double            CalculateZoneStrength(const SLiquidityZone &zone);
    bool              IsZoneValid(const SLiquidityZone &zone);
    ENUM_LIQUIDITY_STATUS GetZoneStatus(const SLiquidityZone &zone);
    
    // Getters
    int               GetZoneCount(void) const { return m_zone_count; }
    SLiquidityZone    GetZone(int index) const;
    SLiquidityZone    GetNearestZone(double price, ENUM_LIQUIDITY_TYPE type = LIQUIDITY_TYPE_ANY);
    
    // Análise de mercado
    bool              IsLiquidityGrab(double price, ENUM_LIQUIDITY_TYPE type);
    double            GetLiquidityDensity(double price_level, double range);
    
private:
    // Métodos auxiliares
    bool              FindSwingHighs(double &highs[], datetime &times[], int &count);
    bool              FindSwingLows(double &lows[], datetime &times[], int &count);
    int               CountTouches(double level, double tolerance, int start_bar, int end_bar);
    bool              CheckVolumeConfirmation(double level, int bar_index);
    void              CleanupOldZones(void);
    void              SortZonesByStrength(void);
};

//+------------------------------------------------------------------+
//| Construtor                                                       |
//+------------------------------------------------------------------+
CLiquidityDetector::CLiquidityDetector(void) :
    m_lookback_bars(100),
    m_min_liquidity_size(20),
    m_min_touches(2),
    m_use_volume_filter(true),
    m_zone_count(0),
    m_last_update(0),
    m_cache_valid(false)
{
    ArrayResize(m_liquidity_zones, 50);
    ArrayInitialize(m_liquidity_zones, 0);
}

//+------------------------------------------------------------------+
//| Destrutor                                                        |
//+------------------------------------------------------------------+
CLiquidityDetector::~CLiquidityDetector(void)
{
    ArrayFree(m_liquidity_zones);
}

//+------------------------------------------------------------------+
//| Inicialização                                                    |
//+------------------------------------------------------------------+
bool CLiquidityDetector::Initialize(void)
{
    CLogger::Log(LOG_INFO, "CLiquidityDetector", "Inicializando detector de liquidez...");
    
    if(Bars(_Symbol, _Period) < m_lookback_bars)
    {
        CLogger::Log(LOG_ERROR, "CLiquidityDetector", "Dados insuficientes para análise");
        return false;
    }
    
    Reset();
    
    CLogger::Log(LOG_INFO, "CLiquidityDetector", "Detector de liquidez inicializado");
    return true;
}

//+------------------------------------------------------------------+
//| Atualização                                                      |
//+------------------------------------------------------------------+
bool CLiquidityDetector::Update(void)
{
    datetime current_time = TimeCurrent();
    
    if(m_cache_valid && current_time == m_last_update)
        return true;
    
    bool result = DetectLiquidityZones();
    
    if(result)
    {
        CleanupOldZones();
        SortZonesByStrength();
        
        m_last_update = current_time;
        m_cache_valid = true;
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Reset                                                            |
//+------------------------------------------------------------------+
void CLiquidityDetector::Reset(void)
{
    m_zone_count = 0;
    m_last_update = 0;
    m_cache_valid = false;
    ArrayInitialize(m_liquidity_zones, 0);
    
    CLogger::Log(LOG_INFO, "CLiquidityDetector", "Detector resetado");
}

//+------------------------------------------------------------------+
//| Detectar zonas de liquidez                                      |
//+------------------------------------------------------------------+
bool CLiquidityDetector::DetectLiquidityZones(void)
{
    m_zone_count = 0;
    
    // Detectar liquidez do lado comprador e vendedor
    DetectBuySideLiquidity();
    DetectSellSideLiquidity();
    
    CLogger::Log(LOG_INFO, "CLiquidityDetector", 
                StringFormat("Detectadas %d zonas de liquidez", m_zone_count));
    
    return true;
}

//+------------------------------------------------------------------+
//| Detectar liquidez do lado comprador                             |
//+------------------------------------------------------------------+
bool CLiquidityDetector::DetectBuySideLiquidity(void)
{
    double highs[100];
    datetime times[100];
    int high_count = 0;
    
    if(!FindSwingHighs(highs, times, high_count))
        return false;
    
    for(int i = 0; i < high_count && m_zone_count < ArraySize(m_liquidity_zones); i++)
    {
        double level = highs[i];
        int touches = CountTouches(level, 5 * _Point, 1, m_lookback_bars);
        
        if(touches >= m_min_touches)
        {
            SLiquidityZone zone;
            zone.type = LIQUIDITY_TYPE_BUY_SIDE;
            zone.level = level;
            zone.upper_bound = level + (m_min_liquidity_size * _Point);
            zone.lower_bound = level - (m_min_liquidity_size * _Point);
            zone.time_created = times[i];
            zone.touches = touches;
            zone.strength = CalculateZoneStrength(zone);
            zone.status = LIQUIDITY_STATUS_ACTIVE;
            zone.volume_confirmation = CheckVolumeConfirmation(level, iBarShift(_Symbol, _Period, times[i]));
            
            if(IsZoneValid(zone))
            {
                m_liquidity_zones[m_zone_count] = zone;
                m_zone_count++;
            }
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Detectar liquidez do lado vendedor                              |
//+------------------------------------------------------------------+
bool CLiquidityDetector::DetectSellSideLiquidity(void)
{
    double lows[100];
    datetime times[100];
    int low_count = 0;
    
    if(!FindSwingLows(lows, times, low_count))
        return false;
    
    for(int i = 0; i < low_count && m_zone_count < ArraySize(m_liquidity_zones); i++)
    {
        double level = lows[i];
        int touches = CountTouches(level, 5 * _Point, 1, m_lookback_bars);
        
        if(touches >= m_min_touches)
        {
            SLiquidityZone zone;
            zone.type = LIQUIDITY_TYPE_SELL_SIDE;
            zone.level = level;
            zone.upper_bound = level + (m_min_liquidity_size * _Point);
            zone.lower_bound = level - (m_min_liquidity_size * _Point);
            zone.time_created = times[i];
            zone.touches = touches;
            zone.strength = CalculateZoneStrength(zone);
            zone.status = LIQUIDITY_STATUS_ACTIVE;
            zone.volume_confirmation = CheckVolumeConfirmation(level, iBarShift(_Symbol, _Period, times[i]));
            
            if(IsZoneValid(zone))
            {
                m_liquidity_zones[m_zone_count] = zone;
                m_zone_count++;
            }
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Encontrar swing highs                                           |
//+------------------------------------------------------------------+
bool CLiquidityDetector::FindSwingHighs(double &highs[], datetime &times[], int &count)
{
    count = 0;
    int swing_period = 5;
    
    for(int i = swing_period; i < m_lookback_bars - swing_period; i++)
    {
        double current_high = iHigh(_Symbol, _Period, i);
        bool is_swing_high = true;
        
        // Verificar se é um swing high
        for(int j = 1; j <= swing_period; j++)
        {
            if(iHigh(_Symbol, _Period, i - j) >= current_high ||
               iHigh(_Symbol, _Period, i + j) >= current_high)
            {
                is_swing_high = false;
                break;
            }
        }
        
        if(is_swing_high && count < ArraySize(highs))
        {
            highs[count] = current_high;
            times[count] = iTime(_Symbol, _Period, i);
            count++;
        }
    }
    
    return (count > 0);
}

//+------------------------------------------------------------------+
//| Encontrar swing lows                                            |
//+------------------------------------------------------------------+
bool CLiquidityDetector::FindSwingLows(double &lows[], datetime &times[], int &count)
{
    count = 0;
    int swing_period = 5;
    
    for(int i = swing_period; i < m_lookback_bars - swing_period; i++)
    {
        double current_low = iLow(_Symbol, _Period, i);
        bool is_swing_low = true;
        
        // Verificar se é um swing low
        for(int j = 1; j <= swing_period; j++)
        {
            if(iLow(_Symbol, _Period, i - j) <= current_low ||
               iLow(_Symbol, _Period, i + j) <= current_low)
            {
                is_swing_low = false;
                break;
            }
        }
        
        if(is_swing_low && count < ArraySize(lows))
        {
            lows[count] = current_low;
            times[count] = iTime(_Symbol, _Period, i);
            count++;
        }
    }
    
    return (count > 0);
}

//+------------------------------------------------------------------+
//| Contar toques no nível                                          |
//+------------------------------------------------------------------+
int CLiquidityDetector::CountTouches(double level, double tolerance, int start_bar, int end_bar)
{
    int touches = 0;
    
    for(int i = start_bar; i <= end_bar; i++)
    {
        double high = iHigh(_Symbol, _Period, i);
        double low = iLow(_Symbol, _Period, i);
        
        if((high >= level - tolerance && high <= level + tolerance) ||
           (low >= level - tolerance && low <= level + tolerance))
        {
            touches++;
        }
    }
    
    return touches;
}

//+------------------------------------------------------------------+
//| Calcular força da zona                                          |
//+------------------------------------------------------------------+
double CLiquidityDetector::CalculateZoneStrength(const SLiquidityZone &zone)
{
    double strength = 0.0;
    
    // Fator de toques (peso 40%)
    strength += (zone.touches * 0.4);
    
    // Fator de idade (peso 20%)
    int age_bars = iBarShift(_Symbol, _Period, zone.time_created);
    double age_factor = MathMax(0.1, 1.0 - (age_bars / 100.0));
    strength += (age_factor * 0.2);
    
    // Fator de volume (peso 40%)
    if(zone.volume_confirmation)
        strength += 0.4;
    
    return strength;
}

//+------------------------------------------------------------------+
//| Verificar se zona é válida                                      |
//+------------------------------------------------------------------+
bool CLiquidityDetector::IsZoneValid(const SLiquidityZone &zone)
{
    // Verificar número mínimo de toques
    if(zone.touches < m_min_touches)
        return false;
    
    // Verificar confirmação de volume se habilitado
    if(m_use_volume_filter && !zone.volume_confirmation)
        return false;
    
    // Verificar força mínima
    if(zone.strength < 0.3)
        return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Verificar confirmação por volume                                 |
//+------------------------------------------------------------------+
bool CLiquidityDetector::CheckVolumeConfirmation(double level, int bar_index)
{
    if(bar_index < 0) return false;
    
    long volume = iVolume(_Symbol, _Period, bar_index);
    long avg_volume = 0;
    
    // Calcular volume médio
    for(int i = 1; i <= 20; i++)
    {
        avg_volume += iVolume(_Symbol, _Period, bar_index + i);
    }
    avg_volume /= 20;
    
    return (volume > avg_volume * 1.5);
}

//+------------------------------------------------------------------+
//| Limpar zonas antigas                                            |
//+------------------------------------------------------------------+
void CLiquidityDetector::CleanupOldZones(void)
{
    int valid_count = 0;
    
    for(int i = 0; i < m_zone_count; i++)
    {
        int age_bars = iBarShift(_Symbol, _Period, m_liquidity_zones[i].time_created);
        
        if(age_bars <= 200 && m_liquidity_zones[i].status == LIQUIDITY_STATUS_ACTIVE)
        {
            if(valid_count != i)
                m_liquidity_zones[valid_count] = m_liquidity_zones[i];
            valid_count++;
        }
    }
    
    m_zone_count = valid_count;
}

//+------------------------------------------------------------------+
//| Ordenar zonas por força                                         |
//+------------------------------------------------------------------+
void CLiquidityDetector::SortZonesByStrength(void)
{
    for(int i = 0; i < m_zone_count - 1; i++)
    {
        for(int j = 0; j < m_zone_count - 1 - i; j++)
        {
            if(m_liquidity_zones[j].strength < m_liquidity_zones[j + 1].strength)
            {
                SLiquidityZone temp = m_liquidity_zones[j];
                m_liquidity_zones[j] = m_liquidity_zones[j + 1];
                m_liquidity_zones[j + 1] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Obter zona por índice                                           |
//+------------------------------------------------------------------+
SLiquidityZone CLiquidityDetector::GetZone(int index) const
{
    SLiquidityZone empty_zone = {0};
    
    if(index >= 0 && index < m_zone_count)
        return m_liquidity_zones[index];
    
    return empty_zone;
}

//+------------------------------------------------------------------+
//| Obter zona mais próxima                                         |
//+------------------------------------------------------------------+
SLiquidityZone CLiquidityDetector::GetNearestZone(double price, ENUM_LIQUIDITY_TYPE type = LIQUIDITY_TYPE_ANY)
{
    SLiquidityZone nearest_zone = {0};
    double min_distance = DBL_MAX;
    
    for(int i = 0; i < m_zone_count; i++)
    {
        if(type != LIQUIDITY_TYPE_ANY && m_liquidity_zones[i].type != type)
            continue;
        
        if(m_liquidity_zones[i].status != LIQUIDITY_STATUS_ACTIVE)
            continue;
        
        double distance = MathAbs(price - m_liquidity_zones[i].level);
        
        if(distance < min_distance)
        {
            min_distance = distance;
            nearest_zone = m_liquidity_zones[i];
        }
    }
    
    return nearest_zone;
}
