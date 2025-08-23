//+------------------------------------------------------------------+
//|                                              FVGDetector.mqh |
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
//| Fair Value Gap (FVG) Detector Class                             |
//+------------------------------------------------------------------+
class CFVGDetector : public IDetector
{
private:
    // Configurações
    int               m_min_gap_points;     // Mínimo de pontos para FVG válido
    double            m_min_gap_ratio;      // Ratio mínimo do gap
    int               m_max_age_bars;       // Idade máxima em barras
    bool              m_filter_by_volume;   // Filtrar por volume
    
    // Arrays de dados
    SFVG              m_fvgs[];             // Array de FVGs detectados
    int               m_fvg_count;          // Contador de FVGs
    
    // Cache
    datetime          m_last_check_time;
    bool              m_cache_valid;
    
public:
    // Construtor/Destrutor
                     CFVGDetector(void);
                    ~CFVGDetector(void);
    
    // Métodos principais
    virtual bool      Initialize(void) override;
    virtual bool      Update(void) override;
    virtual void      Reset(void) override;
    
    // Configuração
    void              SetMinGapPoints(int points) { m_min_gap_points = points; }
    void              SetMinGapRatio(double ratio) { m_min_gap_ratio = ratio; }
    void              SetMaxAge(int bars) { m_max_age_bars = bars; }
    void              SetVolumeFilter(bool enable) { m_filter_by_volume = enable; }
    
    // Detecção de FVGs
    bool              DetectFVGs(int start_bar = 1, int bars_count = 100);
    bool              IsFVGValid(const SFVG &fvg);
    double            CalculateFVGStrength(const SFVG &fvg);
    
    // Análise de FVGs
    bool              IsFVGFilled(const SFVG &fvg, int current_bar = 0);
    double            GetFVGFillPercentage(const SFVG &fvg, int current_bar = 0);
    ENUM_FVG_STATUS   GetFVGStatus(const SFVG &fvg, int current_bar = 0);
    
    // Getters
    int               GetFVGCount(void) const { return m_fvg_count; }
    SFVG              GetFVG(int index) const;
    SFVG              GetNearestFVG(double price, ENUM_FVG_TYPE type = FVG_TYPE_ANY);
    
    // Filtros
    bool              FilterFVGsByTimeframe(ENUM_TIMEFRAMES tf);
    bool              FilterFVGsByStrength(double min_strength);
    
private:
    // Métodos auxiliares
    bool              IsGapValid(double high1, double low1, double high2, double low2, double high3, double low3);
    ENUM_FVG_TYPE     DetermineFVGType(double high1, double low1, double high2, double low2, double high3, double low3);
    double            CalculateGapSize(const SFVG &fvg);
    bool              CheckVolumeConfirmation(int bar_index);
    void              CleanupOldFVGs(void);
    void              SortFVGsByStrength(void);
};

//+------------------------------------------------------------------+
//| Construtor                                                       |
//+------------------------------------------------------------------+
CFVGDetector::CFVGDetector(void) :
    m_min_gap_points(10),
    m_min_gap_ratio(0.001),
    m_max_age_bars(100),
    m_filter_by_volume(true),
    m_fvg_count(0),
    m_last_check_time(0),
    m_cache_valid(false)
{
    ArrayResize(m_fvgs, 100);
    ArrayInitialize(m_fvgs, 0);
}

//+------------------------------------------------------------------+
//| Destrutor                                                        |
//+------------------------------------------------------------------+
CFVGDetector::~CFVGDetector(void)
{
    ArrayFree(m_fvgs);
}

//+------------------------------------------------------------------+
//| Inicialização                                                    |
//+------------------------------------------------------------------+
bool CFVGDetector::Initialize(void)
{
    CLogger::Log(LOG_INFO, "CFVGDetector", "Inicializando detector de FVG...");
    
    // Verificar dados suficientes
    if(Bars(_Symbol, _Period) < 10)
    {
        CLogger::Log(LOG_ERROR, "CFVGDetector", "Dados insuficientes para análise");
        return false;
    }
    
    Reset();
    
    CLogger::Log(LOG_INFO, "CFVGDetector", "Detector de FVG inicializado com sucesso");
    return true;
}

//+------------------------------------------------------------------+
//| Atualização                                                      |
//+------------------------------------------------------------------+
bool CFVGDetector::Update(void)
{
    datetime current_time = TimeCurrent();
    
    // Verificar se precisa atualizar
    if(m_cache_valid && current_time == m_last_check_time)
        return true;
    
    // Detectar novos FVGs
    bool result = DetectFVGs();
    
    if(result)
    {
        CleanupOldFVGs();
        SortFVGsByStrength();
        
        m_last_check_time = current_time;
        m_cache_valid = true;
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Reset                                                            |
//+------------------------------------------------------------------+
void CFVGDetector::Reset(void)
{
    m_fvg_count = 0;
    m_last_check_time = 0;
    m_cache_valid = false;
    ArrayInitialize(m_fvgs, 0);
    
    CLogger::Log(LOG_INFO, "CFVGDetector", "Detector resetado");
}

//+------------------------------------------------------------------+
//| Detectar FVGs                                                    |
//+------------------------------------------------------------------+
bool CFVGDetector::DetectFVGs(int start_bar = 1, int bars_count = 100)
{
    int bars_available = Bars(_Symbol, _Period);
    if(bars_available < 3) return false;
    
    int end_bar = MathMin(start_bar + bars_count, bars_available - 2);
    m_fvg_count = 0;
    
    for(int i = start_bar; i < end_bar; i++)
    {
        // Obter dados das 3 barras
        double high1 = iHigh(_Symbol, _Period, i+1);
        double low1 = iLow(_Symbol, _Period, i+1);
        double high2 = iHigh(_Symbol, _Period, i);
        double low2 = iLow(_Symbol, _Period, i);
        double high3 = iHigh(_Symbol, _Period, i-1);
        double low3 = iLow(_Symbol, _Period, i-1);
        
        // Verificar se há gap válido
        if(IsGapValid(high1, low1, high2, low2, high3, low3))
        {
            SFVG fvg;
            fvg.type = DetermineFVGType(high1, low1, high2, low2, high3, low3);
            fvg.time_created = iTime(_Symbol, _Period, i);
            fvg.bar_index = i;
            
            if(fvg.type == FVG_TYPE_BULLISH)
            {
                fvg.upper_level = low3;
                fvg.lower_level = high1;
            }
            else if(fvg.type == FVG_TYPE_BEARISH)
            {
                fvg.upper_level = low1;
                fvg.lower_level = high3;
            }
            
            fvg.strength = CalculateFVGStrength(fvg);
            fvg.status = FVG_STATUS_ACTIVE;
            fvg.fill_percentage = 0.0;
            
            if(IsFVGValid(fvg))
            {
                if(m_fvg_count < ArraySize(m_fvgs))
                {
                    m_fvgs[m_fvg_count] = fvg;
                    m_fvg_count++;
                }
            }
        }
    }
    
    CLogger::Log(LOG_INFO, "CFVGDetector", StringFormat("Detectados %d FVGs", m_fvg_count));
    return true;
}

//+------------------------------------------------------------------+
//| Verificar se gap é válido                                       |
//+------------------------------------------------------------------+
bool CFVGDetector::IsGapValid(double high1, double low1, double high2, double low2, double high3, double low3)
{
    // FVG Bullish: low3 > high1
    if(low3 > high1)
    {
        double gap_size = low3 - high1;
        double gap_points = gap_size / _Point;
        return (gap_points >= m_min_gap_points);
    }
    
    // FVG Bearish: high3 < low1
    if(high3 < low1)
    {
        double gap_size = low1 - high3;
        double gap_points = gap_size / _Point;
        return (gap_points >= m_min_gap_points);
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Determinar tipo de FVG                                          |
//+------------------------------------------------------------------+
ENUM_FVG_TYPE CFVGDetector::DetermineFVGType(double high1, double low1, double high2, double low2, double high3, double low3)
{
    if(low3 > high1) return FVG_TYPE_BULLISH;
    if(high3 < low1) return FVG_TYPE_BEARISH;
    return FVG_TYPE_NONE;
}

//+------------------------------------------------------------------+
//| Calcular força do FVG                                           |
//+------------------------------------------------------------------+
double CFVGDetector::CalculateFVGStrength(const SFVG &fvg)
{
    double gap_size = MathAbs(fvg.upper_level - fvg.lower_level);
    double atr = iATR(_Symbol, _Period, 14, fvg.bar_index);
    
    if(atr > 0)
        return gap_size / atr;
    
    return 1.0;
}

//+------------------------------------------------------------------+
//| Verificar se FVG é válido                                       |
//+------------------------------------------------------------------+
bool CFVGDetector::IsFVGValid(const SFVG &fvg)
{
    // Verificar tamanho mínimo
    double gap_size = MathAbs(fvg.upper_level - fvg.lower_level);
    if(gap_size < m_min_gap_points * _Point)
        return false;
    
    // Verificar força mínima
    if(fvg.strength < m_min_gap_ratio)
        return false;
    
    // Verificar confirmação por volume se habilitado
    if(m_filter_by_volume && !CheckVolumeConfirmation(fvg.bar_index))
        return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Verificar confirmação por volume                                 |
//+------------------------------------------------------------------+
bool CFVGDetector::CheckVolumeConfirmation(int bar_index)
{
    long volume = iVolume(_Symbol, _Period, bar_index);
    long avg_volume = 0;
    
    // Calcular volume médio das últimas 20 barras
    for(int i = 1; i <= 20; i++)
    {
        avg_volume += iVolume(_Symbol, _Period, bar_index + i);
    }
    avg_volume /= 20;
    
    return (volume > avg_volume * 1.2); // Volume 20% acima da média
}

//+------------------------------------------------------------------+
//| Limpar FVGs antigos                                             |
//+------------------------------------------------------------------+
void CFVGDetector::CleanupOldFVGs(void)
{
    datetime current_time = TimeCurrent();
    int valid_count = 0;
    
    for(int i = 0; i < m_fvg_count; i++)
    {
        // Verificar idade
        int bars_age = iBarShift(_Symbol, _Period, m_fvgs[i].time_created);
        
        if(bars_age <= m_max_age_bars && m_fvgs[i].status != FVG_STATUS_FILLED)
        {
            if(valid_count != i)
                m_fvgs[valid_count] = m_fvgs[i];
            valid_count++;
        }
    }
    
    m_fvg_count = valid_count;
}

//+------------------------------------------------------------------+
//| Ordenar FVGs por força                                          |
//+------------------------------------------------------------------+
void CFVGDetector::SortFVGsByStrength(void)
{
    // Bubble sort simples por força (decrescente)
    for(int i = 0; i < m_fvg_count - 1; i++)
    {
        for(int j = 0; j < m_fvg_count - 1 - i; j++)
        {
            if(m_fvgs[j].strength < m_fvgs[j + 1].strength)
            {
                SFVG temp = m_fvgs[j];
                m_fvgs[j] = m_fvgs[j + 1];
                m_fvgs[j + 1] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Obter FVG por índice                                            |
//+------------------------------------------------------------------+
SFVG CFVGDetector::GetFVG(int index) const
{
    SFVG empty_fvg = {0};
    
    if(index >= 0 && index < m_fvg_count)
        return m_fvgs[index];
    
    return empty_fvg;
}

//+------------------------------------------------------------------+
//| Obter FVG mais próximo                                          |
//+------------------------------------------------------------------+
SFVG CFVGDetector::GetNearestFVG(double price, ENUM_FVG_TYPE type = FVG_TYPE_ANY)
{
    SFVG nearest_fvg = {0};
    double min_distance = DBL_MAX;
    
    for(int i = 0; i < m_fvg_count; i++)
    {
        if(type != FVG_TYPE_ANY && m_fvgs[i].type != type)
            continue;
        
        if(m_fvgs[i].status != FVG_STATUS_ACTIVE)
            continue;
        
        double center = (m_fvgs[i].upper_level + m_fvgs[i].lower_level) / 2.0;
        double distance = MathAbs(price - center);
        
        if(distance < min_distance)
        {
            min_distance = distance;
            nearest_fvg = m_fvgs[i];
        }
    }
    
    return nearest_fvg;
}
