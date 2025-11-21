//+------------------------------------------------------------------+
//|                                                   EliteFVG.mqh |
//|                           Autonomous Expert Advisor for XAUUSD Trading |
//|                                       FVG Detection Component |
//+------------------------------------------------------------------+
#property copyright "Developed by Autonomous AI Agent - FTMO Elite Trading System"
#property strict

#include "Definitions.mqh"

// === ELITE FVG DETECTOR CLASS ===
class CEliteFVGDetector
{
private:
    // Detection parameters
    double              m_min_displacement_size;    // Minimum displacement for valid FVG
    double              m_volume_spike_threshold;   // Volume spike threshold
    bool                m_require_structure_break;  // Structure break confirmation
    double              m_min_gap_size;            // Minimum gap size in points
    double              m_max_gap_size;            // Maximum gap size in points
    
    // Quality assessment parameters
    double              m_institutional_threshold;  // Threshold for institutional FVG
    double              m_confluence_weight;        // Weight for confluence scoring
    
    // Multi-timeframe analysis
    ENUM_TIMEFRAME      m_analysis_timeframes[4];   // Analysis timeframes
    int                 m_timeframe_count;          // Number of timeframes
    
    // Data Storage
    SEliteFairValueGap  m_fvgs[50];
    int                 m_fvg_count;
    
public:
    CEliteFVGDetector();
    ~CEliteFVGDetector();
    
    // Accessors
    int GetCount() { return m_fvg_count; }
    SEliteFairValueGap GetFVG(int index) { if(index >= 0 && index < 50) return m_fvgs[index]; SEliteFairValueGap empty; ZeroMemory(empty); return empty; }
    void UpdateFVG(int index, const SEliteFairValueGap& fvg) { if(index >= 0 && index < 50) m_fvgs[index] = fvg; }
    
    // Main detection methods
    bool DetectEliteFairValueGaps();
    bool DetectInstitutionalFVG();
    bool ValidateWithDisplacement(SEliteFairValueGap& fvg);
    
    // Calculation methods
    double CalculateOptimalFillLevel(const SEliteFairValueGap& fvg);
    double CalculateFVGQualityScore(const SEliteFairValueGap& fvg);
    double CalculateExpectedReaction(const SEliteFairValueGap& fvg);
    ENUM_FVG_QUALITY ClassifyFVGQuality(const SEliteFairValueGap& fvg);
    
    // Analysis methods
    bool IsHighProbabilityFVG(const SEliteFairValueGap& fvg);
    bool IsInstitutionalFVG(const SEliteFairValueGap& fvg);
    
    // Helper methods
    bool DetectBullishFVG(const MqlRates& rates[], int index);
    bool DetectBearishFVG(const MqlRates& rates[], int index);
    bool CreateFVGStructure(const MqlRates& rates[], int index, ENUM_FVG_TYPE type, SEliteFairValueGap& fvg);
    bool ValidateFVG(const SEliteFairValueGap& fvg);
    void SortFVGsByQuality();
    
    // Analysis helpers
    double CalculateDisplacementAfterFVG(const MqlRates& rates[], int index, bool is_bullish);
    bool HasVolumeConfirmation(const MqlRates& rates[], int index);
    bool IsInPremiumZone(double price);
    
    // Management methods
    void UpdateFVGStatus();
    void RemoveFilledFVGs();
    int GetActiveFVGCount();
    double GetBestFVGScore();
};

// ... (Implementation) ...

// Main detection method for elite FVGs
bool CEliteFVGDetector::DetectEliteFairValueGaps()
{
    // Reset current FVG count
    m_fvg_count = 0;
    
    // Get market data for analysis
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 100, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Analyze each potential FVG formation (need 3 consecutive candles)
    for(int i = 2; i < 97; i++)
    {
        // Check for bullish FVG pattern
        if(DetectBullishFVG(rates, i))
        {
            SEliteFairValueGap fvg;
            if(CreateFVGStructure(rates, i, FVG_BULLISH, fvg))
            {
                if(ValidateFVG(fvg))
                {
                    m_fvgs[m_fvg_count] = fvg;
                    m_fvg_count++;
                    if(m_fvg_count >= 50) break; // Increased to 50 to match array size
                }
            }
        }
        
        // Check for bearish FVG pattern
        if(DetectBearishFVG(rates, i))
        {
            SEliteFairValueGap fvg;
            if(CreateFVGStructure(rates, i, FVG_BEARISH, fvg))
            {
                if(ValidateFVG(fvg))
                {
                    m_fvgs[m_fvg_count] = fvg;
                    m_fvg_count++;
                    if(m_fvg_count >= 50) break;
                }
            }
        }
    }
    
    // Sort FVGs by quality and proximity
    SortFVGsByQuality();
    
    return m_fvg_count > 0;
}

// Create FVG structure
bool CEliteFVGDetector::CreateFVGStructure(const MqlRates& rates[], int index, ENUM_FVG_TYPE type, SEliteFairValueGap& fvg)
{
    // Initialize FVG structure
    fvg.formation_time = rates[index].time;
    fvg.type = type;
    fvg.state = FVG_STATE_OPEN;
    fvg.fill_percentage = 0.0;
    fvg.touch_count = 0;
    fvg.is_fresh = true;
    fvg.origin_timeframe = PERIOD_M15;
    
    // Set gap boundaries based on type
    if(type == FVG_BULLISH)
    {
        fvg.upper_level = rates[index+1].high;
        fvg.lower_level = rates[index-1].low;
    }
    else
    {
        fvg.upper_level = rates[index-1].high;
        fvg.lower_level = rates[index+1].low;
    }
    
    // Calculate middle level and optimal entry
    fvg.mid_level = (fvg.upper_level + fvg.lower_level) / 2.0;
    
    if(type == FVG_BULLISH)
        fvg.optimal_entry = fvg.lower_level + (fvg.upper_level - fvg.lower_level) * 0.618; // 61.8% level
    else
        fvg.optimal_entry = fvg.upper_level - (fvg.upper_level - fvg.lower_level) * 0.618; // 61.8% level
    
    // Calculate gap size
    fvg.gap_size_points = (fvg.upper_level - fvg.lower_level) / _Point;
    
    // Calculate advanced properties
    fvg.displacement_size = CalculateDisplacementAfterFVG(rates, index, type == FVG_BULLISH);
    fvg.has_volume_spike = HasVolumeConfirmation(rates, index);
    
    // Confluence factors calculated externally
    fvg.has_ob_confluence = false;
    fvg.has_liquidity_confluence = false;
    fvg.has_structure_confluence = false;
    fvg.confluence_score = 0.0;
    fvg.confluence_count = 0;
    
    fvg.quality_score = CalculateFVGQualityScore(fvg);
    fvg.expected_reaction = CalculateExpectedReaction(fvg);
    fvg.quality = ClassifyFVGQuality(fvg);
    
    // Determine institutional characteristics
    fvg.is_institutional = IsInstitutionalFVG(fvg);
    
    // Set timing properties
    fvg.age_in_bars = 0;
    fvg.expiry_time = fvg.formation_time + 24*3600; // 24 hours expiry
    fvg.time_decay_factor = 1.0;
    
    // Determine premium/discount location
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    fvg.is_in_premium = IsInPremiumZone(current_price);
    
    return true;
}

// Calculate FVG quality score
double CEliteFVGDetector::CalculateFVGQualityScore(const SEliteFairValueGap& fvg)
{
    double score = 50.0; // Base score
    
    // Gap size factor (optimal around 30-100 points for XAUUSD)
    if(fvg.gap_size_points >= 30 && fvg.gap_size_points <= 100)
        score += 20.0;
    else if(fvg.gap_size_points >= 20 && fvg.gap_size_points <= 150)
        score += 15.0;
    else if(fvg.gap_size_points >= 15 && fvg.gap_size_points <= 200)
        score += 10.0;
    
    // Displacement factor
    double displacement_points = fvg.displacement_size / _Point;
    if(displacement_points >= 200) score += 20.0;
    else if(displacement_points >= 150) score += 15.0;
    else if(displacement_points >= 100) score += 10.0;
    
    // Volume confirmation
    if(fvg.has_volume_spike) score += 15.0;
    
    // Institutional characteristics
    if(fvg.is_institutional) score += 10.0;
    
    // Confluence score (added externally)
    score += fvg.confluence_score * 0.15;
    
    return MathMin(score, 100.0);
}


// Calculate expected reaction probability
double CEliteFVGDetector::CalculateExpectedReaction(const SEliteFairValueGap& fvg)
{
    double reaction = 0.5; // Base 50% probability
    
    // Quality bonus
    reaction += fvg.quality_score * 0.005; // Up to 50% bonus from quality
    
    // Gap size factor
    if(fvg.gap_size_points >= 30 && fvg.gap_size_points <= 80)
        reaction += 0.2; // Optimal size range
    
    // Institutional bonus
    if(fvg.is_institutional) reaction += 0.15;
    
    // Fresh FVG bonus
    if(fvg.is_fresh) reaction += 0.1;
    
    // Volume confirmation bonus
    if(fvg.has_volume_spike) reaction += 0.1;
    
    // Premium/discount consideration
    if((fvg.type == FVG_BULLISH && !fvg.is_in_premium) ||
       (fvg.type == FVG_BEARISH && fvg.is_in_premium))
    {
        reaction += 0.15; // Trading from appropriate zones
    }
    
    return MathMin(reaction, 1.0);
}

// Classify FVG quality
ENUM_FVG_QUALITY CEliteFVGDetector::ClassifyFVGQuality(const SEliteFairValueGap& fvg)
{
    if(fvg.quality_score >= 90.0 && fvg.is_institutional) return FVG_QUALITY_ELITE;
    if(fvg.quality_score >= 80.0) return FVG_QUALITY_HIGH;
    if(fvg.quality_score >= 65.0) return FVG_QUALITY_MEDIUM;
    return FVG_QUALITY_LOW;
}

// Check if FVG has institutional characteristics
bool CEliteFVGDetector::IsInstitutionalFVG(const SEliteFairValueGap& fvg)
{
    // Large gap size
    if(fvg.gap_size_points < 200) return false; // 20 pips minimum for institutional
    
    // Strong displacement
    if(fvg.displacement_size < m_institutional_threshold) return false;
    
    // Volume spike confirmation
    if(!fvg.has_volume_spike) return false;
    
    // High quality score
    if(fvg.quality_score < 80.0) return false;
    
    return true;
}

// Calculate confluence score
void CEliteFVGDetector::CalculateConfluenceScore(SEliteFairValueGap& fvg)
{
    double score = 0.0;
    
    // Check Order Block confluence
    fvg.has_ob_confluence = CheckOrderBlockConfluence(fvg);
    if(fvg.has_ob_confluence) score += 35.0;
    
    // Check Liquidity confluence
    fvg.has_liquidity_confluence = CheckLiquidityConfluence(fvg);
    if(fvg.has_liquidity_confluence) score += 30.0;
    
    // Check Structure confluence
    fvg.has_structure_confluence = CheckStructureConfluence(fvg);
    if(fvg.has_structure_confluence) score += 35.0;
    
    fvg.confluence_score = score;
    fvg.confluence_count = (fvg.has_ob_confluence ? 1 : 0) +
                          (fvg.has_liquidity_confluence ? 1 : 0) +
                          (fvg.has_structure_confluence ? 1 : 0);
}

// Helper methods implementation
double CEliteFVGDetector::CalculateDisplacementAfterFVG(const MqlRates& rates[], int index, bool is_bullish)
{
    if(index < 5) return 0.0;
    
    double max_displacement = 0.0;
    double reference_price = is_bullish ? rates[index].close : rates[index].close;
    
    // Check displacement in next 5 candles
    for(int i = index - 1; i >= MathMax(0, index - 5); i--)
    {
        double displacement = 0;
        if(is_bullish)
            displacement = rates[i].high - reference_price;
        else
            displacement = reference_price - rates[i].low;
            
        if(displacement > max_displacement)
            max_displacement = displacement;
    }
    
    return max_displacement;
}

bool CEliteFVGDetector::HasVolumeConfirmation(const MqlRates& rates[], int index)
{
    if(index >= ArraySize(rates) - 2) return true; // Default if insufficient data
    
    // Check volume spike in the FVG formation period
    long total_volume = rates[index-1].tick_volume + rates[index].tick_volume + rates[index+1].tick_volume;
    long avg_volume = 0;
    
    // Calculate average volume of previous 10 bars
    for(int i = index + 2; i < index + 12 && i < ArraySize(rates); i++)
    {
        avg_volume += rates[i].tick_volume;
    }
    avg_volume /= 10;
    
    return total_volume > avg_volume * m_volume_spike_threshold;
}

bool CEliteFVGDetector::CheckOrderBlockConfluence(const SEliteFairValueGap& fvg)
{
    // Check if there's an Order Block near the FVG
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        const SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        if(ob.state == OB_STATE_DISABLED || ob.state == OB_STATE_MITIGATED) continue;
        
        // Check if FVG overlaps or is near Order Block
        double distance = MathMin(
            MathAbs(fvg.upper_level - ob.high_price),
            MathAbs(fvg.lower_level - ob.low_price)
        );
        
        if(distance <= 30 * _Point) // Within 3 pips
            return true;
    }
    return false;
}

bool CEliteFVGDetector::CheckLiquidityConfluence(const SEliteFairValueGap& fvg)
{
    // Check if there's liquidity near the FVG
    for(int i = 0; i < g_institutional_liq_count; i++)
    {
        double distance = MathMin(
            MathAbs(fvg.upper_level - g_institutional_liquidity[i].price_level),
            MathAbs(fvg.lower_level - g_institutional_liquidity[i].price_level)
        );
        
        if(distance <= 25 * _Point) // Within 2.5 pips
            return true;
    }
    return false;
}

bool CEliteFVGDetector::CheckStructureConfluence(const SEliteFairValueGap& fvg)
{
    // Check if FVG aligns with market structure
    MqlRates rates[50];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 50, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Simple trend alignment check
    bool uptrend = rates[0].close > rates[10].close;
    
    if(fvg.type == FVG_BULLISH && uptrend) return true;
    if(fvg.type == FVG_BEARISH && !uptrend) return true;
    
    return false;
}

void CEliteFVGDetector::SortFVGsByQuality()
{
    // Simple bubble sort by quality score
    for(int i = 0; i < g_elite_fvg_count - 1; i++)
    {
        for(int j = 0; j < g_elite_fvg_count - i - 1; j++)
        {
            if(g_elite_fair_value_gaps[j].quality_score < g_elite_fair_value_gaps[j + 1].quality_score)
            {
                SEliteFairValueGap temp = g_elite_fair_value_gaps[j];
                g_elite_fair_value_gaps[j] = g_elite_fair_value_gaps[j + 1];
                g_elite_fair_value_gaps[j + 1] = temp;
            }
        }
    }
}

// Update FVG status
void CEliteFVGDetector::UpdateFVGStatus()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        SEliteFairValueGap& fvg = g_elite_fair_value_gaps[i];
        
        // Check if price has entered the FVG
        if(current_price >= fvg.lower_level && current_price <= fvg.upper_level)
        {
            fvg.touch_count++;
            fvg.is_fresh = false;
            
            // Calculate fill percentage
            if(fvg.type == FVG_BULLISH)
            {
                fvg.fill_percentage = (current_price - fvg.lower_level) / (fvg.upper_level - fvg.lower_level) * 100.0;
            }
            else
            {
                fvg.fill_percentage = (fvg.upper_level - current_price) / (fvg.upper_level - fvg.lower_level) * 100.0;
            }
            
            // Update state based on fill percentage
            if(fvg.fill_percentage >= 100.0)
                fvg.state = FVG_STATE_FILLED;
            else if(fvg.fill_percentage >= 50.0)
                fvg.state = FVG_STATE_PARTIAL;
        }
        
        // Age-based expiry (24 hours for M15 timeframe)
        fvg.age_in_bars = (int)((TimeCurrent() - fvg.formation_time) / (15 * 60)); // 15 minutes per bar
        
        if(TimeCurrent() > fvg.expiry_time && fvg.touch_count == 0)
        {
            fvg.state = FVG_STATE_EXPIRED;
        }
        
        // Calculate time decay factor
        double age_hours = (TimeCurrent() - fvg.formation_time) / 3600.0;
        fvg.time_decay_factor = MathMax(0.1, 1.0 - (age_hours / 24.0)); // Decay over 24 hours
    }
}

// Remove filled/expired FVGs
void CEliteFVGDetector::RemoveFilledFVGs()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state != FVG_STATE_FILLED &&
           g_elite_fair_value_gaps[i].state != FVG_STATE_EXPIRED)
        {
            if(valid_count != i)
            {
                g_elite_fair_value_gaps[valid_count] = g_elite_fair_value_gaps[i];
            }
            valid_count++;
        }
    }
    
    g_elite_fvg_count = valid_count;
}

// Get active FVG count
int CEliteFVGDetector::GetActiveFVGCount()
{
    int active_count = 0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state == FVG_STATE_OPEN ||
           g_elite_fair_value_gaps[i].state == FVG_STATE_PARTIAL)
        {
            active_count++;
        }
    }
    
    return active_count;
}

// Get best FVG score
double CEliteFVGDetector::GetBestFVGScore()
{
    double best_score = 0.0;
    
    for(int i = 0; i < g_elite_fvg_count; i++)
    {
        if(g_elite_fair_value_gaps[i].state == FVG_STATE_OPEN ||
           g_elite_fair_value_gaps[i].state == FVG_STATE_PARTIAL)
        {
            if(g_elite_fair_value_gaps[i].quality_score > best_score)
                best_score = g_elite_fair_value_gaps[i].quality_score;
        }
    }
    
    return best_score;
}

// Placeholder methods for missing implementations
bool CEliteFVGDetector::DetectInstitutionalFVG() { return false; }
bool CEliteFVGDetector::ValidateWithDisplacement(SEliteFairValueGap& fvg) { return true; }
double CEliteFVGDetector::CalculateOptimalFillLevel(const SEliteFairValueGap& fvg) { return fvg.mid_level; }
bool CEliteFVGDetector::IsHighProbabilityFVG(const SEliteFairValueGap& fvg) { return fvg.quality_score > 80.0; }
bool CEliteFVGDetector::IsInPremiumZone(double price) { return false; }
