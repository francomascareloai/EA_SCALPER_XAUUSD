//+------------------------------------------------------------------+
//|                                              EliteOrderBlock.mqh |
//|                           Autonomous Expert Advisor for XAUUSD Trading |
//|                                     Order Block Detection Component |
//+------------------------------------------------------------------+
#property copyright "Developed by Autonomous AI Agent - FTMO Elite Trading System"
#property strict

#include "Definitions.mqh"

// === ELITE ORDER BLOCK DETECTOR CLASS ===
class CEliteOrderBlockDetector
{
private:
    // Detection parameters
    double              m_displacement_threshold;   // Minimum displacement for valid OB
    double              m_volume_threshold;         // Volume threshold for validation
    bool                m_require_structure_break;  // Structure break confirmation
    bool                m_use_liquidity_confirmation; // Liquidity confirmation
    bool                m_use_volume_confirmation;  // Volume confirmation
    
    // Multi-timeframe analysis
    ENUM_TIMEFRAME      m_analysis_timeframes[4];   // Analysis timeframes
    int                 m_timeframe_count;          // Number of timeframes
    
    // Data Storage
    SAdvancedOrderBlock m_order_blocks[50];
    int                 m_ob_count;
    
public:
    CEliteOrderBlockDetector();
    ~CEliteOrderBlockDetector();
    
    // Accessors
    int GetCount() { return m_ob_count; }
    SAdvancedOrderBlock GetOrderBlock(int index) { if(index >= 0 && index < 50) return m_order_blocks[index]; SAdvancedOrderBlock empty; ZeroMemory(empty); return empty; }
    // Helper to update OB in array
    void UpdateOrderBlock(int index, const SAdvancedOrderBlock& ob) { if(index >= 0 && index < 50) m_order_blocks[index] = ob; }
    
    // Main detection methods
    bool DetectEliteOrderBlocks();
    bool DetectPremiumOrderBlocks();
    bool DetectDiscountOrderBlocks();
    
    // Validation methods
    bool ValidateWithLiquidity(SAdvancedOrderBlock& ob);
    bool ValidateWithVolume(SAdvancedOrderBlock& ob);
    bool ValidateWithStructure(SAdvancedOrderBlock& ob);
    
    // Calculation methods
    double CalculateOptimalEntry(const SAdvancedOrderBlock& ob);
    double CalculateOrderBlockStrength(const SAdvancedOrderBlock& ob);
    double CalculateProbabilityScore(const SAdvancedOrderBlock& ob);
    ENUM_OB_QUALITY ClassifyOrderBlockQuality(const SAdvancedOrderBlock& ob);
    
    // Helper methods
    bool DetectBullishOrderBlock(const MqlRates& rates[], int index);
    bool DetectBearishOrderBlock(const MqlRates& rates[], int index);
    bool CreateOrderBlockStructure(const MqlRates& rates[], int index, ENUM_ORDER_BLOCK_TYPE type, SAdvancedOrderBlock& ob);
    bool ValidateOrderBlock(const SAdvancedOrderBlock& ob);
    void SortOrderBlocksByQuality();
    
    // Analysis helpers
    double CalculateAverageBodySize(const MqlRates& rates[], int index, int period);
    bool HasVolumeSpike(const MqlRates& rates[], int index);
    double CalculateDisplacementSize(const MqlRates& rates[], int index);
    double CalculateVolumeProfile(const MqlRates& rates[], int index);
    double CalculateReactionQuality(const MqlRates& rates[], int index);
    bool IsInstitutionalOrderBlock(const SAdvancedOrderBlock& ob);
    bool IsInPremiumZone(double price);
    bool IsInDiscountZone(double price);
    
    // Management methods
    void UpdateOrderBlockStatus();
    void RemoveInvalidOrderBlocks();
    int GetActiveOrderBlockCount();
};

// ... (Implementation) ...

// Create order block structure
bool CEliteOrderBlockDetector::CreateOrderBlockStructure(const MqlRates& rates[], int index, ENUM_ORDER_BLOCK_TYPE type, SAdvancedOrderBlock& ob)
{
    // Initialize order block structure
    ob.formation_time = rates[index].time;
    ob.high_price = rates[index].high;
    ob.low_price = rates[index].low;
    ob.type = type;
    ob.state = OB_STATE_ACTIVE;
    ob.is_fresh = true;
    ob.touch_count = 0;
    ob.origin_timeframe = PERIOD_M15;
    
    // Calculate refined entry
    if(type == OB_BULLISH)
    {
        ob.refined_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    else
    {
        ob.refined_entry = ob.high_price - (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    
    // Calculate advanced properties
    ob.displacement_size = CalculateDisplacementSize(rates, index);
    ob.strength = CalculateOrderBlockStrength(ob);
    ob.volume_profile = CalculateVolumeProfile(rates, index);
    ob.reaction_quality = CalculateReactionQuality(rates, index);
    ob.probability_score = CalculateProbabilityScore(ob);
    ob.quality = ClassifyOrderBlockQuality(ob);
    
    // Determine premium/discount zone
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    ob.is_premium = IsInPremiumZone(current_price);
    
    // Check institutional characteristics
    ob.is_institutional = IsInstitutionalOrderBlock(ob);
    
    // Confluence factors will be calculated externally to avoid circular dependencies
    ob.has_fvg_confluence = false;
    ob.has_liquidity_confluence = false;
    ob.has_structure_confluence = false;
    ob.confluence_score = 0.0;
    
    return true;
}

// Calculate order block strength
double CEliteOrderBlockDetector::CalculateOrderBlockStrength(const SAdvancedOrderBlock& ob)
{
    double strength = 0.0;
    
    // Base strength from displacement size
    strength += MathMin(ob.displacement_size / (100 * _Point), 30.0); // Max 30 points
    
    // Volume strength (if available)
    strength += MathMin(ob.volume_profile * 20.0, 20.0); // Max 20 points
    
    // Reaction quality
    strength += ob.reaction_quality * 25.0; // Max 25 points
    
    // Institutional characteristics
    if(ob.is_institutional) strength += 15.0;
    
    // Confluence bonus (added externally)
    if(ob.has_fvg_confluence) strength += 5.0;
    if(ob.has_liquidity_confluence) strength += 5.0;
    if(ob.has_structure_confluence) strength += 5.0;
    
    return MathMin(strength, 100.0);
}



//+------------------------------------------------------------------+
//| Elite Order Block Detector Implementation                        |
//+------------------------------------------------------------------+

// Constructor
CEliteOrderBlockDetector::CEliteOrderBlockDetector()
{
    // Initialize detection parameters for XAUUSD
    m_displacement_threshold = 200.0 * _Point;  // 20 pips displacement minimum
    m_volume_threshold = 1.5;                   // 1.5x average volume
    m_require_structure_break = true;           // Require structure break
    m_use_liquidity_confirmation = true;        // Use liquidity confirmation
    m_use_volume_confirmation = true;           // Use volume confirmation
    
    // Set analysis timeframes
    m_analysis_timeframes[0] = PERIOD_H4;
    m_analysis_timeframes[1] = PERIOD_H1;
    m_analysis_timeframes[2] = PERIOD_M15;
    m_analysis_timeframes[3] = PERIOD_M5;
    m_timeframe_count = 4;
    
    m_ob_count = 0;
}

// Destructor
CEliteOrderBlockDetector::~CEliteOrderBlockDetector()
{
    // Cleanup if needed
}

// Main detection method for elite order blocks
bool CEliteOrderBlockDetector::DetectEliteOrderBlocks()
{
    // Reset current order blocks count
    m_ob_count = 0;
    
    // Get market data for analysis
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_M15, 0, 100, rates) <= 0)
        return false;
        
    ArraySetAsSeries(rates, true);
    
    // Analyze each potential order block formation
    for(int i = 5; i < 95; i++)
    {
        // Check for bullish order block pattern
        if(DetectBullishOrderBlock(rates, i))
        {
            SAdvancedOrderBlock ob;
            if(CreateOrderBlockStructure(rates, i, OB_BULLISH, ob))
            {
                if(ValidateOrderBlock(ob))
                {
                    m_order_blocks[m_ob_count] = ob;
                    m_ob_count++;
                    if(m_ob_count >= 50) break;
                }
            }
        }
        
        // Check for bearish order block pattern
        if(DetectBearishOrderBlock(rates, i))
        {
            SAdvancedOrderBlock ob;
            if(CreateOrderBlockStructure(rates, i, OB_BEARISH, ob))
            {
                if(ValidateOrderBlock(ob))
                {
                    m_order_blocks[m_ob_count] = ob;
                    m_ob_count++;
                    if(m_ob_count >= 50) break;
                }
            }
        }
    }
    
    // Sort order blocks by quality and proximity
    SortOrderBlocksByQuality();
    
    return m_ob_count > 0;
}

// Detect bullish order block formation
bool CEliteOrderBlockDetector::DetectBullishOrderBlock(const MqlRates& rates[], int index)
{
    if(index < 3 || index >= ArraySize(rates) - 2) return false;
    
    // Look for bullish order block pattern:
    // 1. Strong bullish candle (displacement)
    // 2. Small retracement/consolidation 
    // 3. Strong move away (imbalance creation)
    
    double current_body = rates[index].close - rates[index].open;
    double prev_body = rates[index+1].close - rates[index+1].open;
    double next_displacement = 0;
    
    // Calculate displacement after the order block
    for(int j = index-1; j >= MathMax(0, index-5); j--)
    {
        if(rates[j].close > rates[index].high)
        {
            next_displacement = rates[j].close - rates[index].high;
            break;
        }
    }
    
    // Bullish order block criteria:
    // 1. Current candle is bullish
    if(current_body <= 0) return false;
    
    // 2. Strong bullish body (at least 50% of range)
    double total_range = rates[index].high - rates[index].low;
    if(total_range > 0 && current_body < total_range * 0.5) return false;
    
    // 3. Significant size relative to recent candles
    double avg_body = CalculateAverageBodySize(rates, index, 10);
    if(current_body < avg_body * 1.5) return false;
    
    // 4. Must have displacement after formation
    if(next_displacement < m_displacement_threshold) return false;
    
    // 5. Volume confirmation (if available)
    if(m_use_volume_confirmation)
    {
        if(!HasVolumeSpike(rates, index)) return false;
    }
    
    return true;
}

// Detect bearish order block formation  
bool CEliteOrderBlockDetector::DetectBearishOrderBlock(const MqlRates& rates[], int index)
{
    if(index < 3 || index >= ArraySize(rates) - 2) return false;
    
    double current_body = rates[index].open - rates[index].close;
    double next_displacement = 0;
    
    // Calculate displacement after the order block
    for(int j = index-1; j >= MathMax(0, index-5); j--)
    {
        if(rates[j].close < rates[index].low)
        {
            next_displacement = rates[index].low - rates[j].close;
            break;
        }
    }
    
    // Bearish order block criteria:
    // 1. Current candle is bearish
    if(current_body <= 0) return false;
    
    // 2. Strong bearish body (at least 50% of range)
    double total_range = rates[index].high - rates[index].low;
    if(total_range > 0 && current_body < total_range * 0.5) return false;
    
    // 3. Significant size relative to recent candles
    double avg_body = CalculateAverageBodySize(rates, index, 10);
    if(current_body < avg_body * 1.5) return false;
    
    // 4. Must have displacement after formation
    if(next_displacement < m_displacement_threshold) return false;
    
    // 5. Volume confirmation (if available)
    if(m_use_volume_confirmation)
    {
        if(!HasVolumeSpike(rates, index)) return false;
    }
    
    return true;
}

// Create order block structure
bool CEliteOrderBlockDetector::CreateOrderBlockStructure(const MqlRates& rates[], int index, ENUM_ORDER_BLOCK_TYPE type, SAdvancedOrderBlock& ob)
{
    // Initialize order block structure
    ob.formation_time = rates[index].time;
    ob.high_price = rates[index].high;
    ob.low_price = rates[index].low;
    ob.type = type;
    ob.state = OB_STATE_ACTIVE;
    ob.is_fresh = true;
    ob.touch_count = 0;
    ob.origin_timeframe = PERIOD_M15;
    
    // Calculate refined entry
    if(type == OB_BULLISH)
    {
        ob.refined_entry = ob.low_price + (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    else
    {
        ob.refined_entry = ob.high_price - (ob.high_price - ob.low_price) * 0.5; // 50% level
    }
    
    // Calculate advanced properties
    ob.displacement_size = CalculateDisplacementSize(rates, index);
    ob.strength = CalculateOrderBlockStrength(ob);
    ob.volume_profile = CalculateVolumeProfile(rates, index);
    ob.reaction_quality = CalculateReactionQuality(rates, index);
    ob.probability_score = CalculateProbabilityScore(ob);
    ob.quality = ClassifyOrderBlockQuality(ob);
    
    // Determine premium/discount zone
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    ob.is_premium = IsInPremiumZone(current_price);
    
    // Check institutional characteristics
    ob.is_institutional = IsInstitutionalOrderBlock(ob);
    
    // Calculate confluence factors
    ob.has_fvg_confluence = CheckFVGConfluence(ob);
    ob.has_liquidity_confluence = CheckLiquidityConfluence(ob);
    ob.has_structure_confluence = CheckStructureConfluence(ob);
    ob.confluence_score = CalculateConfluenceScore(ob);
    
    return true;
}

// Validate order block quality and requirements
bool CEliteOrderBlockDetector::ValidateOrderBlock(const SAdvancedOrderBlock& ob)
{
    // Minimum strength requirement
    if(ob.strength < 60.0) return false;
    
    // Minimum probability score
    if(ob.probability_score < 70.0) return false;
    
    // Must be at least medium quality
    if(ob.quality < OB_QUALITY_MEDIUM) return false;
    
    // Must have minimum confluence score
    if(ob.confluence_score < 65.0) return false;
    
    // Additional validations
    if(m_use_liquidity_confirmation && !ob.has_liquidity) return false;
    
    return true;
}

// Calculate order block strength
double CEliteOrderBlockDetector::CalculateOrderBlockStrength(const SAdvancedOrderBlock& ob)
{
    double strength = 0.0;
    
    // Base strength from displacement size
    strength += MathMin(ob.displacement_size / (100 * _Point), 30.0); // Max 30 points
    
    // Volume strength (if available)
    strength += MathMin(ob.volume_profile * 20.0, 20.0); // Max 20 points
    
    // Reaction quality
    strength += ob.reaction_quality * 25.0; // Max 25 points
    
    // Institutional characteristics
    if(ob.is_institutional) strength += 15.0;
    
    // Confluence bonus
    if(ob.has_fvg_confluence) strength += 5.0;
    if(ob.has_liquidity_confluence) strength += 5.0;
    if(ob.has_structure_confluence) strength += 5.0;
    
    return MathMin(strength, 100.0);
}

// Calculate probability score
double CEliteOrderBlockDetector::CalculateProbabilityScore(const SAdvancedOrderBlock& ob)
{
    double probability = 50.0; // Base probability
    
    // Quality bonus
    switch(ob.quality)
    {
        case OB_QUALITY_HIGH: probability += 20.0; break;
        case OB_QUALITY_ELITE: probability += 30.0; break;
        case OB_QUALITY_MEDIUM: probability += 10.0; break;
        default: break;
    }
    
    // Institutional bonus
    if(ob.is_institutional) probability += 15.0;
    
    // Fresh order block bonus
    if(ob.is_fresh) probability += 10.0;
    
    // Premium/discount zone consideration
    if((ob.type == OB_BULLISH && !ob.is_premium) || 
       (ob.type == OB_BEARISH && ob.is_premium))
    {
        probability += 15.0; // Trading from discount/premium appropriately
    }
    
    // Confluence bonus
    probability += ob.confluence_score * 0.2;
    
    return MathMin(probability, 100.0);
}

// Classify order block quality
ENUM_OB_QUALITY CEliteOrderBlockDetector::ClassifyOrderBlockQuality(const SAdvancedOrderBlock& ob)
{
    double quality_score = 0.0;
    
    // Size factor
    if(ob.displacement_size >= 300 * _Point) quality_score += 25.0;
    else if(ob.displacement_size >= 200 * _Point) quality_score += 15.0;
    else if(ob.displacement_size >= 100 * _Point) quality_score += 10.0;
    
    // Volume factor
    if(ob.volume_profile > 1.8) quality_score += 25.0;
    else if(ob.volume_profile > 1.5) quality_score += 15.0;
    else if(ob.volume_profile > 1.2) quality_score += 10.0;
    
    // Reaction quality
    quality_score += ob.reaction_quality * 25.0;
    
    // Institutional characteristics
    if(ob.is_institutional) quality_score += 15.0;
    
    // Confluence score
    quality_score += ob.confluence_score * 0.1;
    
    // Classify based on total score
    if(quality_score >= 85.0) return OB_QUALITY_ELITE;
    if(quality_score >= 70.0) return OB_QUALITY_HIGH;
    if(quality_score >= 55.0) return OB_QUALITY_MEDIUM;
    return OB_QUALITY_LOW;
}

// Helper methods implementation
double CEliteOrderBlockDetector::CalculateAverageBodySize(const MqlRates& rates[], int index, int period)
{
    double total_body = 0.0;
    int count = 0;
    
    for(int i = index; i < index + period && i < ArraySize(rates); i++)
    {
        total_body += MathAbs(rates[i].close - rates[i].open);
        count++;
    }
    
    return (count > 0) ? total_body / count : 0.0;
}

bool CEliteOrderBlockDetector::HasVolumeSpike(const MqlRates& rates[], int index)
{
    // Simple volume spike detection based on tick volume
    if(index >= ArraySize(rates) - 5) return true; // Default to true if insufficient data
    
    long current_volume = rates[index].tick_volume;
    long avg_volume = 0;
    
    // Calculate average volume of previous 10 bars
    for(int i = index + 1; i < index + 11 && i < ArraySize(rates); i++)
    {
        avg_volume += rates[i].tick_volume;
    }
    avg_volume /= 10;
    
    return current_volume > avg_volume * m_volume_threshold;
}

double CEliteOrderBlockDetector::CalculateDisplacementSize(const MqlRates& rates[], int index)
{
    if(index < 5) return 0.0;
    
    double max_displacement = 0.0;
    
    // Check displacement in next 5 candles
    for(int i = index - 1; i >= MathMax(0, index - 5); i--)
    {
        double displacement = MathAbs(rates[i].close - rates[index].close);
        if(displacement > max_displacement)
            max_displacement = displacement;
    }
    
    return max_displacement;
}

double CEliteOrderBlockDetector::CalculateVolumeProfile(const MqlRates& rates[], int index)
{
    if(index >= ArraySize(rates) - 1) return 1.0;
    
    long current_volume = rates[index].tick_volume;
    long prev_volume = rates[index + 1].tick_volume;
    
    return (prev_volume > 0) ? (double)current_volume / prev_volume : 1.0;
}

double CEliteOrderBlockDetector::CalculateReactionQuality(const MqlRates& rates[], int index)
{
    // Measure the quality of market reaction after order block formation
    if(index < 3) return 0.5;
    
    double initial_price = rates[index].close;
    double max_move = 0.0;
    bool found_reaction = false;
    
    // Check next 3 candles for reaction
    for(int i = index - 1; i >= MathMax(0, index - 3); i--)
    {
        double move = MathAbs(rates[i].close - initial_price);
        if(move > max_move) max_move = move;
    }
    
    return MathMin(max_move / (100 * _Point), 1.0);
}

bool CEliteOrderBlockDetector::IsInstitutionalOrderBlock(const SAdvancedOrderBlock& ob)
{
    // Criteria for institutional order blocks
    if(ob.displacement_size > 250 * _Point) return true; // Large displacement
    if(ob.volume_profile > 2.0) return true; // Huge volume spike
    if(ob.origin_timeframe >= PERIOD_H4) return true; // Higher timeframe
    
    return false;
}

bool CEliteOrderBlockDetector::CheckFVGConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if there's an FVG near the order block
    // This would require access to the FVG array, which is global
    // For now, we'll return false as a placeholder or implement a simplified check
    return false; 
}

bool CEliteOrderBlockDetector::CheckLiquidityConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if order block swept liquidity
    // Simplified implementation
    return ob.displacement_size > 150 * _Point;
}

bool CEliteOrderBlockDetector::CheckStructureConfluence(const SAdvancedOrderBlock& ob)
{
    // Check if order block caused a break of structure
    return m_require_structure_break;
}

double CEliteOrderBlockDetector::CalculateConfluenceScore(const SAdvancedOrderBlock& ob)
{
    double score = 0.0;
    
    if(ob.has_fvg_confluence) score += 30.0;
    if(ob.has_liquidity_confluence) score += 30.0;
    if(ob.has_structure_confluence) score += 40.0;
    
    return score;
}

bool CEliteOrderBlockDetector::IsInPremiumZone(double price)
{
    // Simplified implementation - check if price is in upper 50% of recent range
    MqlRates rates[100];
    if(CopyRates(_Symbol, PERIOD_H1, 0, 100, rates) <= 0) return false;
    
    double highest = rates[0].high;
    double lowest = rates[0].low;
    for(int i = 1; i < 100; i++)
    {
        if(rates[i].high > highest) highest = rates[i].high;
        if(rates[i].low < lowest) lowest = rates[i].low;
    }
    
    double range = highest - lowest;
    if(range <= 0) return false;
    
    double mid_point = lowest + range * 0.5;
    return price > mid_point;
}

bool CEliteOrderBlockDetector::IsInDiscountZone(double price)
{
    return !IsInPremiumZone(price);
}

void CEliteOrderBlockDetector::UpdateOrderBlockStatus()
{
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        SAdvancedOrderBlock& ob = g_elite_order_blocks[i];
        
        if(ob.state == OB_STATE_ACTIVE || ob.state == OB_STATE_TESTED)
        {
            // Check for mitigation
            if(ob.type == OB_BULLISH)
            {
                if(current_price < ob.low_price)
                {
                    ob.state = OB_STATE_MITIGATED;
                }
                else if(current_price <= ob.high_price)
                {
                    ob.state = OB_STATE_TESTED;
                    ob.touch_count++;
                }
            }
            else // Bearish
            {
                if(current_price > ob.high_price)
                {
                    ob.state = OB_STATE_MITIGATED;
                }
                else if(current_price >= ob.low_price)
                {
                    ob.state = OB_STATE_TESTED;
                    ob.touch_count++;
                }
            }
        }
    }
}

void CEliteOrderBlockDetector::RemoveInvalidOrderBlocks()
{
    int valid_count = 0;
    
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        if(g_elite_order_blocks[i].state != OB_STATE_MITIGATED && 
           g_elite_order_blocks[i].state != OB_STATE_DISABLED)
        {
            if(valid_count != i)
            {
                g_elite_order_blocks[valid_count] = g_elite_order_blocks[i];
            }
            valid_count++;
        }
    }
    
    g_elite_ob_count = valid_count;
}

int CEliteOrderBlockDetector::GetActiveOrderBlockCount()
{
    int count = 0;
    for(int i = 0; i < g_elite_ob_count; i++)
    {
        if(g_elite_order_blocks[i].state == OB_STATE_ACTIVE)
            count++;
    }
    return count;
}

void CEliteOrderBlockDetector::SortOrderBlocksByQuality()
{
    // Simple bubble sort by probability score (can be optimized)
    for(int i = 0; i < g_elite_ob_count - 1; i++)
    {
        for(int j = 0; j < g_elite_ob_count - i - 1; j++)
        {
            if(g_elite_order_blocks[j].probability_score < g_elite_order_blocks[j + 1].probability_score)
            {
                SAdvancedOrderBlock temp = g_elite_order_blocks[j];
                g_elite_order_blocks[j] = g_elite_order_blocks[j + 1];
                g_elite_order_blocks[j + 1] = temp;
            }
        }
    }
}

// Placeholder methods for missing implementations
bool CEliteOrderBlockDetector::DetectPremiumOrderBlocks() { return false; }
bool CEliteOrderBlockDetector::DetectDiscountOrderBlocks() { return false; }
bool CEliteOrderBlockDetector::ValidateWithLiquidity(SAdvancedOrderBlock& ob) { return true; }
bool CEliteOrderBlockDetector::ValidateWithVolume(SAdvancedOrderBlock& ob) { return true; }
bool CEliteOrderBlockDetector::ValidateWithStructure(SAdvancedOrderBlock& ob) { return true; }
double CEliteOrderBlockDetector::CalculateOptimalEntry(const SAdvancedOrderBlock& ob) { return ob.refined_entry; }
