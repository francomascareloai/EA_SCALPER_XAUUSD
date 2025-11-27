# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 5: Data Structures e Module Stubs

---

## 4.4 CDataStructures.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CDataStructures.mqh
//+------------------------------------------------------------------+
//|                                           CDataStructures.mqh    |
//|                    Shared enums and structures for all modules   |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+

enum ENUM_RISK_STATE
{
    RISK_NORMAL,      // 0-1% DD: risco normal
    RISK_REDUCED,     // 1-2.5% DD: risco reduzido (50%)
    RISK_MINIMAL,     // 2.5-4% DD: risco mínimo (25%)
    RISK_BLOCKED      // 4%+ DD: bloqueado
};

enum ENUM_OB_TYPE
{
    OB_NONE,
    OB_BULLISH,
    OB_BEARISH
};

enum ENUM_SWEEP_TYPE
{
    SWEEP_NONE,
    SWEEP_BULLISH,    // Swept lows → expect up
    SWEEP_BEARISH     // Swept highs → expect down
};

enum ENUM_MARKET_STRUCTURE
{
    STRUCTURE_BULLISH,   // HH + HL
    STRUCTURE_BEARISH,   // LH + LL
    STRUCTURE_RANGING
};

enum ENUM_VOLATILITY_STATE
{
    VOL_LOW,
    VOL_NORMAL,
    VOL_HIGH
};

//+------------------------------------------------------------------+
//| DATA STRUCTURES                                                   |
//+------------------------------------------------------------------+

struct SOrderBlockData
{
    bool           hasValidOB;
    double         obHigh;
    double         obLow;
    ENUM_OB_TYPE   obType;
    int            obStrength;    // 0-100
    datetime       obTime;
    ENUM_TIMEFRAMES obTimeframe;
    
    void Reset()
    {
        hasValidOB = false;
        obHigh = 0; obLow = 0;
        obType = OB_NONE;
        obStrength = 0;
        obTime = 0;
    }
};

struct SFVGData
{
    bool   hasFVG;
    double fvgHigh;
    double fvgLow;
    bool   fvgBullish;
    int    fvgStrength;    // 0-100
    int    fvgAge;         // Bars since formation
    
    void Reset()
    {
        hasFVG = false;
        fvgHigh = 0; fvgLow = 0;
        fvgBullish = false;
        fvgStrength = 0;
        fvgAge = 0;
    }
};

struct SLiquidityData
{
    bool            liquiditySwept;
    double          liquidityLevel;
    ENUM_SWEEP_TYPE sweepType;
    datetime        sweepTime;
    
    void Reset()
    {
        liquiditySwept = false;
        liquidityLevel = 0;
        sweepType = SWEEP_NONE;
        sweepTime = 0;
    }
};

struct SMarketStructureData
{
    ENUM_MARKET_STRUCTURE currentStructure;
    bool   hasBOS;         // Break of Structure
    bool   hasCHoCH;       // Change of Character
    int    structureStrength;  // 0-100
    double lastSwingHigh;
    double lastSwingLow;
    
    void Reset()
    {
        currentStructure = STRUCTURE_RANGING;
        hasBOS = false;
        hasCHoCH = false;
        structureStrength = 0;
        lastSwingHigh = 0;
        lastSwingLow = 0;
    }
};

struct SVolatilityData
{
    double currentATR;
    double avgATR;
    ENUM_VOLATILITY_STATE volState;
    double suggestedSL;
    double suggestedTP;
    
    void Reset()
    {
        currentATR = 0;
        avgATR = 0;
        volState = VOL_NORMAL;
        suggestedSL = 0;
        suggestedTP = 0;
    }
};

struct SPythonHubData
{
    double   techSubscorePy;
    double   fundScore;
    double   sentScore;
    string   fundBias;      // "bullish", "bearish", "neutral"
    string   sentBias;
    string   llmReasoning;
    datetime lastUpdate;
    bool     isValid;
    
    void Reset()
    {
        techSubscorePy = 50;
        fundScore = 50;
        sentScore = 50;
        fundBias = "neutral";
        sentBias = "neutral";
        llmReasoning = "";
        lastUpdate = 0;
        isValid = false;
    }
};
```

---

## 4.5 Module Stubs (Assinaturas com TODOs)

### COrderBlockModule.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\COrderBlockModule.mqh
//+------------------------------------------------------------------+
//|                                         COrderBlockModule.mqh    |
//|                          Order Block Detection (SMC/ICT)         |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

class COrderBlockModule
{
private:
    ENUM_TIMEFRAMES m_HTF;
    ENUM_TIMEFRAMES m_MTF;
    ENUM_TIMEFRAMES m_LTF;
    int m_LookbackBars;
    
    // TODO: Implement helper functions
    bool FindBullishOB(ENUM_TIMEFRAMES tf, double &high, double &low, int &strength);
    bool FindBearishOB(ENUM_TIMEFRAMES tf, double &high, double &low, int &strength);
    bool IsPriceAtOB(double price, double obHigh, double obLow);
    int  CalculateOBStrength(int age, bool mitigated, int touchCount);

public:
    COrderBlockModule(ENUM_TIMEFRAMES htf, ENUM_TIMEFRAMES mtf, ENUM_TIMEFRAMES ltf);
    ~COrderBlockModule() {}
    
    void Analyze(SOrderBlockData &outData);
};

//+------------------------------------------------------------------+
COrderBlockModule::COrderBlockModule(ENUM_TIMEFRAMES htf, 
                                     ENUM_TIMEFRAMES mtf, 
                                     ENUM_TIMEFRAMES ltf)
{
    m_HTF = htf;
    m_MTF = mtf;
    m_LTF = ltf;
    m_LookbackBars = 50;
}

//+------------------------------------------------------------------+
void COrderBlockModule::Analyze(SOrderBlockData &outData)
{
    outData.Reset();
    
    // TODO: Implement Order Block detection logic
    // 1. Identify swing highs/lows on HTF
    // 2. Find last bearish candle before bullish impulse (Bullish OB)
    // 3. Find last bullish candle before bearish impulse (Bearish OB)
    // 4. Check if current price is within OB zone
    // 5. Calculate strength based on:
    //    - Age of OB (newer = stronger)
    //    - Number of times tested
    //    - Size of the impulse move after OB
    
    // PLACEHOLDER: Return neutral data
    outData.hasValidOB = false;
    outData.obStrength = 0;
}
```

### CFVGModule.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CFVGModule.mqh
//+------------------------------------------------------------------+
//|                                              CFVGModule.mqh      |
//|                         Fair Value Gap Detection                 |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

class CFVGModule
{
private:
    ENUM_TIMEFRAMES m_MTF;
    ENUM_TIMEFRAMES m_LTF;
    double m_MinGapSize;  // Minimum gap in points
    
    // TODO: Implement
    bool FindBullishFVG(ENUM_TIMEFRAMES tf, double &high, double &low, int &age);
    bool FindBearishFVG(ENUM_TIMEFRAMES tf, double &high, double &low, int &age);
    bool IsFVGMitigated(double fvgHigh, double fvgLow, bool isBullish);
    int  CalculateFVGStrength(double gapSize, int age, bool partiallyFilled);

public:
    CFVGModule(ENUM_TIMEFRAMES mtf, ENUM_TIMEFRAMES ltf);
    ~CFVGModule() {}
    
    void Analyze(SFVGData &outData);
};

//+------------------------------------------------------------------+
CFVGModule::CFVGModule(ENUM_TIMEFRAMES mtf, ENUM_TIMEFRAMES ltf)
{
    m_MTF = mtf;
    m_LTF = ltf;
    m_MinGapSize = 50;  // 50 points minimum
}

//+------------------------------------------------------------------+
void CFVGModule::Analyze(SFVGData &outData)
{
    outData.Reset();
    
    // TODO: Implement FVG detection
    // FVG = Gap between candle 1 high and candle 3 low (bullish)
    //       or candle 1 low and candle 3 high (bearish)
    // 1. Scan last N bars for 3-candle patterns
    // 2. Check if gap exists and is not mitigated
    // 3. Calculate strength based on gap size and age
    
    outData.hasFVG = false;
    outData.fvgStrength = 0;
}
```

### CLiquidityModule.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CLiquidityModule.mqh
//+------------------------------------------------------------------+
//|                                         CLiquidityModule.mqh     |
//|                    Liquidity Pool Detection & Sweep Analysis     |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

class CLiquidityModule
{
private:
    ENUM_TIMEFRAMES m_HTF;
    ENUM_TIMEFRAMES m_MTF;
    int m_SwingLookback;
    double m_EqualLevelTolerance;  // Points tolerance for equal highs/lows
    
    // TODO: Implement
    bool FindEqualHighs(double &level, int &count);
    bool FindEqualLows(double &level, int &count);
    bool DetectSweep(double liquidityLevel, bool isHighs);

public:
    CLiquidityModule(ENUM_TIMEFRAMES htf, ENUM_TIMEFRAMES mtf);
    ~CLiquidityModule() {}
    
    void Analyze(SLiquidityData &outData);
};

//+------------------------------------------------------------------+
CLiquidityModule::CLiquidityModule(ENUM_TIMEFRAMES htf, ENUM_TIMEFRAMES mtf)
{
    m_HTF = htf;
    m_MTF = mtf;
    m_SwingLookback = 20;
    m_EqualLevelTolerance = 30;  // 30 points
}

//+------------------------------------------------------------------+
void CLiquidityModule::Analyze(SLiquidityData &outData)
{
    outData.Reset();
    
    // TODO: Implement liquidity detection
    // 1. Find swing highs/lows
    // 2. Identify clusters (equal highs/lows = liquidity pools)
    // 3. Detect if price swept through these levels
    // 4. Sweep of lows = bullish signal (stop hunt complete)
    // 5. Sweep of highs = bearish signal
    
    outData.liquiditySwept = false;
    outData.sweepType = SWEEP_NONE;
}
```

### CMarketStructureModule.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CMarketStructureModule.mqh
//+------------------------------------------------------------------+
//|                                   CMarketStructureModule.mqh     |
//|                      Market Structure Analysis (BOS/CHoCH)       |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

class CMarketStructureModule
{
private:
    ENUM_TIMEFRAMES m_HTF;
    ENUM_TIMEFRAMES m_MTF;
    int m_SwingLookback;
    
    // Swing point storage
    double m_SwingHighs[];
    double m_SwingLows[];
    datetime m_SwingHighTimes[];
    datetime m_SwingLowTimes[];
    
    // TODO: Implement
    void FindSwingPoints(ENUM_TIMEFRAMES tf);
    ENUM_MARKET_STRUCTURE DetermineStructure();
    bool DetectBOS(bool &isBullish);
    bool DetectCHoCH(bool &isBullish);
    int  CalculateStructureStrength();

public:
    CMarketStructureModule(ENUM_TIMEFRAMES htf, ENUM_TIMEFRAMES mtf);
    ~CMarketStructureModule() {}
    
    void Analyze(SMarketStructureData &outData);
};

//+------------------------------------------------------------------+
CMarketStructureModule::CMarketStructureModule(ENUM_TIMEFRAMES htf, 
                                               ENUM_TIMEFRAMES mtf)
{
    m_HTF = htf;
    m_MTF = mtf;
    m_SwingLookback = 5;  // Bars left/right to confirm swing
}

//+------------------------------------------------------------------+
void CMarketStructureModule::Analyze(SMarketStructureData &outData)
{
    outData.Reset();
    
    // TODO: Implement market structure analysis
    // 1. Identify swing highs (HH, LH) and swing lows (HL, LL)
    // 2. Bullish structure: HH + HL sequence
    // 3. Bearish structure: LH + LL sequence
    // 4. BOS = Break of significant swing point
    // 5. CHoCH = First sign of structure change
    // 6. Strength based on clarity of structure
    
    outData.currentStructure = STRUCTURE_RANGING;
    outData.hasBOS = false;
    outData.hasCHoCH = false;
    outData.structureStrength = 50;
}
```

### CVolatilityModule.mqh

```mql5
// filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CVolatilityModule.mqh
//+------------------------------------------------------------------+
//|                                        CVolatilityModule.mqh     |
//|                         ATR-Based Volatility Analysis            |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

class CVolatilityModule
{
private:
    int m_ATRPeriod;
    int m_ATRHandle;
    double m_LowVolThreshold;   // ATR ratio for low vol
    double m_HighVolThreshold;  // ATR ratio for high vol

public:
    CVolatilityModule(int atrPeriod);
    ~CVolatilityModule();
    
    void Analyze(SVolatilityData &outData);
};

//+------------------------------------------------------------------+
CVolatilityModule::CVolatilityModule(int atrPeriod)
{
    m_ATRPeriod = atrPeriod;
    m_LowVolThreshold = 0.7;   // ATR < 70% of average
    m_HighVolThreshold = 1.5;  // ATR > 150% of average
    
    m_ATRHandle = iATR(_Symbol, PERIOD_CURRENT, m_ATRPeriod);
    if(m_ATRHandle == INVALID_HANDLE)
        Print("ERROR: Failed to create ATR indicator");
}

//+------------------------------------------------------------------+
CVolatilityModule::~CVolatilityModule()
{
    if(m_ATRHandle != INVALID_HANDLE)
        IndicatorRelease(m_ATRHandle);
}

//+------------------------------------------------------------------+
void CVolatilityModule::Analyze(SVolatilityData &outData)
{
    outData.Reset();
    
    if(m_ATRHandle == INVALID_HANDLE)
        return;
    
    // Get current ATR
    double atrBuffer[];
    ArraySetAsSeries(atrBuffer, true);
    
    if(CopyBuffer(m_ATRHandle, 0, 0, 20, atrBuffer) < 20)
        return;
    
    outData.currentATR = atrBuffer[0];
    
    // Calculate average ATR (last 20 periods)
    double sum = 0;
    for(int i = 0; i < 20; i++)
        sum += atrBuffer[i];
    outData.avgATR = sum / 20.0;
    
    // Determine volatility state
    double ratio = outData.currentATR / outData.avgATR;
    
    if(ratio < m_LowVolThreshold)
        outData.volState = VOL_LOW;
    else if(ratio > m_HighVolThreshold)
        outData.volState = VOL_HIGH;
    else
        outData.volState = VOL_NORMAL;
    
    // Suggested SL/TP based on ATR
    outData.suggestedSL = outData.currentATR * 1.5;
    outData.suggestedTP = outData.currentATR * 3.0;  // 2:1 RR
}
```
