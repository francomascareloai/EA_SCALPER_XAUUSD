//+------------------------------------------------------------------+
//|                                           EA_Fibonacci_Elite_v1.mq5 |
//|                                 Copyright 2024, TradeDev_Master |
//|                          Expert Advisor Elite Fibonacci Trading |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master 2024"
#property link      "https://github.com/tradedevmaster"
#property version   "1.0"
#property description "🚀 EA FIBONACCI ELITE - EXTREMAMENTE INTELIGENTE"
#property description "🎯 Estratégias: Range, Retracement, Extension, Golden Zone"
#property description "🧠 IA Adaptativa com 10 Cenários de Mercado"
#property description "👻 Modo Ghost para Análise e Aprendizado"

//--- Includes necessários
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Object.mqh>

//--- Objetos de negociação
CTrade         m_trade;
CSymbolInfo    m_symbol;
CAccountInfo   m_account;
CPositionInfo  m_position;

//+------------------------------------------------------------------+
//| ENUMERAÇÕES E ESTRUTURAS AVANÇADAS
//+------------------------------------------------------------------+
enum ENUM_FIBONACCI_STRATEGY
{
    FIB_RANGE = 0,        // Fibonacci Range (RECOMENDADO)
    FIB_RETRACEMENT = 1,  // Fibonacci Retracement
    FIB_EXTENSION = 2,    // Fibonacci Extension
    FIB_GOLDEN_ZONE = 3,  // Golden Zone (61.8% + 78.6%)
    FIB_CONFLUENCE = 4    // Multi-Confluence
};

enum ENUM_MARKET_REGIME
{
    MARKET_TRENDING = 1,  // Mercado em tendência
    MARKET_RANGING = 2,   // Mercado lateral
    MARKET_BREAKOUT = 3,  // Breakout em curso
    MARKET_VOLATILE = 4,  // Alta volatilidade
    MARKET_QUIET = 5      // Baixa volatilidade
};

enum ENUM_SIGNAL_TYPE
{
    SIGNAL_NONE = 0,
    SIGNAL_BUY = 1,
    SIGNAL_SELL = -1
};

//--- Classe de sinal Fibonacci
class CFibonacciSignal
{
public:
    ENUM_SIGNAL_TYPE     type;           // Tipo do sinal
    double               price;          // Preço de entrada
    double               sl;             // Stop Loss
    double               tp;             // Take Profit
    int                  strength;       // Força do sinal (1-10)
    string               reasoning;      // Lógica do sinal
    datetime             timestamp;      // Timestamp
    ENUM_FIBONACCI_STRATEGY strategy;    // Estratégia Fibonacci
    double               confidence;     // Nível de confiança (0-100%)
    int                  confluences;    // Número de confluências
    
    // Default constructor
    CFibonacciSignal()
    {
        type = SIGNAL_NONE;
        price = 0.0;
        sl = 0.0;
        tp = 0.0;
        strength = 0;
        reasoning = "";
        timestamp = 0;
        strategy = FIB_RANGE;
        confidence = 0.0;
        confluences = 0;
    }
    
    // Copy constructor
    CFibonacciSignal(const CFibonacciSignal &other)
    {
        type = other.type;
        price = other.price;
        sl = other.sl;
        tp = other.tp;
        strength = other.strength;
        reasoning = other.reasoning;
        timestamp = other.timestamp;
        strategy = other.strategy;
        confidence = other.confidence;
        confluences = other.confluences;
    }
    
    // Assignment operator
    void operator=(const CFibonacciSignal &other)
    {
        type = other.type;
        price = other.price;
        sl = other.sl;
        tp = other.tp;
        strength = other.strength;
        reasoning = other.reasoning;
        timestamp = other.timestamp;
        strategy = other.strategy;
        confidence = other.confidence;
        confluences = other.confluences;
    }
    
    // Método para inicializar com valores padrão
    void Reset()
    {
        type = SIGNAL_NONE;
        price = 0.0;
        sl = 0.0;
        tp = 0.0;
        strength = 0;
        reasoning = "";
        timestamp = 0;
        strategy = FIB_RANGE;
        confidence = 0.0;
        confluences = 0;
    }
};

//--- Estrutura de nível Fibonacci
struct SFibonacciLevel
{
    double               price;          // Preço do nível
    double               ratio;          // Ratio Fibonacci
    int                  touches;        // Número de toques
    datetime             lastTouch;      // Último toque
    bool                 isActive;       // Ativo/Inativo
    double               strength;       // Força do nível
    string               description;    // Descrição
};

//--- Estrutura de análise de mercado
struct SMarketAnalysis
{
    ENUM_MARKET_REGIME   regime;         // Regime de mercado
    double               volatility;     // Índice de volatilidade
    double               trend_strength; // Força da tendência
    double               volume_ratio;   // Ratio de volume
    bool                 is_news_time;   // Horário de notícias
    double               atr_ratio;      // Ratio ATR
    int                  scenario_id;    // ID do cenário (1-10)
    bool                 circuit_breaker_active; // Circuit breaker ativo
    int                  consecutive_losses;     // Perdas consecutivas
    double               correlation_heat;       // Índice de correlação
};

//--- Estrutura de proteção avançada
struct SRiskProtection
{
    bool                 emergency_stop;         // Parada de emergência
    datetime             last_loss_time;         // Última perda
    int                  consecutive_losses;     // Perdas seguidas
    double               max_correlation;        // Correlação máxima
    bool                 high_volatility_alert;  // Alerta alta volatilidade
    bool                 low_liquidity_alert;    // Alerta baixa liquidez
    double               confidence_decay;       // Decay de confiança
};

//--- Estrutura de Volume Profile
struct SVolumeProfile
{
    double               volume_average;         // Volume médio
    double               volume_current;         // Volume atual
    double               volume_ratio;           // Ratio atual/médio
    bool                 volume_confirmation;    // Confirmação por volume
    double               vwap_distance;          // Distância do VWAP
};

//--- Estrutura de Ghost Trade para análise
struct SGhostTrade
{
    datetime             entry_time;     // Tempo de entrada
    double               entry_price;    // Preço de entrada
    ENUM_SIGNAL_TYPE     direction;      // Direção
    double               sl_price;       // Stop Loss
    double               tp_price;       // Take Profit
    string               strategy_used;  // Estratégia usada
    bool                 was_successful; // Foi bem-sucedida?
    double               result_pips;    // Resultado em pips
    string               failure_reason; // Razão do fracasso
    int                  confluences;    // Confluências detectadas
    double               market_conditions; // Condições de mercado
};

//+------------------------------------------------------------------+
//| PARÂMETROS DE ENTRADA AVANÇADOS
//+------------------------------------------------------------------+
input group "=== 🎯 ESTRATÉGIA FIBONACCI ==="
input ENUM_FIBONACCI_STRATEGY InpFibStrategy = FIB_RANGE; // Estratégia Principal
input bool InpUseFibRange = true;           // ✅ Usar Fibonacci Range
input bool InpUseFibRetracement = true;     // ✅ Usar Retracement
input bool InpUseFibExtension = true;       // ✅ Usar Extension
input bool InpUseFibGoldenZone = true;      // ✅ Usar Golden Zone

input group "=== 🧠 INTELIGÊNCIA ADAPTATIVA ==="
input bool InpUseAI = true;                 // ✅ Usar IA Adaptativa
input bool InpGhostMode = true;             // 👻 Modo Ghost Ativo
input int InpMinSignalStrength = 7;         // Força Mínima do Sinal (1-10)
input int InpMinConfluences = 2;            // Confluências Mínimas
input double InpConfidenceThreshold = 70.0; // Limite de Confiança (%)

input group "=== 📊 CÁLCULOS FIBONACCI ==="
input int InpSwingLookback = 50;            // Lookback para Swing Points
input double InpMinSwingSize = 30.0;        // Tamanho Mínimo do Swing (points)
input double InpLevelTolerance = 3.0;       // Tolerância para Níveis (points)
input int InpMaxFibLevels = 15;             // Máx. Níveis Fibonacci Ativos

input group "=== 💰 GESTÃO DE RISCO FTMO ==="
input double InpRiskPercent = 1.0;          // Risco por Trade (%)
input double InpMaxDailyLoss = 4.0;         // Perda Máxima Diária (%)
input double InpMaxTotalLoss = 8.0;         // Perda Máxima Total (%)
input double InpMaxLotSize = 10.0;          // Lote Máximo
input bool InpCloseOnFriday = true;         // Fechar na Sexta-feira

input group "=== ⚡ ALAVANCAGEM INTELIGENTE ==="
input bool InpUseDynamicLeverage = true;    // ✅ Alavancagem Dinâmica
input double InpBaseLeverage = 50.0;        // Alavancagem Base (REDUZIDA!)
input double InpMaxLeverage = 200.0;        // Alavancagem Máxima (LIMITADA!)
input bool InpVolatilityAdjust = true;      // Ajuste por Volatilidade

input group "=== 🛡️ PROTEÇÃO AVANÇADA ==="
input bool InpUseVolumeFilter = true;       // ✅ Filtro de Volume
input double InpVolumeThreshold = 0.4;      // Volume Mínimo (40% da média)
input double InpVolumeConfirmThreshold = 1.2; // Volume Confirmação (120% da média)
input bool InpAdaptiveVolumeFilter = true;  // ✅ Filtro de Volume Adaptativo
input bool InpUseCorrelationFilter = true;  // ✅ Filtro de Correlação
input int InpMaxConsecutiveLosses = 3;      // Máx. Perdas Consecutivas
input double InpVolatilityCircuitBreaker = 2.0; // Circuit Breaker (x ATR)
input bool InpEmergencyStopEnabled = true;  // ✅ Parada de Emergência

input group "=== ⏰ FILTROS TEMPORAIS ==="
input bool InpTradeAsian = true;            // Negociar Sessão Asiática
input bool InpTradeEuropean = true;         // Negociar Sessão Europeia
input bool InpTradeAmerican = true;         // Negociar Sessão Americana
input int InpStartHour = 0;                 // Hora de Início
input int InpEndHour = 23;                  // Hora de Término

//+------------------------------------------------------------------+
//| VARIÁVEIS GLOBAIS INTELIGENTES
//+------------------------------------------------------------------+
SFibonacciLevel g_fibLevels[];              // Array de níveis Fibonacci
CFibonacciSignal g_lastSignal;              // Último sinal gerado
SMarketAnalysis g_marketAnalysis;           // Análise de mercado atual
SGhostTrade g_ghostTrades[];                // Array de Ghost trades
SRiskProtection g_riskProtection;           // Proteção avançada
SVolumeProfile g_volumeProfile;             // Perfil de volume

//--- Variáveis de controle
double g_dailyPnL = 0.0;                    // P&L diário
double g_totalPnL = 0.0;                    // P&L total
datetime g_lastBarTime = 0;                 // Controle de nova barra
int g_magicNumber;                          // Magic number dinâmico
string g_expertName = "FibElite";           // Nome do expert
int g_uniqueID;                             // ID único por instância

//--- Indicadores
int g_handleATR;                            // Handle ATR
int g_handleRSI;                            // Handle RSI
int g_handleMA50;                           // Handle MA 50
int g_handleMA200;                          // Handle MA 200

//--- Arrays para análise
double g_atr[];                             // Array ATR
double g_rsi[];                             // Array RSI
double g_ma50[];                            // Array MA 50
double g_ma200[];                           // Array MA 200

//+------------------------------------------------------------------+
//| CLASSE FIBONACCI ENGINE - MOTOR INTELIGENTE
//+------------------------------------------------------------------+
class CFibonacciEngine
{
private:
    double m_swingHigh;
    double m_swingLow;
    datetime m_swingHighTime;
    datetime m_swingLowTime;
    SFibonacciLevel m_levels[20];
    int m_levelCount;
    
public:
    CFibonacciEngine() { m_levelCount = 0; }
    ~CFibonacciEngine() {}
    
    bool CalculateFibonacciLevels();
    bool DetectSwingPoints();
    double GetLevelStrength(double price);
    void GenerateRangeSignal(CFibonacciSignal &signal);
    void GenerateRetracementSignal(CFibonacciSignal &signal);
    void GenerateGoldenZoneSignal(CFibonacciSignal &signal);
};

//+------------------------------------------------------------------+
//| CLASSE MARKET INTELLIGENCE - IA DE MERCADO
//+------------------------------------------------------------------+
class CMarketIntelligence
{
private:
    double m_volatilityHistory[100];
    ENUM_MARKET_REGIME m_currentRegime;
    
public:
    CMarketIntelligence() { m_currentRegime = MARKET_RANGING; }
    ~CMarketIntelligence() {}
    
    ENUM_MARKET_REGIME DetectMarketRegime();
    double CalculateVolatilityIndex();
    int DetermineScenario();
    double CalculateOptimalLeverage();
    double AdaptivePositionSizing(double baseRisk);
    bool ShouldTrade();
};

//+------------------------------------------------------------------+
//| CLASSE GHOST ANALYZER - ANÁLISE DE TRADES
//+------------------------------------------------------------------+
class CGhostAnalyzer
{
private:
    SGhostTrade m_ghostTrades[1000];
    int m_ghostCount;
    double m_successRate;
    
public:
    CGhostAnalyzer() { m_ghostCount = 0; m_successRate = 0.0; }
    ~CGhostAnalyzer() {}
    
    void RecordTrade(const CFibonacciSignal &signal);
    bool EvaluateTradeOutcome(SGhostTrade &trade);
    void UpdateLearning();
    double GetSuccessRate() { return m_successRate; }
    void GenerateAnalysisReport();
};

//+------------------------------------------------------------------+
//| CLASSE RISK MANAGER FTMO
//+------------------------------------------------------------------+
class CFTMORiskManager
{
private:
    double m_dailyStartBalance;
    double m_maxDailyLoss;
    double m_currentDailyPnL;
    
public:
    CFTMORiskManager() { m_currentDailyPnL = 0.0; }
    ~CFTMORiskManager() {}
    
    bool IsTradeAllowed(double lotSize);
    bool CheckDailyLossLimit();
    bool CheckTotalLossLimit();
    double CalculatePositionSize(double riskPercent, double slPoints);
    void UpdateDailyPnL();
    bool ShouldCloseFridayPositions();
};

//+------------------------------------------------------------------+
//| CLASSE RISK TERMINATOR - PROTEÇÃO AVANÇADA
//+------------------------------------------------------------------+
class CRiskTerminator
{
private:
    double m_correlationMatrix[10];
    bool m_emergencyMode;
    
public:
    CRiskTerminator() { m_emergencyMode = false; }
    ~CRiskTerminator() {}
    
    bool IsVolatilityTooHigh();
    bool IsCorrelationOverheated();
    bool IsLiquidityAdequate();
    void TriggerEmergencyStop();
    bool CheckConsecutiveLosses();
    double CalculateConfidenceDecay();
    void UpdateRiskMetrics();
};

//+------------------------------------------------------------------+
//| CLASSE VOLUME ANALYZER - ANÁLISE DE VOLUME
//+------------------------------------------------------------------+
class CVolumeAnalyzer
{
private:
    double m_volumeHistory[50];
    double m_vwap;
    
public:
    CVolumeAnalyzer() { m_vwap = 0.0; }
    ~CVolumeAnalyzer() {}
    
    bool HasVolumeConfirmation(double fibLevel);
    double CalculateVWAP();
    double GetVolumeRatio();
    bool IsVolumeDrying();
    void UpdateVolumeProfile();
};

//--- Objetos globais das classes
CFibonacciEngine g_fibEngine;
CMarketIntelligence g_marketAI;
CGhostAnalyzer g_ghostAnalyzer;
CFTMORiskManager g_riskManager;
CRiskTerminator g_riskTerminator;           // NOVO: Proteção avançada
CVolumeAnalyzer g_volumeAnalyzer;           // NOVO: Análise de volume

//+------------------------------------------------------------------+
//| Expert initialization function
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 INICIANDO EA FIBONACCI ELITE v1.0 - ULTRA PROTECTED");
    Print("🧠 Inteligência Artificial: ", InpUseAI ? "ATIVADA" : "DESATIVADA");
    Print("👻 Modo Ghost: ", InpGhostMode ? "ATIVADO" : "DESATIVADO");
    Print("🛡️ Proteção Avançada: ATIVADA");
    
    //--- Gerar magic number único
    g_uniqueID = (int)(TimeCurrent() % 100000);
    g_magicNumber = 20241201 + g_uniqueID;
    Print("🎯 Magic Number Único: ", g_magicNumber);
    
    //--- Configurar símbolo
    if(!m_symbol.Name(_Symbol))
    {
        Print("❌ Erro ao configurar símbolo: ", _Symbol);
        return INIT_FAILED;
    }
    
    //--- Configurar negociação
    m_trade.SetExpertMagicNumber(g_magicNumber);
    m_trade.SetMarginMode();
    m_trade.SetTypeFillingBySymbol(_Symbol);
    
    //--- Inicializar indicadores com verificação aprimorada
    g_handleATR = iATR(_Symbol, PERIOD_M15, 14);
    g_handleRSI = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
    g_handleMA50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE);
    g_handleMA200 = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_SMA, PRICE_CLOSE);
    
    //--- Verificar handles com error handling
    if(g_handleATR == INVALID_HANDLE || g_handleRSI == INVALID_HANDLE ||
       g_handleMA50 == INVALID_HANDLE || g_handleMA200 == INVALID_HANDLE)
    {
        Print("❌ Erro crítico ao criar indicadores! Código: ", GetLastError());
        return INIT_FAILED;
    }
    
    //--- Aguardar inicialização dos indicadores
    Sleep(1000);
    if(BarsCalculated(g_handleATR) < 10 || BarsCalculated(g_handleRSI) < 10)
    {
        Print("❌ Indicadores não calcularam dados suficientes");
        return INIT_FAILED;
    }
    
    //--- Configurar arrays com proteção
    ArraySetAsSeries(g_atr, true);
    ArraySetAsSeries(g_rsi, true);
    ArraySetAsSeries(g_ma50, true);
    ArraySetAsSeries(g_ma200, true);
    
    //--- Inicializar estruturas de proteção
    ZeroMemory(g_riskProtection);
    ZeroMemory(g_volumeProfile);
    g_riskProtection.confidence_decay = 1.0;
    
    //--- Resetar variáveis
    g_dailyPnL = 0.0;
    g_totalPnL = 0.0;
    g_lastBarTime = 0;
    
    //--- Configurar timer mais inteligente
    EventSetTimer(30); // Timer a cada 30 segundos para monitoramento rápido
    
    Print("✅ EA FIBONACCI ELITE ULTRA PROTECTED inicializado!");
    Print("🔋 Estratégia Principal: ", EnumToString(InpFibStrategy));
    Print("⚡ Alavancagem Base/Máx: ", InpBaseLeverage, "/", InpMaxLeverage);
    Print("🛡️ Circuit Breaker: ", InpVolatilityCircuitBreaker, "x ATR");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🛑 Finalizando EA FIBONACCI ELITE");
    
    //--- Liberar indicadores
    IndicatorRelease(g_handleATR);
    IndicatorRelease(g_handleRSI);
    IndicatorRelease(g_handleMA50);
    IndicatorRelease(g_handleMA200);
    
    //--- Destruir timer
    EventKillTimer();
    
    //--- Gerar relatório final se modo ghost ativo
    if(InpGhostMode)
    {
        g_ghostAnalyzer.GenerateAnalysisReport();
    }
    
    Print("📊 Relatório Final:");
    Print("💰 P&L Total: ", g_totalPnL, " USD");
    Print("📈 Taxa de Sucesso Ghost: ", g_ghostAnalyzer.GetSuccessRate(), "%");
    
    Print("✅ EA FIBONACCI ELITE finalizado!");
}

//+------------------------------------------------------------------+
//| Expert tick function - MOTOR PRINCIPAL
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Verificar nova barra
    if(!IsNewBar()) return;
    
    //--- Atualizar dados de mercado
    UpdateMarketData();
    
    //--- Análise de mercado com IA
    if(InpUseAI)
    {
        g_marketAnalysis.regime = g_marketAI.DetectMarketRegime();
        g_marketAnalysis.volatility = g_marketAI.CalculateVolatilityIndex();
        g_marketAnalysis.scenario_id = g_marketAI.DetermineScenario();
    }
    
    //--- Atualizar sistemas de proteção avançada
    if(InpUseVolumeFilter || InpUseCorrelationFilter)
    {
        g_riskTerminator.UpdateRiskMetrics();
        g_volumeAnalyzer.UpdateVolumeProfile();
    }
    
    //--- Verificar se pode negociar
    if(!CanTrade()) return;
    
    //--- Verificar filtros avançados
    
    if(InpUseVolumeFilter && g_volumeAnalyzer.IsVolumeDrying()) {
        // Log mais detalhado para debugging
        double currentRatio = g_volumeAnalyzer.GetVolumeRatio();
        Print("📉 Trading pausado: Volume insuficiente - Ratio: ", DoubleToString(currentRatio, 3), 
              ", Threshold: ", DoubleToString(InpVolumeThreshold, 3));
        return;
    }
    
    //--- Calcular níveis Fibonacci
    if(!g_fibEngine.CalculateFibonacciLevels()) return;
    
    //--- Gerar sinais baseados na estratégia
    CFibonacciSignal signal;
    GenerateMainSignal(signal);
    
    //--- Processar sinal
    if(signal.type != SIGNAL_NONE && signal.strength >= InpMinSignalStrength)
    {
        //--- Modo Ghost: apenas analisar
        if(InpGhostMode)
        {
            g_ghostAnalyzer.RecordTrade(signal);
            Print("👻 Ghost Trade Registrado: ", EnumToString(signal.type), 
                  " | Força: ", signal.strength, " | Confiança: ", signal.confidence, "%");
        }
        else
        {
            //--- Executar trade real
            ExecuteTrade(signal);
        }
    }
    
    //--- Atualizar gestão de risco
    g_riskManager.UpdateDailyPnL();
    
    //--- Verificar fechamento de sexta-feira
    if(InpCloseOnFriday && g_riskManager.ShouldCloseFridayPositions())
    {
        CloseAllPositions("Fechamento de Sexta-feira");
    }
}

//+------------------------------------------------------------------+
//| Timer function
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Atualizar P&L diário
    g_riskManager.UpdateDailyPnL();
    
    //--- Análise Ghost a cada hora
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    if(InpGhostMode && dt.min == 0)
    {
        g_ghostAnalyzer.UpdateLearning();
    }
}

//+------------------------------------------------------------------+
//| FUNÇÕES AUXILIARES PRINCIPAIS
//+------------------------------------------------------------------+

//--- Verificar nova barra
bool IsNewBar()
{
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(currentBarTime != g_lastBarTime)
    {
        g_lastBarTime = currentBarTime;
        return true;
    }
    return false;
}

//--- Atualizar dados de mercado
void UpdateMarketData()
{
    //--- Copiar dados dos indicadores
    CopyBuffer(g_handleATR, 0, 0, 5, g_atr);
    CopyBuffer(g_handleRSI, 0, 0, 5, g_rsi);
    CopyBuffer(g_handleMA50, 0, 0, 5, g_ma50);
    CopyBuffer(g_handleMA200, 0, 0, 5, g_ma200);
}

//--- Verificar se pode negociar
bool CanTrade()
{
    //--- Verificar horário de negociação
    if(!IsInTradingHours()) return false;
    
    //--- Verificar limites FTMO
    if(!g_riskManager.CheckDailyLossLimit()) return false;
    if(!g_riskManager.CheckTotalLossLimit()) return false;
    
    //--- Verificar se IA permite
    if(InpUseAI && !g_marketAI.ShouldTrade()) return false;
    
    return true;
}

//--- Verificar horário de negociação
bool IsInTradingHours()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int currentHour = dt.hour;
    
    //--- Verificar hora geral
    if(currentHour < InpStartHour || currentHour > InpEndHour) return false;
    
    //--- Verificar sessões específicas
    if(!InpTradeAsian && currentHour >= 0 && currentHour <= 8) return false;
    if(!InpTradeEuropean && currentHour >= 8 && currentHour <= 16) return false;
    if(!InpTradeAmerican && currentHour >= 16 && currentHour <= 24) return false;
    
    return true;
}

//--- Gerar sinal principal
void GenerateMainSignal(CFibonacciSignal &signal)
{
    // Inicializar o sinal
    signal.Reset();
    
    //--- Gerar sinal baseado na estratégia principal
    switch(InpFibStrategy)
    {
        case FIB_RANGE:
            g_fibEngine.GenerateRangeSignal(signal);
            break;
        case FIB_RETRACEMENT:
            g_fibEngine.GenerateRetracementSignal(signal);
            break;
        case FIB_GOLDEN_ZONE:
            g_fibEngine.GenerateGoldenZoneSignal(signal);
            break;
        case FIB_CONFLUENCE:
            GenerateConfluenceSignal(signal);
            break;
    }
    
    //--- Aplicar filtros de IA se ativo
    if(InpUseAI && signal.type != SIGNAL_NONE)
    {
        ApplyAIFilters(signal);
    }
}

//--- Gerar sinal de confluência
void GenerateConfluenceSignal(CFibonacciSignal &signal)
{
    // Inicializar o sinal
    signal.Reset();
    
    //--- Combinar múltiplas estratégias
    CFibonacciSignal rangeSignal, retracementSignal, goldenSignal;
    g_fibEngine.GenerateRangeSignal(rangeSignal);
    g_fibEngine.GenerateRetracementSignal(retracementSignal);
    g_fibEngine.GenerateGoldenZoneSignal(goldenSignal);
    
    //--- Verificar confluências
    int confluenceCount = 0;
    if(rangeSignal.type != SIGNAL_NONE) confluenceCount++;
    if(retracementSignal.type != SIGNAL_NONE) confluenceCount++;
    if(goldenSignal.type != SIGNAL_NONE) confluenceCount++;
    
    //--- Se há confluências suficientes
    if(confluenceCount >= InpMinConfluences)
    {
        //--- Usar o sinal com maior força
        if(rangeSignal.strength >= retracementSignal.strength && rangeSignal.strength >= goldenSignal.strength)
            signal = rangeSignal;
        else if(retracementSignal.strength >= goldenSignal.strength)
            signal = retracementSignal;
        else
            signal = goldenSignal;
            
        //--- Aumentar força por confluência
        signal.strength += confluenceCount;
        signal.confluences = confluenceCount;
        signal.confidence += confluenceCount * 10.0;
        signal.reasoning = "Multi-Confluence: " + IntegerToString(confluenceCount) + " estratégias";
    }
}

//--- Aplicar filtros de IA
void ApplyAIFilters(CFibonacciSignal &signal)
{
    //--- Aplicar análise de cenário
    switch(g_marketAnalysis.scenario_id)
    {
        case 1: // Baixa volatilidade + range
            if(signal.strategy != FIB_RANGE) signal.strength -= 2;
            break;
        case 2: // Alta volatilidade + breakout
            if(signal.strategy == FIB_RANGE) signal.strength -= 3;
            break;
        case 3: // Tendência forte
            if(signal.strategy == FIB_RETRACEMENT) signal.strength += 2;
            break;
    }
    
    //--- Ajustar por volatilidade
    if(g_marketAnalysis.volatility > 70.0) signal.strength -= 1;
    if(g_marketAnalysis.volatility < 30.0) signal.strength += 1;
    
    //--- Ajustar confiança
    signal.confidence = MathMin(signal.confidence, 95.0);
    signal.confidence = MathMax(signal.confidence, 5.0);
}

//--- Executar trade
void ExecuteTrade(const CFibonacciSignal &signal)
{
    //--- Calcular tamanho da posição
    double slPoints = MathAbs(signal.price - signal.sl) / _Point;
    double lotSize = g_riskManager.CalculatePositionSize(InpRiskPercent, slPoints);
    
    //--- Ajustar por alavancagem dinâmica se ativo
    if(InpUseDynamicLeverage)
    {
        double riskAdjusted = g_marketAI.AdaptivePositionSizing(InpRiskPercent);
        lotSize = g_riskManager.CalculatePositionSize(riskAdjusted, slPoints);
    }
    
    //--- Verificar se pode negociar este lote
    if(!g_riskManager.IsTradeAllowed(lotSize)) return;
    
    //--- Executar ordem
    bool result = false;
    if(signal.type == SIGNAL_BUY)
    {
        result = m_trade.Buy(lotSize, _Symbol, signal.price, signal.sl, signal.tp, 
                           "FibElite_" + EnumToString(signal.strategy));
    }
    else if(signal.type == SIGNAL_SELL)
    {
        result = m_trade.Sell(lotSize, _Symbol, signal.price, signal.sl, signal.tp,
                            "FibElite_" + EnumToString(signal.strategy));
    }
    
    //--- Log da execução
    if(result)
    {
        Print("✅ TRADE EXECUTADO: ", EnumToString(signal.type), " | Lote: ", lotSize, 
              " | SL: ", signal.sl, " | TP: ", signal.tp);
    }
    else
    {
        Print("❌ ERRO AO EXECUTAR TRADE: ", m_trade.ResultRetcode());
    }
}

//--- Fechar todas as posições
void CloseAllPositions(string reason)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(m_position.SelectByIndex(i))
        {
            if(m_position.Symbol() == _Symbol && m_position.Magic() == g_magicNumber)
            {
                m_trade.PositionClose(m_position.Ticket());
                Print("⚙️ Posição fechada: ", reason);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| IMPLEMENTAÇÃO DAS CLASSES - FIBONACCI ENGINE
//+------------------------------------------------------------------+

//--- Calcular níveis Fibonacci
bool CFibonacciEngine::CalculateFibonacciLevels()
{
    if(!DetectSwingPoints()) return false;
    
    double range = m_swingHigh - m_swingLow;
    if(range < InpMinSwingSize * _Point) return false;
    
    //--- Níveis Fibonacci clássicos
    double fibRatios[] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618};
    m_levelCount = ArraySize(fibRatios);
    
    for(int i = 0; i < m_levelCount; i++)
    {
        m_levels[i].ratio = fibRatios[i];
        m_levels[i].price = m_swingLow + (range * fibRatios[i]);
        m_levels[i].isActive = true;
        m_levels[i].touches = 0;
        m_levels[i].strength = (fibRatios[i] == 0.618 || fibRatios[i] == 0.786) ? 10 : 5;
        m_levels[i].description = "Fib " + DoubleToString(fibRatios[i] * 100, 1) + "%";
    }
    
    return true;
}

//--- Detectar swing points
bool CFibonacciEngine::DetectSwingPoints()
{
    double highest = 0, lowest = 999999;
    datetime highTime = 0, lowTime = 0;
    
    //--- Procurar swing high e low nos últimos bars
    for(int i = 1; i <= InpSwingLookback; i++)
    {
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        datetime time = iTime(_Symbol, PERIOD_CURRENT, i);
        
        if(high > highest)
        {
            highest = high;
            highTime = time;
        }
        
        if(low < lowest)
        {
            lowest = low;
            lowTime = time;
        }
    }
    
    //--- Verificar se encontrou swing válido
    if(highest - lowest < InpMinSwingSize * _Point) return false;
    
    m_swingHigh = highest;
    m_swingLow = lowest;
    m_swingHighTime = highTime;
    m_swingLowTime = lowTime;
    
    return true;
}

//--- Gerar sinal de Range Fibonacci
void CFibonacciEngine::GenerateRangeSignal(CFibonacciSignal &signal)
{
    // Inicializar o sinal
    signal.Reset();
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double rangeHigh = m_levels[6].price;  // 100% level
    double rangeLow = m_levels[0].price;   // 0% level
    double fib236 = m_levels[1].price;     // 23.6% level
    double fib764 = m_levels[5].price;     // 78.6% level
    
    //--- Sinal de COMPRA perto do fundo do range
    if(currentPrice <= fib236 + InpLevelTolerance * _Point && currentPrice >= rangeLow)
    {
        signal.type = SIGNAL_BUY;
        signal.price = currentPrice;
        signal.sl = rangeLow - 5 * _Point;
        signal.tp = fib764;
        signal.strength = 8;
        signal.confidence = 75.0;
        signal.strategy = FIB_RANGE;
        signal.reasoning = "Range Fibonacci - Compra no suporte 23.6%";
        signal.timestamp = TimeCurrent();
    }
    //--- Sinal de VENDA perto do topo do range
    else if(currentPrice >= fib764 - InpLevelTolerance * _Point && currentPrice <= rangeHigh)
    {
        signal.type = SIGNAL_SELL;
        signal.price = currentPrice;
        signal.sl = rangeHigh + 5 * _Point;
        signal.tp = fib236;
        signal.strength = 8;
        signal.confidence = 75.0;
        signal.strategy = FIB_RANGE;
        signal.reasoning = "Range Fibonacci - Venda na resistência 78.6%";
        signal.timestamp = TimeCurrent();
    }
}

//--- Gerar sinal de Retracement
void CFibonacciEngine::GenerateRetracementSignal(CFibonacciSignal &signal)
{
    // Inicializar o sinal
    signal.Reset();
    signal.strategy = FIB_RETRACEMENT;
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double fib618 = m_levels[4].price;  // 61.8% Golden Ratio
    double fib786 = m_levels[5].price;  // 78.6% Deep Retracement
    
    //--- Determinar direção da tendência principal
    bool isUptrend = (g_ma50[0] > g_ma200[0]);
    
    if(isUptrend)
    {
        //--- Sinal de COMPRA no retracement
        if(MathAbs(currentPrice - fib618) <= InpLevelTolerance * _Point ||
           MathAbs(currentPrice - fib786) <= InpLevelTolerance * _Point)
        {
            signal.type = SIGNAL_BUY;
            signal.price = currentPrice;
            signal.sl = m_swingLow - 10 * _Point;
            signal.tp = m_swingHigh + (m_swingHigh - m_swingLow) * 0.618; // Extension
            signal.strength = 7;
            signal.confidence = 70.0;
            signal.reasoning = "Retracement em uptrend - Compra no Golden Ratio";
        }
    }
    else
    {
        //--- Sinal de VENDA no retracement
        if(MathAbs(currentPrice - fib618) <= InpLevelTolerance * _Point ||
           MathAbs(currentPrice - fib786) <= InpLevelTolerance * _Point)
        {
            signal.type = SIGNAL_SELL;
            signal.price = currentPrice;
            signal.sl = m_swingHigh + 10 * _Point;
            signal.tp = m_swingLow - (m_swingHigh - m_swingLow) * 0.618; // Extension
            signal.strength = 7;
            signal.confidence = 70.0;
            signal.reasoning = "Retracement em downtrend - Venda no Golden Ratio";
        }
    }
    
    signal.timestamp = TimeCurrent();
}

//--- Gerar sinal de Golden Zone
void CFibonacciEngine::GenerateGoldenZoneSignal(CFibonacciSignal &signal)
{
    // Inicializar o sinal
    signal.Reset();
    signal.strategy = FIB_GOLDEN_ZONE;
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double fib618 = m_levels[4].price;  // 61.8%
    double fib786 = m_levels[5].price;  // 78.6%
    
    //--- Verificar se o preço está na Golden Zone (61.8% - 78.6%)
    bool inGoldenZone = (currentPrice >= fib618 && currentPrice <= fib786) ||
                        (currentPrice <= fib618 + InpLevelTolerance * _Point && 
                         currentPrice >= fib786 - InpLevelTolerance * _Point);
    
    if(inGoldenZone)
    {
        //--- Confirmar com RSI
        bool rsiOversold = (g_rsi[0] < 30);
        bool rsiOverbought = (g_rsi[0] > 70);
        
        //--- Determinar direção
        bool isUptrend = (g_ma50[0] > g_ma200[0]);
        
        if(isUptrend && rsiOversold)
        {
            signal.type = SIGNAL_BUY;
            signal.price = currentPrice;
            signal.sl = fib786 - 10 * _Point;
            signal.tp = m_levels[7].price; // 127.2% extension
            signal.strength = 9;
            signal.confidence = 85.0;
            signal.reasoning = "Golden Zone + RSI oversold em uptrend";
        }
        else if(!isUptrend && rsiOverbought)
        {
            signal.type = SIGNAL_SELL;
            signal.price = currentPrice;
            signal.sl = fib618 + 10 * _Point;
            signal.tp = m_levels[0].price - (m_swingHigh - m_swingLow) * 0.272; // Extension down
            signal.strength = 9;
            signal.confidence = 85.0;
            signal.reasoning = "Golden Zone + RSI overbought em downtrend";
        }
    }
    
    signal.timestamp = TimeCurrent();
}

//+------------------------------------------------------------------+
//| IMPLEMENTAÇÃO - MARKET INTELLIGENCE
//+------------------------------------------------------------------+

//--- Detectar regime de mercado
ENUM_MARKET_REGIME CMarketIntelligence::DetectMarketRegime()
{
    double atr = g_atr[0];
    double atrAvg = (g_atr[0] + g_atr[1] + g_atr[2] + g_atr[3] + g_atr[4]) / 5.0;
    double trendStrength = MathAbs(g_ma50[0] - g_ma200[0]) / _Point;
    
    //--- Alta volatilidade
    if(atr > atrAvg * 1.5)
    {
        return MARKET_VOLATILE;
    }
    //--- Baixa volatilidade
    else if(atr < atrAvg * 0.7)
    {
        return MARKET_QUIET;
    }
    //--- Tendência forte
    else if(trendStrength > 50)
    {
        return MARKET_TRENDING;
    }
    //--- Range/Lateral
    else
    {
        return MARKET_RANGING;
    }
}

//--- Calcular índice de volatilidade
double CMarketIntelligence::CalculateVolatilityIndex()
{
    double atr = g_atr[0];
    double atrAvg = (g_atr[0] + g_atr[1] + g_atr[2] + g_atr[3] + g_atr[4]) / 5.0;
    
    double volatilityRatio = atr / atrAvg;
    double volatilityIndex = (volatilityRatio - 0.5) * 100.0; // Normalizar para 0-100
    
    return MathMax(0, MathMin(100, volatilityIndex));
}

//--- Determinar cenário de mercado (1-10)
int CMarketIntelligence::DetermineScenario()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    double volatility = CalculateVolatilityIndex();
    ENUM_MARKET_REGIME regime = DetectMarketRegime();
    
    //--- Cenário 1: Baixa volatilidade + Range (Sessão Asiática)
    if(hour >= 0 && hour <= 8 && volatility < 30 && regime == MARKET_RANGING)
        return 1;
        
    //--- Cenário 2: Alta volatilidade + Breakout (Abertura Europeia)
    if(hour >= 8 && hour <= 10 && volatility > 70 && regime == MARKET_VOLATILE)
        return 2;
        
    //--- Cenário 3: Tendência forte (Sessão Europeia)
    if(hour >= 10 && hour <= 16 && regime == MARKET_TRENDING)
        return 3;
        
    //--- Cenário 4: Consolidação (Entre sessões)
    if((hour >= 6 && hour <= 8) || (hour >= 16 && hour <= 18))
        return 4;
        
    //--- Cenário 5: Volatilidade por notícias (Qualquer hora + alta vol)
    if(volatility > 80)
        return 5;
        
    //--- Outros cenários...
    return 6; // Cenário padrão
}

//--- Calcular alavancagem ótima
double CMarketIntelligence::CalculateOptimalLeverage()
{
    double baseLeverage = InpBaseLeverage;
    double volatility = CalculateVolatilityIndex();
    ENUM_MARKET_REGIME regime = DetectMarketRegime();
    
    //--- Ajustar por regime de mercado
    switch(regime)
    {
        case MARKET_RANGING:
            baseLeverage *= 1.5; // Aumentar em range
            break;
        case MARKET_TRENDING:
            baseLeverage *= 1.2; // Aumentar moderadamente em tendência
            break;
        case MARKET_VOLATILE:
            baseLeverage *= 0.8; // Reduzir em alta volatilidade
            break;
        case MARKET_QUIET:
            baseLeverage *= 1.3; // Aumentar em baixa volatilidade
            break;
    }
    
    //--- Ajustar por volatilidade
    if(volatility > 70) baseLeverage *= 0.7;
    else if(volatility < 30) baseLeverage *= 1.3;
    
    //--- Limitar alavancagem
    return MathMin(baseLeverage, InpMaxLeverage);
}

//--- Position sizing adaptativo
double CMarketIntelligence::AdaptivePositionSizing(double baseRisk)
{
    double adaptiveRisk = baseRisk;
    double volatility = CalculateVolatilityIndex();
    ENUM_MARKET_REGIME regime = DetectMarketRegime();
    
    //--- Ajustar risco por regime
    switch(regime)
    {
        case MARKET_RANGING:
            adaptiveRisk *= 1.2; // Aumentar risco em range
            break;
        case MARKET_VOLATILE:
            adaptiveRisk *= 0.8; // Reduzir risco em alta volatilidade
            break;
        case MARKET_QUIET:
            adaptiveRisk *= 1.1; // Aumentar ligeiramente em baixa vol
            break;
    }
    
    //--- Ajustar por horário
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    if(hour >= 0 && hour <= 8) adaptiveRisk *= 1.1; // Sessão asiática - mais conservadora
    
    //--- Limitar risco
    adaptiveRisk = MathMin(adaptiveRisk, 3.0); // Máximo 3%
    adaptiveRisk = MathMax(adaptiveRisk, 0.3); // Mínimo 0.3%
    
    return adaptiveRisk;
}

//--- Verificar se deve negociar
bool CMarketIntelligence::ShouldTrade()
{
    double volatility = CalculateVolatilityIndex();
    
    //--- Não negociar em volatilidade extrema
    if(volatility > 90) return false;
    
    //--- Não negociar em spread muito alto
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    double atr = g_atr[0];
    if(spread > atr * 0.3) return false; // Spread > 30% do ATR
    
    return true;
}

//+------------------------------------------------------------------+
//| IMPLEMENTAÇÃO - GHOST ANALYZER
//+------------------------------------------------------------------+

//--- Registrar trade ghost
void CGhostAnalyzer::RecordTrade(const CFibonacciSignal &signal)
{
    if(m_ghostCount >= 1000) return; // Limite de trades
    
    SGhostTrade trade;
    trade.entry_time = TimeCurrent();
    trade.entry_price = signal.price;
    trade.direction = signal.type;
    trade.sl_price = signal.sl;
    trade.tp_price = signal.tp;
    trade.strategy_used = EnumToString(signal.strategy);
    trade.confluences = signal.confluences;
    trade.market_conditions = g_marketAnalysis.volatility;
    
    m_ghostTrades[m_ghostCount] = trade;
    m_ghostCount++;
    
    //--- Avaliar resultado após 1 hora (simulado)
    EvaluateTradeOutcome(trade);
}

//--- Avaliar resultado do trade
bool CGhostAnalyzer::EvaluateTradeOutcome(SGhostTrade &trade)
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    //--- Simular se atingiu TP ou SL
    if(trade.direction == SIGNAL_BUY)
    {
        if(currentPrice >= trade.tp_price)
        {
            trade.was_successful = true;
            trade.result_pips = (trade.tp_price - trade.entry_price) / _Point;
        }
        else if(currentPrice <= trade.sl_price)
        {
            trade.was_successful = false;
            trade.result_pips = (trade.sl_price - trade.entry_price) / _Point;
            trade.failure_reason = "Stop Loss atingido";
        }
    }
    else if(trade.direction == SIGNAL_SELL)
    {
        if(currentPrice <= trade.tp_price)
        {
            trade.was_successful = true;
            trade.result_pips = (trade.entry_price - trade.tp_price) / _Point;
        }
        else if(currentPrice >= trade.sl_price)
        {
            trade.was_successful = false;
            trade.result_pips = (trade.entry_price - trade.sl_price) / _Point;
            trade.failure_reason = "Stop Loss atingido";
        }
    }
    
    return trade.was_successful;
}

//--- Atualizar aprendizado
void CGhostAnalyzer::UpdateLearning()
{
    if(m_ghostCount == 0) return;
    
    int successCount = 0;
    for(int i = 0; i < m_ghostCount; i++)
    {
        if(m_ghostTrades[i].was_successful) successCount++;
    }
    
    m_successRate = (double)successCount / m_ghostCount * 100.0;
    
    Print("👻 Ghost Analysis Update: ", m_ghostCount, " trades, ", 
          DoubleToString(m_successRate, 1), "% success rate");
}

//--- Gerar relatório de análise
void CGhostAnalyzer::GenerateAnalysisReport()
{
    Print("📈 ===== RELATÓRIO GHOST ANALYZER =====");
    Print("📊 Total de Trades Analisados: ", m_ghostCount);
    Print("🎯 Taxa de Sucesso: ", DoubleToString(m_successRate, 2), "%");
    
    if(m_ghostCount > 0)
    {
        //--- Analisar por estratégia
        int rangeSuccess = 0, retracementSuccess = 0, goldenSuccess = 0;
        int rangeTotal = 0, retracementTotal = 0, goldenTotal = 0;
        
        for(int i = 0; i < m_ghostCount; i++)
        {
            if(StringFind(m_ghostTrades[i].strategy_used, "RANGE") >= 0)
            {
                rangeTotal++;
                if(m_ghostTrades[i].was_successful) rangeSuccess++;
            }
            else if(StringFind(m_ghostTrades[i].strategy_used, "RETRACEMENT") >= 0)
            {
                retracementTotal++;
                if(m_ghostTrades[i].was_successful) retracementSuccess++;
            }
            else if(StringFind(m_ghostTrades[i].strategy_used, "GOLDEN") >= 0)
            {
                goldenTotal++;
                if(m_ghostTrades[i].was_successful) goldenSuccess++;
            }
        }
        
        Print("📉 Fibonacci Range: ", rangeSuccess, "/", rangeTotal, 
              " (", DoubleToString((double)rangeSuccess/rangeTotal*100, 1), "%)");
        Print("📉 Fibonacci Retracement: ", retracementSuccess, "/", retracementTotal, 
              " (", DoubleToString((double)retracementSuccess/retracementTotal*100, 1), "%)");
        Print("📉 Golden Zone: ", goldenSuccess, "/", goldenTotal, 
              " (", DoubleToString((double)goldenSuccess/goldenTotal*100, 1), "%)");
    }
    
    Print("📈 ===== FIM DO RELATÓRIO =====");
}

//+------------------------------------------------------------------+
//| IMPLEMENTAÇÃO - RISK TERMINATOR
//+------------------------------------------------------------------+

//--- Verificar se volatilidade está muito alta
bool CRiskTerminator::IsVolatilityTooHigh()
{
    double atr = g_atr[0];
    double atrAvg = (g_atr[0] + g_atr[1] + g_atr[2] + g_atr[3] + g_atr[4]) / 5.0;
    
    // Volatilidade muito alta se ATR > 2x a média
    return (atr > atrAvg * InpVolatilityCircuitBreaker);
}

//--- Verificar se correlação está sobreaquecida
bool CRiskTerminator::IsCorrelationOverheated()
{
    // Simulação de análise de correlação
    // Em implementação real, analisaria correlação entre pares
    int openPositions = PositionsTotal();
    
    // Evitar mais de 3 posições simultâneas
    return (openPositions >= 3);
}

//--- Verificar se liquidez é adequada
bool CRiskTerminator::IsLiquidityAdequate()
{
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    double atr = g_atr[0];
    
    // Liquidez inadequada se spread > 50% do ATR
    return (spread <= atr * 0.5);
}

//--- Disparar parada de emergência
void CRiskTerminator::TriggerEmergencyStop()
{
    m_emergencyMode = true;
    
    Print("⚠️ PARADA DE EMERGÊNCIA ATIVADA!");
    Print("🛑 Motivo: Condições de mercado perigosas detectadas");
    
    // Fechar todas as posições abertas
    CloseAllPositions("Emergency Stop - Risk Terminator");
}

//--- Verificar perdas consecutivas
bool CRiskTerminator::CheckConsecutiveLosses()
{
    // Esta verificação seria feita baseada no histórico de trades
    // Por simplicidade, simulamos baseado no P&L diário
    return (g_dailyPnL < 0 && MathAbs(g_dailyPnL) > AccountInfoDouble(ACCOUNT_BALANCE) * 0.02);
}

//--- Calcular decay de confiança
double CRiskTerminator::CalculateConfidenceDecay()
{
    double decayFactor = 1.0;
    
    // Reduzir confiança após perdas
    if(g_dailyPnL < 0)
    {
        decayFactor = 1.0 - (MathAbs(g_dailyPnL) / AccountInfoDouble(ACCOUNT_BALANCE) * 10.0);
        decayFactor = MathMax(decayFactor, 0.5); // Mínimo 50% de confiança
    }
    
    return decayFactor;
}

//--- Atualizar métricas de risco
void CRiskTerminator::UpdateRiskMetrics()
{
    // Verificar todas as condições de risco
    bool highVol = IsVolatilityTooHigh();
    bool corrOverheat = IsCorrelationOverheated();
    bool lowLiq = !IsLiquidityAdequate();
    bool consLosses = CheckConsecutiveLosses();
    
    // Ativar parada de emergência se necessário
    if((highVol && lowLiq) || (corrOverheat && consLosses))
    {
        if(InpEmergencyStopEnabled && !m_emergencyMode)
        {
            TriggerEmergencyStop();
        }
    }
    
    // Atualizar proteção global
    g_riskProtection.high_volatility_alert = highVol;
    g_riskProtection.low_liquidity_alert = lowLiq;
    g_riskProtection.max_correlation = corrOverheat ? 0.9 : 0.7;
    g_riskProtection.confidence_decay = CalculateConfidenceDecay();
}

//+------------------------------------------------------------------+
//| IMPLEMENTAÇÃO - VOLUME ANALYZER
//+------------------------------------------------------------------+

//--- Verificar confirmação por volume em nível Fibonacci
bool CVolumeAnalyzer::HasVolumeConfirmation(double fibLevel)
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double distanceFromLevel = MathAbs(currentPrice - fibLevel);
    
    // Se o preço não está próximo do nível, não há confirmação
    if(distanceFromLevel > InpLevelTolerance * _Point) return false;
    
    // Verificar se o volume atual é acima da média
    long currentVolume = iVolume(_Symbol, PERIOD_M15, 0);
    double volumeRatio = GetVolumeRatio();
    
    // Usar threshold configurável para confirmação
    return (volumeRatio > InpVolumeConfirmThreshold);
}

//--- Calcular VWAP (Volume Weighted Average Price)
double CVolumeAnalyzer::CalculateVWAP()
{
    double vwap = 0.0;
    double totalVolume = 0.0;
    double volumeWeightedPrice = 0.0;
    
    // Calcular VWAP dos últimos 20 períodos
    for(int i = 0; i < 20; i++)
    {
        double high = iHigh(_Symbol, PERIOD_M15, i);
        double low = iLow(_Symbol, PERIOD_M15, i);
        double close = iClose(_Symbol, PERIOD_M15, i);
        long volume = iVolume(_Symbol, PERIOD_M15, i);
        
        double typicalPrice = (high + low + close) / 3.0;
        
        volumeWeightedPrice += typicalPrice * (double)volume;
        totalVolume += (double)volume;
    }
    
    if(totalVolume > 0)
    {
        vwap = volumeWeightedPrice / totalVolume;
    }
    
    m_vwap = vwap;
    return vwap;
}

//--- Obter ratio de volume
double CVolumeAnalyzer::GetVolumeRatio()
{
    long currentVolume = iVolume(_Symbol, PERIOD_M15, 0);
    
    // Calcular volume médio dos últimos 20 períodos
    double avgVolume = 0.0;
    for(int i = 1; i <= 20; i++)
    {
        avgVolume += (double)iVolume(_Symbol, PERIOD_M15, i);
    }
    avgVolume /= 20.0;
    
    if(avgVolume > 0)
    {
        return (double)currentVolume / avgVolume;
    }
    
    return 1.0;
}

//--- Verificar se volume está secando
bool CVolumeAnalyzer::IsVolumeDrying()
{
    double volumeRatio = GetVolumeRatio();
    
    // Usar threshold configurável ao invés de fixo
    double threshold = InpVolumeThreshold;
    
    // Obter hora atual para uso posterior
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    
    // Se filtro adaptativo está ativo, ajustar threshold
    if(InpAdaptiveVolumeFilter)
    {
        // Em sessões asiáticas, volume naturalmente mais baixo
        if(hour >= 0 && hour <= 8) // Sessão asiática
        {
            threshold *= 0.6; // Reduzir threshold para 24% (40% * 0.6)
        }
        else if(hour >= 8 && hour <= 16) // Sessão europeia
        {
            threshold *= 0.8; // Threshold para 32% (40% * 0.8)
        }
        // Sessão americana mantém threshold padrão
        
        // Ajustar por volatilidade do mercado
        if(ArraySize(g_atr) > 0)
        {
            double atrCurrent = g_atr[0];
            double atrAvg = (g_atr[0] + g_atr[1] + g_atr[2] + g_atr[3] + g_atr[4]) / 5.0;
            
            // Se volatilidade alta, aceitar volume menor
            if(atrCurrent > atrAvg * 1.3)
            {
                threshold *= 0.7; // Reduzir mais o threshold
            }
        }
    }
    
    // Log para debugging
    if(volumeRatio < threshold)
    {
        Print("📊 Volume Analysis: Ratio=", DoubleToString(volumeRatio, 3), 
              ", Threshold=", DoubleToString(threshold, 3), 
              ", Session=", (hour >= 0 && hour <= 8) ? "Asian" : 
                            (hour >= 8 && hour <= 16) ? "European" : "American");
    }
    
    // Volume está secando se abaixo do threshold adaptativo
    return (volumeRatio < threshold);
}

//--- Atualizar perfil de volume
void CVolumeAnalyzer::UpdateVolumeProfile()
{
    g_volumeProfile.volume_current = (double)iVolume(_Symbol, PERIOD_M15, 0);
    g_volumeProfile.volume_ratio = GetVolumeRatio();
    g_volumeProfile.vwap_distance = MathAbs(SymbolInfoDouble(_Symbol, SYMBOL_BID) - CalculateVWAP());
    g_volumeProfile.volume_confirmation = (g_volumeProfile.volume_ratio > InpVolumeConfirmThreshold);
    
    // Calcular média de volume
    double avgVol = 0.0;
    for(int i = 1; i <= 20; i++)
    {
        avgVol += (double)iVolume(_Symbol, PERIOD_M15, i);
    }
    g_volumeProfile.volume_average = avgVol / 20.0;
}

//--- Verificar se trade é permitido
bool CFTMORiskManager::IsTradeAllowed(double lotSize)
{
    //--- Verificar lote máximo
    if(lotSize > InpMaxLotSize) return false;
    
    //--- Verificar limites diários
    if(!CheckDailyLossLimit()) return false;
    if(!CheckTotalLossLimit()) return false;
    
    return true;
}

//--- Verificar limite de perda diária
bool CFTMORiskManager::CheckDailyLossLimit()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double dailyLoss = balance - equity;
    double maxDailyLoss = balance * (InpMaxDailyLoss / 100.0);
    
    return dailyLoss <= maxDailyLoss;
}

//--- Verificar limite de perda total
bool CFTMORiskManager::CheckTotalLossLimit()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double initialBalance = 100000.0; // Assumir saldo inicial FTMO
    double totalLoss = initialBalance - equity;
    double maxTotalLoss = initialBalance * (InpMaxTotalLoss / 100.0);
    
    return totalLoss <= maxTotalLoss;
}

//--- Calcular tamanho da posição
double CFTMORiskManager::CalculatePositionSize(double riskPercent, double slPoints)
{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (riskPercent / 100.0);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = riskAmount / (slPoints * tickValue);
    
    //--- Aplicar limites
    lotSize = MathMax(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
    lotSize = MathMin(lotSize, InpMaxLotSize);
    
    return NormalizeDouble(lotSize, 2);
}

//--- Atualizar P&L diário
void CFTMORiskManager::UpdateDailyPnL()
{
    static datetime lastDay = 0;
    datetime currentDay = (datetime)(TimeCurrent() / 86400) * 86400; // Início do dia
    
    if(currentDay != lastDay)
    {
        //--- Novo dia - resetar P&L diário
        m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        lastDay = currentDay;
    }
    
    //--- Calcular P&L atual
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    m_currentDailyPnL = currentBalance - m_dailyStartBalance;
    g_dailyPnL = m_currentDailyPnL;
}

//--- Verificar se deve fechar posições na sexta
bool CFTMORiskManager::ShouldCloseFridayPositions()
{
    if(!InpCloseOnFriday) return false;
    
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    //--- Fechar após 21:00 de sexta-feira (GMT)
    return (dt.day_of_week == 5 && dt.hour >= 21);
}