//+------------------------------------------------------------------+
//|                                    EA_Fibonacci_Visual_v1.mq5   |
//|                                 Copyright 2024, TradeDev_Master |
//|                  Expert Advisor Fibonacci com Interface Visual |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master 2024"
#property link      "https://github.com/tradedevmaster"
#property version   "1.0"
#property description "🚀 EA FIBONACCI VISUAL - XAUUSD SPECIALIST"
#property description "📊 Interface Visual para Acompanhar Análise"
#property description "🎯 Especialista em OURO (XAUUSD)"

//--- Includes necessários
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Objetos de negociação
CTrade         m_trade;
CSymbolInfo    m_symbol;
CAccountInfo   m_account;

//--- Enumerações
enum ENUM_FIBONACCI_STRATEGY
{
    FIB_RANGE = 0,        // Fibonacci Range Strategy
    FIB_RETRACEMENT = 1,  // Fibonacci Retracement
    FIB_GOLDEN_ZONE = 2   // Golden Zone (61.8% + 78.6%)
};

enum ENUM_SIGNAL_TYPE
{
    SIGNAL_NONE = 0,
    SIGNAL_BUY = 1,
    SIGNAL_SELL = -1
};

//--- Parâmetros de entrada
input group "=== 🎯 ESTRATÉGIA FIBONACCI ==="
input ENUM_FIBONACCI_STRATEGY InpFibStrategy = FIB_RANGE; // Estratégia Principal
input bool InpShowVisualInfo = true;        // ✅ Mostrar Informações Visuais

input group "=== 📊 ANÁLISE FIBONACCI ==="
input int InpSwingLookback = 50;            // Lookback para Swing Points
input double InpMinSwingSize = 30.0;        // Tamanho Mínimo do Swing (points)
input double InpLevelTolerance = 3.0;       // Tolerância para Níveis (points)

input group "=== 💰 GESTÃO DE RISCO XAUUSD ==="
input double InpRiskPercent = 1.0;          // Risco por Trade (%)
input double InpMaxDailyLoss = 4.0;         // Perda Máxima Diária (%)
input bool InpCloseOnFriday = true;         // Fechar na Sexta-feira

input group "=== ⏰ HORÁRIOS DE NEGOCIAÇÃO ==="
input bool InpTradeAsian = false;           // Negociar Sessão Asiática
input bool InpTradeEuropean = true;         // Negociar Sessão Europeia  
input bool InpTradeAmerican = true;         // Negociar Sessão Americana

//--- Variáveis globais
datetime g_lastBarTime = 0;
double g_swingHigh = 0;
double g_swingLow = 0;
double g_fibLevels[9];
string g_fibDescriptions[9] = {"0.0%", "23.6%", "38.2%", "50.0%", "61.8%", "78.6%", "100.0%", "127.2%", "161.8%"};
double g_fibRatios[9] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618};

string g_currentAnalysis = "";
string g_marketCondition = "";
ENUM_SIGNAL_TYPE g_lastSignal = SIGNAL_NONE;
double g_signalStrength = 0;
int g_magicNumber = 123456;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 INICIANDO EA FIBONACCI VISUAL - ESPECIALISTA XAUUSD v1.0");
    
    //--- Verificar se é XAUUSD
    if(_Symbol != "XAUUSD" && _Symbol != "GOLD" && _Symbol != "XAU")
    {
        Alert("⚠️ ATENÇÃO: Este EA foi desenvolvido especificamente para XAUUSD!");
        Print("📢 Símbolo atual: ", _Symbol, " - Recomendado: XAUUSD");
    }
    
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
    
    //--- Configurar timer para updates visuais
    EventSetTimer(1); // Update a cada segundo
    
    //--- Inicializar análise
    g_currentAnalysis = "Iniciando análise...";
    g_marketCondition = "Aguardando dados";
    
    Print("✅ EA FIBONACCI VISUAL inicializado com sucesso!");
    Print("📊 Interface Visual: ", InpShowVisualInfo ? "ATIVADA" : "DESATIVADA");
    Print("🎯 Estratégia: ", EnumToString(InpFibStrategy));
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🛑 EA FIBONACCI VISUAL finalizado");
    
    //--- Limpar objetos visuais
    ObjectsDeleteAll(0, "FibVisual_");
    
    //--- Destruir timer
    EventKillTimer();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Verificar nova barra
    if(!IsNewBar()) return;
    
    //--- Analisar mercado
    AnalyzeMarket();
    
    //--- Detectar swing points
    if(DetectSwingPoints())
    {
        //--- Calcular níveis Fibonacci
        CalculateFibonacciLevels();
        
        //--- Desenhar níveis se visual ativo
        if(InpShowVisualInfo)
        {
            DrawFibonacciLevels();
        }
    }
    
    //--- Gerar sinal
    ENUM_SIGNAL_TYPE signal = GenerateFibonacciSignal();
    
    //--- Atualizar informações visuais
    if(InpShowVisualInfo)
    {
        UpdateVisualInfo();
    }
    
    //--- Executar trade se sinal válido
    if(signal != SIGNAL_NONE && CanTrade())
    {
        ExecuteTrade(signal);
    }
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Atualizar informações visuais a cada segundo
    if(InpShowVisualInfo)
    {
        UpdateRealtimeInfo();
    }
}

//+------------------------------------------------------------------+
//| Verificar nova barra                                            |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| Analisar condições de mercado                                   |
//+------------------------------------------------------------------+
void AnalyzeMarket()
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double spread = ask - bid;
    
    //--- Analisar spread (importante para XAUUSD)
    if(spread > 1.0) // Spread alto para ouro
    {
        g_marketCondition = "🔴 SPREAD ALTO - Cuidado";
    }
    else if(spread < 0.3) // Spread bom para ouro
    {
        g_marketCondition = "🟢 SPREAD BOM - Favorável";
    }
    else
    {
        g_marketCondition = "🟡 SPREAD NORMAL";
    }
    
    //--- Analisar horário
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    if(dt.hour >= 8 && dt.hour <= 17) // Horário europeu/americano
    {
        g_marketCondition += " | 🕐 HORÁRIO ATIVO";
    }
    else
    {
        g_marketCondition += " | 🌙 HORÁRIO QUIETO";
    }
    
    //--- Determinar análise atual
    if(g_swingHigh > 0 && g_swingLow > 0)
    {
        double currentPrice = bid;
        double rangeSize = g_swingHigh - g_swingLow;
        double pricePosition = (currentPrice - g_swingLow) / rangeSize;
        
        if(pricePosition < 0.25)
            g_currentAnalysis = "📉 Preço na ZONA BAIXA do range";
        else if(pricePosition > 0.75)
            g_currentAnalysis = "📈 Preço na ZONA ALTA do range";
        else if(pricePosition >= 0.58 && pricePosition <= 0.68) // Zona 61.8%
            g_currentAnalysis = "🎯 Preço na GOLDEN ZONE (61.8%)";
        else
            g_currentAnalysis = "➡️ Preço no MEIO do range";
    }
    else
    {
        g_currentAnalysis = "🔍 Aguardando formação de swing points...";
    }
}

//+------------------------------------------------------------------+
//| Detectar swing points                                           |
//+------------------------------------------------------------------+
bool DetectSwingPoints()
{
    double highest = 0, lowest = 999999;
    
    //--- Procurar swing high e low
    for(int i = 1; i <= InpSwingLookback; i++)
    {
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        
        if(high > highest) highest = high;
        if(low < lowest) lowest = low;
    }
    
    //--- Verificar se é um swing válido
    if(highest - lowest < InpMinSwingSize * _Point) return false;
    
    //--- Atualizar swing points
    g_swingHigh = highest;
    g_swingLow = lowest;
    
    return true;
}

//+------------------------------------------------------------------+
//| Calcular níveis Fibonacci                                       |
//+------------------------------------------------------------------+
void CalculateFibonacciLevels()
{
    if(g_swingHigh <= g_swingLow) return;
    
    double range = g_swingHigh - g_swingLow;
    
    //--- Calcular todos os níveis
    for(int i = 0; i < 9; i++)
    {
        g_fibLevels[i] = g_swingLow + (range * g_fibRatios[i]);
    }
}

//+------------------------------------------------------------------+
//| Desenhar níveis Fibonacci no gráfico                            |
//+------------------------------------------------------------------+
void DrawFibonacciLevels()
{
    //--- Limpar níveis anteriores
    ObjectsDeleteAll(0, "FibLevel_");
    
    if(g_swingHigh <= g_swingLow) return;
    
    //--- Desenhar cada nível
    for(int i = 0; i < 9; i++)
    {
        string objName = "FibLevel_" + IntegerToString(i);
        
        //--- Criar linha horizontal
        ObjectCreate(0, objName, OBJ_HLINE, 0, 0, g_fibLevels[i]);
        
        //--- Configurar cor baseada no nível
        color lineColor = clrGray;
        int lineWidth = 1;
        
        if(g_fibRatios[i] == 0.618) // Golden Ratio
        {
            lineColor = clrGold;
            lineWidth = 2;
        }
        else if(g_fibRatios[i] == 0.786) // 78.6%
        {
            lineColor = clrOrange;
            lineWidth = 2;
        }
        else if(g_fibRatios[i] == 0.5) // 50%
        {
            lineColor = clrBlue;
            lineWidth = 1;
        }
        else if(g_fibRatios[i] == 0.0 || g_fibRatios[i] == 1.0) // 0% e 100%
        {
            lineColor = clrRed;
            lineWidth = 2;
        }
        
        ObjectSetInteger(0, objName, OBJPROP_COLOR, lineColor);
        ObjectSetInteger(0, objName, OBJPROP_WIDTH, lineWidth);
        ObjectSetInteger(0, objName, OBJPROP_STYLE, STYLE_SOLID);
        
        //--- Adicionar texto do nível
        string textName = "FibText_" + IntegerToString(i);
        ObjectCreate(0, textName, OBJ_TEXT, 0, TimeCurrent(), g_fibLevels[i]);
        ObjectSetString(0, textName, OBJPROP_TEXT, g_fibDescriptions[i] + " (" + DoubleToString(g_fibLevels[i], _Digits) + ")");
        ObjectSetInteger(0, textName, OBJPROP_COLOR, lineColor);
        ObjectSetInteger(0, textName, OBJPROP_FONTSIZE, 8);
    }
}

//+------------------------------------------------------------------+
//| Gerar sinal Fibonacci                                           |
//+------------------------------------------------------------------+
ENUM_SIGNAL_TYPE GenerateFibonacciSignal()
{
    if(g_swingHigh <= g_swingLow) return SIGNAL_NONE;
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    ENUM_SIGNAL_TYPE signal = SIGNAL_NONE;
    g_signalStrength = 0;
    
    //--- Estratégia Range Fibonacci (melhor para XAUUSD)
    if(InpFibStrategy == FIB_RANGE)
    {
        double fib236 = g_fibLevels[1]; // 23.6%
        double fib786 = g_fibLevels[5]; // 78.6%
        
        //--- Sinal de COMPRA perto do suporte
        if(MathAbs(currentPrice - fib236) <= InpLevelTolerance * _Point)
        {
            signal = SIGNAL_BUY;
            g_signalStrength = 8;
        }
        //--- Sinal de VENDA perto da resistência
        else if(MathAbs(currentPrice - fib786) <= InpLevelTolerance * _Point)
        {
            signal = SIGNAL_SELL;
            g_signalStrength = 8;
        }
    }
    //--- Estratégia Golden Zone
    else if(InpFibStrategy == FIB_GOLDEN_ZONE)
    {
        double fib618 = g_fibLevels[4]; // 61.8%
        double fib786 = g_fibLevels[5]; // 78.6%
        
        //--- Verificar se está na Golden Zone
        if(currentPrice >= fib618 && currentPrice <= fib786)
        {
            signal = SIGNAL_BUY; // Assumindo bounce para cima
            g_signalStrength = 7;
        }
    }
    
    g_lastSignal = signal;
    return signal;
}

//+------------------------------------------------------------------+
//| Verificar se pode negociar                                      |
//+------------------------------------------------------------------+
bool CanTrade()
{
    //--- Verificar horário
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    
    //--- Verificar sessões permitidas
    bool canTradeTime = false;
    if(InpTradeAsian && hour >= 0 && hour <= 8) canTradeTime = true;
    if(InpTradeEuropean && hour >= 8 && hour <= 16) canTradeTime = true;
    if(InpTradeAmerican && hour >= 16 && hour <= 24) canTradeTime = true;
    
    if(!canTradeTime) return false;
    
    //--- Verificar spread (importante para XAUUSD)
    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    if(spread > 2.0) return false; // Spread muito alto
    
    //--- Verificar se não é sexta-feira tarde
    if(InpCloseOnFriday && dt.day_of_week == 5 && hour >= 20) return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Executar trade                                                   |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_SIGNAL_TYPE signal)
{
    double lotSize = 0.01; // Lote conservador para teste
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    if(signal == SIGNAL_BUY)
    {
        double sl = g_fibLevels[0] - 10 * _Point; // SL abaixo do 0%
        double tp = g_fibLevels[5]; // TP no 78.6%
        
        if(m_trade.Buy(lotSize, _Symbol, 0, sl, tp, "Fib_Buy_" + g_fibDescriptions[4]))
        {
            Print("✅ COMPRA EXECUTADA: Fibonacci Range Strategy");
        }
    }
    else if(signal == SIGNAL_SELL)
    {
        double sl = g_fibLevels[6] + 10 * _Point; // SL acima do 100%
        double tp = g_fibLevels[1]; // TP no 23.6%
        
        if(m_trade.Sell(lotSize, _Symbol, 0, sl, tp, "Fib_Sell_" + g_fibDescriptions[1]))
        {
            Print("✅ VENDA EXECUTADA: Fibonacci Range Strategy");
        }
    }
}

//+------------------------------------------------------------------+
//| Atualizar informações visuais                                   |
//+------------------------------------------------------------------+
void UpdateVisualInfo()
{
    //--- Limpar textos anteriores
    ObjectsDeleteAll(0, "FibVisual_");
    
    //--- Criar painel de informações
    int yPos = 30;
    
    //--- Título
    CreateTextLabel("FibVisual_Title", 20, yPos, "🚀 EA FIBONACCI XAUUSD - ANÁLISE EM TEMPO REAL", clrWhite, 12);
    yPos += 25;
    
    //--- Análise atual
    CreateTextLabel("FibVisual_Analysis", 20, yPos, "📊 " + g_currentAnalysis, clrYellow, 10);
    yPos += 20;
    
    //--- Condição de mercado
    CreateTextLabel("FibVisual_Market", 20, yPos, "🌐 " + g_marketCondition, clrLightBlue, 10);
    yPos += 20;
    
    //--- Último sinal
    string signalText = "🎯 Último Sinal: ";
    color signalColor = clrGray;
    
    if(g_lastSignal == SIGNAL_BUY)
    {
        signalText += "COMPRA (Força: " + DoubleToString(g_signalStrength, 0) + "/10)";
        signalColor = clrLime;
    }
    else if(g_lastSignal == SIGNAL_SELL)
    {
        signalText += "VENDA (Força: " + DoubleToString(g_signalStrength, 0) + "/10)";
        signalColor = clrRed;
    }
    else
    {
        signalText += "AGUARDANDO...";
        signalColor = clrGray;
    }
    
    CreateTextLabel("FibVisual_Signal", 20, yPos, signalText, signalColor, 10);
    yPos += 20;
    
    //--- Estratégia ativa
    CreateTextLabel("FibVisual_Strategy", 20, yPos, "⚙️ Estratégia: " + EnumToString(InpFibStrategy), clrWhite, 9);
    yPos += 20;
    
    //--- Swing points
    if(g_swingHigh > 0 && g_swingLow > 0)
    {
        string swingInfo = "📈 Swing High: " + DoubleToString(g_swingHigh, _Digits) + 
                          " | 📉 Swing Low: " + DoubleToString(g_swingLow, _Digits);
        CreateTextLabel("FibVisual_Swing", 20, yPos, swingInfo, clrOrange, 9);
    }
}

//+------------------------------------------------------------------+
//| Atualizar informações em tempo real                             |
//+------------------------------------------------------------------+
void UpdateRealtimeInfo()
{
    //--- Atualizar apenas informações que mudam rapidamente
    string priceInfo = "💰 Preço Atual: " + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);
    
    if(ObjectFind(0, "FibVisual_Price") < 0)
    {
        CreateTextLabel("FibVisual_Price", 20, 200, priceInfo, clrYellow, 10);
    }
    else
    {
        ObjectSetString(0, "FibVisual_Price", OBJPROP_TEXT, priceInfo);
    }
    
    //--- Atualizar timestamp
    string timeInfo = "🕐 " + TimeToString(TimeCurrent(), TIME_SECONDS);
    
    if(ObjectFind(0, "FibVisual_Time") < 0)
    {
        CreateTextLabel("FibVisual_Time", 20, 220, timeInfo, clrSilver, 8);
    }
    else
    {
        ObjectSetString(0, "FibVisual_Time", OBJPROP_TEXT, timeInfo);
    }
}

//+------------------------------------------------------------------+
//| Criar label de texto                                            |
//+------------------------------------------------------------------+
void CreateTextLabel(string name, int x, int y, string text, color clr, int fontSize)
{
    ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
    ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold");
}