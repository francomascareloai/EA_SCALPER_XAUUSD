//+------------------------------------------------------------------+
//|                                      EA_Fibonacci_Simple_v1.mq5 |
//|                                 Copyright 2024, TradeDev_Master |
//|                          Expert Advisor Elite Fibonacci Trading |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master 2024"
#property link      "https://github.com/tradedevmaster"
#property version   "1.0"
#property description "🚀 EA FIBONACCI SIMPLE - TESTE DE COMPILAÇÃO"

//--- Includes necessários
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Objetos de negociação
CTrade         m_trade;
CSymbolInfo    m_symbol;
CAccountInfo   m_account;

//--- Parâmetros de entrada
input double InpRiskPercent = 1.0;          // Risco por Trade (%)
input int InpMagicNumber = 123456;          // Magic Number

//--- Variáveis globais
datetime g_lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 Iniciando EA Fibonacci Simple v1.0");
    
    //--- Configurar símbolo
    if(!m_symbol.Name(_Symbol))
    {
        Print("❌ Erro ao configurar símbolo: ", _Symbol);
        return INIT_FAILED;
    }
    
    //--- Configurar negociação
    m_trade.SetExpertMagicNumber(InpMagicNumber);
    m_trade.SetMarginMode();
    m_trade.SetTypeFillingBySymbol(_Symbol);
    
    Print("✅ EA Fibonacci Simple inicializado com sucesso!");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🛑 EA Fibonacci Simple finalizado");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Verificar nova barra
    if(!IsNewBar()) return;
    
    //--- Log simples
    Print("📊 Nova barra - ", TimeToString(TimeCurrent()));
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