//+------------------------------------------------------------------+
//|                                                   test_enums.mq5 |
//+------------------------------------------------------------------+
#property copyright "Test"
#property version   "1.0"

// Enum para níveis de log
enum ENUM_LOG_LEVEL
{
   LOG_ERROR = 0,
   LOG_WARNING = 1,
   LOG_INFO = 2,
   LOG_DEBUG = 3
};

// Enum para métodos de cálculo de Stop Loss
enum ENUM_SL_CALCULATION_METHOD
{
   SL_FIXED = 0,
   SL_ATR = 1,
   SL_HYBRID = 2
};

// Enum para métodos de cálculo de Take Profit
enum ENUM_TP_CALCULATION_METHOD
{
   TP_FIXED = 0,
   TP_RR = 1,
   TP_STRUCTURE = 2
};

// Enum para métodos de Trailing Stop
enum ENUM_TRAILING_METHOD
{
   TRAILING_FIXED = 0,
   TRAILING_ATR = 1,
   TRAILING_STRUCTURE_BREAKS = 2
};

// Teste de input
input ENUM_LOG_LEVEL Log_Level = LOG_INFO;
input ENUM_SL_CALCULATION_METHOD SL_Method = SL_HYBRID;
input ENUM_TP_CALCULATION_METHOD TP_Method = TP_STRUCTURE;
input ENUM_TRAILING_METHOD Trailing_Method = TRAILING_STRUCTURE_BREAKS;

void OnTick()
{
   // Teste simples
}